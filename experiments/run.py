"""Command generator/runner derived from the registry: expanding n never means re-deriving a recipe.

    python -m experiments.run --n 1000            # print the full plan (obs + every cell x method)
    python -m experiments.run --n 1000 --gpus 0,1,2,3 --exec   # run it, sharded across GPUs

Every command below is BUILT FROM registry.py -- change a recipe there, not here. Commands are grouped
so you can eyeball the plan and the guard (`experiments.score`) catches any recipe drift after running.
DPS is emitted at n=min(100,n) (subset). External baselines run from their own backbone dir.
"""
import argparse, itertools, os, subprocess, sys
from . import registry as R
SC = "/tmp/claude-1002/-home-linw-diffusion/ff682d3d-9c94-4e49-b113-d4d50b4cdf86/scratchpad"  # matched kernels


def obs_cmds(n):
    c = []
    for tag, sy in R.DEBLUR_NOISE.items():
        c.append(("gaussian", f"python scripts/deblur.py obs --operator gaussian --source celebahq "
                  f"--image_size 256 --blur_sigma 4 --noise {sy} --n {n} --out {R.ROOT}/gaussian_{tag}"))
    shared = f"{R.ROOT}/gaussian_s05/clean"
    for op, extra in [("motion", "--kernel_npy results/motion_reflect/kernel.npy"), ("defocus", "--radius 7")]:
        for tag, sy in R.DEBLUR_NOISE.items():
            c.append((op, f"python scripts/deblur.py obs --operator {op} --clean {shared} {extra} "
                      f"--noise {sy} --n {n} --out {R.ROOT}/{op}_{tag}"))
    return c


def method_cmds(n):
    """Yield (label, cmd). One place, all recipes."""
    CK = "checkpoint/ihdm/ihdm_ffhq256_full.pth"; CFG = "img_size_256_full"
    COLD = "checkpoint/cold_diffusion/ffhq256.pth"; MK = "results/motion_reflect/kernel.npy"
    out = []
    for name, op, tag, sy in R.deblur_cells():
        cell = f"{R.ROOT}/{name}"; io = f"--clean_dir {cell}/clean --observation_dir {cell}/observation"
        g = R.IHDM_GAMMA_DEBLUR[op][tag]
        # IHDM
        base = (f"python scripts/deblur.py restore --operator {op} --prior ihdm --ihdm_config {CFG} "
                f"--ckpt {CK} --image_size 256 --n {n} {io} --sigma_y {sy} --freq_reg {g} --out {cell}/ihdm")
        out.append((f"IHDM {name}", base + (" --blur_sigma 4" if op == "gaussian" else
                    f" --kernel_npy {MK} --cg_iters 12" if op == "motion" else f" --kernel_npy {cell}/kernel.npy")))
        # cold (freq_reg 0)
        cb = (f"python scripts/deblur.py restore --operator {op} --prior cold_diffusion --ckpt {COLD} "
              f"--image_size 256 --ch 128 --ch_mult 1 1 2 2 4 --num_res_blocks 2 --n {n} {io} "
              f"--sigma_y {sy} --freq_reg 0 --out {cell}/cold")
        out.append((f"cold {name}", cb + (" --blur_sigma 4" if op == "gaussian" else
                    f" --kernel_npy {MK} --cg_iters 12" if op == "motion" else f" --kernel_npy {cell}/kernel.npy")))
        # TV (per-operator script + beta)
        tv = R.TV_RECIPE[op]; beta = tv["beta"][tag]
        k = MK if op == "motion" else (f"{cell}/kernel.npy" if op == "defocus" else None)
        if tv["script"] == "run_tv_cg":
            out.append((f"TV {name}", f"python scripts/run_tv_cg.py --kernel_npy {k} {io} --beta {beta} --sigma_y {sy} --n {n} --out {cell}/tv"))
        else:
            kk = f"--kernel_npy {k}" if k else ""
            out.append((f"TV {name}", f"python scripts/run_tv_hqs.py --operator {tv['op']} {kk} {tv['extra']} --beta {beta} {io} --sigma_y {sy} --n {n} --out {cell}/tv"))
        # DPS (subset n=100)
        ndps = min(R.METHODS["DPS"]["subset"], n)
        dop = "heat --degradation_sigma 4" if op == "gaussian" else f"motion --kernel_npy {MK if op=='motion' else cell+'/kernel.npy'}"
        out.append((f"DPS {name}", f"python scripts/run_dps.py --operator {dop} --clean_dir {cell}/clean "
                    f"--observation_dir {cell}/observation --num_images {ndps} --scale 0.3 --gpu 0 --save_dir {cell}/dps"))
        # DDRM (gaussian only, deblur_aa, timesteps 20)
        if op in R.DDRM_DEG and op == "gaussian":
            nm = {"s05": "ddrm_gaa05", "s10": "ddrm_gaa10", "s20": "ddrm_gaa20"}[tag]
            out.append((f"DDRM {name}", f"cd model/ddrm_backbone && python main.py --ni --config ffhq_256.yml "
                        f"--doc ffhq --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_aa --sigma_0 {R.half(sy)} -i {nm}"))
        # DiffPIR
        out.append((f"DiffPIR {name}", f"cd model/DiffPIR_backbone && DIFFPIR_TESTSET=our{n} DIFFPIR_NOISE={R.half(sy)} "
                    f"DIFFPIR_DIYK=0 DIFFPIR_ITER=100 DIFFPIR_BLURMODE={ {'gaussian':'gauss','motion':'motion','defocus':'defocus'}[op] } "
                    f"DIFFPIR_KERNEL_NPY={_dp_kernel(op)} python main_ddpir_deblur.py"))
    # SR handled by sr.py freqreg (IHDM+obs) + externals; see registry for gamma/lambda/deg
    for name, kind, tag, sy in R.sr_cells():
        g = R.IHDM_GAMMA_SR[kind][tag]; aa = "--aa_sigma 0" if kind == "box" else ""
        out.append((f"IHDM+obs {name}", f"python scripts/sr.py freqreg --scale 4 {aa} --decimation avgpool "
                    f"--abs_noise --data_weight 64 --sigmas {sy} --regs {g} --n {n} --clean_dir {R.SHARED_CLEAN} --out {R.ROOT}/{name}"))
        deg = R.DDRM_DEG[kind]; nm = f"ddrm_{kind}{'01' if sy==0.01 else '05'}"
        out.append((f"DDRM {name}", f"cd model/ddrm_backbone && python main.py --ni --config ffhq_256.yml --doc ffhq "
                    f"--timesteps 20 --eta 0.85 --etaB 1 --deg {deg} --sigma_0 {R.half(sy)} -i {nm}"))
        if kind == "box":
            nm = f"ddnm_box{'01' if sy==0.01 else '05'}"
            out.append((f"DDNM {name}", f"cd model/ddnm_backbone && python main.py --config ffhq_gd.yml --deg sr_averagepooling "
                        f"--deg_scale 4 --sigma_y {R.half(sy)} --add_noise --ni --path_y our{n}_deblur --eta 0.85 -i {nm} --exp exp"))
        out.append((f"DiffPIR {name}", f"cd model/DiffPIR_backbone && DIFFPIR_TESTSET=our{n} DIFFPIR_NOISE={R.half(sy)} "
                    f"DIFFPIR_ITER=100 DIFFPIR_CLASSICAL=1 DIFFPIR_LAMBDA_MULT=5 DIFFPIR_SR_KERNEL_NPY={_dp_kernel(kind)} "
                    f"DIFFPIR_TAG={kind} python main_ddpir_sisr.py"))
    return out


def _dp_kernel(op):
    return {"gaussian": f"{SC}/gauss_sigma4.npy", "motion": "results/motion_reflect/kernel.npy",
            "defocus": "results200/defocus_s05/kernel.npy", "box": f"{SC}/box4.npy", "aa": f"{SC}/aa4.npy"}[op]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--exec", action="store_true", help="actually run (else dry-print the plan)")
    args = ap.parse_args()
    print(f"# ---- obs (n={args.n}) ----")
    for op, c in obs_cmds(args.n): print(c)
    print(f"\n# ---- methods (n={args.n}; DPS n={min(100,args.n)}) ----  shard across GPUs {args.gpus}")
    for label, c in method_cmds(args.n): print(f"# {label}\n{c}")
    if args.exec:
        print("\n# --exec: run externals' dataset staging first (testsets/our{n}, ood_celeba, our{n}_deblur), "
              "then execute the above sharded one-job-per-GPU. Kept manual on purpose -- see docs/PITFALLS.md.")


if __name__ == "__main__":
    main()
