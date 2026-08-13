"""SMDC super-resolution, one script per task with four sub-commands:

  demo      SMDC + IHDM reconstruction (heat prior, ||A||=1 CG data step) + Clean|LR|SMDC figure.
            Also writes clean/ and observation/ (the physical LR) shared by `baselines`/`figure`.
  baselines TV+CG (Chambolle) and DPS on the SAME operator + observations as `demo`.
  freqreg   Frequency-aware (DDRM/Wiener) data regularization sweep over (sigma_y, gamma).
  figure    Assemble the Clean|LR|DPS|TV|SMDC comparison grid from the outputs above.

The forward is A = ops.superres.SuperResolution (antialias Gaussian + avg-pool decimation); the
scale-matched target is L_t y in LR space (companion L_t = LR-grid heat blur, std sigma_t/s), from
the decimation intertwining A(K_t x) = L_t(A x). Per step (wp I + wy AᵀA) x = wp mu + wy Aᵀ(L y),
solved by CG. `freqreg` adds a DCT-diagonal prior-precision boost r(f)=gamma·wp·(1-â_n(f)) that
kills the transition-band noise ("speckle"). Shared scaffolding is in utils/pipeline.py.

  python scripts/sr.py demo      --scale 4 --n 4 --out results/sr_demo
  python scripts/sr.py baselines --scale 4 --methods tv dps --in_dir results/sr_demo
  python scripts/sr.py freqreg   --n 4 --sigmas 0 0.01 --regs 0 16 64
  python scripts/sr.py figure    --dir results/sr_demo --scale 4 --n 4
"""
import os, sys, glob, types, argparse
from pathlib import Path
from functools import partial
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ops.transforms import DCTTransform
from ops.superres import SuperResolution, lr_heat_schedule, sr_scale_matched_target
from utils import pipeline as P


# --------------------------------------------------------------------------- demo (SMDC + IHDM)
def cmd_demo(args):
    dev, dt = args.device, torch.float32
    torch.manual_seed(args.seed)
    tf = DCTTransform()
    prior, sch, H = P.build_prior(args, tf, dev)
    lr_sch = lr_heat_schedule(sch, args.scale)                  # LR-grid companion (sigma_t/s)
    N = sch.num_levels - 1
    A_raw = SuperResolution(args.scale, H, channels=3, aa_sigma=args.aa_sigma,
                            decimation=args.decimation, device=dev, dtype=dt)
    A, lam = P.normalize_operator(A_raw, H, dev, dt)            # ||A||=1

    sig_w = max(args.sigma_y, 0.01)                 # data-weight sigma (floored so sigma_y=0 -> noiseless obs, same wy)
    wp, wy0 = 1.0 / args.delta ** 2, args.data_weight / sig_w ** 2
    # frequency-aware prior-precision boost gamma*wp*(1-a_n(f)): suppresses transition-band speckle in
    # A's null space (see docs/hqs_report.tex Table tab:sr; best gamma is per-cell, 0.5 at sigma_y=0.01)
    reg = P.make_freq_reg((1.0 - A_raw.transfer_profile()).clamp_min(0.0), args.freq_reg, wp)
    times = list(range(N, -1, -1))
    for s in ("recon", "clean", "observation"):
        os.makedirs(os.path.join(args.out, s), exist_ok=True)
    print(f"SR-SMDC: H={H} scale={args.scale} aa={A_raw.aa_sigma} ||A||={lam:.3f} "
          f"sigma_y={args.sigma_y} N={N}", flush=True)

    paths = sorted(glob.glob(os.path.join(args.clean_dir, "*.png")))[:args.n]
    panels, S = [], P.new_scores("in", "out", "ssim", "lpips")
    for idx, p in enumerate(paths):
        x0 = P.load_png(p, H, dev)
        with torch.no_grad():
            y = A.forward(x0)
            if args.sigma_y > 0:
                y = y + args.sigma_y * y.abs().mean() * torch.randn_like(y)
            # bicubic baseline / LR panel from the *physical* LR (A is DC-preserving; y*lam undoes
            # the ||A||=1 rescaling so intensities match x0). SMDC loop keeps the normalized y.
            bic = F.interpolate(y * lam, size=(H, H), mode="bicubic", align_corners=False)
            # scale-matched target L_{t-1} y (LR space); optionally add the exact QMF alias
            # correction Delta = A(K x) - L(A x), plug-in x = the current prior mean mu (has high-freq).
            def target(tn, mu=None):
                return sr_scale_matched_target(A, sch, lr_sch, tn, y,
                                               x_hr=mu if args.exact_intertwine else None)
            x = P.smdc_cg(A, prior, target, times, wp, wy0, N,
                          x_init=sch.apply_K(bic, times[0]).clamp(-1, 1), reg=reg, cg_iters=args.cg_iters)
        P.save_img(os.path.join(args.out, "recon", f"{idx:05d}.png"), x)
        P.save_img(os.path.join(args.out, "clean", f"{idx:05d}.png"), x0)
        P.save_img(os.path.join(args.out, "observation", f"{idx:05d}.png"), y * lam)
        P.add_scores(S, x, x0, dev)
        S["in"].append(P.psnr(bic, x0))
        panels.append((P.to_img(x0), P.to_img(bic), P.to_img(x), S["out"][-1], S["ssim"][-1]))
        print(f"  img {idx}: bicubic {S['in'][-1]:.2f} -> SMDC {S['out'][-1]:.2f} dB  "
              f"SSIM {S['ssim'][-1]:.3f}  LPIPS {S['lpips'][-1]:.3f}", flush=True)

    # figure: rows = images, cols = Clean | LR (bicubic up) | SMDC
    cols = ["Clean (HR)", f"LR x{args.scale} (bicubic up)", "SMDC + IHDM"]
    fig, ax = plt.subplots(len(panels), 3, figsize=(9, 3 * len(panels)))
    ax = np.atleast_2d(ax)
    for r, (cl, lo, rec, ps, ss) in enumerate(panels):
        for c, im in enumerate([cl, lo, rec]):
            a = ax[r, c]
            a.imshow(im); a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(cols[c], fontsize=12)
        ax[r, 2].set_xlabel(f"{ps:.1f} dB / SSIM {ss:.2f}", fontsize=10)
    fig.suptitle(f"SMDC single-image super-resolution (x{args.scale}, sigma_y={args.sigma_y}) - FFHQ 256",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figure_sr.png"), dpi=110, bbox_inches="tight")
    m = P.mean_scores(S)
    print(f"[SR-SMDC x{args.scale}]  bicubic {m['in']:.2f} -> SMDC {m['out']:.2f} dB  "
          f"SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  -> {args.out}/figure_sr.png", flush=True)


# --------------------------------------------------------------------------- baselines (TV, DPS)
def _report(name, S, out_png):
    m = P.mean_scores(S)
    print(f"[SR {name}]  PSNR {m['out']:.2f} dB  SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  "
          f"-> {out_png}", flush=True)


def _run_tv(cleans, obs_lr, A, H, dev, args, out_dir):
    rec_dir = os.path.join(out_dir, "tv_recon"); os.makedirs(rec_dir, exist_ok=True)
    S = P.new_scores("out", "ssim", "lpips")
    for i, (x0, y) in enumerate(zip(cleans, obs_lr)):
        sig2 = (args.sigma_y * y.abs().mean()) ** 2                      # matches demo's relative noise
        x_init = F.interpolate(y, size=(H, H), mode="bicubic", align_corners=False)
        x = P.tv_cg_restore(A, y, sig2, args.beta, args.rho0, args.gamma, args.T, args.cg_iters, dev,
                            x_init=x_init)
        P.save_img(os.path.join(rec_dir, f"{i:05d}.png"), x)
        P.add_scores(S, x, x0, dev)
        print(f"  TV  img {i}: {S['out'][-1]:.2f} dB  SSIM {S['ssim'][-1]:.3f}", flush=True)
    _report(f"TV+CG beta={args.beta}", S, rec_dir)


def _run_dps(cleans, obs_lr, scale, dev, args, out_dir):
    import yaml
    REPO = Path(__file__).resolve().parent.parent
    DPS_ROOT = REPO / "model" / "dps_backbone"
    sys.path.insert(0, str(DPS_ROOT))
    mb = types.ModuleType("motionblur"); mbm = types.ModuleType("motionblur.motionblur")
    mbm.Kernel = type("K", (), {}); mb.motionblur = mbm
    sys.modules.setdefault("motionblur", mb); sys.modules.setdefault("motionblur.motionblur", mbm)
    from guided_diffusion.condition_methods import get_conditioning_method
    from guided_diffusion.gaussian_diffusion import create_sampler
    from guided_diffusion.unet import create_model

    mc = yaml.safe_load(open(str(DPS_ROOT / "configs/model_config.yaml")))
    mc["model_path"] = str(DPS_ROOT / "models" / "ffhq_10m.pt")
    H = int(mc["image_size"])
    model = create_model(**mc).to(dev).eval()
    A = SuperResolution(scale, H, channels=3, device=dev, dtype=torch.float32)   # SAME operator

    class Operator:                                                              # DPS forward/transpose API
        __name__ = "sr"
        def forward(self, data, **kw): return A.forward(data)
        def transpose(self, data, **kw): return A.adjoint(data)
    class GaussianNoiser: __name__ = "gaussian"

    cond = get_conditioning_method("ps", Operator(), GaussianNoiser(), scale=args.dps_scale)
    sampler = create_sampler(**yaml.safe_load(open(str(DPS_ROOT / "configs/diffusion_config.yaml"))))
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=cond.conditioning)

    rec_dir = os.path.join(out_dir, "dps_recon"); os.makedirs(rec_dir, exist_ok=True)
    S = P.new_scores("out", "ssim", "lpips")
    for i, (x0, y) in enumerate(zip(cleans, obs_lr)):
        x_start = torch.randn(1, 3, H, H, device=dev).requires_grad_()
        rec = sample_fn(x_start=x_start, measurement=y, record=False, save_root=str(rec_dir))
        rec = rec.detach().clamp(-1, 1)
        P.save_img(os.path.join(rec_dir, f"{i:05d}.png"), rec)
        P.add_scores(S, rec, x0, dev)
        print(f"  DPS img {i}: {S['out'][-1]:.2f} dB  SSIM {S['ssim'][-1]:.3f}", flush=True)
    _report(f"DPS scale={args.dps_scale}", S, rec_dir)


def cmd_baselines(args):
    dev, H, s = args.device, args.image_size, args.scale
    cpaths = sorted(glob.glob(os.path.join(args.in_dir, "clean", "*.png")))[:args.n]
    opaths = sorted(glob.glob(os.path.join(args.in_dir, "observation", "*.png")))[:args.n]
    assert cpaths and len(cpaths) == len(opaths), f"need matching clean/observation in {args.in_dir}"
    cleans = [P.load_png(p, H, dev) for p in cpaths]
    obs_lr = [P.load_png(p, H // s, dev) for p in opaths]                # physical LR at H/s
    print(f"SR baselines: n={len(cleans)} scale={s} obs={tuple(obs_lr[0].shape[-2:])} "
          f"methods={args.methods}", flush=True)

    A = SuperResolution(s, H, channels=3, device=dev, dtype=torch.float32)
    if "tv" in args.methods:
        _run_tv(cleans, obs_lr, A, H, dev, args, args.out)
    if "dps" in args.methods:
        _run_dps(cleans, obs_lr, s, dev, args, args.out)


# --------------------------------------------------------------------------- freqreg sweep
def cmd_freqreg(args):
    dev, dt, s = args.device, torch.float32, args.scale
    tf = DCTTransform()
    prior, sch, H = P.build_prior(args, tf, dev)
    lr_sch = lr_heat_schedule(sch, s); N = sch.num_levels - 1
    A_raw = SuperResolution(s, H, channels=3, aa_sigma=args.aa_sigma,
                            decimation=args.decimation, device=dev, dtype=dt)
    A, lam = P.normalize_operator(A_raw, H, dev, dt)

    one_minus_an = (1.0 - A_raw.transfer_profile()).clamp_min(0.0)
    wp = 1.0 / args.delta ** 2
    times = list(range(N, -1, -1))
    x0s = [P.load_png(p, H, dev) for p in sorted(glob.glob(os.path.join(args.clean_dir, "*.png")))[:args.n]]
    os.makedirs(args.out, exist_ok=True)
    print(f"SR freq-reg sweep: n={len(x0s)} H={H} scale={s} ||A||={lam:.3f}", flush=True)
    print(f"\n{'sigma_y':>7} {'freq_reg':>8} | {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6}")
    print("-" * 46)

    for sy in args.sigmas:
        # --abs_noise: sy is an absolute Gaussian std on the physical [-1,1] LR (DDRM/DPS
        # convention). The loop works on the ||A||=1-normalized measurement, so that same noise is
        # (sy/lam)*randn there and the calibrated data-term noise level is sig_w = sy/lam. Without
        # the flag sy is our relative std (sy*mean|y|), sig_w = sy (floored at 0.01).
        sig_w = max(sy / lam, 1e-4) if args.abs_noise else max(sy, 0.01)
        wy0 = args.data_weight / sig_w ** 2
        for gamma in args.regs:
            reg = P.make_freq_reg(one_minus_an, gamma, wp)
            S = P.new_scores("out", "ssim", "lpips")
            for k, x0 in enumerate(x0s):
                torch.manual_seed(args.seed + k)                   # same noise draw across gamma
                with torch.no_grad():
                    y = A.forward(x0)
                    if sy > 0:
                        y = y + (sy / lam if args.abs_noise else sy * y.abs().mean()) * torch.randn_like(y)
                    x_init = sch.apply_K(F.interpolate(y * lam, size=(H, H), mode="bicubic",
                                                       align_corners=False), times[0]).clamp(-1, 1)
                    x = P.smdc_cg(A, prior, lambda tn, mu: lr_sch.apply_K(y, tn), times, wp, wy0, N,
                                  x_init=x_init, reg=reg, cg_iters=args.cg_iters)
                P.add_scores(S, x, x0, dev)
                d = os.path.join(args.out, f"sy{sy:g}_reg{gamma:g}", "recon"); os.makedirs(d, exist_ok=True)
                P.save_img(os.path.join(d, f"{k:05d}.png"), x)
            m = P.mean_scores(S)
            print(f"{sy:>7.3g} {gamma:>8.3g} | {m['out']:>6.2f} {m['ssim']:>6.3f} {m['lpips']:>6.3f}", flush=True)


# --------------------------------------------------------------------------- comparison figure
def cmd_figure(args):
    H, s = args.image_size, args.scale
    recons = [("DPS", "dps_recon"), ("TV+CG", "tv_recon"), ("SMDC+IHDM", "recon")]
    cols = ["Clean (HR)", f"LR x{s} (bicubic up)"] + [n for n, _ in recons]

    cpaths = sorted(glob.glob(os.path.join(args.dir, "clean", "*.png")))[:args.n]
    rows = []
    for i, cp in enumerate(cpaths):
        x0 = P.load_png(cp, H, "cpu")
        lr = P.load_png(os.path.join(args.dir, "observation", f"{i:05d}.png"), H // s, "cpu")
        bic = F.interpolate(lr, size=(H, H), mode="bicubic", align_corners=False)
        panels = [(P.to_img(x0), None), (P.to_img(bic), None)]
        for _, sub in recons:
            r = P.load_png(os.path.join(args.dir, sub, f"{i:05d}.png"), H, "cpu")
            panels.append((P.to_img(r), (P.psnr(r, x0), P.ssim(r, x0))))
        rows.append(panels)

    fig, ax = plt.subplots(len(rows), len(cols), figsize=(3 * len(cols), 3 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, panels in enumerate(rows):
        for c, (im, sc) in enumerate(panels):
            a = ax[r, c]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(cols[c], fontsize=12)
            if sc is not None:
                a.set_xlabel(f"{sc[0]:.1f} dB / SSIM {sc[1]:.2f}", fontsize=10)
    fig.suptitle(f"Super-resolution x{s} on FFHQ-256 - method comparison (same operator + observation)",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(args.dir, "figure_sr_compare.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}", flush=True)


# --------------------------------------------------------------------------- CLI
def _add_common(p):
    p.add_argument("--scale", type=int, default=4, help="downsampling factor s")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--device", default="cuda:0")


def _add_sr_operator(p):
    p.add_argument("--aa_sigma", type=float, default=None,
                   help="antialias std (HR px); default=scale. Set 0 for pure (box avg-pool) downsample.")
    p.add_argument("--decimation", choices=["avgpool", "stride"], default="avgpool")


def _add_prior(p):
    p.add_argument("--prior", choices=["ihdm", "cold_diffusion"], default="ihdm")
    p.add_argument("--ckpt", default="checkpoint/ihdm/ihdm_ffhq256_full.pth")
    p.add_argument("--ihdm_config", default="img_size_256_full")
    p.add_argument("--clean_dir", default="results/gaussian/clean")
    p.add_argument("--cg_iters", type=int, default=12)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--data_weight", type=float, default=64.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("demo", help="SMDC + IHDM reconstruction + figure (+ shared clean/observation)")
    _add_common(pd); _add_prior(pd); _add_sr_operator(pd)
    pd.add_argument("--exact_intertwine", action="store_true",
                    help="add the QMF alias correction to the data target (plug-in: current prior mean)")
    pd.add_argument("--sigma_y", type=float, default=0.01, help="LR noise std (relative, x |y|.mean())")
    pd.add_argument("--freq_reg", type=float, default=0.0,
                    help="frequency-aware prior-precision boost gamma (0 = plain data step; best per cell "
                         "in Table tab:sr, e.g. 0.5 for the anti-alias operator at sigma_y=0.01)")
    pd.add_argument("--out", default="results/sr_demo")
    pd.add_argument("--seed", type=int, default=0, help="fixes the LR noise so baselines share the obs")
    pd.set_defaults(func=cmd_demo)

    pb = sub.add_parser("baselines", help="TV+CG and DPS on the same operator + observations")
    _add_common(pb)
    pb.add_argument("--methods", nargs="+", default=["tv", "dps"], choices=["tv", "dps"])
    pb.add_argument("--in_dir", default="results/sr_demo", help="has clean/ and observation/ from demo")
    pb.add_argument("--out", default="results/sr_demo")
    pb.add_argument("--image_size", type=int, default=256)
    pb.add_argument("--sigma_y", type=float, default=0.01)
    pb.add_argument("--beta", type=float, default=2.0)          # TV knobs
    pb.add_argument("--rho0", type=float, default=2.0)
    pb.add_argument("--gamma", type=float, default=1.2)
    pb.add_argument("--T", type=int, default=40)
    pb.add_argument("--cg_iters", type=int, default=12)
    pb.add_argument("--dps_scale", type=float, default=0.3)     # DPS knob
    pb.set_defaults(func=cmd_baselines)

    pf = sub.add_parser("freqreg", help="frequency-aware data-regularization sweep over (sigma_y, gamma)")
    _add_common(pf); _add_prior(pf)
    pf.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.01])
    pf.add_argument("--abs_noise", action="store_true",
                    help="interpret --sigmas as absolute Gaussian std on the physical [-1,1] LR "
                         "(DDRM/DPS convention) instead of our relative std (sigma*mean|y|)")
    pf.add_argument("--regs", type=float, nargs="+", default=[0.0, 16.0, 64.0])
    _add_sr_operator(pf)
    pf.add_argument("--out", default="results/sr_freqreg")
    pf.add_argument("--seed", type=int, default=0)
    pf.set_defaults(func=cmd_freqreg)

    pg = sub.add_parser("figure", help="assemble Clean|LR|DPS|TV|SMDC comparison grid")
    pg.add_argument("--dir", default="results/sr_demo")
    pg.add_argument("--scale", type=int, default=4)
    pg.add_argument("--n", type=int, default=4)
    pg.add_argument("--image_size", type=int, default=256)
    pg.set_defaults(func=cmd_figure)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
