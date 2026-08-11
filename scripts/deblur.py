"""SMDC deblurring, one script for the whole blur family with the degradation chosen by
`--operator {gaussian, motion, defocus}` and the action by the sub-command:

  obs      Make clean/ + observation/ pairs for the operator (gaussian: DCT-heat blur of a held-out
           dataset; motion: reflect-boundary Levin-kernel conv; defocus: reflect-boundary disk conv).
  restore  SMDC + IHDM/cold reconstruction. gaussian & defocus are DCT-diagonal -> closed-form
           per-step MAP (solver/hqs.py, no CG); motion is reflect-spatial -> per-step spatial CG.
  freqreg  Frequency-aware (DDRM/Wiener) data-regularization sweep over gamma (per-step CG, any op).
  noise    Post-data-consistency noise-injection sweep (gaussian; deterministic vs stochastic reverse).

Scale-matched target is K_{t-1} y (the DCT-heat companion L=K for every plain convolution here).
Shared scaffolding (IO, prior, ||A||=1, the CG loop, freq-reg, metrics) is in utils/pipeline.py.
Baselines live in scripts/run_tv_hqs.py, run_tv_cg.py, run_dps.py (operator-parameterized).

  python scripts/deblur.py obs     --operator defocus --clean results/gaussian/clean --radius 7 --out results/defocus
  python scripts/deblur.py restore --operator gaussian --clean_dir results/gaussian/clean \
     --observation_dir results/gaussian/observation --sigma_y 0.05 --out results/gaussian/ihdm_hqs
  python scripts/deblur.py restore --operator motion --ckpt checkpoint/ihdm/ihdm_ffhq256_full.pth \
     --clean_dir results/motion_reflect/clean --observation_dir results/motion_reflect/observation \
     --kernel_npy results/motion_reflect/kernel.npy --sigma_y 0.05 --out results/motion_reflect/ihdm_cg
  python scripts/deblur.py freqreg --operators gaussian motion defocus --regs 0 0.5 --n 4
  python scripts/deblur.py noise   --scales 0 0.01 0.02 0.05 --kind anneal --n 8
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from PIL import Image
from scipy.signal import convolve2d
from ops.transforms import DCTTransform
from ops.deblur import build_deblur, gaussian_blur_transfer
from ops.spectral import SpectralOperator
from ops.motion_spatial import SpatialMotionBlur
from solver.init import terminal_init
from solver.hqs import MAPCorrection
from solver.base import scale_matched_solver
from utils.seed import set_seed
from utils.logging import RunLogger
from utils import pipeline as P


# ============================================================================ observation generators
def disk_psf(radius, oversample=8):
    """Uniform disk (pillbox) of `radius` px, anti-aliased by `oversample`x supersampling;
    sum-normalized (2r+1)x(2r+1)."""
    n = 2 * radius + 1
    hi = n * oversample
    ys, xs = np.mgrid[0:hi, 0:hi].astype(np.float64)
    c = (hi - 1) / 2.0
    inside = ((xs - c) ** 2 + (ys - c) ** 2) <= (radius * oversample) ** 2
    k = inside.reshape(n, oversample, n, oversample).mean(axis=(1, 3))
    return (k / k.sum()).astype(np.float32)


def reflect_blur(xi, k):
    """xi:(1,3,H,W) in [-1,1], k:(h,w) sum-normalized -> reflect-boundary 'same' convolution."""
    img = xi[0].numpy()
    out = np.stack([convolve2d(img[c], k, mode="same", boundary="symm") for c in range(3)], 0)
    return torch.from_numpy(out.astype(np.float32))[None]


def _load_clean(d):
    xs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)).permute(2, 0, 1)
          / 127.5 - 1.0 for p in sorted(glob.glob(os.path.join(d, "*.png")))]
    return torch.stack(xs)


def cmd_obs(args):
    set_seed(args.seed)
    cdir = os.path.join(args.out, "clean"); odir = os.path.join(args.out, "observation")
    os.makedirs(cdir, exist_ok=True); os.makedirs(odir, exist_ok=True)

    if args.operator == "gaussian":
        H = args.image_size
        if args.source == "celebahq":
            from huggingface_hub import hf_hub_download
            from data.loaders import load_parquet_faces
            p = hf_hub_download(repo_id="korexyz/celeba-hq-256x256", repo_type="dataset",
                                filename="data/validation-00000-of-00001.parquet",
                                local_dir=os.path.join(args.out, "hf_cache"))
            x0 = load_parquet_faces([p], image_size=H, limit=args.n).float().div(127.5).sub(1.0)
        else:
            t = torch.load(args.cache, map_location="cpu", mmap=True)
            x0 = t[-args.n:].float().div(127.5).sub(1.0)
        A, _ = build_deblur(H, H, args.blur_sigma, transform=DCTTransform(), device="cpu", dtype=torch.float32)
        for i in range(x0.shape[0]):
            xi = x0[i:i + 1]
            y = (A.forward(xi) + args.noise * torch.randn_like(xi)).clamp(-1, 1)
            P.save_u8(os.path.join(cdir, f"{i:05d}.png"), xi[0])
            P.save_u8(os.path.join(odir, f"{i:05d}.png"), y[0])
        print(f"wrote {x0.shape[0]} gaussian pairs (blur_sigma={args.blur_sigma}, noise={args.noise}) "
              f"-> {args.out}/", flush=True)
        return

    # motion / defocus: reflect-boundary convolution of an existing clean set
    x0 = _load_clean(args.clean)
    if args.operator == "motion":
        k = np.load(args.kernel_npy).astype(np.float32); k = k / k.sum()
        tag = f"motion (Levin {k.shape[0]}x{k.shape[1]})"
    else:
        k = disk_psf(args.radius); tag = f"defocus (disk r={args.radius}, {k.shape[0]}x{k.shape[1]})"
    np.save(os.path.join(args.out, "kernel.npy"), k)
    for i in range(x0.shape[0]):
        xi = x0[i:i + 1]
        y = (reflect_blur(xi, k) + args.noise * torch.randn_like(xi)).clamp(-1, 1)
        P.save_u8(os.path.join(cdir, f"{i:05d}.png"), xi[0])
        P.save_u8(os.path.join(odir, f"{i:05d}.png"), y[0])
    print(f"wrote {x0.shape[0]} REFLECT-boundary {tag} pairs (noise={args.noise}) -> {args.out}/", flush=True)


# ============================================================================ operator builders
def build_operator(operator, H, tf, dev, dt, blur_sigma=4.0, kernel_npy=None):
    """Return (A, an) where A is the forward operator and `an` its normalized DCT transfer (â_n,
    1 in passband -> 0 in the weak/null band) used by freqreg. Closed-form ops also carry the
    DCT transfer needed by MAPCorrection as A._dct_transfer."""
    if operator == "gaussian":
        A, gA = build_deblur(H, H, blur_sigma, transform=tf, device=dev, dtype=dt)
        an = gaussian_blur_transfer(H, H, blur_sigma, dev, dt)
        A._dct_transfer = gA
        return A, an
    if operator == "defocus":
        k = torch.from_numpy(np.load(kernel_npy))
        a_hat = tf.fwd(SpatialMotionBlur(k, 3, device=dev, dtype=dt).forward(
            tf.inv(torch.ones(1, 3, H, H, device=dev))))[0, 0]
        A = SpectralOperator(a_hat, tf)
        A._dct_transfer = a_hat
        an = a_hat.abs() / a_hat.abs().reshape(-1)[0].clamp_min(1e-12)
        return A, an
    # motion: reflect-spatial, not DCT-diagonal
    k = torch.from_numpy(np.load(kernel_npy))
    A = SpatialMotionBlur(k, channels=3, device=dev, dtype=dt)
    an = P.hutchinson_transfer(A, H, 3, dev, dt)
    return A, an


# ============================================================================ restore
def cmd_restore(args):
    dev, dt = args.device, torch.float32
    tf = DCTTransform()
    os.makedirs(os.path.join(args.out, "recon"), exist_ok=True)
    prior, sch, H = P.build_prior(args, tf, dev)
    times = list(range(sch.num_levels - 1, -1, -1))
    A, an = build_operator(args.operator, H, tf, dev, dt, args.blur_sigma, args.kernel_npy)
    x0 = P.load_stack(args.clean_dir, H, args.n, dev)
    ys = P.load_stack(args.observation_dir, H, args.n, dev)

    if args.operator == "motion":                                   # per-step spatial CG (no FFT)
        wp = args.prior_weight / args.delta ** 2
        wy0 = args.data_weight / args.sigma_y ** 2
        N = sch.num_levels - 1
        reg = P.make_freq_reg((1.0 - an).clamp_min(0.0), args.freq_reg, wp)   # freq-aware CG reg (gamma)
        print(f"  MOTION-CG prior={args.prior} res={H} K={N} cg_iters={args.cg_iters} "
              f"freq_reg={args.freq_reg} (spatial reflect A, exact adjoint, no FFT)", flush=True)
        S = P.new_scores("in", "out", "ssim", "lpips", "mc")
        for k in range(x0.shape[0]):
            xi, y = x0[k:k + 1], ys[k:k + 1]
            with torch.no_grad():
                x = P.smdc_cg(A, prior, lambda tn, mu: sch.apply_K(y, tn), times, wp, wy0, N,
                              x_init=sch.apply_K(y, times[0]).clamp(-1, 1), reg=reg, cg_iters=args.cg_iters)
            P.save_img(os.path.join(args.out, "recon", f"{k:05d}.png"), x)
            P.add_scores(S, x, xi, dev, y=y, A=A)
        m = P.mean_scores(S)
        print(f"[motion-CG {args.prior}]  PSNR {m['in']:.2f}->{m['out']:.2f}dB  SSIM {m['ssim']:.3f}  "
              f"LPIPS {m['lpips']:.3f}  MC {m['mc']:.3f}", flush=True)
        return

    # gaussian / defocus: closed-form per-step MAP (DCT-Wiener), with frequency-aware reg built in
    corr = MAPCorrection(sch, A._dct_transfer, delta=args.delta, sigma_y=args.sigma_y,
                         prior_weight=args.prior_weight, data_weight=args.data_weight,
                         schedule_kind=args.map_schedule, freq_reg=args.freq_reg)
    print(f"  {args.operator.upper()}-HQS(closed form) prior={args.prior} res={H} freq_reg={args.freq_reg}",
          flush=True)
    S = P.new_scores("in", "out", "ssim", "lpips", "mc")
    for k in range(x0.shape[0]):
        xi, y = x0[k:k + 1], ys[k:k + 1]
        x_init = terminal_init("matched_measurement", y, sch, times[0])
        x = scale_matched_solver(y, x_init, times, A, sch, prior, corr, clamp=(-1, 1),
                                 logger=RunLogger(verbose=False), x0_ref=xi,
                                 noise_scale=args.noise_scale, noise_kind=args.noise_kind)
        P.save_img(os.path.join(args.out, "recon", f"{k:05d}.png"), x)
        P.add_scores(S, x, xi, dev, y=y, A=A)
    m = P.mean_scores(S)
    print(f"[{args.operator}-HQS {args.prior}]  blur={args.blur_sigma}  PSNR {m['in']:.2f}->{m['out']:.2f}dB  "
          f"SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  MC {m['mc']:.3f}", flush=True)


# ============================================================================ freqreg sweep
def cmd_freqreg(args):
    dev, dt = args.device, torch.float32
    tf = DCTTransform()
    prior, sch, H = P.build_prior(args, tf, dev)
    N = sch.num_levels - 1
    wp, wy0 = 1.0 / args.delta ** 2, args.data_weight / args.sigma_y ** 2
    times = list(range(N, -1, -1))
    print(f"deblur freq-reg sweep: n={args.n} H={H} sigma_y={args.sigma_y}", flush=True)
    print(f"\n{'operator':>9} {'freq_reg':>8} | {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6} {'MC':>6}")
    print("-" * 52)

    for opname in args.operators:
        kernel = os.path.join(getattr(args, f"{opname}_dir"), "kernel.npy") if opname != "gaussian" else None
        A, an = build_operator(opname, H, tf, dev, dt, args.blur_sigma, kernel)
        one_minus_an = (1.0 - an).clamp_min(0.0)
        src = getattr(args, f"{opname}_dir")
        cleans = P.load_stack(os.path.join(src, "clean"), H, args.n, dev)
        obses = P.load_stack(os.path.join(src, "observation"), H, args.n, dev)
        for gamma in args.regs:
            reg = P.make_freq_reg(one_minus_an, gamma, wp)
            S = P.new_scores("out", "ssim", "lpips", "mc")
            for j in range(cleans.shape[0]):
                x0, y = cleans[j:j + 1], obses[j:j + 1]
                with torch.no_grad():
                    x = P.smdc_cg(A, prior, lambda tn, mu: sch.apply_K(y, tn), times, wp, wy0, N,
                                  x_init=sch.apply_K(y, times[0]).clamp(-1, 1), reg=reg, cg_iters=args.cg_iters)
                P.add_scores(S, x, x0, dev, y=y, A=A)
                if j == 0:
                    d = os.path.join(args.out, f"{opname}_reg{gamma:g}"); os.makedirs(d, exist_ok=True)
                    P.save_img(os.path.join(d, "00000.png"), x)
            m = P.mean_scores(S)
            print(f"{opname:>9} {gamma:>8.3g} | {m['out']:>6.2f} {m['ssim']:>6.3f} {m['lpips']:>6.3f} "
                  f"{m['mc']:>6.3f}", flush=True)


# ============================================================================ noise-injection sweep
def cmd_noise(args):
    dev, dt = args.device, torch.float32
    tf = DCTTransform()
    prior, sch, H = P.build_prior(args, tf, dev)
    A, _ = build_operator("gaussian", H, tf, dev, dt, args.blur_sigma)
    times = list(range(sch.num_levels - 1, -1, -1))
    corr = MAPCorrection(sch, A._dct_transfer, delta=args.delta, sigma_y=args.sigma_y,
                         prior_weight=1.0, data_weight=args.data_weight, schedule_kind="late")
    x0 = P.load_stack(args.clean_dir, H, args.n, dev)
    ys = P.load_stack(args.observation_dir, H, args.n, dev)
    print(f"deblur noise sweep: n={x0.shape[0]} H={H} blur={args.blur_sigma} sigma_y={args.sigma_y} "
          f"kind={args.kind}", flush=True)
    print(f"\n{'noise':>7} | {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6} {'MC':>6}")
    print("-" * 42)
    for ns in args.scales:
        S = P.new_scores("out", "ssim", "lpips", "mc")
        for k in range(x0.shape[0]):
            xi, y = x0[k:k + 1], ys[k:k + 1]
            rng = torch.Generator(device=dev).manual_seed(args.seed + k)   # same noise draw across scales
            x_init = terminal_init("matched_measurement", y, sch, times[0])
            x = scale_matched_solver(y, x_init, times, A, sch, prior, corr, clamp=(-1, 1),
                                     logger=RunLogger(verbose=False), rng=rng,
                                     noise_scale=ns, noise_kind=args.kind)
            P.add_scores(S, x, xi, dev, y=y, A=A)
            if k == 0:
                d = os.path.join(args.out, f"ns{ns:g}"); os.makedirs(d, exist_ok=True)
                P.save_img(os.path.join(d, f"{k:05d}.png"), x)
        m = P.mean_scores(S)
        print(f"{ns:>7.3g} | {m['out']:>6.2f} {m['ssim']:>6.3f} {m['lpips']:>6.3f} {m['mc']:>6.3f}", flush=True)


# ============================================================================ CLI
def _add_prior(p):
    p.add_argument("--prior", choices=["ihdm", "cold_diffusion"], default="ihdm")
    p.add_argument("--ckpt", default="checkpoint/ihdm/ihdm_ffhq256_full.pth")
    p.add_argument("--ihdm_config", default="img_size_256_full")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--ch", type=int, default=128)
    p.add_argument("--ch_mult", type=int, nargs="+", default=[1, 1, 2, 2, 4])
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--attn_res", type=int, default=16)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--sigma_min", type=float, default=0.5)
    p.add_argument("--sigma_max", type=float, default=128.0)
    p.add_argument("--device", default="cuda:0")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    po = sub.add_parser("obs", help="generate clean/ + observation/ pairs")
    po.add_argument("--operator", choices=["gaussian", "motion", "defocus"], required=True)
    po.add_argument("--out", required=True)
    po.add_argument("--noise", type=float, default=0.05)
    po.add_argument("--seed", type=int, default=0)
    po.add_argument("--blur_sigma", type=float, default=4.0)                  # gaussian
    po.add_argument("--image_size", type=int, default=256)
    po.add_argument("--source", choices=["ffhq_pt", "celebahq"], default="celebahq")
    po.add_argument("--cache", default="data/ffhq256/ffhq256_uint8.pt")
    po.add_argument("--n", type=int, default=8)
    po.add_argument("--clean", default="results/gaussian/clean")             # motion/defocus source
    po.add_argument("--kernel_npy", default="results/motion/kernel.npy")     # motion
    po.add_argument("--radius", type=int, default=7)                         # defocus
    po.set_defaults(func=cmd_obs)

    pr = sub.add_parser("restore", help="SMDC + IHDM/cold reconstruction")
    _add_prior(pr)
    pr.add_argument("--operator", choices=["gaussian", "motion", "defocus"], default="gaussian")
    pr.add_argument("--clean_dir", required=True)
    pr.add_argument("--observation_dir", required=True)
    pr.add_argument("--kernel_npy", default="results/motion_reflect/kernel.npy")   # motion/defocus
    pr.add_argument("--blur_sigma", type=float, default=4.0)                        # gaussian
    pr.add_argument("--n", type=int, default=16)
    pr.add_argument("--out", default="results/restore_out")
    pr.add_argument("--delta", type=float, default=0.01)
    pr.add_argument("--sigma_y", type=float, default=0.05)
    pr.add_argument("--prior_weight", type=float, default=1.0)
    pr.add_argument("--data_weight", type=float, default=64.0)
    pr.add_argument("--cg_iters", type=int, default=12)                            # motion
    pr.add_argument("--map_schedule", choices=["late", "const"], default="late")   # closed-form
    pr.add_argument("--freq_reg", type=float, default=0.5,
                    help="frequency-aware prior-precision boost gamma (0 = plain MAP; 0.5 default)")
    pr.add_argument("--noise_scale", type=float, default=0.0)
    pr.add_argument("--noise_kind", choices=["anneal", "sigma", "const"], default="anneal")
    pr.set_defaults(func=cmd_restore)

    pf = sub.add_parser("freqreg", help="frequency-aware data-regularization sweep (per-step CG)")
    _add_prior(pf)
    pf.add_argument("--operators", nargs="+", default=["gaussian", "motion", "defocus"],
                    choices=["gaussian", "motion", "defocus"])
    pf.add_argument("--regs", type=float, nargs="+", default=[0.0, 0.5])
    pf.add_argument("--n", type=int, default=4)
    pf.add_argument("--blur_sigma", type=float, default=4.0)
    pf.add_argument("--sigma_y", type=float, default=0.05)
    pf.add_argument("--delta", type=float, default=0.01)
    pf.add_argument("--data_weight", type=float, default=64.0)
    pf.add_argument("--cg_iters", type=int, default=12)
    pf.add_argument("--gaussian_dir", default="results/gaussian")
    pf.add_argument("--motion_dir", default="results/motion_reflect")
    pf.add_argument("--defocus_dir", default="results/defocus")
    pf.add_argument("--out", default="results/deblur_freqreg")
    pf.set_defaults(func=cmd_freqreg)

    pn = sub.add_parser("noise", help="post-data-consistency noise-injection sweep (gaussian)")
    _add_prior(pn)
    pn.add_argument("--clean_dir", default="results/gaussian/clean")
    pn.add_argument("--observation_dir", default="results/gaussian/observation")
    pn.add_argument("--blur_sigma", type=float, default=4.0)
    pn.add_argument("--sigma_y", type=float, default=0.05)
    pn.add_argument("--delta", type=float, default=0.01)
    pn.add_argument("--data_weight", type=float, default=64.0)
    pn.add_argument("--n", type=int, default=8)
    pn.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05])
    pn.add_argument("--kind", choices=["anneal", "sigma", "const"], default="anneal")
    pn.add_argument("--seed", type=int, default=0)
    pn.add_argument("--out", default="results/gaussian/noise_sweep")
    pn.set_defaults(func=cmd_noise)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
