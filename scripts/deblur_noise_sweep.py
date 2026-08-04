"""Sweep post-data-consistency noise injection on the Gaussian-deblur SMDC pipeline.

Loads the IHDM prior + observations ONCE, then runs the scale-matched HQS solver at several
`noise_scale` values (small white noise added AFTER each per-step data-consistency step, never on
the final step -> the returned image has no added noise). Reports PSNR/SSIM/LPIPS/MC per setting
on identical inputs, so the only variable is the injected noise.

  python scripts/deblur_noise_sweep.py --n 8 --scales 0 0.01 0.02 0.05 --kind anneal
"""
import os, sys, glob, argparse, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from PIL import Image
from torchvision.utils import save_image
from ops.transforms import DCTTransform
from ops.heat import HeatSchedule
from ops.deblur import build_deblur
from model.ihdm import load_ihdm
from solver.init import terminal_init
from solver.hqs import MAPCorrection
from solver.base import scale_matched_solver
from utils.logging import RunLogger
from utils.metrics import psnr, ssim, lpips_metric, measurement_consistency


def load_stack(d, H, n):
    xs = []
    for p in sorted(glob.glob(os.path.join(d, "*.png")))[:n]:
        im = Image.open(p).convert("RGB")
        if im.size != (H, H):
            im = im.resize((H, H), Image.LANCZOS)
        xs.append(torch.from_numpy(np.asarray(im, np.float32)).permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoint/ihdm/ihdm_ffhq256_full.pth")
    ap.add_argument("--ihdm_config", default="img_size_256_full")
    ap.add_argument("--clean_dir", default="results/gaussian/clean")
    ap.add_argument("--observation_dir", default="results/gaussian/observation")
    ap.add_argument("--blur_sigma", type=float, default=4.0)
    ap.add_argument("--sigma_y", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--data_weight", type=float, default=64.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05])
    ap.add_argument("--kind", choices=["anneal", "sigma", "const"], default="anneal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gaussian/noise_sweep")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    tf = DCTTransform()
    prior, sigmas, cfg = load_ihdm(ckpt=args.ckpt, config_name=args.ihdm_config, device=dev)
    H = int(cfg.data.image_size)
    sch = HeatSchedule(H, H, sigmas, transform=tf, device=dev, dtype=torch.float32)
    A, gA = build_deblur(H, H, args.blur_sigma, transform=tf, device=dev, dtype=torch.float32)
    times = list(range(sch.num_levels - 1, -1, -1))
    corrector = MAPCorrection(sch, gA, delta=args.delta, sigma_y=args.sigma_y,
                              prior_weight=1.0, data_weight=args.data_weight, schedule_kind="late")
    x0 = load_stack(args.clean_dir, H, args.n).to(dev)
    ys = load_stack(args.observation_dir, H, args.n).to(dev)
    print(f"deblur noise sweep: n={x0.shape[0]} H={H} blur={args.blur_sigma} sigma_y={args.sigma_y} "
          f"kind={args.kind}", flush=True)

    print(f"\n{'noise':>7} | {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6} {'MC':>6}")
    print("-" * 42)
    for ns in args.scales:
        S = {"out": [], "ssim": [], "lpips": [], "mc": []}
        for k in range(x0.shape[0]):
            xi, y = x0[k:k + 1], ys[k:k + 1]
            rng = torch.Generator(device=dev).manual_seed(args.seed + k)      # same noise draw across scales
            x_init = terminal_init("matched_measurement", y, sch, times[0])
            x_hat = scale_matched_solver(y, x_init, times, A, sch, prior, corrector, clamp=(-1, 1),
                                         logger=RunLogger(verbose=False), rng=rng,
                                         noise_scale=ns, noise_kind=args.kind)
            S["out"].append(psnr(x_hat, xi)); S["ssim"].append(ssim(x_hat, xi))
            S["lpips"].append(lpips_metric(x_hat, xi, dev)); S["mc"].append(measurement_consistency(y, A, x_hat))
            if k == 0:
                d = os.path.join(args.out, f"ns{ns:g}"); os.makedirs(d, exist_ok=True)
                save_image((x_hat[0].clamp(-1, 1) + 1) / 2, os.path.join(d, f"{k:05d}.png"))
        m = {kk: float(np.mean(v)) for kk, v in S.items()}
        print(f"{ns:>7.3g} | {m['out']:>6.2f} {m['ssim']:>6.3f} {m['lpips']:>6.3f} {m['mc']:>6.3f}", flush=True)


if __name__ == "__main__":
    main()
