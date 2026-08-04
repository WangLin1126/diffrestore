"""Unified Gaussian-deblur restoration entrypoint:  prior  x  HQS (per-step MAP) solver.

  prior  : cold_diffusion (our x0-predictor) | ihdm (heat, vendored)
  solver : hqs (closed-form per-step MAP data consistency; the scale-matched data step)

Deblurs external observations (clean_dir + observation_dir, same DCT-heat A both were made with).
For motion blur under realistic reflect boundaries, use scripts/restore_motion_cg.py (spatial CG).

  # IHDM prior + HQS (full MAP) solver
  python scripts/restore.py --prior ihdm --solver hqs --ihdm_config img_size_256_full \
     --ckpt checkpoint/ihdm/ihdm_ffhq256_full.pth --image_size 256 \
     --clean_dir results/.../clean --observation_dir results/.../observation
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from ops.transforms import DCTTransform
from ops.heat import HeatSchedule
from ops.deblur import build_deblur
from model.cold_diffusion import ColdDiffusionPrior
from model.unet import UNet
from model.ihdm import load_ihdm
from solver.init import terminal_init
from solver.hqs import MAPCorrection
from solver.base import scale_matched_solver
from utils.seed import set_seed
from utils.logging import RunLogger
from utils.metrics import psnr, ssim, lpips_metric, measurement_consistency


def load_png_stack(d, image_size, n=None):
    paths = sorted(glob.glob(os.path.join(d, "*.png")))
    if n:
        paths = paths[:n]
    xs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if im.size != (image_size, image_size):
            im = im.resize((image_size, image_size), Image.LANCZOS)
        xs.append(torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(xs)


def build_prior(args, tf, device):
    """Return (prior, schedule, image_size)."""
    if args.prior == "ihdm":
        prior, sigmas, config = load_ihdm(ckpt=args.ckpt or None, config_name=args.ihdm_config, device=device) \
            if args.ckpt else load_ihdm(config_name=args.ihdm_config, device=device)
        H = int(config.data.image_size)
        sch = HeatSchedule(H, H, sigmas, transform=tf, device=device, dtype=torch.float32)
        return prior, sch, H
    # cold_diffusion
    H = args.image_size
    model = UNet(ch=args.ch, out_ch=3, ch_mult=tuple(args.ch_mult), num_res_blocks=args.num_res_blocks,
                 attn_resolutions=(args.attn_res,), in_channels=3, resolution=H).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); model.eval()
    sch = HeatSchedule.ihdm(H, H, K=args.K, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                            transform=tf, device=device, dtype=torch.float32)
    return ColdDiffusionPrior(model, sch), sch, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["cold_diffusion", "ihdm"], required=True)
    ap.add_argument("--solver", choices=["hqs"], default="hqs")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--observation_dir", required=True)
    ap.add_argument("--blur_sigma", type=float, default=4.0)
    ap.add_argument("--noise_std", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="results/restore_out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    # cold-diffusion prior architecture / schedule
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--ch_mult", type=int, nargs="+", default=[1, 1, 2, 2, 4])
    ap.add_argument("--num_res_blocks", type=int, default=2)
    ap.add_argument("--attn_res", type=int, default=16)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--sigma_min", type=float, default=0.5)
    ap.add_argument("--sigma_max", type=float, default=128.0)
    ap.add_argument("--ihdm_config", default="img_size_128_maxblur128")
    # HQS solver knobs
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--sigma_y", type=float, default=0.05)
    ap.add_argument("--prior_weight", type=float, default=1.0)
    ap.add_argument("--data_weight", type=float, default=64.0)
    ap.add_argument("--map_schedule", choices=["late", "const"], default="late")
    ap.add_argument("--freq_reg", type=float, default=0.5,
                    help="frequency-aware prior-precision boost gamma (0 = plain MAP; 0.5 default)")
    # stochastic reverse: small noise injected AFTER the data-consistency step (0 = deterministic)
    ap.add_argument("--noise_scale", type=float, default=0.0)
    ap.add_argument("--noise_kind", choices=["anneal", "sigma", "const"], default="anneal")
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device
    tf = DCTTransform()
    os.makedirs(os.path.join(args.out, "recon"), exist_ok=True)

    prior, sch, H = build_prior(args, tf, device)
    A, gA = build_deblur(H, H, args.blur_sigma, transform=tf, device=device, dtype=torch.float32)
    times = list(range(sch.num_levels - 1, -1, -1))
    corrector = MAPCorrection(sch, gA, delta=args.delta, sigma_y=args.sigma_y,
                              prior_weight=args.prior_weight, data_weight=args.data_weight,
                              schedule_kind=args.map_schedule, freq_reg=args.freq_reg)
    print(f"  prior={args.prior} solver={args.solver} res={H} K={sch.num_levels-1}")

    x0 = load_png_stack(args.clean_dir, H, args.n).to(device)
    ys = load_png_stack(args.observation_dir, H, args.n).to(device)
    S = {"in": [], "out": [], "ssim": [], "lpips": [], "mc": []}
    for k in range(x0.shape[0]):
        xi, y = x0[k:k + 1], ys[k:k + 1]
        x_init = terminal_init("matched_measurement", y, sch, times[0])
        x_hat = scale_matched_solver(y, x_init, times, A, sch, prior, corrector,
                                     clamp=(-1, 1), logger=RunLogger(verbose=False), x0_ref=xi,
                                     noise_scale=args.noise_scale, noise_kind=args.noise_kind)
        save_image((x_hat[0].clamp(-1, 1) + 1) / 2, os.path.join(args.out, "recon", f"{k:05d}.png"))
        S["in"].append(psnr(y, xi)); S["out"].append(psnr(x_hat, xi)); S["ssim"].append(ssim(x_hat, xi))
        S["lpips"].append(lpips_metric(x_hat, xi, device)); S["mc"].append(measurement_consistency(y, A, x_hat))
    m = {k: sum(v) / len(v) for k, v in S.items()}
    print(f"[{args.prior} x {args.solver}] blur={args.blur_sigma} noise={args.noise_std}  "
          f"PSNR {m['in']:.2f}->{m['out']:.2f}dB  SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  MC {m['mc']:.3f}")


if __name__ == "__main__":
    main()
