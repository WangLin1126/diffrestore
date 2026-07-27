"""Score + visualize a head-to-head: clean | observation | DPS | SMDC.

  python smdc/scripts/compare_grid.py --clean compare_dps_smdc/clean \
     --obs compare_dps_smdc/observation \
     --recon DPS=compare_dps_smdc/dps_out/heat_blur/recon \
     --recon SMDC=compare_dps_smdc/smdc_surrogate_l2/recon \
     --out compare_dps_smdc/grid.png
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

from utils.metrics import psnr, ssim, lpips_metric


def load(d, n=None):
    paths = sorted(glob.glob(os.path.join(d, "*.png")))
    if n:
        paths = paths[:n]
    xs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        xs.append(torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(xs)


def score(name, clean, recon, device):
    ps = [psnr(recon[i], clean[i]) for i in range(len(clean))]
    ss = [ssim(recon[i], clean[i]) for i in range(len(clean))]
    lp = [lpips_metric(recon[i], clean[i], device) for i in range(len(clean))]
    m = (sum(ps) / len(ps), sum(ss) / len(ss), sum(lp) / len(lp))
    print(f"  {name:14s} PSNR {m[0]:6.2f}  SSIM {m[1]:.3f}  LPIPS {m[2]:.3f}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--recon", action="append", default=[], help="Name=dir (repeatable)")
    ap.add_argument("--out", default="compare_dps_smdc/grid.png")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    clean = load(args.clean)
    obs = load(args.obs)
    n = len(clean)
    methods = [("input(obs)", obs)]
    for spec in args.recon:
        name, d = spec.split("=", 1)
        methods.append((name, load(d, n)))

    print("=" * 52)
    print(f"Head-to-head on {n} held-out CelebA-HQ-256 images (blur σ=4, noise 0.05)")
    print("=" * 52)
    for name, r in methods:
        score(name, clean, r, args.device)

    def to01(x):
        return (x.clamp(-1, 1) + 1) / 2
    tiles = []
    for i in range(n):
        tiles.append(to01(clean[i]))
        for _, r in methods:
            tiles.append(to01(r[i]))
    save_image(torch.stack(tiles), args.out, nrow=len(methods) + 1)
    print(f"saved grid -> {args.out}  (cols: clean, {', '.join(m[0] for m in methods)})")


if __name__ == "__main__":
    main()
