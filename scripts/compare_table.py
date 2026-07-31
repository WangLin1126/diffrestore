"""Uniform metric table across restoration methods (recon dir vs shared clean dir).

Computes PSNR / SSIM / LPIPS (recon vs clean) and measurement-consistency
(‖y - A x̂‖ / ‖y‖, using the SAME DCT-heat A the observations were made with) for
each named recon directory, then prints a comparison table. Every method — SMDC, HQS,
cold-diffusion, DPS — is scored identically from its output PNGs, so the numbers are
directly comparable regardless of solver paradigm.

  python scripts/compare_table.py --clean results/gaussian/clean --obs results/gaussian/observation \
    --blur_sigma 4 --recon "cold+SMDC"=results/gaussian/cold_smdc/recon ...
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image

from ops.transforms import DCTTransform
from ops.deblur import build_deblur
from utils.metrics import psnr, ssim, lpips_metric, measurement_consistency


def load(d, n=None):
    paths = sorted(glob.glob(os.path.join(d, "*.png")))
    if n:
        paths = paths[:n]
    xs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32))
          .permute(2, 0, 1) / 127.5 - 1.0 for p in paths]
    return torch.stack(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--recon", action="append", default=[], help="Name=dir (repeatable)")
    ap.add_argument("--blur_sigma", type=float, default=4.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    clean = load(args.clean)
    obs = load(args.obs)
    n = len(clean)
    H = clean.shape[-1]
    A, _ = build_deblur(H, H, args.blur_sigma, transform=DCTTransform(), device="cpu", dtype=torch.float32)

    def score(imgs):
        ps = np.mean([psnr(imgs[i], clean[i]) for i in range(n)])
        ss = np.mean([ssim(imgs[i], clean[i]) for i in range(n)])
        lp = np.mean([lpips_metric(imgs[i], clean[i], args.device) for i in range(n)])
        mc = np.mean([measurement_consistency(obs[i:i+1], A, imgs[i:i+1]) for i in range(n)])
        return ps, ss, lp, mc

    rows = [("Observation", *score(obs))]
    for spec in args.recon:
        name, d = spec.split("=", 1)
        rows.append((name, *score(load(d, n))))

    print(f"\n  Heat deblur (sigma={args.blur_sigma}, noise=0.05) — held-out CelebA-HQ 256, n={n}\n")
    print(f"  {'method':<22} {'PSNR↑':>7} {'SSIM↑':>7} {'LPIPS↓':>7} {'MC↓':>7}")
    print("  " + "-" * 54)
    for name, ps, ss, lp, mc in rows:
        print(f"  {name:<22} {ps:>7.2f} {ss:>7.3f} {lp:>7.3f} {mc:>7.3f}")
    print()


if __name__ == "__main__":
    main()
