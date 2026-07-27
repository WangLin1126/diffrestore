"""Presentation figure: labeled image montage with per-method metrics.

Columns = Clean | Observation | <methods...>, rows = sample images. Each method column header
shows mean PSNR/SSIM/LPIPS over all images; each cell shows its PSNR/LPIPS.

  python smdc/scripts/make_figure.py --clean compare_dps_smdc/clean --obs compare_dps_smdc/observation \
    --recon "DPS (ffhq_10m)"=compare_dps_smdc/dps_out/heat_blur/recon \
    --recon "SMDC (ours)"=compare_dps_smdc/smdc_surrogate_l2/recon \
    --rows 4 --title "Heat deblur (sigma=4, noise=0.05) - held-out CelebA-HQ 256" \
    --out compare_dps_smdc/figure_dps_vs_smdc.png
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from utils.metrics import psnr, ssim, lpips_metric


def load(d, n=None):
    paths = sorted(glob.glob(os.path.join(d, "*.png")))
    if n:
        paths = paths[:n]
    xs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32))
          .permute(2, 0, 1) / 127.5 - 1.0 for p in paths]
    return torch.stack(xs)


def to01(x):
    return ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--recon", action="append", default=[], help="Name=dir (repeatable)")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    clean = load(args.clean)
    obs = load(args.obs)
    n = len(clean)
    cols = [("Clean", clean, False), ("Observation", obs, True)]
    for spec in args.recon:
        name, d = spec.split("=", 1)
        cols.append((name, load(d, n), True))

    # metrics
    headers = []
    per_cell = {}
    for j, (name, imgs, scored) in enumerate(cols):
        if not scored:
            headers.append(name)
            continue
        ps = [psnr(imgs[i], clean[i]) for i in range(n)]
        ss = [ssim(imgs[i], clean[i]) for i in range(n)]
        lp = [lpips_metric(imgs[i], clean[i], args.device) for i in range(n)]
        per_cell[j] = (ps, lp)
        headers.append(f"{name}\n{np.mean(ps):.2f} / {np.mean(ss):.3f} / {np.mean(lp):.3f}")

    R, C = min(args.rows, n), len(cols)
    idx = np.linspace(0, n - 1, R).astype(int)
    fig, axes = plt.subplots(R, C, figsize=(C * 2.3, R * 2.5 + 0.8))
    if R == 1:
        axes = axes[None, :]
    for r, i in enumerate(idx):
        for j, (name, imgs, scored) in enumerate(cols):
            ax = axes[r, j]
            ax.imshow(to01(imgs[i])); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(headers[j], fontsize=8.5, fontweight="bold", pad=4)
            if scored:
                ps, lp = per_cell[j]
                ax.text(0.03, 0.03, f"{ps[i]:.2f}dB\n{lp[i]:.3f}", transform=ax.transAxes,
                        fontsize=7.5, va="bottom", ha="left", color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0))
    if args.title:
        fig.suptitle(args.title + "\n(per-method mean: PSNR / SSIM / LPIPS   ·   per image: PSNR, LPIPS)",
                     fontsize=10, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"saved figure -> {args.out}  ({R}x{C}, cols: {[c[0] for c in cols]})")


if __name__ == "__main__":
    main()
