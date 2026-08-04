"""Assemble the SR method-comparison figure from scripts/{sr_demo,sr_baselines}.py outputs:
Clean | LR (bicubic up) | DPS | TV+CG | SMDC+IHDM, one row per image, PSNR/SSIM under each recon.

  python scripts/sr_compare_fig.py --dir results/sr_demo --scale 4 --n 4
"""
import os, sys, glob, argparse, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from utils.metrics import psnr, ssim


def load(path, size):
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im, np.float32)).permute(2, 0, 1)[None] / 127.5 - 1.0


def img(t): return ((t[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/sr_demo")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=256)
    args = ap.parse_args()
    H, s = args.image_size, args.scale
    recons = [("DPS", "dps_recon"), ("TV+CG", "tv_recon"), ("SMDC+IHDM", "recon")]
    cols = ["Clean (HR)", f"LR x{s} (bicubic up)"] + [n for n, _ in recons]

    cpaths = sorted(glob.glob(os.path.join(args.dir, "clean", "*.png")))[:args.n]
    rows = []
    for i, cp in enumerate(cpaths):
        x0 = load(cp, H)
        lr = load(os.path.join(args.dir, "observation", f"{i:05d}.png"), H // s)
        bic = F.interpolate(lr, size=(H, H), mode="bicubic", align_corners=False)
        panels = [(img(x0), None), (img(bic), None)]
        for _, sub in recons:
            r = load(os.path.join(args.dir, sub, f"{i:05d}.png"), H)
            panels.append((img(r), (psnr(r, x0), ssim(r, x0))))
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


if __name__ == "__main__":
    main()
