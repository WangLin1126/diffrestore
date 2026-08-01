"""Central-crop metric helper: score recon dir against clean dir on the central CxC crop.

Used for the motion 'interior' table (boundary artifacts excluded).
  python scripts/crop_score.py --clean results/motion_reflect/clean \
      --recon results/motion_reflect/ihdm_hqs/recon --crop 128
Also (optionally) writes the cropped recon PNGs to --save_dir for the figure.
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from PIL import Image
from utils.metrics import psnr, ssim, lpips_metric


def load(d, n=None):
    paths = sorted(glob.glob(os.path.join(d, "*.png")))
    if n:
        paths = paths[:n]
    xs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32))
          .permute(2, 0, 1) / 127.5 - 1.0 for p in paths]
    return torch.stack(xs), paths


def center_crop(x, c):
    H = x.shape[-1]
    s = (H - c) // 2
    return x[..., s:s + c, s:s + c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--label", default="recon")
    args = ap.parse_args()

    clean, _ = load(args.clean)
    recon, rpaths = load(args.recon, len(clean))
    n = len(clean)
    cc = lambda x: center_crop(x, args.crop)
    ps = np.mean([psnr(cc(recon[i]), cc(clean[i])) for i in range(n)])
    ss = np.mean([ssim(cc(recon[i]), cc(clean[i])) for i in range(n)])
    lp = np.mean([lpips_metric(cc(recon[i]), cc(clean[i]), args.device) for i in range(n)])
    print(f"[{args.label} crop{args.crop}]  PSNR {ps:.2f}  SSIM {ss:.3f}  LPIPS {lp:.3f}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for i in range(n):
            c = ((cc(recon[i]).clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()
            Image.fromarray(c).save(os.path.join(args.save_dir, f"{i:05d}.png"))


if __name__ == "__main__":
    main()
