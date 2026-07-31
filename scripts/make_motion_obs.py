"""Make motion-blur observations from an existing clean set (same faces as the Gaussian run).

Applies Levin09 kernel #1 (circular conv) + Gaussian noise, so the ONLY difference vs the
Gaussian experiment is the degradation operator -> directly comparable.

  python scripts/make_motion_obs.py --clean results/gaussian/clean --out results/motion --noise 0.05
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image

from ops.motion import load_levin_kernel, MotionBlurOperator
from utils.seed import set_seed


def load(d):
    paths = sorted(glob.glob(os.path.join(d, "*.png")))
    xs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32))
          .permute(2, 0, 1) / 127.5 - 1.0 for p in paths]
    return torch.stack(xs), paths


def save_png(path, x):
    a = ((x.clamp(-1, 1) + 1) / 2 * 255 + 0.5).byte().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(a).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="results/gaussian/clean")
    ap.add_argument("--out", default="results/motion")
    ap.add_argument("--kernel", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    x0, _ = load(args.clean)
    H = x0.shape[-1]
    k = torch.from_numpy(load_levin_kernel(args.kernel))
    A = MotionBlurOperator(k, (H, H), device="cpu", dtype=torch.float32)

    cdir = os.path.join(args.out, "clean"); odir = os.path.join(args.out, "observation")
    os.makedirs(cdir, exist_ok=True); os.makedirs(odir, exist_ok=True)
    np.save(os.path.join(args.out, "kernel.npy"), k.numpy())
    for i in range(x0.shape[0]):
        xi = x0[i:i + 1]
        y = (A.forward(xi) + args.noise * torch.randn_like(xi)).clamp(-1, 1)
        save_png(os.path.join(cdir, f"{i:05d}.png"), xi[0])
        save_png(os.path.join(odir, f"{i:05d}.png"), y[0])
    print(f"wrote {x0.shape[0]} motion-blur pairs (Levin#{args.kernel}, {k.shape[0]}x{k.shape[1]}, "
          f"noise={args.noise}) -> {args.out}/")


if __name__ == "__main__":
    main()
