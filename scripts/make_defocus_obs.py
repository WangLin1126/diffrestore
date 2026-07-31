"""Defocus-blur observations: out-of-focus = convolution with a uniform DISK (pillbox) PSF,
applied with a REALISTIC reflect (symmetric) boundary — same convention as the motion pipeline
(reflect-pad h//2, convolve, crop == scipy convolve2d mode='same', boundary='symm').

The disk is anti-aliased by supersampling its edge. Because it is a plain convolution it
commutes with the DCT-heat K_t (L_t = K_t), so restoration reuses scripts/restore_motion_cg.py
verbatim with --kernel_npy pointing at the disk saved here.

  python scripts/make_defocus_obs.py --clean results/gaussian/clean \
      --radius 7 --out results/defocus --noise 0.05
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from PIL import Image
from scipy.signal import convolve2d
from utils.seed import set_seed


def disk_psf(radius: int, oversample: int = 8) -> np.ndarray:
    """Uniform disk (pillbox) of the given pixel radius, anti-aliased by `oversample`x
    supersampling. Returns a sum-normalized (2r+1)x(2r+1) kernel."""
    n = 2 * radius + 1
    hi = n * oversample
    ys, xs = np.mgrid[0:hi, 0:hi].astype(np.float64)
    c = (hi - 1) / 2.0
    inside = ((xs - c) ** 2 + (ys - c) ** 2) <= (radius * oversample) ** 2
    k = inside.reshape(n, oversample, n, oversample).mean(axis=(1, 3))
    return (k / k.sum()).astype(np.float32)


def load(d):
    paths = sorted(glob.glob(os.path.join(d, "*.png")))
    xs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32))
          .permute(2, 0, 1) / 127.5 - 1.0 for p in paths]
    return torch.stack(xs), paths


def save_png(path, x):
    a = ((x.clamp(-1, 1) + 1) / 2 * 255 + 0.5).byte().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(a).save(path)


def reflect_blur(xi, k):
    img = xi[0].numpy()
    out = np.stack([convolve2d(img[c], k, mode="same", boundary="symm") for c in range(3)], 0)
    return torch.from_numpy(out.astype(np.float32))[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="results/gaussian/clean")
    ap.add_argument("--radius", type=int, default=7)          # 15x15 disk
    ap.add_argument("--out", default="results/defocus")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    x0, _ = load(args.clean)
    k = disk_psf(args.radius)

    cdir = os.path.join(args.out, "clean"); odir = os.path.join(args.out, "observation")
    os.makedirs(cdir, exist_ok=True); os.makedirs(odir, exist_ok=True)
    np.save(os.path.join(args.out, "kernel.npy"), k)
    for i in range(x0.shape[0]):
        xi = x0[i:i + 1]
        blur = reflect_blur(xi, k)
        y = (blur + args.noise * torch.randn_like(xi)).clamp(-1, 1)
        save_png(os.path.join(cdir, f"{i:05d}.png"), xi[0])
        save_png(os.path.join(odir, f"{i:05d}.png"), y[0])
    print(f"wrote {x0.shape[0]} REFLECT-boundary defocus pairs (disk r={args.radius}, "
          f"{k.shape[0]}x{k.shape[1]}, noise={args.noise}) -> {args.out}/", flush=True)


if __name__ == "__main__":
    main()
