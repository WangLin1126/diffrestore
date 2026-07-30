"""Motion-blur observations with a REALISTIC reflect (symmetric) boundary — the traditional
convention: reflect-pad by half the kernel, convolve, crop to original size (scipy
convolve2d mode='same', boundary='symm'). Contrast with make_motion_obs.py, which uses a
periodic/circular FFT convolution (wrap-around).

Same clean faces, same Levin kernel (reused from results/motion/kernel.npy), and the same
noise seed/order as the circular run, so the ONLY difference is the blur's boundary handling.

  python scripts/make_motion_obs_reflect.py --clean results/motion/clean \
      --kernel_npy results/motion/kernel.npy --out results/motion_reflect --noise 0.05
"""
import os, sys, glob, argparse, shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from PIL import Image
from scipy.signal import convolve2d
from utils.seed import set_seed


def load(d):
    paths = sorted(glob.glob(os.path.join(d, "*.png")))
    xs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32))
          .permute(2, 0, 1) / 127.5 - 1.0 for p in paths]
    return torch.stack(xs), paths


def save_png(path, x):
    a = ((x.clamp(-1, 1) + 1) / 2 * 255 + 0.5).byte().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(a).save(path)


def reflect_blur(xi, k):
    """xi: (1,3,H,W) in [-1,1]; k: (h,w) sum-normalized. True convolution, symmetric (reflect)
    boundary, same size — i.e. reflect-pad h//2 -> convolve -> crop."""
    img = xi[0].numpy()                                   # (3,H,W)
    out = np.stack([convolve2d(img[c], k, mode="same", boundary="symm") for c in range(3)], 0)
    return torch.from_numpy(out.astype(np.float32))[None]  # (1,3,H,W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="results/motion/clean")
    ap.add_argument("--kernel_npy", default="results/motion/kernel.npy")
    ap.add_argument("--out", default="results/motion_reflect")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    x0, _ = load(args.clean)
    k = np.load(args.kernel_npy).astype(np.float32)
    k = k / k.sum()

    cdir = os.path.join(args.out, "clean"); odir = os.path.join(args.out, "observation")
    os.makedirs(cdir, exist_ok=True); os.makedirs(odir, exist_ok=True)
    np.save(os.path.join(args.out, "kernel.npy"), k)
    for i in range(x0.shape[0]):
        xi = x0[i:i + 1]
        blur = reflect_blur(xi, k)
        y = (blur + args.noise * torch.randn_like(xi)).clamp(-1, 1)   # same noise order as circular run
        save_png(os.path.join(cdir, f"{i:05d}.png"), xi[0])
        save_png(os.path.join(odir, f"{i:05d}.png"), y[0])
    print(f"wrote {x0.shape[0]} REFLECT-boundary motion pairs (Levin {k.shape[0]}x{k.shape[1]}, "
          f"noise={args.noise}) -> {args.out}/", flush=True)


if __name__ == "__main__":
    main()
