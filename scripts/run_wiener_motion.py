"""Oracle Wiener deconvolution baseline for motion blur (DFT, circular). lambda chosen by best
mean test-set PSNR (Wiener's strongest setting), matching the report's convention.

  python scripts/run_wiener_motion.py --clean results/motion_reflect/clean \
     --observation_dir results/motion_reflect/observation --kernel_npy results/motion_reflect/kernel.npy \
     --out results/motion_reflect/wiener
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from ops.motion import MotionBlurOperator
from utils.metrics import psnr, ssim, lpips_metric


def load(d, H, n=None):
    xs = []
    for p in sorted(glob.glob(os.path.join(d, "*.png")))[:n]:
        im = Image.open(p).convert("RGB")
        if im.size != (H, H):
            im = im.resize((H, H), Image.LANCZOS)
        xs.append(torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--observation_dir", required=True)
    ap.add_argument("--kernel_npy", default="results/motion_reflect/kernel.npy")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    H, device = args.image_size, args.device
    os.makedirs(os.path.join(args.out, "recon"), exist_ok=True)
    clean = load(args.clean, H, args.n).to(device)
    ys = load(args.observation_dir, H, args.n).to(device)
    k = torch.from_numpy(np.load(args.kernel_npy))
    A = MotionBlurOperator(k, (H, H), device=device, dtype=torch.float32)
    otf = A.otf

    def wiener(y, lam):
        Y = torch.fft.fft2(y)
        return torch.fft.ifft2(torch.conj(otf) * Y / (torch.abs(otf) ** 2 + lam)).real.clamp(-1, 1)

    lams = np.logspace(-3, 0, 25)
    best = max(lams, key=lambda l: np.mean([psnr(wiener(ys[i:i+1], l), clean[i:i+1]) for i in range(len(ys))]))
    S = {"out": [], "ssim": [], "lpips": [], "mc": []}
    for i in range(len(ys)):
        x = wiener(ys[i:i+1], best)
        save_image((x[0] + 1) / 2, os.path.join(args.out, "recon", f"{i:05d}.png"))
        S["out"].append(psnr(x, clean[i:i+1])); S["ssim"].append(ssim(x, clean[i:i+1]))
        S["lpips"].append(lpips_metric(x, clean[i:i+1], device))
        S["mc"].append(((ys[i:i+1] - A.forward(x)).norm() / ys[i:i+1].norm()).item())
    m = {kk: float(np.mean(v)) for kk, v in S.items()}
    print(f"[wiener motion oracle lam={best:.3g}]  PSNR {m['out']:.2f}dB  SSIM {m['ssim']:.3f}  "
          f"LPIPS {m['lpips']:.3f}  MC {m['mc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
