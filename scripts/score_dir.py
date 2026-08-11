"""A1-1 scorer: per-image PSNR/SSIM/LPIPS for a recon dir vs clean dir.
Reports full-set mean+/-std and the first-16 mean (the reproduction guard vs the published n=16).

  python scripts/score_dir.py --clean results200/gaussian_s05/clean \
      --recon results200/gaussian_s05/ihdm/recon --tag "gaussian_s05 IHDM"
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from utils.pipeline import load_dir
from utils.metrics import psnr, ssim, lpips_metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--guard", type=int, default=16, help="first-k mean printed as repro guard")
    args = ap.parse_args()

    cp = sorted(glob.glob(os.path.join(args.clean, "*.png")))
    rp = sorted(glob.glob(os.path.join(args.recon, "*.png")))
    n = min(len(cp), len(rp))
    assert n > 0, f"no pairs: clean={len(cp)} recon={len(rp)}"
    C = load_dir(args.clean, n, args.device)
    R = load_dir(args.recon, n, args.device)
    P = np.array([psnr(R[i], C[i]) for i in range(n)])
    S = np.array([ssim(R[i], C[i]) for i in range(n)])
    L = np.array([lpips_metric(R[i], C[i], args.device) for i in range(n)])

    k = min(args.guard, n)
    def line(name, idx):
        return (f"  {name:<10} PSNR {P[idx].mean():6.2f}  SSIM {S[idx].mean():.3f}  "
                f"LPIPS {L[idx].mean():.3f}")
    print(f"[{args.tag}]  n={n}")
    print(line(f"first{k}", slice(0, k)) + "   (guard vs published)")
    print(f"  full n={n}  PSNR {P.mean():6.2f}+/-{P.std():.2f}  "
          f"SSIM {S.mean():.3f}+/-{S.std():.3f}  LPIPS {L.mean():.3f}+/-{L.std():.3f}")


if __name__ == "__main__":
    main()
