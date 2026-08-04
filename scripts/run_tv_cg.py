"""HQS + Total-Variation baseline with a SPATIAL reflect-boundary CG data step — the fair TV
counterpart to the IHDM/cold CG pipeline. Same plug-and-play splitting as run_tv_hqs.py, but the
data subproblem is solved by conjugate gradient with the SAME reflect-boundary operator
(ops.motion_spatial.SpatialMotionBlur, exact autograd adjoint) that deblur.py (--operator motion)
uses — NOT the circular DFT-Wiener filter. So TV and IHDM differ ONLY in the prior (Chambolle
TV-prox vs learned reverse step); both use reflect A + spatial CG, no FFT, no wrap-around ring.

The TV+CG loop itself is utils/pipeline.py::tv_cg_restore (shared with sr.py baselines).

  python scripts/run_tv_cg.py --kernel_npy results/defocus/kernel.npy \
     --clean_dir results/defocus/clean --observation_dir results/defocus/observation \
     --beta 4 --sigma_y 0.05 --out results/defocus/tv_cg
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from ops.motion_spatial import SpatialMotionBlur
from utils import pipeline as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--observation_dir", required=True)
    ap.add_argument("--kernel_npy", required=True)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--rho0", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=1.2)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--cg_iters", type=int, default=12)
    ap.add_argument("--sigma_y", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    H, sig2, dev = args.image_size, args.sigma_y ** 2, args.device
    os.makedirs(os.path.join(args.out, "recon"), exist_ok=True)

    k = torch.from_numpy(np.load(args.kernel_npy))
    A = SpatialMotionBlur(k, channels=3, device=dev, dtype=torch.float32)
    clean = P.load_stack(args.clean_dir, H, args.n, dev)
    obs = P.load_stack(args.observation_dir, H, args.n, dev)
    S = P.new_scores("in", "out", "ssim", "lpips", "mc")
    for i in range(clean.shape[0]):
        xi, y = clean[i:i + 1], obs[i:i + 1]
        xh = P.tv_cg_restore(A, y, sig2, args.beta, args.rho0, args.gamma, args.T, args.cg_iters, dev)
        P.save_img(os.path.join(args.out, "recon", f"{i:05d}.png"), xh)
        P.add_scores(S, xh, xi, dev, y=y, A=A)
    m = P.mean_scores(S)
    print(f"[TV+CG beta={args.beta} sigma_y={args.sigma_y}]  PSNR {m['in']:.2f}->{m['out']:.2f}  "
          f"SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  MC {m['mc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
