"""HQS + Total-Variation baseline with a SPATIAL reflect-boundary CG data step — the fair TV
counterpart to the IHDM/cold CG pipeline. Identical plug-and-play splitting as run_tv_hqs.py,
but the data subproblem is solved by conjugate gradient with the SAME reflect-boundary operator
(ops.motion_spatial.SpatialMotionBlur, exact autograd adjoint) that restore_motion_cg.py uses —
NOT the circular DFT-Wiener filter. So TV and IHDM now differ ONLY in the prior (Chambolle
TV-prox vs learned reverse step); both use reflect A + spatial CG, no FFT, no wrap-around ring.

    for it in 1..T:
        z = TV_denoise(x, weight = beta/rho)                       # prior prox
        # data step:  min_x (1/2 sigma_y^2)||A x - y||^2 + (rho/2)||x - z||^2   via CG
        #   ( (1/sigma_y^2) A^T A + rho I ) x = (1/sigma_y^2) A^T y + rho z
        x = CG(...)
        rho *= gamma

  python scripts/run_tv_cg.py --kernel_npy results/defocus/kernel.npy \
     --clean_dir results/defocus/clean --observation_dir results/defocus/observation \
     --beta 4 --sigma_y 0.05 --out results/defocus/tv_cg
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from PIL import Image
from torchvision.utils import save_image
from skimage.restoration import denoise_tv_chambolle
from ops.motion_spatial import SpatialMotionBlur, cg_solve
from utils.metrics import psnr, ssim, lpips_metric


def load(d, H, n=None):
    ps = sorted(glob.glob(os.path.join(d, "*.png")))[:n]
    xs = []
    for p in ps:
        im = Image.open(p).convert("RGB")
        if im.size != (H, H):
            im = im.resize((H, H), Image.LANCZOS)
        xs.append(torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1)[None] / 127.5 - 1)
    return xs


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

    def tvprox(x, w):                                    # Chambolle prox on CPU (skimage), back to device
        z = denoise_tv_chambolle(x[0].detach().cpu().permute(1, 2, 0).numpy(), weight=w, channel_axis=-1)
        return torch.from_numpy(z).permute(2, 0, 1)[None].float().to(dev)

    def hqs_tv_cg(y):
        x = y.clone(); rho = args.rho0
        Aty = A.adjoint(y)
        for _ in range(args.T):
            z = tvprox(x, args.beta / rho)
            def M(v, rho=rho):
                return (1.0 / sig2) * A.adjoint(A.forward(v)) + rho * v
            c = (1.0 / sig2) * Aty + rho * z
            x = cg_solve(M, c, z, iters=args.cg_iters)
            rho *= args.gamma
        return x.clamp(-1, 1)

    clean = load(args.clean_dir, H, args.n); obs = load(args.observation_dir, H, args.n)
    S = {"in": [], "out": [], "ssim": [], "lpips": [], "mc": []}
    for i, (xi, y) in enumerate(zip(clean, obs)):
        xi, y = xi.to(dev), y.to(dev)
        xh = hqs_tv_cg(y)
        save_image((xh[0] + 1) / 2, os.path.join(args.out, "recon", f"{i:05d}.png"))
        mc = ((y - A.forward(xh)).norm() / y.norm()).item()
        S["in"].append(psnr(y, xi)); S["out"].append(psnr(xh, xi)); S["ssim"].append(ssim(xh, xi))
        S["lpips"].append(lpips_metric(xh, xi, dev)); S["mc"].append(mc)
    m = {kk: sum(v) / len(v) for kk, v in S.items()}
    print(f"[TV+CG beta={args.beta} sigma_y={args.sigma_y}]  PSNR {m['in']:.2f}->{m['out']:.2f}  "
          f"SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  MC {m['mc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
