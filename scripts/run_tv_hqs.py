"""HQS + Total-Variation baseline: the same variable-splitting loop as the diffusion solver,
but the prior-proximal step is a classical TV (Chambolle) denoiser.

    for it in 1..T:
        z = TV_denoise(x, weight = beta/rho)          # prior prox (plug-and-play)
        x = argmin (1/2 sigma_y^2)||A x - y||^2 + (rho/2)||x - z||^2   # Wiener data step (basis Phi)
        rho *= gamma                                  # continuation

Data step is closed-form per frequency: DCT (Gaussian, real transfer) or DFT (motion, complex OTF).

  python scripts/run_tv_hqs.py --operator motion --kernel_npy results/motion/kernel.npy \
     --clean_dir results/motion/clean --observation_dir results/motion/observation \
     --beta 4 --sigma_y 0.05 --out results/motion/tv_hqs
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from PIL import Image
from torchvision.utils import save_image
from skimage.restoration import denoise_tv_chambolle
from ops.transforms import DCTTransform
from ops.deblur import build_deblur
from ops.motion import MotionBlurOperator
from utils.metrics import psnr, ssim, lpips_metric, measurement_consistency


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
    ap.add_argument("--operator", choices=["gaussian", "motion", "disk"], required=True)
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--observation_dir", required=True)
    ap.add_argument("--blur_sigma", type=float, default=4.0)
    ap.add_argument("--kernel_npy", default="results/motion/kernel.npy")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--rho0", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=1.2)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--sigma_y", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    H = args.image_size
    sig2 = args.sigma_y ** 2
    os.makedirs(os.path.join(args.out, "recon"), exist_ok=True)

    if args.operator in ("gaussian", "disk"):
        tf = DCTTransform()
        if args.operator == "gaussian":
            A, g = build_deblur(H, H, args.blur_sigma, transform=tf, device="cpu", dtype=torch.float32)
        else:  # disk: symmetric kernel is DCT-diagonal; transfer via all-ones-DCT probe
            from ops.motion_spatial import SpatialMotionBlur
            from ops.spectral import SpectralOperator
            kk = torch.from_numpy(np.load(args.kernel_npy))
            A_sp = SpatialMotionBlur(kk, 3, "cpu", torch.float32)
            g = tf.fwd(A_sp.forward(tf.inv(torch.ones(1, 3, H, H))))[0, 0]
            A = SpectralOperator(g, tf)
        fwd, inv = tf.fwd, tf.inv
        num_a = lambda Y: g * Y                       # conj(g)=g (real)
        den_a = g * g
    else:
        k = torch.from_numpy(np.load(args.kernel_npy))
        A = MotionBlurOperator(k, (H, H), device="cpu", dtype=torch.float32)
        otf = A.otf
        fwd = lambda x: torch.fft.fft2(x)
        inv = lambda X: torch.fft.ifft2(X).real
        num_a = lambda Y: torch.conj(otf) * Y
        den_a = torch.abs(otf) ** 2

    def tvprox(x, w):
        z = denoise_tv_chambolle(x[0].permute(1, 2, 0).numpy(), weight=w, channel_axis=-1)
        return torch.from_numpy(z).permute(2, 0, 1)[None].float()

    def hqs_tv(y):
        Y = fwd(y); x = y.clone(); rho = args.rho0
        for _ in range(args.T):
            Z = fwd(tvprox(x, args.beta / rho))
            x = inv((num_a(Y) / sig2 + rho * Z) / (den_a / sig2 + rho))
            rho *= args.gamma
        return x.clamp(-1, 1)

    clean = load(args.clean_dir, H, args.n); obs = load(args.observation_dir, H, args.n)
    S = {"in": [], "out": [], "ssim": [], "lpips": [], "mc": []}
    for i, (xi, y) in enumerate(zip(clean, obs)):
        xh = hqs_tv(y)
        save_image((xh[0] + 1) / 2, os.path.join(args.out, "recon", f"{i:05d}.png"))
        S["in"].append(psnr(y, xi)); S["out"].append(psnr(xh, xi)); S["ssim"].append(ssim(xh, xi))
        S["lpips"].append(lpips_metric(xh, xi, "cpu")); S["mc"].append(measurement_consistency(y, A, xh))
    m = {kk: sum(v) / len(v) for kk, v in S.items()}
    print(f"[TV+HQS {args.operator} beta={args.beta} sigma_y={args.sigma_y}]  "
          f"PSNR {m['in']:.2f}->{m['out']:.2f}  SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  MC {m['mc']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
