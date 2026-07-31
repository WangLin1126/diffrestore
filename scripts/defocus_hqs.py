"""Defocus deblur with the CLOSED-FORM DCT-HQS data step (no CG).

A symmetric disk (pillbox) PSF under reflect boundaries is diagonalized by the DCT-II (verified
||A x - IDCT(a_hat . DCT x)||/||A x|| ~ 9e-6), exactly like the Gaussian operator. So the disk's
DCT transfer a_hat is extracted once (via the all-ones-DCT probe, a_hat = DCT(A . IDCT(1))) and the
per-step MAP data step is the same closed-form per-frequency Wiener filter used for Gaussian
(solver/hqs.py) — exact, iteration-free, no adjoint. Contrast the asymmetric motion kernel, which
is NOT DCT-diagonal and needs spatial CG (scripts/restore_motion_cg.py).

  python scripts/defocus_hqs.py --clean_dir results/defocus/clean \
     --observation_dir results/defocus/observation --kernel_npy results/defocus/kernel.npy \
     --sigma_y 0.05 --n 16 --out results/defocus/ihdm_hqs
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from PIL import Image
from torchvision.utils import save_image
from ops.transforms import DCTTransform
from ops.heat import HeatSchedule
from ops.spectral import SpectralOperator
from ops.motion_spatial import SpatialMotionBlur
from model.ihdm import load_ihdm
from model.cold_diffusion import ColdDiffusionPrior
from model.unet import UNet
from solver.init import terminal_init
from solver.hqs import MAPCorrection
from solver.base import scale_matched_solver
from utils.logging import RunLogger
from utils.metrics import psnr, ssim, lpips_metric, measurement_consistency


def load(d, H, n=None):
    xs = []
    for p in sorted(glob.glob(os.path.join(d, "*.png")))[:n]:
        im = Image.open(p).convert("RGB")
        if im.size != (H, H):
            im = im.resize((H, H), Image.LANCZOS)
        xs.append(torch.from_numpy(np.asarray(im, np.float32)).permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(xs)


def dct_transfer(kernel, H, tf, device):
    """DCT transfer a_hat of the reflect-boundary symmetric-kernel convolution, via
    a_hat = DCT(A . IDCT(ones))  (exact iff A is DCT-diagonal, i.e. kernel symmetric)."""
    A_sp = SpatialMotionBlur(kernel, 3, device=device, dtype=torch.float32)
    e = tf.inv(torch.ones(1, 3, H, H, device=device))
    a_hat = tf.fwd(A_sp.forward(e))[0, 0]                      # (H,W)
    # diagnostic: how well does a_hat diagonalize A?
    x = torch.randn(1, 3, H, H, device=device)
    err = (A_sp.forward(x) - tf.inv(a_hat * tf.fwd(x))).norm() / A_sp.forward(x).norm()
    return a_hat, err.item()


def build_prior(args, tf, device):
    """Return (prior, schedule, H) for ihdm or cold_diffusion (both DCT-heat)."""
    if args.prior == "ihdm":
        prior, sigmas, cfg = load_ihdm(ckpt=args.ckpt, config_name=args.ihdm_config, device=device)
        H = int(cfg.data.image_size)
        return prior, HeatSchedule(H, H, sigmas, transform=tf, device=device, dtype=torch.float32), H
    H = args.image_size
    model = UNet(ch=args.ch, out_ch=3, ch_mult=tuple(args.ch_mult), num_res_blocks=args.num_res_blocks,
                 attn_resolutions=(args.attn_res,), in_channels=3, resolution=H).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); model.eval()
    sch = HeatSchedule.ihdm(H, H, K=args.K, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                            transform=tf, device=device, dtype=torch.float32)
    return ColdDiffusionPrior(model, sch), sch, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["ihdm", "cold_diffusion"], default="ihdm")
    ap.add_argument("--ckpt", default="checkpoint/ihdm/ihdm_ffhq256_full.pth")
    ap.add_argument("--ihdm_config", default="img_size_256_full")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--ch_mult", type=int, nargs="+", default=[1, 1, 2, 2, 4])
    ap.add_argument("--num_res_blocks", type=int, default=2)
    ap.add_argument("--attn_res", type=int, default=16)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--sigma_min", type=float, default=0.5)
    ap.add_argument("--sigma_max", type=float, default=128.0)
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--observation_dir", required=True)
    ap.add_argument("--kernel_npy", required=True)
    ap.add_argument("--sigma_y", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--prior_weight", type=float, default=1.0)
    ap.add_argument("--data_weight", type=float, default=64.0)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device
    tf = DCTTransform()
    os.makedirs(os.path.join(args.out, "recon"), exist_ok=True)

    prior, sch, H = build_prior(args, tf, device)
    k = torch.from_numpy(np.load(args.kernel_npy))
    a_hat, diag_err = dct_transfer(k, H, tf, device)
    A = SpectralOperator(a_hat, tf)
    corr = MAPCorrection(sch, a_hat, delta=args.delta, sigma_y=args.sigma_y,
                         prior_weight=args.prior_weight, data_weight=args.data_weight, schedule_kind="late")
    times = list(range(sch.num_levels - 1, -1, -1))
    print(f"  DEFOCUS-HQS(DCT closed-form) prior={args.prior} res={H} kernel={tuple(k.shape)} "
          f"DCT-diagonalization err={diag_err:.2e}", flush=True)

    x0 = load(args.clean_dir, H, args.n).to(device)
    ys = load(args.observation_dir, H, args.n).to(device)
    S = {"in": [], "out": [], "ssim": [], "lpips": [], "mc": []}
    for i in range(x0.shape[0]):
        xi, y = x0[i:i+1], ys[i:i+1]
        x_init = terminal_init("matched_measurement", y, sch, times[0])
        xh = scale_matched_solver(y, x_init, times, A, sch, prior, corr,
                                  clamp=(-1, 1), logger=RunLogger(verbose=False), x0_ref=xi)
        save_image((xh[0].clamp(-1, 1) + 1) / 2, os.path.join(args.out, "recon", f"{i:05d}.png"))
        S["in"].append(psnr(y, xi)); S["out"].append(psnr(xh, xi)); S["ssim"].append(ssim(xh, xi))
        S["lpips"].append(lpips_metric(xh, xi, device)); S["mc"].append(measurement_consistency(y, A, xh))
    m = {kk: float(np.mean(v)) for kk, v in S.items()}
    print(f"[defocus-HQS {args.prior}]  sigma_y={args.sigma_y}  PSNR {m['in']:.2f}->{m['out']:.2f}dB  "
          f"SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  MC {m['mc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
