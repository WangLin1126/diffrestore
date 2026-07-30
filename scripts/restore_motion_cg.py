"""Motion deblur with a SPATIAL reflect-boundary operator + per-step CG data step (no FFT).

Same per-step MAP objective as the DFT-HQS pipeline —
    min_x  (wp/2)||x - mu||^2 + (wy/2)||A x - K_{t-1} y||^2
— but A is the true reflect-boundary motion convolution (ops.motion_spatial.SpatialMotionBlur,
exact autograd adjoint), and the data step is solved by CONJUGATE GRADIENT in the spatial
domain instead of the closed-form DFT Wiener filter. No iFFT over the image => no periodic
boundary => no wrap-around ringing. K_{t-1} y (the scale-matched target) is the DCT-heat blur,
which is itself a Neumann/reflect operator, so the whole pipeline is boundary-consistent.

  python scripts/restore_motion_cg.py --prior ihdm --ihdm_config img_size_256_full \
     --ckpt checkpoint/ihdm/ihdm_ffhq256_full.pth --clean_dir results/motion_reflect/clean \
     --observation_dir results/motion_reflect/observation --kernel_npy results/motion_reflect/kernel.npy \
     --sigma_y 0.05 --cg_iters 12 --out results/motion_reflect/ihdm_hqs_cg
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from ops.transforms import DCTTransform
from ops.heat import HeatSchedule
from ops.motion_spatial import SpatialMotionBlur, cg_solve
from model.ihdm import load_ihdm
from model.cold_diffusion import ColdDiffusionPrior
from model.unet import UNet
from utils.metrics import psnr, ssim, lpips_metric


def load_stack(d, H, n=None):
    paths = sorted(glob.glob(os.path.join(d, "*.png")))[:n]
    xs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if im.size != (H, H):
            im = im.resize((H, H), Image.LANCZOS)
        xs.append(torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(xs)


def build_prior(args, tf, device):
    """Return (prior, schedule, H) for ihdm or cold_diffusion."""
    if args.prior == "ihdm":
        prior, sigmas, config = load_ihdm(ckpt=args.ckpt, config_name=args.ihdm_config, device=device)
        H = int(config.data.image_size)
        sch = HeatSchedule(H, H, sigmas, transform=tf, device=device, dtype=torch.float32)
        return prior, sch, H
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
    ap.add_argument("--ckpt", required=True)
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
    ap.add_argument("--kernel_npy", default="results/motion_reflect/kernel.npy")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="results/motion_reflect/ihdm_hqs_cg")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--sigma_y", type=float, default=0.05)
    ap.add_argument("--prior_weight", type=float, default=1.0)
    ap.add_argument("--data_weight", type=float, default=64.0)
    ap.add_argument("--cg_iters", type=int, default=12)
    args = ap.parse_args()

    device = args.device
    tf = DCTTransform()
    os.makedirs(os.path.join(args.out, "recon"), exist_ok=True)
    prior, sch, H = build_prior(args, tf, device)
    N = sch.num_levels - 1

    k = torch.from_numpy(np.load(args.kernel_npy))
    A = SpatialMotionBlur(k, channels=3, device=device, dtype=torch.float32)
    wp = args.prior_weight / (args.delta ** 2)
    wy0 = args.data_weight / (args.sigma_y ** 2)
    times = list(range(N, -1, -1))
    print(f"  MOTION-CG prior={args.prior} res={H} K={N} kernel={tuple(k.shape)} "
          f"cg_iters={args.cg_iters} (spatial reflect A, exact adjoint, no FFT)", flush=True)

    x0 = load_stack(args.clean_dir, H, args.n).to(device)
    ys = load_stack(args.observation_dir, H, args.n).to(device)
    S = {"in": [], "out": [], "ssim": [], "lpips": [], "mc": []}
    for idx in range(x0.shape[0]):
        xi, y = x0[idx:idx + 1], ys[idx:idx + 1]
        with torch.no_grad():
            x = sch.apply_K(y, times[0])                                  # init: heavy heat-blur of obs
            for (t, t_next) in zip(times[:-1], times[1:]):
                mu = prior.reverse_step(x, t, t_next)                     # learned prior mean
                b = sch.apply_K(y, t_next)                                # scale-matched target K_{t-1} y
                wy = wy0 * (1.0 - float(t_next) / N)
                def M(v, wy=wy):
                    return wp * v + wy * A.adjoint(A.forward(v))
                c = wp * mu + wy * A.adjoint(b)
                x = cg_solve(M, c, mu, iters=args.cg_iters).clamp(-1, 1)  # spatial CG, warm-start at mu
        save_image((x[0] + 1) / 2, os.path.join(args.out, "recon", f"{idx:05d}.png"))
        mc = ((y - A.forward(x)).norm() / y.norm()).item()
        S["in"].append(psnr(y, xi)); S["out"].append(psnr(x, xi)); S["ssim"].append(ssim(x, xi))
        S["lpips"].append(lpips_metric(x, xi, device)); S["mc"].append(mc)
    m = {kk: sum(v) / len(v) for kk, v in S.items()}
    print(f"[motion-CG {args.prior} x cg]  PSNR {m['in']:.2f}->{m['out']:.2f}dB  "
          f"SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  MC {m['mc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
