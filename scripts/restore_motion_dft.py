"""Motion deblur with a FULLY DFT-CONSISTENT pipeline (the "assume a DFT-basis prior" test).

Everything lives in the periodic/DFT basis so that the heat kernel K_t and the circular motion
operator A COMMUTE (both DFT-diagonal) — restoring the exact scale-match L_t = K_t that the
SMDC/HQS framework needs. The learned network (trained on DCT-heat) is used as a stand-in for a
DFT-heat prior (interior heat diffusion is nearly identical; only boundaries differ).

  K_t^{DFT}(x) = ifft2( exp(-|w|^2 * sigma_t^2/2) * fft2(x) ),  w = 2*pi*fftfreq(N)   [real symbol]
  scale-match : L_t y = K_t^{DFT} y                                    (exact, since A K = K A)
  HQS MAP     : x = ifft2( (wp*fft2(mu) + wy*conj(OTF)*fft2(L_t y)) / (wp + wy*|OTF|^2) )

Mirrors scripts/restore.py's Gaussian HQS exactly, but in DFT with the motion operator.
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from ops.transforms import DCTTransform
from ops.heat import HeatSchedule
from ops.motion import MotionBlurOperator, edgetaper
from model.cold_diffusion import ColdDiffusionPrior
from model.unet import UNet
from model.ihdm import load_ihdm
from utils.metrics import psnr, ssim, lpips_metric, measurement_consistency


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
    if args.prior == "ihdm":
        prior, sigmas, config = load_ihdm(ckpt=args.ckpt, config_name=args.ihdm_config, device=device)
        H = int(config.data.image_size)
        return prior, np.asarray(sigmas, dtype=np.float64), H
    H = args.image_size
    model = UNet(ch=args.ch, out_ch=3, ch_mult=tuple(args.ch_mult), num_res_blocks=args.num_res_blocks,
                 attn_resolutions=(args.attn_res,), in_channels=3, resolution=H).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); model.eval()
    sch = HeatSchedule.ihdm(H, H, K=args.K, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                            transform=tf, device=device, dtype=torch.float32)
    sig = getattr(sch, "sigmas", getattr(sch, "blur_sigmas", None))
    if torch.is_tensor(sig):
        sig = sig.detach().cpu().numpy()
    return ColdDiffusionPrior(model, sch), np.asarray(sig, dtype=np.float64), H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["cold_diffusion", "ihdm"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--observation_dir", required=True)
    ap.add_argument("--kernel_npy", default="results/motion/kernel.npy")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="results/motion/ihdm_hqs_dft")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--ch_mult", type=int, nargs="+", default=[1, 1, 2, 2, 4])
    ap.add_argument("--num_res_blocks", type=int, default=2)
    ap.add_argument("--attn_res", type=int, default=16)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--sigma_min", type=float, default=0.5)
    ap.add_argument("--sigma_max", type=float, default=128.0)
    ap.add_argument("--ihdm_config", default="img_size_128_maxblur128")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--sigma_y", type=float, default=0.05)
    ap.add_argument("--prior_weight", type=float, default=1.0)
    ap.add_argument("--data_weight", type=float, default=64.0)
    ap.add_argument("--edgetaper", action="store_true", help="edgetaper the obs before FFT (no padding)")
    args = ap.parse_args()

    device = args.device
    tf = DCTTransform()  # only used to build the cold schedule; solver is pure DFT
    os.makedirs(os.path.join(args.out, "recon"), exist_ok=True)
    prior, sigmas, H = build_prior(args, tf, device)
    N = len(sigmas) - 1

    # DFT heat symbol h_t (real, periodic) — the SAME Gaussian as DCTBlur but on angular DFT freqs
    w = 2.0 * np.pi * torch.fft.fftfreq(H, device=device)
    freq2 = (w[:, None] ** 2 + w[None, :] ** 2)                         # (H,W)
    sig_t = torch.tensor(sigmas, device=device, dtype=torch.float32)
    def Ksym(level):                                                   # DFT symbol of K at a level
        return torch.exp(-freq2 * (sig_t[level] ** 2) / 2.0)

    k = torch.from_numpy(np.load(args.kernel_npy))
    A = MotionBlurOperator(k, (H, H), device=device, dtype=torch.float32)
    otf = A.otf
    wp = args.prior_weight / (args.delta ** 2)
    wy0 = args.data_weight / (args.sigma_y ** 2)
    times = list(range(N, -1, -1))
    print(f"  MOTION-DFT prior={args.prior} res={H} K={N} kernel={tuple(k.shape)} "
          f"(A·K commute: |AK-KA| exact in DFT)", flush=True)

    x0 = load_stack(args.clean_dir, H, args.n).to(device)
    ys = load_stack(args.observation_dir, H, args.n).to(device)
    S = {"in": [], "out": [], "ssim": [], "lpips": [], "mc": []}
    for idx in range(x0.shape[0]):
        xi, y = x0[idx:idx + 1], ys[idx:idx + 1]
        y_in = edgetaper(y, k) if args.edgetaper else y
        Y = torch.fft.fft2(y_in)
        with torch.no_grad():
            x = torch.fft.ifft2(Ksym(times[0]) * Y).real                # init: heat-blur obs (DFT)
            for (t, t_next) in zip(times[:-1], times[1:]):
                mu = prior.reverse_step(x, t, t_next)                   # image-domain net (assumed DFT prior)
                Lt_y = Ksym(t_next) * Y                                 # scale-matched measurement (DFT), exact
                wy = wy0 * (1.0 - float(t_next) / N)
                num = wp * torch.fft.fft2(mu) + wy * torch.conj(otf) * Lt_y
                den = wp + wy * (torch.abs(otf) ** 2)
                x = torch.fft.ifft2(num / den).real.clamp(-1, 1)
        save_image((x[0] + 1) / 2, os.path.join(args.out, "recon", f"{idx:05d}.png"))
        S["in"].append(psnr(y, xi)); S["out"].append(psnr(x, xi)); S["ssim"].append(ssim(x, xi))
        S["lpips"].append(lpips_metric(x, xi, device)); S["mc"].append(measurement_consistency(y, A, x))
    m = {kk: sum(v) / len(v) for kk, v in S.items()}
    print(f"[motion-DFT {args.prior} x hqs]  PSNR {m['in']:.2f}->{m['out']:.2f}dB  "
          f"SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  MC {m['mc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
