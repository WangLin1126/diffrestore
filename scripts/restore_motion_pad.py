"""Motion deblur (DFT-consistent HQS) with EXPAND -> restore -> CROP to kill boundary ringing.

Same fully-DFT HQS pipeline as restore_motion_dft.py, but the whole solve runs on a
reflect-padded canvas of size M = 256 + 2*pad and the result is center-cropped back to 256.

Why this removes the border banding: the periodic (circular) Wiener/HQS update assumes the
signal wraps around. A natural image does not, so the deconvolution injects Gibbs ringing at
the four borders. Reflect-padding makes the extended signal even-symmetric at each edge, so its
periodic continuation is continuous (no wrap jump) -> the ringing is pushed into the padded
margin, which we then crop away. Metrics are still computed on the full original 256x256 region.

  M = 256 + 2*pad,  y_pad = reflect_pad(y, pad)
  OTF, K_t^{DFT} rebuilt at MxM ; HQS MAP solved on the MxM canvas ; net runs at MxM
  x_256 = x_pad[..., pad:pad+256, pad:pad+256]
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

from ops.transforms import DCTTransform
from ops.heat import HeatSchedule
from ops.motion import MotionBlurOperator, psf2otf, edgetaper
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
    ap.add_argument("--out", default="results/motion/ihdm_hqs_pad")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--pad", type=int, default=32, help="reflect-pad width per side; M=256+2*pad")
    ap.add_argument("--pad_mode", default="reflect", choices=["reflect", "replicate", "circular"])
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--ch_mult", type=int, nargs="+", default=[1, 1, 2, 2, 4])
    ap.add_argument("--num_res_blocks", type=int, default=2)
    ap.add_argument("--attn_res", type=int, default=16)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--sigma_min", type=float, default=0.5)
    ap.add_argument("--sigma_max", type=float, default=128.0)
    ap.add_argument("--ihdm_config", default="img_size_256_full")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--sigma_y", type=float, default=0.05)
    ap.add_argument("--prior_weight", type=float, default=1.0)
    ap.add_argument("--data_weight", type=float, default=64.0)
    ap.add_argument("--edgetaper", action="store_true", help="edgetaper the padded canvas before FFT")
    args = ap.parse_args()

    device = args.device
    tf = DCTTransform()
    os.makedirs(os.path.join(args.out, "recon"), exist_ok=True)
    prior, sigmas, H0 = build_prior(args, tf, device)
    N = len(sigmas) - 1
    pad = args.pad
    M = H0 + 2 * pad                                   # padded canvas

    # DFT heat symbol on the MxM canvas (same Gaussian, angular DFT freqs of size M)
    w = 2.0 * np.pi * torch.fft.fftfreq(M, device=device)
    freq2 = (w[:, None] ** 2 + w[None, :] ** 2)
    sig_t = torch.tensor(sigmas, device=device, dtype=torch.float32)
    def Ksym(level):
        return torch.exp(-freq2 * (sig_t[level] ** 2) / 2.0)

    k = torch.from_numpy(np.load(args.kernel_npy))
    otf = psf2otf(k.to(device=device, dtype=torch.float32), (M, M)).to(device)   # OTF at MxM
    A256 = MotionBlurOperator(k, (H0, H0), device=device, dtype=torch.float32)    # for MC on 256
    wp = args.prior_weight / (args.delta ** 2)
    wy0 = args.data_weight / (args.sigma_y ** 2)
    times = list(range(N, -1, -1))
    print(f"  MOTION-PAD prior={args.prior} res={H0} canvas={M} pad={pad}({args.pad_mode}) "
          f"K={N} kernel={tuple(k.shape)}", flush=True)

    x0 = load_stack(args.clean_dir, H0, args.n).to(device)
    ys = load_stack(args.observation_dir, H0, args.n).to(device)
    S = {"in": [], "out": [], "ssim": [], "lpips": [], "mc": []}
    for idx in range(x0.shape[0]):
        xi, y = x0[idx:idx + 1], ys[idx:idx + 1]
        yp = F.pad(y, (pad, pad, pad, pad), mode=args.pad_mode)   # (1,3,M,M)
        if args.edgetaper:
            yp = edgetaper(yp, k)                                 # taper the padded-canvas wrap
        Y = torch.fft.fft2(yp)
        with torch.no_grad():
            x = torch.fft.ifft2(Ksym(times[0]) * Y).real          # init: heat-blur padded obs
            for (t, t_next) in zip(times[:-1], times[1:]):
                mu = prior.reverse_step(x, t, t_next)             # net runs at MxM
                Lt_y = Ksym(t_next) * Y                           # scale-matched (padded, DFT)
                wy = wy0 * (1.0 - float(t_next) / N)
                num = wp * torch.fft.fft2(mu) + wy * torch.conj(otf) * Lt_y
                den = wp + wy * (torch.abs(otf) ** 2)
                x = torch.fft.ifft2(num / den).real.clamp(-1, 1)
        xc = x[..., pad:pad + H0, pad:pad + H0].contiguous()      # crop back to 256
        save_image((xc[0] + 1) / 2, os.path.join(args.out, "recon", f"{idx:05d}.png"))
        S["in"].append(psnr(y, xi)); S["out"].append(psnr(xc, xi)); S["ssim"].append(ssim(xc, xi))
        S["lpips"].append(lpips_metric(xc, xi, device)); S["mc"].append(measurement_consistency(y, A256, xc))
    m = {kk: sum(v) / len(v) for kk, v in S.items()}
    print(f"[motion-PAD {args.prior} x hqs pad={pad}]  PSNR {m['in']:.2f}->{m['out']:.2f}dB  "
          f"SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  MC {m['mc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
