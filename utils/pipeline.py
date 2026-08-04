"""Shared building blocks for the degradation restoration scripts (scripts/{sr,deblur,ct}.py).

Everything common across degradation operators lives here so each entry script only has to build
its operator A and the scale-matched target: image IO, prior construction (ihdm/cold), ||A||=1
normalization, the per-step scale-matched CG data step, the DDRM/Wiener frequency-aware
regularizer, the TV+CG baseline, and metric bookkeeping. The closed-form DCT-Wiener data step for
DCT-diagonal operators lives in solver/{hqs,base}.py and is used directly by the scripts.
"""
import os
import glob
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from ops.heat import HeatSchedule
from ops.dct import dct2, idct2
from ops.motion_spatial import cg_solve
from model.ihdm import load_ihdm
from model.cold_diffusion import ColdDiffusionPrior
from model.unet import UNet
from utils.metrics import psnr, ssim, lpips_metric, measurement_consistency


# ----------------------------------------------------------------------- image IO
def load_png(path, size, dev="cpu"):
    """Load one PNG as a (1,3,size,size) tensor in [-1,1]."""
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(im, np.float32)).permute(2, 0, 1) / 127.5 - 1.0
    return x[None].to(dev)


def load_stack(d, size, n=None, dev="cpu"):
    """Stack the first n PNGs in directory d into (N,3,size,size) (resized to `size`)."""
    paths = sorted(glob.glob(os.path.join(d, "*.png")))[:n]
    return torch.cat([load_png(p, size, dev) for p in paths], 0)


def load_dir(d, n=None, dev="cpu"):
    """Stack the first n PNGs in d at their native resolution (no resize) into (N,3,H,W).
    For scoring/figure tools that read recon dirs already at the right size."""
    paths = sorted(glob.glob(os.path.join(d, "*.png")))[:n]
    xs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32)).permute(2, 0, 1)
          / 127.5 - 1.0 for p in paths]
    return torch.stack(xs).to(dev)


def to_img(t):
    """(1,C,H,W) or (C,H,W) in [-1,1] -> HxWx3 numpy in [0,1] for matplotlib."""
    t = t[0] if t.dim() == 4 else t
    return ((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


def save_img(path, x):
    """(1,C,H,W) or (C,H,W) in [-1,1] -> PNG in [0,1] (torchvision, antialiased float)."""
    t = x[0] if x.dim() == 4 else x
    save_image((t.clamp(-1, 1) + 1) / 2, path)


def save_u8(path, x):
    """(C,H,W) in [-1,1] -> uint8 PNG (byte-exact; used by the observation generators)."""
    a = ((x.clamp(-1, 1) + 1) / 2 * 255 + 0.5).byte().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(a).save(path)


# ----------------------------------------------------------------------- prior
def build_prior(args, tf, device):
    """Return (prior, HeatSchedule, image_size) for the ihdm or cold_diffusion prior (both DCT-heat).
    Reads args.prior (default 'ihdm') plus the standard checkpoint/architecture knobs off `args`."""
    if getattr(args, "prior", "ihdm") == "ihdm":
        ckpt = getattr(args, "ckpt", None)
        if ckpt:
            prior, sigmas, cfg = load_ihdm(ckpt=ckpt, config_name=args.ihdm_config, device=device)
        else:
            prior, sigmas, cfg = load_ihdm(config_name=args.ihdm_config, device=device)
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


# ----------------------------------------------------------------------- operator ||A||=1
class _Normalized:
    """Wraps a forward/adjoint operator, dividing both by a fixed scalar so ||A||=1."""
    def __init__(self, A_raw, lam): self._A, self._lam = A_raw, lam
    def forward(self, x): return self._A.forward(x) / self._lam
    def adjoint(self, y): return self._A.adjoint(y) / self._lam


def normalize_operator(A_raw, H, device, dtype=torch.float32, channels=3, iters=20):
    """Power-iterate the top singular value of A_raw; return (A with ||A||=1, lam) so the
    per-step data weights transfer between operators."""
    with torch.no_grad():
        v = torch.randn(1, channels, H, H, device=device, dtype=dtype)
        for _ in range(iters):
            v = A_raw.adjoint(A_raw.forward(v)); v = v / v.norm()
        lam = (v * A_raw.adjoint(A_raw.forward(v))).sum().sqrt().item()
    return _Normalized(A_raw, lam), lam


# ----------------------------------------------------------------------- frequency-aware reg
def make_freq_reg(one_minus_an, gamma, wp):
    """DCT-diagonal prior-precision boost r(f)=gamma*wp*(1-â_n(f)) (DDRM/Wiener: trust the prior in
    A's weak/null band). Returns reg(x)=idct2(r*dct2(x)), or None when gamma==0 (plain MAP)."""
    if gamma == 0:
        return None
    r = (gamma * wp) * one_minus_an[None, None]
    return lambda u: idct2(r * dct2(u))


def hutchinson_transfer(A, H, channels, device, dtype, n_probe=24, seed=0):
    """â_n(f)=sqrt(diag(AᵀA))/max, diagonal in the DCT basis, via Hutchinson probes (works for any
    A, DCT-diagonal or not)."""
    g = torch.Generator(device=device).manual_seed(seed)
    acc = torch.zeros(H, H, device=device, dtype=dtype)
    for _ in range(n_probe):
        xi = (torch.randint(0, 2, (1, channels, H, H), generator=g, device=device).to(dtype) * 2 - 1)
        acc += (dct2(A.adjoint(A.forward(idct2(xi)))) * xi).mean(1)[0]
    an = (acc / n_probe).clamp_min(0).sqrt()
    return an / an.max().clamp_min(1e-12)


# ----------------------------------------------------------------------- SMDC per-step CG solver
def smdc_cg(A, prior, target_fn, times, wp, wy0, N, x_init,
            reg=None, cg_iters=12, clamp=(-1, 1)):
    """Reverse SMDC loop with a per-step CG data step (A, Aᵀ only; no FFT / no closed form):
        (wp I + R + wy AᵀA) x = wp mu + R mu + wy Aᵀ b,   b = target_fn(t_next),  wy = wy0(1 - t/N).
    `target_fn(t_next, mu)` returns the scale-matched target b in measurement space (mu = the
    current prior mean, for operators whose target depends on it); `reg` is an optional
    DCT-diagonal prior-precision boost from make_freq_reg."""
    x = x_init
    for (t, tn) in zip(times[:-1], times[1:]):
        mu = prior.reverse_step(x, t, tn)
        b = target_fn(tn, mu)
        wy = wy0 * (1.0 - float(tn) / N)
        if reg is None:
            M = lambda v, wy=wy: wp * v + wy * A.adjoint(A.forward(v))
            c = wp * mu + wy * A.adjoint(b)
        else:
            M = lambda v, wy=wy: wp * v + reg(v) + wy * A.adjoint(A.forward(v))
            c = wp * mu + reg(mu) + wy * A.adjoint(b)
        x = cg_solve(M, c, mu, iters=cg_iters).clamp(*clamp)
    return x


# ----------------------------------------------------------------------- TV+CG baseline
def tv_cg_restore(A, y, sig2, beta, rho0, gamma, T, cg_iters, dev, x_init=None):
    """Plug-and-play HQS with a Chambolle TV prox and a spatial-CG data step (reflect A, no FFT):
        z = TV(x, beta/rho);   ((1/sig2)AᵀA + rho I) x = (1/sig2)Aᵀy + rho z;   rho *= gamma."""
    from skimage.restoration import denoise_tv_chambolle
    def tvprox(x, w):
        z = denoise_tv_chambolle(x[0].detach().cpu().permute(1, 2, 0).numpy(), weight=w, channel_axis=-1)
        return torch.from_numpy(z).permute(2, 0, 1)[None].float().to(dev)
    x = y.clone() if x_init is None else x_init
    rho = rho0; Aty = A.adjoint(y)
    for _ in range(T):
        z = tvprox(x, beta / rho)
        def M(v, rho=rho): return (1.0 / sig2) * A.adjoint(A.forward(v)) + rho * v
        x = cg_solve(M, (1.0 / sig2) * Aty + rho * z, z, iters=cg_iters)
        rho *= gamma
    return x.clamp(-1, 1)


# ----------------------------------------------------------------------- metric bookkeeping
def new_scores(*keys):
    """Fresh dict-of-lists accumulator, e.g. new_scores('in','out','ssim','lpips','mc')."""
    return {k: [] for k in keys}


def add_scores(S, x, x0, dev, y=None, A=None):
    """Append the metrics present as keys in S (in/out/ssim/lpips/mc) for one (recon, clean) pair."""
    if "in" in S and y is not None: S["in"].append(psnr(y, x0))
    if "out" in S: S["out"].append(psnr(x, x0))
    if "ssim" in S: S["ssim"].append(ssim(x, x0))
    if "lpips" in S: S["lpips"].append(lpips_metric(x, x0, dev))
    if "mc" in S and A is not None and y is not None: S["mc"].append(measurement_consistency(y, A, x))


def mean_scores(S):
    return {k: float(np.mean(v)) for k, v in S.items()}
