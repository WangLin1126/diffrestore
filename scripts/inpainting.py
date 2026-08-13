"""SMDC inpainting via the re-blur companion + RePaint-style resampling.

A pixel mask ``M`` (1 = observed, 0 = hole) is an orthogonal projection that does **not** commute with
the heat blur (``M K_t != K_t M``), so no exact scale-matched companion ``L_t`` exists. We approximate
it with a *re-blur* step: at each reverse level ``t`` composite the observed data with the current hole
estimate, blur that composite to level ``t``, and pin the observed region to it -- so the injected data
lives at the *state's own blur scale* instead of as sharp values in a blurry field. ``--resample U``
RePaint-style passes per level then propagate the observed data into the holes.

Regime (see docs/hqs_report.tex, Table ``tab:inpaint``): robust on **scattered/random** masks (beats
DDRM), fails on large **contiguous** holes -- a heat prior deblurs, it does not *synthesize*. Cost is
``U`` x the NFE (``U=5`` -> ~1000). Noiseless (clean) inpainting.

  python scripts/inpainting.py --mask random --keep 0.5 --resample 5 --n 50 \
      --clean_dir results200/gaussian_s05/clean --out results/inpaint/rand50
  python scripts/inpainting.py --mask box --box_frac 0.375 --n 16 \
      --clean_dir results200/gaussian_s05/clean --out results/inpaint/box
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from ops.transforms import DCTTransform
from ops.operators import LinearOperator
from utils import pipeline as P
from utils.metrics import ssim as ssim_metric, lpips_metric


class Mask(LinearOperator):
    """Inpainting operator: elementwise multiply by a 0/1 mask. Self-adjoint idempotent projection."""
    def __init__(self, m): self.m = m
    def forward(self, x): return x * self.m
    def adjoint(self, y): return y * self.m


def make_mask(kind, H, k, dev, keep=0.5, box_frac=0.375, seed=1000):
    """Per-image mask (1 = observed). `random`: Bernoulli(keep), seeded by index k. `box`: centered hole."""
    if kind == "box":
        m = torch.ones(1, 1, H, H, device=dev)
        s = int(H * box_frac); a = (H - s) // 2
        m[..., a:a + s, a:a + s] = 0.0
        return m
    g = torch.Generator(device=dev).manual_seed(seed + k)
    return (torch.rand(1, 1, H, H, device=dev, generator=g) < keep).float()


def inpaint(prior, sch, tf, y, m, times, U=5, clamp=(-1.0, 1.0)):
    """Reverse loop: re-blur companion + U RePaint-style resampling passes per level.

    y = observed data (holes zeroed), m = 1/0 mask, times = decreasing blur levels ending at 0.
    Each level t -> t_next: the network deblurs one level (xu); the observed region is pinned to the
    composite (observed data + current hole estimate) *blurred to level t_next* so it matches the
    state's scale; the hole keeps xu. Between passes we re-blur back up to level t and repeat.
    """
    lmb, sig = sch.lmbda, sch.sigmas

    def reblur_up(x, tn, t):                                   # incremental blur level tn -> t (t > tn)
        return tf.inv(torch.exp(-0.5 * (sig[t] ** 2 - sig[tn] ** 2) * lmb) * tf.fwd(x))

    fill = (y.sum() / m.sum().clamp_min(1)).detach()          # mean of observed pixels
    x = sch.apply_K(y + (1.0 - m) * fill, times[0]).clamp(*clamp)   # init: filled holes, blurred to top level
    for (t, tn) in zip(times[:-1], times[1:]):
        for u in range(U):
            xu = prior.reverse_step(x, t, tn)                 # prior deblurs one level
            comp = m * y + (1.0 - m) * xu                     # observed = sharp data, hole = current estimate
            x = (m * sch.apply_K(comp, tn) + (1.0 - m) * xu).clamp(*clamp)   # re-blur companion
            if u < U - 1:
                x = reblur_up(x, tn, t).clamp(*clamp)         # jump back up, resample
    return x


def _psnr(x, xi, region=None):
    """PSNR on [-1,1] (peak 2). `region` (broadcastable 0/1) restricts to a subset, e.g. the hole."""
    d = (x - xi) ** 2
    if region is not None:
        d = d[region.expand_as(d) > 0]
    return 10.0 * np.log10(4.0 / max(d.mean().item(), 1e-12))


def cmd_inpaint(args):
    dev = args.device
    tf = DCTTransform()
    prior, sch, H = P.build_prior(args, tf, dev)
    N = sch.num_levels - 1
    times = list(range(N, -1, -1))
    x0 = P.load_stack(args.clean_dir, H, args.n, dev)
    os.makedirs(os.path.join(args.out, "recon"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "mask"), exist_ok=True)
    fixed = torch.load(args.mask_file).to(dev) if args.mask_file else None   # one shared mask for all
    tag = (f"file:{os.path.basename(args.mask_file)}" if fixed is not None else
           f"{args.mask}({'keep '+str(args.keep) if args.mask=='random' else 'frac '+str(args.box_frac)})")
    print(f"inpaint {tag}  n={x0.shape[0]}  resample U={args.resample}  H={H}  prior={args.prior}", flush=True)
    S = P.new_scores("full", "hole", "ssim", "lpips", "mc")
    for k in range(x0.shape[0]):
        xi = x0[k:k + 1]
        m = fixed if fixed is not None else make_mask(args.mask, H, k, dev, args.keep, args.box_frac, args.mask_seed)
        A = Mask(m)
        y = A.forward(xi)
        if args.noise > 0:                                    # optional obs noise on observed pixels
            y = y + args.noise * m * torch.randn_like(y)
        x = inpaint(prior, sch, tf, y, m, times, U=args.resample)
        S["full"].append(_psnr(x, xi))
        S["hole"].append(_psnr(x, xi, region=(1.0 - m)))
        S["ssim"].append(float(ssim_metric(x, xi)))
        S["lpips"].append(float(lpips_metric(x, xi, dev)))
        S["mc"].append((y - A.forward(x)).norm().item() / (y.norm().item() + 1e-12))
        P.save_img(os.path.join(args.out, "recon", f"{k:05d}.png"), x)
        torch.save(m.cpu(), os.path.join(args.out, "mask", f"{k:05d}.pt"))
    def ms(key):
        v = np.asarray(S[key]); return f"{v.mean():.2f}±{v.std():.2f}"
    def m3(key):
        v = np.asarray(S[key]); return f"{v.mean():.3f}"
    nfail = int(sum(1 for v in S["full"] if v < 15))
    print(f"[{tag}]  full {ms('full')}  hole {ms('hole')}  SSIM {m3('ssim')}  LPIPS {m3('lpips')}  "
          f"MC {m3('mc')}  #<15dB {nfail}/{x0.shape[0]}", flush=True)


def _add_prior(p):
    p.add_argument("--prior", choices=["ihdm", "cold_diffusion"], default="ihdm")
    p.add_argument("--ckpt", default="checkpoint/ihdm/ihdm_ffhq256_full.pth")
    p.add_argument("--ihdm_config", default="img_size_256_full")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--ch", type=int, default=128)
    p.add_argument("--ch_mult", type=int, nargs="+", default=[1, 1, 2, 2, 4])
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--attn_res", type=int, default=16)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--sigma_min", type=float, default=0.5)
    p.add_argument("--sigma_max", type=float, default=128.0)
    p.add_argument("--device", default="cuda:0")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    _add_prior(ap)
    ap.add_argument("--mask", choices=["random", "box"], default="random")
    ap.add_argument("--keep", type=float, default=0.5, help="observed fraction for --mask random")
    ap.add_argument("--box_frac", type=float, default=0.375, help="hole side fraction for --mask box")
    ap.add_argument("--mask_seed", type=int, default=1000, help="base seed; per-image seed = mask_seed + k")
    ap.add_argument("--mask_file", default=None, help="load one fixed .pt mask for all images (baseline head-to-head)")
    ap.add_argument("--resample", type=int, default=5, help="RePaint resampling passes per level (U)")
    ap.add_argument("--noise", type=float, default=0.0, help="optional Gaussian noise std on observed pixels")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--out", default="results/inpaint")
    ap.set_defaults(func=cmd_inpaint)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
