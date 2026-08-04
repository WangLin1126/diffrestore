"""SMDC single-image super-resolution with the IHDM heat prior (proof of concept).

Reuses the motion-CG data step verbatim, only swapping the operator: A = SuperResolution
(HR image -> LR image, antialias Gaussian blur then area decimation) and the scale-matched
target is L_t y in *LR* space, where the companion L_t is the LR-grid heat blur (std sigma_t/s)
given by the decimation intertwining  A(K_t x) = L_t(A x)  (ops/superres.py, verified by
tests.gates.gate_intertwining_sr to ~2e-4).

Per step:   (wp I + wy A^T A) x = wp mu + wy A^T (L_{t-1} y),   solved by CG (A,A^T only).
A is normalized to ||A||=1 (power iteration) so the deblur weights transfer. Writes recon PNGs
and a Clean|LR (bicubic-up)|SMDC comparison figure.

  python scripts/sr_demo.py --scale 4 --n 4 --out results/sr_demo
"""
import os, sys, glob, argparse, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
from ops.transforms import DCTTransform
from ops.heat import HeatSchedule
from ops.superres import SuperResolution, lr_heat_schedule, sr_scale_matched_target
from ops.motion_spatial import cg_solve
from model.ihdm import load_ihdm
from utils.metrics import psnr, ssim, lpips_metric


def to_img(t):
    return ((t[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=4, help="downsampling factor s")
    ap.add_argument("--aa_sigma", type=float, default=None, help="antialias std (HR px); default=scale")
    ap.add_argument("--decimation", choices=["avgpool", "stride"], default="avgpool")
    ap.add_argument("--exact_intertwine", action="store_true",
                    help="add the QMF alias correction to the data target (plug-in: current prior mean)")
    ap.add_argument("--sigma_y", type=float, default=0.01, help="LR noise std (relative, x |y|.mean())")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cg_iters", type=int, default=12)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--data_weight", type=float, default=64.0)
    ap.add_argument("--ckpt", default="checkpoint/ihdm/ihdm_ffhq256_full.pth")
    ap.add_argument("--ihdm_config", default="img_size_256_full")
    ap.add_argument("--clean_dir", default="results/gaussian/clean")
    ap.add_argument("--out", default="results/sr_demo")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0, help="fixes the LR noise so baselines share the obs")
    args = ap.parse_args()
    dev, dt = args.device, torch.float32
    torch.manual_seed(args.seed)
    tf = DCTTransform()
    prior, sigmas, cfg = load_ihdm(ckpt=args.ckpt, config_name=args.ihdm_config, device=dev)
    H = int(cfg.data.image_size)
    sch = HeatSchedule(H, H, sigmas, transform=tf, device=dev, dtype=dt)
    lr_sch = lr_heat_schedule(sch, args.scale)                  # LR-grid companion (sigma_t/s)
    N = sch.num_levels - 1
    A_raw = SuperResolution(args.scale, H, channels=3, aa_sigma=args.aa_sigma,
                            decimation=args.decimation, device=dev, dtype=dt)
    with torch.no_grad():                                       # normalize ||A||=1
        v = torch.randn(1, 3, H, H, device=dev, dtype=dt)
        for _ in range(20):
            v = A_raw.adjoint(A_raw.forward(v)); v = v / v.norm()
        lam = (v * A_raw.adjoint(A_raw.forward(v))).sum().sqrt().item()

    class A:
        @staticmethod
        def forward(x): return A_raw.forward(x) / lam
        @staticmethod
        def adjoint(y): return A_raw.adjoint(y) / lam

    sig_w = max(args.sigma_y, 0.01)                 # data-weight sigma (floored so sigma_y=0 -> noiseless obs, same wy)
    wp, wy0 = 1.0 / args.delta ** 2, args.data_weight / sig_w ** 2
    times = list(range(N, -1, -1))
    os.makedirs(os.path.join(args.out, "recon"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "clean"), exist_ok=True)          # shared inputs for baselines
    os.makedirs(os.path.join(args.out, "observation"), exist_ok=True)    # physical LR (H/s), un-normalized
    print(f"SR-SMDC: H={H} scale={args.scale} aa={A_raw.aa_sigma} ||A||={lam:.3f} "
          f"sigma_y={args.sigma_y} N={N}", flush=True)

    paths = sorted(glob.glob(os.path.join(args.clean_dir, "*.png")))[:args.n]
    panels, S = [], {"in": [], "out": [], "ssim": [], "lpips": []}
    for idx, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((H, H), Image.LANCZOS)
        x0 = (torch.from_numpy(np.asarray(im, np.float32)).permute(2, 0, 1) / 127.5 - 1.0)[None].to(dev)
        with torch.no_grad():
            y = A.forward(x0)
            if args.sigma_y > 0:
                y = y + args.sigma_y * y.abs().mean() * torch.randn_like(y)
            # bicubic baseline / LR panel from the *physical* LR (A is DC-preserving; y*lam undoes
            # the ||A||=1 rescaling so intensities match x0). SMDC loop keeps the normalized y.
            bic = F.interpolate(y * lam, size=(H, H), mode="bicubic", align_corners=False)
            x = sch.apply_K(bic, times[0]).clamp(-1, 1)                               # heavy-blur init
            for (t, t_next) in zip(times[:-1], times[1:]):
                mu = prior.reverse_step(x, t, t_next)
                # scale-matched target L_{t-1} y (LR space); optionally add the exact QMF alias
                # correction Delta = A(K x) - L(A x), plug-in x = the prior mean mu (has high-freq).
                b = sr_scale_matched_target(A, sch, lr_sch, t_next, y,
                                            x_hr=mu if args.exact_intertwine else None)
                wy = wy0 * (1.0 - float(t_next) / N)
                x = cg_solve(lambda v, wy=wy: wp * v + wy * A.adjoint(A.forward(v)),
                             wp * mu + wy * A.adjoint(b), mu, iters=args.cg_iters).clamp(-1, 1)
        save_image((x[0].clamp(-1, 1) + 1) / 2, os.path.join(args.out, "recon", f"{idx:05d}.png"))
        save_image((x0[0].clamp(-1, 1) + 1) / 2, os.path.join(args.out, "clean", f"{idx:05d}.png"))
        save_image(((y * lam)[0].clamp(-1, 1) + 1) / 2, os.path.join(args.out, "observation", f"{idx:05d}.png"))
        S["in"].append(psnr(bic, x0)); S["out"].append(psnr(x, x0))
        S["ssim"].append(ssim(x, x0)); S["lpips"].append(lpips_metric(x, x0, dev))
        panels.append((to_img(x0), to_img(bic), to_img(x), S["out"][-1], S["ssim"][-1]))
        print(f"  img {idx}: bicubic {S['in'][-1]:.2f} -> SMDC {S['out'][-1]:.2f} dB  "
              f"SSIM {S['ssim'][-1]:.3f}  LPIPS {S['lpips'][-1]:.3f}", flush=True)

    # figure: rows = images, cols = Clean | LR (bicubic up) | SMDC
    cols = ["Clean (HR)", f"LR x{args.scale} (bicubic up)", "SMDC + IHDM"]
    fig, ax = plt.subplots(len(panels), 3, figsize=(9, 3 * len(panels)))
    ax = np.atleast_2d(ax)
    for r, (cl, lo, rec, ps, ss) in enumerate(panels):
        for c, img in enumerate([cl, lo, rec]):
            a = ax[r, c]
            a.imshow(img); a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(cols[c], fontsize=12)
        ax[r, 2].set_xlabel(f"{ps:.1f} dB / SSIM {ss:.2f}", fontsize=10)
    fig.suptitle(f"SMDC single-image super-resolution (x{args.scale}, sigma_y={args.sigma_y}) - FFHQ 256",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figure_sr.png"), dpi=110, bbox_inches="tight")
    m = {k: float(np.mean(v)) for k, v in S.items()}
    print(f"[SR-SMDC x{args.scale}]  bicubic {m['in']:.2f} -> SMDC {m['out']:.2f} dB  "
          f"SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  -> {args.out}/figure_sr.png", flush=True)


if __name__ == "__main__":
    main()
