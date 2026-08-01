"""SMDC parallel-beam CT reconstruction with the IHDM heat prior (proof of concept).

Reuses the motion-CG data step verbatim, only swapping the operator: A = ParallelBeamRadon
(image -> sinogram) and the scale-matched target is L_t y in *sinogram* space, where the
companion L_t = DetectorHeatBlur is the 1-D detector-axis heat blur given by the Fourier-slice
intertwining  R(K_t x) = L_t(R x)  (ops/ct.py, verified by tests.gates.gate_intertwining_ct).

Per step:   (wp I + wy R^T R) x = wp mu + wy R^T (L_{t-1} y),   solved by CG (R,R^T only).
The Radon operator is normalized to ||R||=1 (power iteration) so the deblur weights transfer.
Writes recon PNGs and a Clean|Sinogram|Backproj|SMDC comparison figure.

  python scripts/ct_demo.py --n_angles 180 --n 2 --out results/ct_demo
"""
import os, sys, glob, argparse, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
from ops.transforms import DCTTransform
from ops.heat import HeatSchedule
from ops.ct import ParallelBeamRadon, DetectorHeatBlur
from ops.motion_spatial import cg_solve
from model.ihdm import load_ihdm
from utils.metrics import psnr, ssim, lpips_metric


def to_img(t):
    return ((t[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_angles", type=int, default=180)
    ap.add_argument("--sigma_y", type=float, default=0.01)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--cg_iters", type=int, default=12)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--data_weight", type=float, default=64.0)
    ap.add_argument("--ckpt", default="checkpoint/ihdm/ihdm_ffhq256_full.pth")
    ap.add_argument("--ihdm_config", default="img_size_256_full")
    ap.add_argument("--clean_dir", default="results/gaussian/clean")
    ap.add_argument("--out", default="results/ct_demo")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev, dt = args.device, torch.float32
    tf = DCTTransform()
    prior, sigmas, cfg = load_ihdm(ckpt=args.ckpt, config_name=args.ihdm_config, device=dev)
    H = int(cfg.data.image_size)
    sch = HeatSchedule(H, H, sigmas, transform=tf, device=dev, dtype=dt)
    N = sch.num_levels - 1
    R_raw = ParallelBeamRadon(args.n_angles, H, channels=3, device=dev, dtype=dt)
    with torch.no_grad():                                       # normalize ||R||=1
        v = torch.randn(1, 3, H, H, device=dev, dtype=dt)
        for _ in range(20):
            v = R_raw.adjoint(R_raw.forward(v)); v = v / v.norm()
        lam = (v * R_raw.adjoint(R_raw.forward(v))).sum().sqrt().item()

    class R:
        @staticmethod
        def forward(x): return R_raw.forward(x) / lam
        @staticmethod
        def adjoint(s): return R_raw.adjoint(s) / lam
    Ls = [DetectorHeatBlur(float(sch.sigmas[i]), H, device=dev, dtype=dt) for i in range(sch.num_levels)]

    wp, wy0 = 1.0 / args.delta ** 2, args.data_weight / args.sigma_y ** 2
    times = list(range(N, -1, -1))
    os.makedirs(os.path.join(args.out, "recon"), exist_ok=True)
    print(f"CT-SMDC: H={H} angles={args.n_angles} ||R||={lam:.1f} sigma_y={args.sigma_y} N={N}", flush=True)

    paths = sorted(glob.glob(os.path.join(args.clean_dir, "*.png")))[:args.n]
    panels, S = [], {"out": [], "ssim": [], "lpips": []}
    for idx, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((H, H), Image.LANCZOS)
        x0 = (torch.from_numpy(np.asarray(im, np.float32)).permute(2, 0, 1) / 127.5 - 1.0)[None].to(dev)
        with torch.no_grad():
            y = R.forward(x0)
            y = y + args.sigma_y * y.abs().mean() * torch.randn_like(y)
            bp = R.adjoint(y)                                   # un-filtered back-projection (for the figure)
            x = cg_solve(lambda v: R.adjoint(R.forward(v)) + 1e-2 * v,
                         R.adjoint(Ls[times[0]].forward(y)), torch.zeros_like(x0), iters=30).clamp(-1, 1)
            for (t, t_next) in zip(times[:-1], times[1:]):
                mu = prior.reverse_step(x, t, t_next)
                b = Ls[t_next].forward(y)
                wy = wy0 * (1.0 - float(t_next) / N)
                x = cg_solve(lambda v, wy=wy: wp * v + wy * R.adjoint(R.forward(v)),
                             wp * mu + wy * R.adjoint(b), mu, iters=args.cg_iters).clamp(-1, 1)
        save_image((x[0].clamp(-1, 1) + 1) / 2, os.path.join(args.out, "recon", f"{idx:05d}.png"))
        S["out"].append(psnr(x, x0)); S["ssim"].append(ssim(x, x0)); S["lpips"].append(lpips_metric(x, x0, dev))
        sino = y[0].mean(0).cpu().numpy()                       # (A,W) for display
        bpn = (bp - bp.min()) / (bp.max() - bp.min()) * 2 - 1
        panels.append((to_img(x0), sino, to_img(bpn), to_img(x), S["out"][-1], S["ssim"][-1]))
        print(f"  img {idx}: SMDC {S['out'][-1]:.2f} dB  SSIM {S['ssim'][-1]:.3f}  LPIPS {S['lpips'][-1]:.3f}", flush=True)

    # figure: rows = images, cols = Clean | Sinogram | Backprojection | SMDC
    cols = ["Clean", f"Sinogram ({args.n_angles} views)", "Back-projection", "SMDC + IHDM"]
    fig, ax = plt.subplots(len(panels), 4, figsize=(12, 3 * len(panels)))
    ax = np.atleast_2d(ax)
    for r, (cl, si, bpv, rec, ps, ss) in enumerate(panels):
        for c, img in enumerate([cl, si, bpv, rec]):
            a = ax[r, c]
            a.imshow(img, cmap="gray" if c == 1 else None, aspect="auto" if c == 1 else "equal")
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(cols[c], fontsize=12)
        ax[r, 3].set_xlabel(f"{ps:.1f} dB / SSIM {ss:.2f}", fontsize=10)
    fig.suptitle(f"SMDC parallel-beam CT ({args.n_angles} views, sigma_y={args.sigma_y}) - CelebA-HQ 256",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figure_ct.png"), dpi=110, bbox_inches="tight")
    m = {k: float(np.mean(v)) for k, v in S.items()}
    print(f"[CT-SMDC {args.n_angles}-view]  PSNR {m['out']:.2f} dB  SSIM {m['ssim']:.3f}  "
          f"LPIPS {m['lpips']:.3f}  -> {args.out}/figure_ct.png", flush=True)


if __name__ == "__main__":
    main()
