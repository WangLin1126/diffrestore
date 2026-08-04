"""SMDC parallel-beam CT reconstruction with the IHDM heat prior (proof of concept).

Reuses the shared per-step CG data step, only swapping the operator: A = ParallelBeamRadon
(image -> sinogram) and the scale-matched target is L_t y in *sinogram* space, where the companion
L_t = DetectorHeatBlur is the 1-D detector-axis heat blur given by the Fourier-slice intertwining
R(K_t x) = L_t(R x) (ops/ct.py, verified by tests.gates.gate_intertwining_ct).

Per step:  (wp I + wy RᵀR) x = wp mu + wy Rᵀ(L_{t-1} y),  solved by CG (R,Rᵀ only). R is normalized
to ||R||=1 (power iteration) so the deblur weights transfer. Writes recon PNGs and a
Clean|Sinogram|Backproj|SMDC figure. Shared scaffolding is in utils/pipeline.py.

  python scripts/ct.py --n_angles 180 --n 2 --out results/ct_demo
"""
import os, sys, glob, argparse, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ops.transforms import DCTTransform
from ops.ct import ParallelBeamRadon, DetectorHeatBlur
from ops.motion_spatial import cg_solve
from utils import pipeline as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_angles", type=int, default=180)
    ap.add_argument("--sigma_y", type=float, default=0.01)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--cg_iters", type=int, default=12)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--data_weight", type=float, default=64.0)
    ap.add_argument("--prior", choices=["ihdm", "cold_diffusion"], default="ihdm")
    ap.add_argument("--ckpt", default="checkpoint/ihdm/ihdm_ffhq256_full.pth")
    ap.add_argument("--ihdm_config", default="img_size_256_full")
    ap.add_argument("--clean_dir", default="results/gaussian/clean")
    ap.add_argument("--out", default="results/ct_demo")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev, dt = args.device, torch.float32
    tf = DCTTransform()
    prior, sch, H = P.build_prior(args, tf, dev)
    N = sch.num_levels - 1
    R_raw = ParallelBeamRadon(args.n_angles, H, channels=3, device=dev, dtype=dt)
    R, lam = P.normalize_operator(R_raw, H, dev, dt)               # ||R||=1
    Ls = [DetectorHeatBlur(float(sch.sigmas[i]), H, device=dev, dtype=dt) for i in range(sch.num_levels)]

    wp, wy0 = 1.0 / args.delta ** 2, args.data_weight / args.sigma_y ** 2
    times = list(range(N, -1, -1))
    os.makedirs(os.path.join(args.out, "recon"), exist_ok=True)
    print(f"CT-SMDC: H={H} angles={args.n_angles} ||R||={lam:.1f} sigma_y={args.sigma_y} N={N}", flush=True)

    paths = sorted(glob.glob(os.path.join(args.clean_dir, "*.png")))[:args.n]
    panels, S = [], P.new_scores("out", "ssim", "lpips")
    for idx, p in enumerate(paths):
        x0 = P.load_png(p, H, dev)
        with torch.no_grad():
            y = R.forward(x0)
            y = y + args.sigma_y * y.abs().mean() * torch.randn_like(y)
            bp = R.adjoint(y)                                      # un-filtered back-projection (for the figure)
            x_init = cg_solve(lambda v: R.adjoint(R.forward(v)) + 1e-2 * v,
                              R.adjoint(Ls[times[0]].forward(y)), torch.zeros_like(x0), iters=30).clamp(-1, 1)
            x = P.smdc_cg(R, prior, lambda tn, mu: Ls[tn].forward(y), times, wp, wy0, N,
                          x_init=x_init, cg_iters=args.cg_iters)
        P.save_img(os.path.join(args.out, "recon", f"{idx:05d}.png"), x)
        P.add_scores(S, x, x0, dev)
        sino = y[0].mean(0).cpu().numpy()                          # (A,W) for display
        bpn = (bp - bp.min()) / (bp.max() - bp.min()) * 2 - 1
        panels.append((P.to_img(x0), sino, P.to_img(bpn), P.to_img(x), S["out"][-1], S["ssim"][-1]))
        print(f"  img {idx}: SMDC {S['out'][-1]:.2f} dB  SSIM {S['ssim'][-1]:.3f}  LPIPS {S['lpips'][-1]:.3f}", flush=True)

    # figure: rows = images, cols = Clean | Sinogram | Backprojection | SMDC
    cols = ["Clean", f"Sinogram ({args.n_angles} views)", "Back-projection", "SMDC + IHDM"]
    fig, ax = plt.subplots(len(panels), 4, figsize=(12, 3 * len(panels)))
    ax = np.atleast_2d(ax)
    for r, (cl, si, bpv, rec, ps, ss) in enumerate(panels):
        for c, im in enumerate([cl, si, bpv, rec]):
            a = ax[r, c]
            a.imshow(im, cmap="gray" if c == 1 else None, aspect="auto" if c == 1 else "equal")
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(cols[c], fontsize=12)
        ax[r, 3].set_xlabel(f"{ps:.1f} dB / SSIM {ss:.2f}", fontsize=10)
    fig.suptitle(f"SMDC parallel-beam CT ({args.n_angles} views, sigma_y={args.sigma_y}) - CelebA-HQ 256",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figure_ct.png"), dpi=110, bbox_inches="tight")
    m = P.mean_scores(S)
    print(f"[CT-SMDC {args.n_angles}-view]  PSNR {m['out']:.2f} dB  SSIM {m['ssim']:.3f}  "
          f"LPIPS {m['lpips']:.3f}  -> {args.out}/figure_ct.png", flush=True)


if __name__ == "__main__":
    main()
