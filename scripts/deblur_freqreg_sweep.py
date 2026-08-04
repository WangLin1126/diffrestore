"""Test the frequency-aware (DDRM/Wiener) data regularization on DEBLUR (Gaussian + motion), the
same fix that removed the SR speckle. Same per-step CG data step as SR/motion, with a DCT-diagonal
prior-precision boost in the operator's weak band:
    (w_p I + r + w_y AᵀA) x = w_p μ + r μ + w_y Aᵀ b,   r(f) = γ·w_p·(1 - â_n(f)),
where â_n(f) = normalized per-DCT-frequency gain of A (1 in passband → 0 where A is weak/null):
Gaussian A is DCT-diagonal so â_n = its transfer analytically; motion A is reflect-spatial (not
diagonal) so â_n is Hutchinson-estimated diag(AᵀA). Scale-matched target b = K_{t-1} y (L=K).

  python scripts/deblur_freqreg_sweep.py --n 4 --regs 0 16 64 --operators gaussian motion
"""
import os, sys, glob, argparse, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from PIL import Image
from torchvision.utils import save_image
from ops.transforms import DCTTransform
from ops.heat import HeatSchedule
from ops.dct import dct2, idct2
from ops.deblur import build_deblur, gaussian_blur_transfer
from ops.spectral import SpectralOperator
from ops.motion_spatial import SpatialMotionBlur, cg_solve
from model.ihdm import load_ihdm
from utils.metrics import psnr, ssim, lpips_metric, measurement_consistency


def load_stack(d, H, n, dev):
    xs = []
    for p in sorted(glob.glob(os.path.join(d, "*.png")))[:n]:
        im = Image.open(p).convert("RGB")
        if im.size != (H, H): im = im.resize((H, H), Image.LANCZOS)
        xs.append((torch.from_numpy(np.asarray(im, np.float32)).permute(2, 0, 1) / 127.5 - 1.0)[None])
    return [x.to(dev) for x in xs]


@torch.no_grad()
def hutch_an(A, H, C, dev, dt, n_probe=24, seed=0):
    """â_n(f) = sqrt(diag(AᵀA))/max, diagonal in DCT basis via Hutchinson (works for any A)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    acc = torch.zeros(H, H, device=dev, dtype=dt)
    for _ in range(n_probe):
        xi = (torch.randint(0, 2, (1, C, H, H), generator=g, device=dev).to(dt) * 2 - 1)
        acc += (dct2(A.adjoint(A.forward(idct2(xi)))) * xi).mean(1)[0]
    an = (acc / n_probe).clamp_min(0).sqrt()
    return an / an.max().clamp_min(1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operators", nargs="+", default=["gaussian", "motion", "defocus"],
                    choices=["gaussian", "motion", "defocus"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--regs", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--blur_sigma", type=float, default=4.0)
    ap.add_argument("--sigma_y", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--data_weight", type=float, default=64.0)
    ap.add_argument("--cg_iters", type=int, default=12)
    ap.add_argument("--ckpt", default="checkpoint/ihdm/ihdm_ffhq256_full.pth")
    ap.add_argument("--ihdm_config", default="img_size_256_full")
    ap.add_argument("--gaussian_dir", default="results/gaussian")
    ap.add_argument("--motion_dir", default="results/motion_reflect")
    ap.add_argument("--defocus_dir", default="results/defocus")
    ap.add_argument("--out", default="results/deblur_freqreg")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev, dt = args.device, torch.float32
    tf = DCTTransform()
    prior, sigmas, cfg = load_ihdm(ckpt=args.ckpt, config_name=args.ihdm_config, device=dev)
    H = int(cfg.data.image_size)
    sch = HeatSchedule(H, H, sigmas, transform=tf, device=dev, dtype=dt)
    N = sch.num_levels - 1
    wp, wy0 = 1.0 / args.delta ** 2, args.data_weight / args.sigma_y ** 2
    times = list(range(N, -1, -1))
    print(f"deblur freq-reg sweep: n={args.n} H={H} sigma_y={args.sigma_y}", flush=True)
    print(f"\n{'operator':>9} {'freq_reg':>8} | {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6} {'MC':>6}")
    print("-" * 52)

    for opname in args.operators:
        if opname == "gaussian":
            A, _ = build_deblur(H, H, args.blur_sigma, transform=tf, device=dev, dtype=dt)
            an = gaussian_blur_transfer(H, H, args.blur_sigma, dev, dt)      # DCT-diagonal transfer = â_n
            src = args.gaussian_dir
        elif opname == "motion":
            k = torch.from_numpy(np.load(os.path.join(args.motion_dir, "kernel.npy")))
            A = SpatialMotionBlur(k, channels=3, device=dev, dtype=dt)
            an = hutch_an(A, H, 3, dev, dt)                                  # probed diag(AᵀA)
            src = args.motion_dir
        else:  # defocus: symmetric disk -> DCT-diagonal (a_hat = DCT(A.IDCT(1))), like gaussian
            k = torch.from_numpy(np.load(os.path.join(args.defocus_dir, "kernel.npy")))
            a_hat = tf.fwd(SpatialMotionBlur(k, 3, device=dev, dtype=dt).forward(
                tf.inv(torch.ones(1, 3, H, H, device=dev))))[0, 0]
            A = SpectralOperator(a_hat, tf)
            an = a_hat.abs() / a_hat.abs().reshape(-1)[0].clamp_min(1e-12)
            src = args.defocus_dir
        one_minus_an = (1.0 - an).clamp_min(0.0)[None, None]
        cleans = load_stack(os.path.join(src, "clean"), H, args.n, dev)
        obses = load_stack(os.path.join(src, "observation"), H, args.n, dev)

        for gamma in args.regs:
            r = (gamma * wp) * one_minus_an
            def reg(u, r=r): return idct2(r * dct2(u))
            S = {"out": [], "ssim": [], "lpips": [], "mc": []}
            for j, (x0, y) in enumerate(zip(cleans, obses)):
                with torch.no_grad():
                    x = sch.apply_K(y, times[0]).clamp(-1, 1)
                    for (t, tn) in zip(times[:-1], times[1:]):
                        mu = prior.reverse_step(x, t, tn)
                        b = sch.apply_K(y, tn)
                        wy = wy0 * (1.0 - float(tn) / N)
                        M = lambda v, wy=wy: wp * v + reg(v) + wy * A.adjoint(A.forward(v))
                        c = wp * mu + reg(mu) + wy * A.adjoint(b)
                        x = cg_solve(M, c, mu, iters=args.cg_iters).clamp(-1, 1)
                S["out"].append(psnr(x, x0)); S["ssim"].append(ssim(x, x0))
                S["lpips"].append(lpips_metric(x, x0, dev)); S["mc"].append(measurement_consistency(y, A, x))
                if j == 0:
                    d = os.path.join(args.out, f"{opname}_reg{gamma:g}"); os.makedirs(d, exist_ok=True)
                    save_image((x[0].clamp(-1, 1) + 1) / 2, os.path.join(d, "00000.png"))
            m = {kk: float(np.mean(vv)) for kk, vv in S.items()}
            print(f"{opname:>9} {gamma:>8.3g} | {m['out']:>6.2f} {m['ssim']:>6.3f} {m['lpips']:>6.3f} {m['mc']:>6.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
