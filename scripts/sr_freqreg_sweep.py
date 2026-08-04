"""Frequency-aware (DDRM/Wiener) data-term regularization for SMDC super-res.

The plain SR data step (w_p I + w_y AᵀA)x = w_p μ + w_y Aᵀb has a noise gain
w_y|Â|σ_y/(w_p+w_y|Â|²) that PEAKS at |Â|=√(w_p/w_y) (the [32,64) transition band) with an
amplitude set by the prior precision, ~independent of σ_y -> that is the stippling. Fix: boost the
prior precision ONLY where the operator can't measure, via a DCT-diagonal weight
    r(f) = freq_reg · w_p · (1 - â_n(f)),   â_n = SuperResolution.transfer_profile()  (1 in passband, 0 in stop band),
added to the per-step normal equations as a roughness-free 'trust the prior in A's null space'
term (DDRM's small-singular-value behaviour):
    (w_p I + r(f) + w_y AᵀA) x = w_p μ + r(f) μ + w_y Aᵀb.
This leaves the well-measured low band (â_n≈1 -> r≈0) untouched. Tests freq_reg on σ_y ∈ {0, 0.01}.

  python scripts/sr_freqreg_sweep.py --n 4 --sigmas 0 0.01 --regs 0 16 64
"""
import os, sys, glob, argparse, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image
from ops.transforms import DCTTransform
from ops.heat import HeatSchedule
from ops.dct import dct2, idct2
from ops.superres import SuperResolution, lr_heat_schedule
from ops.motion_spatial import cg_solve
from model.ihdm import load_ihdm
from utils.metrics import psnr, ssim, lpips_metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.01])
    ap.add_argument("--regs", type=float, nargs="+", default=[0.0, 16.0, 64.0])
    ap.add_argument("--cg_iters", type=int, default=12)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--data_weight", type=float, default=64.0)
    ap.add_argument("--ckpt", default="checkpoint/ihdm/ihdm_ffhq256_full.pth")
    ap.add_argument("--ihdm_config", default="img_size_256_full")
    ap.add_argument("--clean_dir", default="results/gaussian/clean")
    ap.add_argument("--out", default="results/sr_freqreg")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev, dt, s = args.device, torch.float32, args.scale
    tf = DCTTransform()
    prior, sigmas, cfg = load_ihdm(ckpt=args.ckpt, config_name=args.ihdm_config, device=dev)
    H = int(cfg.data.image_size)
    sch = HeatSchedule(H, H, sigmas, transform=tf, device=dev, dtype=dt)
    lr_sch = lr_heat_schedule(sch, s); N = sch.num_levels - 1
    A_raw = SuperResolution(s, H, channels=3, device=dev, dtype=dt)
    with torch.no_grad():                                          # ||A||=1
        v = torch.randn(1, 3, H, H, device=dev)
        for _ in range(20): v = A_raw.adjoint(A_raw.forward(v)); v = v / v.norm()
        lam = (v * A_raw.adjoint(A_raw.forward(v))).sum().sqrt().item()

    class A:
        @staticmethod
        def forward(x): return A_raw.forward(x) / lam
        @staticmethod
        def adjoint(y): return A_raw.adjoint(y) / lam

    one_minus_an = (1.0 - A_raw.transfer_profile()).clamp_min(0.0)[None, None]   # (1,1,H,W)
    wp = 1.0 / args.delta ** 2
    times = list(range(N, -1, -1))
    x0s = []
    for p in sorted(glob.glob(os.path.join(args.clean_dir, "*.png")))[:args.n]:
        im = Image.open(p).convert("RGB").resize((H, H), Image.LANCZOS)
        x0s.append((torch.from_numpy(np.asarray(im, np.float32)).permute(2, 0, 1) / 127.5 - 1.0)[None].to(dev))
    os.makedirs(args.out, exist_ok=True)
    print(f"SR freq-reg sweep: n={len(x0s)} H={H} scale={s} ||A||={lam:.3f}", flush=True)
    print(f"\n{'sigma_y':>7} {'freq_reg':>8} | {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6}")
    print("-" * 46)

    for sy in args.sigmas:
        sig_w = max(sy, 0.01)
        wy0 = args.data_weight / sig_w ** 2
        for gamma in args.regs:
            r = (gamma * wp) * one_minus_an                        # (1,1,H,W) DCT-diag prior boost
            def reg(u, r=r): return idct2(r * dct2(u))
            S = {"out": [], "ssim": [], "lpips": []}
            for k, x0 in enumerate(x0s):
                torch.manual_seed(args.seed + k)                   # same noise draw across gamma
                with torch.no_grad():
                    y = A.forward(x0)
                    if sy > 0: y = y + sy * y.abs().mean() * torch.randn_like(y)
                    x = sch.apply_K(F.interpolate(y * lam, size=(H, H), mode="bicubic",
                                                  align_corners=False), times[0]).clamp(-1, 1)
                    for (t, tn) in zip(times[:-1], times[1:]):
                        mu = prior.reverse_step(x, t, tn)
                        b = lr_sch.apply_K(y, tn)
                        wy = wy0 * (1.0 - float(tn) / N)
                        M = lambda v, wy=wy: wp * v + reg(v) + wy * A.adjoint(A.forward(v))
                        c = wp * mu + reg(mu) + wy * A.adjoint(b)
                        x = cg_solve(M, c, mu, iters=args.cg_iters).clamp(-1, 1)
                S["out"].append(psnr(x, x0)); S["ssim"].append(ssim(x, x0)); S["lpips"].append(lpips_metric(x, x0, dev))
                if k == 0:
                    d = os.path.join(args.out, f"sy{sy:g}_reg{gamma:g}"); os.makedirs(d, exist_ok=True)
                    save_image((x[0].clamp(-1, 1) + 1) / 2, os.path.join(d, "00000.png"))
            m = {kk: float(np.mean(vv)) for kk, vv in S.items()}
            print(f"{sy:>7.3g} {gamma:>8.3g} | {m['out']:>6.2f} {m['ssim']:>6.3f} {m['lpips']:>6.3f}", flush=True)


if __name__ == "__main__":
    main()
