"""Single-image boundary ablation for motion deblur (IHDM prior, noise=0.05).

Three independent boundary choices, each circular (C) or reflect (R) -> 2^3 = 8 runs:
  A1  synth   : how the degradation A is applied to GT x   -> y = A x + n   (C: circular OTF | R: reflect conv)
  A2  K_t on y: how the scale-matched target K_t y is formed (C: DFT heat symbol | R: DCT heat, Neumann)
  A3  solver  : how the data step inverts A  (C: closed-form iFFT Wiener + edgetaper | R: spatial CG, reflect A)

Also reports the commutation error ||A K_t x - K_t A x|| / ||x|| (the exchangeability the scale-match assumes),
for both the reflect (A_R,K_R) and circular (A_C,K_C) operator pairs.
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from PIL import Image
from ops.motion import MotionBlurOperator, edgetaper
from ops.motion_spatial import SpatialMotionBlur, cg_solve
from model.ihdm import load_ihdm
from ops.heat import HeatSchedule
from ops.transforms import DCTTransform
from utils.metrics import psnr
from utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="results/motion/clean/00000.png")
    ap.add_argument("--ckpt", default="checkpoint/ihdm/ihdm_ffhq256_full.pth")
    ap.add_argument("--kernel_npy", default="results/motion/kernel.npy")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--cg_iters", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    set_seed(0)

    im = Image.open(args.image).convert("RGB").resize((256, 256), Image.LANCZOS)
    x0 = (torch.from_numpy(np.asarray(im, np.float32)).permute(2, 0, 1) / 127.5 - 1)[None].to(dev)
    H = 256
    prior, sigmas, cfg = load_ihdm(ckpt=args.ckpt, config_name="img_size_256_full", device=dev)
    sch = HeatSchedule(H, H, sigmas, transform=DCTTransform(), device=dev, dtype=torch.float32)
    N = sch.num_levels - 1
    k = torch.from_numpy(np.load(args.kernel_npy))
    Ac = MotionBlurOperator(k, (H, H), device=dev, dtype=torch.float32)     # circular
    Ar = SpatialMotionBlur(k, 3, dev, torch.float32)                        # reflect
    otf = Ac.otf

    # DFT heat symbol (circular K)
    w = 2 * np.pi * torch.fft.fftfreq(H, device=dev)
    freq2 = w[:, None] ** 2 + w[None, :] ** 2
    sig = torch.tensor(np.asarray(sigmas), device=dev, dtype=torch.float32)
    def Kc(x, lvl): return torch.fft.ifft2(torch.exp(-freq2 * sig[lvl] ** 2 / 2) * torch.fft.fft2(x)).real
    def Kr(x, lvl): return sch.apply_K(x, lvl)

    Afwd = {"C": lambda x: Ac.forward(x), "R": lambda x: Ar.forward(x)}
    Kap  = {"C": Kc, "R": Kr}

    # ---- commutation error ||A K_t x - K_t A x|| / ||x|| ----
    print("commutation rel-err  ||A K_t x - K_t A x|| / ||x||  (x = GT):")
    for lvl in [40, 100, 160]:
        eR = (Ar.forward(Kr(x0, lvl)) - Kr(Ar.forward(x0), lvl)).norm() / x0.norm()
        eC = (Ac.forward(Kc(x0, lvl)) - Kc(Ac.forward(x0), lvl)).norm() / x0.norm()
        print(f"  t={lvl:3d} (sigma={sig[lvl]:.1f}):  reflect={eR:.2e}   circular={eC:.2e}")

    # ---- data steps ----
    wp = 1.0 / (0.01 ** 2)
    wy0 = 64.0 / (args.noise ** 2)
    def step_C(mu, b):
        num = wp * torch.fft.fft2(mu) + wy0_t * torch.conj(otf) * torch.fft.fft2(b)
        den = wp + wy0_t * (torch.abs(otf) ** 2)
        return torch.fft.ifft2(num / den).real
    def step_R(mu, b):
        M = lambda v: wp * v + wy0_t * Ar.adjoint(Ar.forward(v))
        return cg_solve(M, wp * mu + wy0_t * Ar.adjoint(b), mu, iters=args.cg_iters)

    noise = args.noise * torch.randn_like(x0)
    cc = lambda z: z[..., 64:192, 64:192]
    print("\n A1(synth) A2(K_t.y) A3(solver) | PSNR_full  PSNR_crop128")
    times = list(range(N, -1, -1))
    for a1 in ["C", "R"]:
        y = (Afwd[a1](x0) + noise).clamp(-1, 1)
        for a2 in ["C", "R"]:
            for a3 in ["C", "R"]:
                global wy0_t
                y_used = edgetaper(y, k) if a3 == "C" else y
                with torch.no_grad():
                    x = Kap[a2](y_used, times[0])
                    for (t, t_next) in zip(times[:-1], times[1:]):
                        mu = prior.reverse_step(x, t, t_next)
                        b = Kap[a2](y_used, t_next)
                        wy0_t = wy0 * (1.0 - float(t_next) / N)
                        x = (step_C(mu, b) if a3 == "C" else step_R(mu, b)).clamp(-1, 1)
                pf = psnr(x, x0); pc = psnr(cc(x), cc(x0))
                print(f"    {a1}        {a2}        {a3}      |  {pf:6.2f}     {pc:6.2f}")


if __name__ == "__main__":
    main()
