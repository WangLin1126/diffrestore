"""Correctness checks for the BDM reparameterization (run: python -m model.bdm_backbone.sanity_check).

1. Variance preservation of the scalar noise schedule:  a_t^2 + sigma_t^2 = 1.
2. Boundary:  at t=0  alpha=1 (all freqs) and sigma~0, so diffuse(x,0) ~= x.
3. Oracle reverse chain:  feeding the *true* epsilon into denoise() must invert the forward
   process (z_1 -> z_0 ~= x).  This validates the mu / posterior-variance coefficients (Eq. 18/23).
4. Model + loss smoke: BDMDenoiser runs and the loss backpropagates.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from ops.dct import dct2 as DCT, idct2 as IDCT
from utils.metrics import psnr
from model.bdm_backbone.schedule import BlurringSchedule, BlurringScheduleConfig
from model.bdm_backbone.diffusion import BlurringDiffusion
from model.bdm_backbone.model import build_bdm_model


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dt = torch.float64
    H = 32
    cfg = BlurringScheduleConfig(image_size=H, sigma_blur_max=20.0)
    sch = BlurringSchedule(cfg, device=dev, dtype=dt)
    diff = BlurringDiffusion(sch)

    # (1) variance preservation
    tt = torch.linspace(0, 1, 11, device=dev, dtype=dt)
    a, s = sch.noise_scaling_cosine(tt)
    vp = (a ** 2 + s ** 2 - 1).abs().max().item()
    print(f"[1] variance preserving  max|a^2+sigma^2-1| = {vp:.2e}  {'OK' if vp < 1e-10 else 'FAIL'}")

    # (2) boundary t=0: blur fully off (d_0 = 1); noise = VP cosine endpoint sqrt(sigmoid(-logsnr_max))
    d0 = sch.frequency_scaling(torch.zeros(1, device=dev, dtype=dt))
    _, sigma0 = sch.get_alpha_sigma(torch.zeros(1, device=dev, dtype=dt))
    dd = (d0 - 1).abs().max().item()
    sig_expected = torch.sigmoid(torch.tensor(-cfg.logsnr_max, dtype=dt)).sqrt().item()
    ds = abs(sigma0.item() - sig_expected)
    print(f"[2] t=0: max|d_0-1| = {dd:.2e}  sigma_0 = {sigma0.item():.2e} (expected {sig_expected:.2e})  "
          f"{'OK' if dd < 1e-10 and ds < 1e-6 else 'FAIL'}")

    # a smooth test image (band-limited), values in [-1,1]
    torch.manual_seed(0)
    x = IDCT(DCT(torch.randn(1, 3, H, H, device=dev, dtype=dt)) *
             torch.exp(-sch.lmbda[None, None] * 1.5))
    x = (x / x.abs().max()).clamp(-1, 1)

    # (3) oracle reverse chain: model returns the exact epsilon for the current z_t given x
    class Oracle:
        def __call__(self, z_t, t):
            alpha, sigma = sch.get_alpha_sigma(t)
            return (z_t - IDCT(alpha * DCT(x))) / sigma          # exact pixel-space eps

    T = 200
    z, _ = diff.diffuse(x, torch.ones(1, device=dev, dtype=dt))   # start at the true forward z_1
    ts = torch.linspace(1.0, 1.0 / T, T, device=dev, dtype=dt)
    for i in range(T):
        z = diff.denoise(Oracle(), z, ts[i].expand(1), (ts[i] - 1.0 / T).expand(1),
                         add_noise=(i != T - 1))
    p = psnr(z.clamp(-1, 1), x)
    print(f"[3] oracle reverse chain (z_1->z_0)  PSNR(z_0, x) = {p:.1f} dB  {'OK' if p > 45 else 'FAIL'}")

    # (4) model + loss smoke (float32, tiny net)
    m = build_bdm_model(image_size=H, in_channels=3, ch=32, ch_mult=(1, 2),
                        num_res_blocks=1, attn_resolutions=(16,)).to(dev)
    diff32 = BlurringDiffusion(BlurringSchedule(cfg, device=dev, dtype=torch.float32))
    xin = x.float()
    loss = diff32.loss(m, xin)
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
    print(f"[4] model+loss smoke  loss={loss.item():.4f}  grad_norm={gnorm:.2f}  "
          f"{'OK' if gnorm > 0 else 'FAIL'}")
    with torch.no_grad():
        samp = diff32.sample(m, (2, 3, H, H), num_steps=8, device=dev)
    print(f"    sample shape {tuple(samp.shape)}  finite={torch.isfinite(samp).all().item()}")


if __name__ == "__main__":
    main()
