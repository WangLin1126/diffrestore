"""Blurring-Diffusion reparameterization: forward `diffuse`, `loss`, reverse `denoise` step, and
ancestral `sample` (Hoogeboom & Salimans, ICLR 2023, Alg. 1/2 + Appendix A.1 pseudo-code).

The neural net `f_theta(z_t, t)` predicts epsilon in **pixel space**; all schedule coefficients
are diagonal in the DCT frequency space, so every data-dependent step is a DCT -> per-frequency
multiply -> IDCT.  We reuse ops.dct.dct2 (= V^T, analysis) and ops.dct.idct2 (= V, synthesis).
"""
from __future__ import annotations
import torch

from ops.dct import dct2 as DCT, idct2 as IDCT
from .schedule import BlurringSchedule


class BlurringDiffusion:
    def __init__(self, schedule: BlurringSchedule, delta: float = 1e-8):
        self.sch = schedule
        self.delta = delta          # numerical floor for the denoising-variance divisions

    # ---- forward process: z_t = V(alpha_t . V^T x) + sigma_t eps  (Appendix A.1 `diffuse`) --
    def diffuse(self, x: torch.Tensor, t: torch.Tensor):
        """Sample z_t ~ q(z_t | x).  Returns (z_t, eps).  eps is pixel-space (sigma_t scalar)."""
        alpha, sigma = self.sch.get_alpha_sigma(t)          # (B,1,H,W), (B,1,1,1)
        eps = torch.randn_like(x)
        z_t = IDCT(alpha * DCT(x)) + sigma * eps
        return z_t, eps

    # ---- training loss: unweighted MSE on epsilon (Eq. 24) ---------------------------------
    def loss(self, model, x: torch.Tensor) -> torch.Tensor:
        t = torch.rand(x.shape[0], device=x.device, dtype=x.dtype)   # t ~ U(0,1) per sample
        z_t, eps = self.diffuse(x, t)
        pred = model(z_t, t)
        return ((eps - pred) ** 2).mean()

    # ---- one reverse step z_t -> z_s  (s < t), Eq. (18)/(22)/(23) ---------------------------
    @torch.no_grad()
    def denoise(self, model, z_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor,
                add_noise: bool = True) -> torch.Tensor:
        d = self.delta
        alpha_s, sigma_s = self.sch.get_alpha_sigma(s)
        alpha_t, sigma_t = self.sch.get_alpha_sigma(t)
        alpha_ts = alpha_t / alpha_s                         # (B,1,H,W)
        # analytically >= 0; clamp_min(0) guards float32 round-off so coeff_term1's (sigma2_ts + d)
        # denominator can never flip sign (the paper writes `sigma2_ts + delta`).
        sigma2_ts = (sigma_t ** 2 - alpha_ts ** 2 * sigma_s ** 2).clamp_min(0.0)

        # posterior (denoising) variance, diagonal in frequency; clamp divisions (Appendix A.1)
        sigma2_denoise = 1.0 / torch.clamp(
            1.0 / torch.clamp(sigma_s ** 2, min=d)
            + 1.0 / torch.clamp(sigma_t ** 2 / alpha_ts ** 2 - sigma_s ** 2, min=d),
            min=d)
        coeff_term1 = alpha_ts * sigma2_denoise / (sigma2_ts + d)
        coeff_term2 = sigma2_denoise / (alpha_ts * torch.clamp(sigma_s ** 2, min=d))   # alpha_st = 1/alpha_ts

        # everything is diagonal in the DCT basis: assemble the posterior mean (and add the
        # per-frequency noise) in frequency space, then a single IDCT back to pixels.
        hat_eps = model(z_t, t)                              # pixel-space epsilon prediction
        u_t = DCT(z_t)
        mu_freq = coeff_term1 * u_t + coeff_term2 * (u_t - sigma_t * DCT(hat_eps))
        if add_noise:
            mu_freq = mu_freq + torch.sqrt(sigma2_denoise) * torch.randn_like(u_t)
        return IDCT(mu_freq)

    # ---- ancestral sampling z_T ~ N(0,I) -> z_0 (Algorithm 1) -------------------------------
    @torch.no_grad()
    def sample(self, model, shape, num_steps: int, device=None, dtype=torch.float32,
               last_step_noiseless: bool = True, progress=False):
        device = device or self.sch.device
        z = torch.randn(shape, device=device, dtype=dtype)
        ts = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device, dtype=dtype)
        it = range(num_steps)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc="BDM sampling")
        B = shape[0]
        for i in it:
            t = ts[i].expand(B)
            s = (ts[i] - 1.0 / num_steps).expand(B)
            add_noise = not (last_step_noiseless and i == num_steps - 1)
            z = self.denoise(model, z, t, s, add_noise=add_noise)
        return z
