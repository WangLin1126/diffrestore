"""Blurring-Diffusion noise/blur schedules (Hoogeboom & Salimans, ICLR 2023, Appendix A.1).

A blurring diffusion process is an ordinary Gaussian diffusion *in the DCT frequency space*
        q(u_t | u_x) = N(u_t | alpha_t . u_x,  sigma_t^2 I),      u_x = V^T x  (DCT of the image)
with a **per-frequency** signal scale `alpha_t = a_t . d_t` (scalar cosine noise-schedule `a_t`
times the heat-dissipation blur vector `d_t = exp(-lambda . tau_t)`) and a **scalar** noise level
`sigma_t` shared across frequencies.  Eq. (19)-(27) of the paper; `lambda` is the Neumann-DCT
Laplacian spectrum (same as ops.dct.neumann_laplacian_eigs).

`get_alpha_sigma(t)` returns `alpha` of shape (B,1,H,W) (broadcast over channels) and `sigma`
of shape (B,1,1,1); both accept a continuous time tensor `t in [0,1]` of shape (B,).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch

from ops.dct import neumann_laplacian_eigs


@dataclass
class BlurringScheduleConfig:
    image_size: int = 32
    sigma_blur_max: float = 20.0     # max Gaussian blur std applied at t=1 (0 -> plain DDPM)
    logsnr_min: float = -10.0        # variance-preserving cosine noise schedule limits
    logsnr_max: float = 10.0
    min_scale: float = 0.001         # d_min: floor on the frequency damping (Eq. 26)


class BlurringSchedule:
    def __init__(self, cfg: BlurringScheduleConfig, device="cpu", dtype=torch.float32):
        self.cfg = cfg
        self.device, self.dtype = device, dtype
        H = W = cfg.image_size
        # lambda[k,l] = (pi k / H)^2 + (pi l / W)^2  == paper's `labda` (Rissanen et al. 2022)
        self.lmbda = neumann_laplacian_eigs(H, W, device=device, dtype=dtype)   # (H, W)
        # cosine-schedule endpoints depend only on logsnr_min/max -> precompute (scalars)
        self._limit_max = math.atan(math.exp(-0.5 * cfg.logsnr_max))
        self._limit_min = math.atan(math.exp(-0.5 * cfg.logsnr_min)) - self._limit_max

    # ---- blur schedule d_t (per-frequency), Eq. (25),(26) ----------------------------------
    def frequency_scaling(self, t: torch.Tensor) -> torch.Tensor:
        """d_t of shape (B, H, W).  t: (B,) in [0,1]."""
        sigma_blur = self.cfg.sigma_blur_max * torch.sin(t * torch.pi / 2.0) ** 2      # (B,)
        dissipation_time = (sigma_blur ** 2) / 2.0                                     # (B,) = tau_t
        scaling = torch.exp(-self.lmbda[None] * dissipation_time[:, None, None])       # (B,H,W)
        return scaling * (1.0 - self.cfg.min_scale) + self.cfg.min_scale

    # ---- scalar variance-preserving cosine noise schedule a_t, sigma_t ---------------------
    def noise_scaling_cosine(self, t: torch.Tensor):
        """(a_t, sigma_t), each (B,).  a_t^2 + sigma_t^2 = 1 (variance preserving)."""
        logsnr = -2.0 * torch.log(torch.tan(self._limit_min * t + self._limit_max))   # (B,)
        return torch.sqrt(torch.sigmoid(logsnr)), torch.sqrt(torch.sigmoid(-logsnr))

    # ---- combined alpha_t (per-frequency), sigma_t (scalar), Eq. (27) ----------------------
    def get_alpha_sigma(self, t: torch.Tensor):
        """alpha: (B,1,H,W)   sigma: (B,1,1,1).  t: (B,) in [0,1]."""
        t = t.to(device=self.device, dtype=self.dtype)
        d_t = self.frequency_scaling(t)                       # (B,H,W)
        a_t, sigma_t = self.noise_scaling_cosine(t)           # (B,), (B,)
        alpha = (a_t[:, None, None] * d_t)[:, None]           # (B,1,H,W)  broadcast over channels
        sigma = sigma_t[:, None, None, None]                  # (B,1,1,1)
        return alpha, sigma
