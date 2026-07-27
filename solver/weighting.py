"""Data-consistency weighting W_t applied to the scale-matched residual.

All three modes are diagonal in the shared basis (MATH.md sec. 5):
  surrogate_l2  : W = I                              (frequency-annealed likelihood)
  regularized   : W = 1 / (sigma_n^2 |g_K|^2 + lam)  (tempered whitening; default)
  exact         : W = 1 / (sigma_n^2 |g_K|^2), support-thresholded (transformed MLE)
"""
from __future__ import annotations
import torch

from ops.heat import HeatSchedule


class Weighting:
    def __init__(self, schedule: HeatSchedule, mode: str, sigma_n: float,
                 A_transfer=None, sigma_xi: float = 0.0, regularizer: float = 1e-3,
                 support_floor: float = 1e-3, use_sigma_eff: bool = False):
        assert mode in ("surrogate_l2", "regularized", "exact")
        self.sch = schedule
        self.mode = mode
        self.sigma_n = float(sigma_n)
        self.A_transfer = A_transfer
        self.sigma_xi = float(sigma_xi)
        self.reg = float(regularizer)
        self.floor = float(support_floor)
        self.use_sigma_eff = use_sigma_eff

    def transfer(self, i: int) -> torch.Tensor:
        g = self.sch.transfer(i)
        if self.mode == "surrogate_l2":
            return torch.ones_like(g)
        s2 = self.sigma_n ** 2
        base = s2 * g * g
        if self.use_sigma_eff and self.A_transfer is not None:
            base = base + (self.sigma_xi ** 2) * (self.A_transfer ** 2)
        if self.mode == "regularized":
            return 1.0 / (base + self.reg)
        # exact: pseudo-inverse restricted to the surviving band (MATH.md sec. 4.2)
        w = torch.zeros_like(g)
        mask = g.abs() > self.floor
        w[mask] = 1.0 / base[mask]
        return w

    def apply(self, residual: torch.Tensor, i: int) -> torch.Tensor:
        w = self.transfer(i)
        if w.device != residual.device:
            w = w.to(residual.device)
        if not torch.is_complex(residual) and w.dtype != residual.dtype:
            w = w.to(residual.dtype)
        return self.sch.transform.inv(w * self.sch.transform.fwd(residual))
