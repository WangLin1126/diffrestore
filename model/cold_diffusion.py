"""Non-hot IHDM-style prior: an x0-predictor F_theta wrapped as a reverse-degradation step.

Interface B (MATH.md sec. 9 / TASK.md): predict the clean image, then re-degrade
('x0_step_down'):   x_{t'} = x_t - K_t x0_hat + K_{t'} x0_hat,   x0_hat = F_theta(x_t, t).
Because K_t here IS the framework's DCT-heat operator, the prior's degradation and the
scale-matching are identical by construction (no approximation).
"""
from __future__ import annotations
import torch

from model.base import ReversePrior
from ops.heat import HeatSchedule


class ColdDiffusionPrior(ReversePrior):
    def __init__(self, model, schedule: HeatSchedule, clamp_x0=(-1.0, 1.0)):
        self.model = model.eval()
        self.sch = schedule
        self.clamp = clamp_x0

    @torch.no_grad()
    def predict_x0(self, x: torch.Tensor, t: int) -> torch.Tensor:
        step = torch.full((x.shape[0],), float(t), device=x.device)
        x0 = self.model(x, step)
        if self.clamp is not None:
            x0 = x0.clamp(*self.clamp)
        return x0

    @torch.no_grad()
    def reverse_step(self, x, t, t_next, rng=None):
        x0 = self.predict_x0(x, t)
        return x - self.sch.apply_K(x0, t) + self.sch.apply_K(x0, t_next)
