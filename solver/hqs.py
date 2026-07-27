"""IHDM-HQS-style per-step MAP correction (their data-consistency solver).

Replaces SMDC's 1-step gradient with the full MAP solve balancing the prior reverse-mean mu
and the scale-matched measurement:

    x* = argmin_x  (lambda_p / 2 delta^2) ||x - mu||^2  +  (lambda_y / 2 sigma_y^2) ||A x - L_t y||^2

Because A (and L_t) are DCT-diagonal here, HQS converges to the exact closed form (per frequency):
    x_hat = [ w_p * mu_hat + w_y * a * (L_t y)_hat ] / [ w_p + w_y * a^2 ]
with w_p = lambda_p/delta^2, w_y = lambda_y/sigma_y^2. `schedule='late'` ramps w_y up toward t=0.
"""
from __future__ import annotations
import torch

from ops.heat import HeatSchedule


class MAPCorrection:
    def __init__(self, schedule: HeatSchedule, A_transfer, delta=0.01, sigma_y=0.05,
                 prior_weight=1.0, data_weight=16.0, schedule_kind="late"):
        self.sch = schedule
        self.a = A_transfer
        self.wp0 = float(prior_weight) / (float(delta) ** 2)
        self.wy0 = float(data_weight) / (float(sigma_y) ** 2)
        self.schedule_kind = schedule_kind
        self.N = schedule.num_levels - 1

    def apply(self, mu, y_next, t_next):
        wp, wy = self.wp0, self.wy0
        if self.schedule_kind == "late":
            wy = wy * (1.0 - float(t_next) / self.N)     # data consistency strong near t=0
        a = self.a
        if a.device != mu.device:
            a = a.to(mu.device)
        if a.dtype != mu.dtype:
            a = a.to(mu.dtype)
        tf = self.sch.transform
        num = wp * tf.fwd(mu) + wy * a * tf.fwd(y_next)   # a real (DCT) -> conj(a)=a
        den = wp + wy * (a * a)
        return tf.inv(num / den)
