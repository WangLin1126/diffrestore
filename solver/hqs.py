"""IHDM-HQS-style per-step MAP correction (their data-consistency solver).

Replaces SMDC's 1-step gradient with the full MAP solve balancing the prior reverse-mean mu
and the scale-matched measurement:

    x* = argmin_x  (lambda_p / 2 delta^2) ||x - mu||^2  +  (lambda_y / 2 sigma_y^2) ||A x - L_t y||^2

Because A (and L_t) are DCT-diagonal here, HQS converges to the exact closed form (per frequency):
    x_hat = [ (w_p + r) * mu_hat + w_y * a * (L_t y)_hat ] / [ w_p + r + w_y * a^2 ]
with w_p = lambda_p/delta^2, w_y = lambda_y/sigma_y^2. `schedule='late'` ramps w_y up toward t=0.

`freq_reg` (gamma) adds a frequency-aware prior-precision term r(f) = gamma * w_p * (1 - |a(f)|/|a(0)|)
that boosts the prior exactly where the operator gain |a| is small (A's weak/null band) -- the
DDRM/Wiener "trust the prior where the singular values vanish" rule. It flattens the transition-band
noise-amplification peak w_y|a|sigma/(w_p+w_y|a|^2) (peak at |a|=sqrt(w_p/w_y)) that shows up as
speckle. gamma=0 recovers the plain MAP; gamma~0.5 is the deblur default (small, ~free on distortion).
"""
from __future__ import annotations
import torch

from ops.heat import HeatSchedule


class MAPCorrection:
    def __init__(self, schedule: HeatSchedule, A_transfer, delta=0.01, sigma_y=0.05,
                 prior_weight=1.0, data_weight=16.0, schedule_kind="late", freq_reg=0.5):
        self.sch = schedule
        self.a = A_transfer
        self.wp0 = float(prior_weight) / (float(delta) ** 2)
        self.wy0 = float(data_weight) / (float(sigma_y) ** 2)
        self.schedule_kind = schedule_kind
        self.N = schedule.num_levels - 1
        self.freq_reg = float(freq_reg)
        # frequency-aware prior-precision boost r(f) = gamma*w_p*(1 - |a|/|a_dc|), large where |a| small
        amag = A_transfer.abs()
        a_dc = amag.reshape(-1)[0].clamp_min(1e-12)
        self.r = (self.freq_reg * self.wp0 * (1.0 - amag / a_dc)).clamp_min(0.0)   # (H,W)

    def apply(self, mu, y_next, t_next):
        wp, wy = self.wp0, self.wy0
        if self.schedule_kind == "late":
            wy = wy * (1.0 - float(t_next) / self.N)     # data consistency strong near t=0
        a, r = self.a, self.r
        if a.device != mu.device:
            a, r = a.to(mu.device), r.to(mu.device)
        if a.dtype != mu.dtype:
            a, r = a.to(mu.dtype), r.to(mu.dtype)
        tf = self.sch.transform
        wp_r = wp + r                                     # (H,W) frequency-aware prior precision
        num = wp_r * tf.fwd(mu) + wy * a * tf.fwd(y_next) # a real (DCT) -> conj(a)=a
        den = wp_r + wy * (a * a)
        return tf.inv(num / den)
