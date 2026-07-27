"""Model-free priors for plumbing tests (Phase 0). No trained network involved."""
from __future__ import annotations
import torch

from model.base import ReversePrior
from ops.heat import HeatSchedule


class NoOpPrior(ReversePrior):
    """Returns the state unchanged; the data-consistency term alone drives the loop."""
    def reverse_step(self, x, t, t_next, rng=None):
        return x


class OracleReblurPrior(ReversePrior):
    """Perfect oracle: re-degrades a known clean reference to level t_next, K_{t_next} x0.

    Used only to verify solver plumbing end-to-end (shapes, indexing, clamp, logging):
    with a perfect prior the reconstruction must equal the reference.
    """
    def __init__(self, x0_ref: torch.Tensor, schedule: HeatSchedule):
        self.x0 = x0_ref
        self.sch = schedule

    def reverse_step(self, x, t, t_next, rng=None):
        return self.sch.apply_K(self.x0, t_next)
