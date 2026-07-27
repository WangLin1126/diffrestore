"""Step-size schedules for the data-consistency correction (MATH.md sec. 6)."""
from __future__ import annotations
import torch


def make_step_fn(kind: str, base: float, a2_max: float | None = None, eps: float = 1e-8):
    """Return step_fn(i, x, grad) -> float.

      fixed               : constant `base`
      spectral_safe       : base / a2_max, with base in (0, 2)  (surrogate bound, eq. 15)
      residual_normalized : base * ||x|| / (||grad|| + eps)      (robust across scales)
    """
    if kind == "fixed":
        return lambda i, x, grad: base
    if kind == "spectral_safe":
        assert a2_max is not None
        return lambda i, x, grad: base / a2_max
    if kind == "residual_normalized":
        def f(i, x, grad):
            return base * x.norm().item() / (grad.norm().item() + eps)
        return f
    raise ValueError(f"unknown step kind: {kind}")
