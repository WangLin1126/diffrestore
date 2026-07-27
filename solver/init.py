"""Terminal-state initialization x_T (MATH.md sec. 8). No universal Gaussian for cold paths."""
from __future__ import annotations
import torch

from ops.heat import HeatSchedule


def terminal_init(mode: str, y: torch.Tensor, schedule: HeatSchedule, t_top: int,
                  A_transfer=None, support_floor: float = 1e-2) -> torch.Tensor:
    if mode == "matched_measurement":
        # x_T = L_T y  (recommended): on the surviving band A ~ 1 and noise is killed.
        return schedule.apply_L(y, t_top)
    if mode == "measurement":
        return y.clone()
    if mode == "zeros":
        return torch.zeros_like(y)
    if mode == "coarse_ls":
        # pseudo-inverse of A applied to y_T (diagonal): g_A / (g_A^2) on the support.
        assert A_transfer is not None
        yT = schedule.apply_L(y, t_top)
        w = torch.zeros_like(A_transfer)
        m = A_transfer.abs() > support_floor
        w[m] = 1.0 / A_transfer[m]
        tf = schedule.transform
        return tf.inv((w.to(yT.dtype)) * tf.fwd(yT))
    raise ValueError(f"unknown init mode: {mode}")
