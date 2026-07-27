"""Scale-matched data-consistency solver: prior-first operator splitting (MATH.md eq. 18).

    for (t, t') in consecutive decreasing levels:
        x~  = R_theta(x, t, t')                       # learned reverse-degradation step
        y'  = L_t' y                                  # scale-matched measurement
        x~ += eta * A^T W_t'(y' - A x~)   (xN)        # data-consistency correction(s)
        x   = clamp(x~).detach()                      # no autograd through the prior
"""
from __future__ import annotations
from typing import Sequence, Optional
import torch

from ops.operators import LinearOperator
from ops.heat import HeatSchedule
from model.base import ReversePrior
from solver.weighting import Weighting
from utils.logging import RunLogger, StepLog


def scale_matched_solver(
    y: torch.Tensor,
    x_init: torch.Tensor,
    times: Sequence[int],
    A: LinearOperator,
    schedule: HeatSchedule,
    prior: ReversePrior,
    weighting: Weighting,
    step_fn,
    inner_steps: int = 1,
    clamp: Optional[tuple] = (-1.0, 1.0),
    logger: Optional[RunLogger] = None,
    rng=None,
    x0_ref: Optional[torch.Tensor] = None,
    scale_match: bool = True,
    map_correction=None,
) -> torch.Tensor:
    if len(times) < 2:
        raise ValueError("need >= 2 levels")
    if any(times[i] <= times[i + 1] for i in range(len(times) - 1)):
        raise ValueError("times must be strictly decreasing")
    if int(times[-1]) != 0:
        raise ValueError("final level must be 0")

    x = x_init
    for step, (t, t_next) in enumerate(zip(times[:-1], times[1:])):
        with torch.no_grad():
            x_prior = prior.reverse_step(x, t, t_next, rng)
            prior_norm = (x_prior - x).norm().item()

            # scale_match=False is the ablation: compare A x_t against the *untransformed* y
            y_next = schedule.apply_L(y, t_next) if scale_match else y
            x_corr = x_prior
            if map_correction is not None:                       # IHDM-HQS MAP solve
                x_corr = map_correction.apply(x_prior, y_next, t_next)
                r = y_next - A.forward(x_corr)
            else:                                                # SMDC gradient correction
                r = y_next - A.forward(x_corr)
                for _ in range(inner_steps):
                    r = y_next - A.forward(x_corr)
                    grad = A.adjoint(weighting.apply(r, t_next))
                    eta = step_fn(t_next, x_corr, grad)
                    x_corr = x_corr + eta * grad
            corr_norm = (x_corr - x_prior).norm().item()
            if clamp is not None:
                x_corr = x_corr.clamp(clamp[0], clamp[1])
            x = x_corr

        if logger is not None:
            sc = float("nan")
            if x0_ref is not None:
                ref = schedule.apply_K(x0_ref, t_next)
                sc = (x - ref).norm().item() / (ref.norm().item() + 1e-12)
            logger.log(StepLog(step, float(t), float(t_next),
                               residual_norm=r.norm().item(), correction_norm=corr_norm,
                               prior_step_norm=prior_norm,
                               state_min=x.min().item(), state_max=x.max().item(),
                               state_consistency=sc))
    return x
