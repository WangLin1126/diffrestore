"""Scale-matched data-consistency solver — per-step MAP (HQS).

    for (t, t') in consecutive decreasing levels:
        x~  = R_theta(x, t, t')            # learned reverse-degradation step
        y'  = L_t' y                       # scale-matched measurement
        x~  = MAP(x~, y', t')              # closed-form per-step MAP data-consistency step
        x   = clamp(x~).detach()           # no autograd through the prior
"""
from __future__ import annotations
from typing import Sequence, Optional
import torch

from ops.operators import LinearOperator
from ops.heat import HeatSchedule
from model.base import ReversePrior
from utils.logging import RunLogger, StepLog


def scale_matched_solver(
    y: torch.Tensor,
    x_init: torch.Tensor,
    times: Sequence[int],
    A: LinearOperator,
    schedule: HeatSchedule,
    prior: ReversePrior,
    map_correction,
    clamp: Optional[tuple] = (-1.0, 1.0),
    logger: Optional[RunLogger] = None,
    rng=None,
    x0_ref: Optional[torch.Tensor] = None,
    scale_match: bool = True,
    noise_scale: float = 0.0,
    noise_kind: str = "anneal",
) -> torch.Tensor:
    """noise_scale>0 injects small white noise *after* the per-step data-consistency step
    (turning the deterministic reverse into a stochastic one), never on the final step (t_next==0),
    so the returned image carries no added noise. The per-step std is
        std = noise_scale * w(t_next),   w = t_next/N ("anneal") | sigma_{t_next}/sigma_max ("sigma")
              | 1 ("const"),
    so "anneal"/"sigma" fade the noise out toward fine scales. noise_scale=0 == the original
    deterministic solver."""
    if len(times) < 2:
        raise ValueError("need >= 2 levels")
    if any(times[i] <= times[i + 1] for i in range(len(times) - 1)):
        raise ValueError("times must be strictly decreasing")
    if int(times[-1]) != 0:
        raise ValueError("final level must be 0")

    N = max(int(times[0]), 1)
    sig_max = float(schedule.sigmas.max().item())
    x = x_init
    for step, (t, t_next) in enumerate(zip(times[:-1], times[1:])):
        with torch.no_grad():
            x_prior = prior.reverse_step(x, t, t_next, rng)
            prior_norm = (x_prior - x).norm().item()

            # scale_match=False is the ablation: compare A x_t against the *untransformed* y
            y_next = schedule.apply_L(y, t_next) if scale_match else y
            x_corr = map_correction.apply(x_prior, y_next, t_next)     # closed-form per-step MAP
            r = y_next - A.forward(x_corr)
            corr_norm = (x_corr - x_prior).norm().item()
            if clamp is not None:
                x_corr = x_corr.clamp(clamp[0], clamp[1])
            x = x_corr

            # optional stochastic reverse: small noise AFTER data consistency, not on the last step
            if noise_scale > 0.0 and int(t_next) > 0:
                if noise_kind == "anneal":
                    w = float(t_next) / N
                elif noise_kind == "sigma":
                    w = float(schedule.sigmas[int(t_next)].item()) / max(sig_max, 1e-12)
                elif noise_kind == "const":
                    w = 1.0
                else:
                    raise ValueError(f"unknown noise_kind {noise_kind}")
                eps = torch.randn(x.shape, generator=rng, device=x.device, dtype=x.dtype) \
                    if rng is not None else torch.randn_like(x)
                x = x + noise_scale * w * eps
                if clamp is not None:
                    x = x.clamp(clamp[0], clamp[1])

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
