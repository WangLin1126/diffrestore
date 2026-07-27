"""Numerical acceptance gates (TASK.md sec. 8 / MATH.md [TEST] markers).

Run:  python -m smdc.scripts.run_tests
All model-free gates (1-5) plus a solver plumbing check. Gate 6 (state semantics) needs a
trained prior and is exercised in Phase 1.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from ops.transforms import DCTTransform
from ops.heat import HeatSchedule
from ops.deblur import build_deblur
from solver.weighting import Weighting
from solver.step import make_step_fn
from solver.init import terminal_init
from model.identity import OracleReblurPrior
from solver.base import scale_matched_solver
from utils.logging import RunLogger


@dataclass
class GateResult:
    name: str
    value: float
    tol: float
    passed: bool


def _setup(H=32, W=32, dtype=torch.float64, device="cpu", blur_sigma=2.0,
           K=40, sigma_max=32.0):
    tf = DCTTransform()
    sch = HeatSchedule.ihdm(H, W, K=K, sigma_min=0.5, sigma_max=sigma_max,
                            transform=tf, device=device, dtype=dtype)
    A, gA = build_deblur(H, W, blur_sigma, transform=tf, device=device, dtype=dtype)
    return tf, sch, A, gA


def gate_adjoint(H=32, W=32, dtype=torch.float64) -> GateResult:
    _, _, A, _ = _setup(H, W, dtype)
    v = A.adjoint_test((1, 3, H, W), dtype=dtype)
    return GateResult("adjoint  <Ax,z>=<x,A^Tz>", v, 1e-9, v < 1e-9)


def gate_intertwining(H=32, W=32, dtype=torch.float64) -> GateResult:
    _, sch, A, _ = _setup(H, W, dtype)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3, H, W, generator=g, dtype=dtype)
    worst = 0.0
    for i in range(0, sch.num_levels, max(1, sch.num_levels // 8)):
        lhs = sch.apply_L(A.forward(x), i)      # L_i A x
        rhs = A.forward(sch.apply_K(x, i))      # A K_i x
        rel = (lhs - rhs).norm().item() / max(1.0, rhs.norm().item())
        worst = max(worst, rel)
    return GateResult("intertwining L_tA = AK_t", worst, 1e-9, worst < 1e-9)


def gate_limits(H=32, W=32, dtype=torch.float64) -> GateResult:
    _, sch, A, _ = _setup(H, W, dtype)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 3, H, W, generator=g, dtype=dtype)
    e_k0 = (sch.apply_K(x, 0) - x).norm().item() / x.norm().item()
    e_l0 = (sch.apply_L(x, 0) - x).norm().item() / x.norm().item()
    worst = max(e_k0, e_l0)
    return GateResult("limits K_0=L_0=I", worst, 1e-9, worst < 1e-9)


def gate_gradient(H=16, W=16, dtype=torch.float64) -> GateResult:
    """Analytic ascent A^T W r must equal -grad of D = 0.5 <r, W r>. Both modes."""
    tf, sch, A, gA = _setup(H, W, dtype)
    i = sch.num_levels // 2
    g = torch.Generator().manual_seed(2)
    y_t = torch.randn(1, 3, H, W, generator=g, dtype=dtype)
    worst = 0.0
    for mode in ("surrogate_l2", "regularized"):
        w = Weighting(sch, mode, sigma_n=0.05, A_transfer=gA, regularizer=1e-2)
        x = torch.randn(1, 3, H, W, generator=g, dtype=dtype, requires_grad=True)
        r = y_t - A.forward(x)
        wr = w.apply(r, i)
        D = 0.5 * (r * wr).sum()
        (grad_ad,) = torch.autograd.grad(D, x)
        with torch.no_grad():
            r2 = y_t - A.forward(x)
            ascent = A.adjoint(w.apply(r2, i))          # A^T W r
        rel = (ascent - (-grad_ad)).norm().item() / max(1.0, grad_ad.norm().item())
        worst = max(worst, rel)
    return GateResult("gradient analytic=autograd", worst, 1e-8, worst < 1e-8)


def gate_noise_cov(H=8, W=8, dtype=torch.float64) -> GateResult:
    """Empirical per-frequency power of n_t=L_t n vs sigma_n^2 |g_t|^2 (MATH.md eq. 8)."""
    tf, sch, A, _ = _setup(H, W, dtype)
    i = sch.num_levels // 3
    sigma_n = 0.1
    B = 20000
    g = torch.Generator().manual_seed(3)
    n = sigma_n * torch.randn(B, 1, H, W, generator=g, dtype=dtype)
    n_t = sch.apply_L(n, i)
    P_emp = (tf.fwd(n_t) ** 2).mean(dim=0).squeeze(0)          # (H,W)
    P_mod = sch.sigma_transfer(i, sigma_n)                     # (H,W)
    mask = sch.transfer(i) > 1e-3
    rel = ((P_emp - P_mod).abs()[mask] / (P_mod[mask] + 1e-12)).mean().item()
    return GateResult("noise cov Cov(L_t n)", rel, 0.05, rel < 0.05)


def gate_solver_plumbing(H=32, W=32, dtype=torch.float64) -> GateResult:
    """With a perfect oracle prior the reconstruction must equal the reference image."""
    tf, sch, A, gA = _setup(H, W, dtype)
    g = torch.Generator().manual_seed(4)
    x0 = 0.5 * torch.randn(1, 3, H, W, generator=g, dtype=dtype).clamp(-1, 1)
    y = A.forward(x0)  # noiseless: perfect prior + zero residual must recover x0 exactly
    times = list(range(sch.num_levels - 1, -1, -1))
    prior = OracleReblurPrior(x0, sch)
    w = Weighting(sch, "regularized", sigma_n=0.02, A_transfer=gA, regularizer=1e-3)
    step_fn = make_step_fn("residual_normalized", base=0.2)
    x_init = terminal_init("matched_measurement", y, sch, times[0])
    x_hat = scale_matched_solver(y, x_init, times, A, sch, prior, w, step_fn,
                                 inner_steps=1, clamp=(-1, 1))
    rel = (x_hat - x0).norm().item() / x0.norm().item()
    return GateResult("solver plumbing (oracle)", rel, 1e-2, rel < 1e-2)


ALL_GATES = [
    gate_adjoint, gate_intertwining, gate_limits,
    gate_gradient, gate_noise_cov, gate_solver_plumbing,
]


def run_all() -> bool:
    print("=" * 66)
    print("SMDC numerical gates (Phase 0)")
    print("=" * 66)
    ok = True
    for fn in ALL_GATES:
        res = fn()
        ok &= res.passed
        flag = "PASS" if res.passed else "FAIL"
        print(f"  [{flag}] {res.name:32s} value={res.value:.3e} (tol {res.tol:.0e})")
    print("=" * 66)
    print("ALL PASSED" if ok else "SOME GATES FAILED")
    return ok
