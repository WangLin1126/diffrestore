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
from ops.ct import ParallelBeamRadon, DetectorHeatBlur


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


def gate_ct_adjoint(H=48, n_angles=100, dtype=torch.float64) -> GateResult:
    """Parallel-beam Radon back-projection is the exact adjoint of the forward projector:
    <R x, s> = <x, R^T s>  (R^T obtained by autograd VJP, ops/ct.py)."""
    R = ParallelBeamRadon(n_angles, H, channels=1, dtype=dtype)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 1, H, H, generator=g, dtype=dtype)
    s = torch.randn(1, 1, n_angles, H, generator=g, dtype=dtype)
    lhs = (R.forward(x) * s).sum().item()
    rhs = (x * R.adjoint(s)).sum().item()
    rel = abs(lhs - rhs) / max(1.0, abs(rhs))
    return GateResult("CT adjoint <Rx,s>=<x,R^Ts>", rel, 1e-9, rel < 1e-9)


def gate_intertwining_ct(H=96, n_angles=140, dtype=torch.float64) -> GateResult:
    """Fourier-slice intertwining  R(K_t x) = L_t(R x)  at a FINE heat scale, where L_t is the
    1-D detector-axis heat blur (ops/ct.py DetectorHeatBlur). This is a *continuum* identity, so
    on a discrete rotate-and-sum projector it holds only up to interpolation/boundary error, which
    shrinks with resolution (~4% here at H=96, ~0.2% at the operational H=256 -- see EXPERIMENTS)
    and with blur scale; we assert a few-percent bound at a fine scale on a band-limited field
    (the operational input) -- the regime the SMDC continuation weights most heavily."""
    tf = DCTTransform()
    sch = HeatSchedule.ihdm(H, H, K=40, sigma_min=0.5, sigma_max=16.0, transform=tf, dtype=dtype)
    R = ParallelBeamRadon(n_angles, H, channels=1, dtype=dtype)
    g = torch.Generator().manual_seed(1)
    x = sch.apply_K(torch.randn(1, 1, H, H, generator=g, dtype=dtype), sch.num_levels // 6)  # band-limited
    i = sch.num_levels // 6                                     # fine scale (sigma ~ 1)
    L = DetectorHeatBlur(float(sch.sigmas[i]), H, dtype=dtype)
    rel = (R.forward(sch.apply_K(x, i)) - L.forward(R.forward(x))).norm().item() \
        / max(1e-12, L.forward(R.forward(x)).norm().item())
    return GateResult("CT intertwining R K_t=L_t R (fine)", rel, 6e-2, rel < 6e-2)


ALL_GATES = [
    gate_adjoint, gate_intertwining, gate_limits, gate_noise_cov,
    gate_ct_adjoint, gate_intertwining_ct,
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
