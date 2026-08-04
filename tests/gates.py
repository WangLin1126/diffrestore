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
from ops.superres import SuperResolution, lr_heat_schedule, sr_scale_matched_target


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


def gate_sr_adjoint(H=64, scale=2, dtype=torch.float64) -> GateResult:
    """SR forward A = avg_pool_s . B_aa has an exact autograd-VJP adjoint (ops/superres.py):
    <A x, y> = <x, A^T y>."""
    A = SuperResolution(scale, H, channels=1, dtype=dtype)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 1, H, H, generator=g, dtype=dtype)
    y = torch.randn(1, 1, H // scale, H // scale, generator=g, dtype=dtype)
    lhs = (A.forward(x) * y).sum().item()
    rhs = (x * A.adjoint(y)).sum().item()
    rel = abs(lhs - rhs) / max(1.0, abs(rhs))
    return GateResult("SR adjoint <Ax,y>=<x,A^Ty>", rel, 1e-9, rel < 1e-9)


def gate_intertwining_sr(H=64, scale=2, dtype=torch.float64) -> GateResult:
    """SR intertwining  A(K_t x) = L_t(A x)  with L_t = LR-grid heat blur (sigma_t/s). Decimating
    an isotropically heat-blurred image equals heat-blurring the decimated image on the LR grid.
    Because the antialias B_aa commutes with K_t exactly (shared DCT basis) and avg-pool lands on
    the LR half-sample grid, the only residual is the aliasing B_aa suppresses -- ~2e-4 here, far
    tighter than CT's few percent (see EXPERIMENTS)."""
    tf = DCTTransform()
    sch = HeatSchedule.ihdm(H, H, K=40, sigma_min=0.5, sigma_max=16.0, transform=tf, dtype=dtype)
    A = SuperResolution(scale, H, channels=1, dtype=dtype)
    lr = lr_heat_schedule(sch, scale)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 1, H, H, generator=g, dtype=dtype)
    worst = 0.0
    for i in range(1, sch.num_levels, max(1, sch.num_levels // 8)):
        lhs = A.forward(sch.apply_K(x, i))     # A K_t x  (HR blur then decimate)
        rhs = lr.apply_K(A.forward(x), i)      # L_t A x  (decimate then LR heat blur)
        worst = max(worst, (lhs - rhs).norm().item() / max(1e-12, rhs.norm().item()))
    return GateResult("SR intertwining A K_t=L_t A", worst, 1e-3, worst < 1e-3)


def gate_sr_intertwining_exact(H=64, scale=2, dtype=torch.float64) -> GateResult:
    """The QMF alias correction (Deblur-INR Prop. 1; ops/superres.py sr_scale_matched_target with
    x_hr) makes the SR data target exact for ANY decimation/antialias. Uses STRIDED decimation with
    no antialias -- the WORST case, where the plain target L_t(A x) is ~8% off -- and checks the
    corrected target L_t(A x)+Delta_t(x0) equals A(K_t x0) to machine precision (oracle x_hr=x0)."""
    tf = DCTTransform()
    sch = HeatSchedule.ihdm(H, H, K=40, sigma_min=0.5, sigma_max=16.0, transform=tf, dtype=dtype)
    A = SuperResolution(scale, H, channels=1, aa_sigma=1e-6, decimation="stride", dtype=dtype)
    lr = lr_heat_schedule(sch, scale)
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(1, 1, H, H, generator=g, dtype=dtype)
    y = A.forward(x0)
    worst = 0.0
    for i in range(1, sch.num_levels, max(1, sch.num_levels // 8)):
        truth = A.forward(sch.apply_K(x0, i))                        # A(K_t x0)
        b = sr_scale_matched_target(A, sch, lr, i, y, x_hr=x0)       # L_t y + Delta_t(x0)
        worst = max(worst, (b - truth).norm().item() / max(1e-12, truth.norm().item()))
    return GateResult("SR intertwining exact (QMF corr.)", worst, 1e-9, worst < 1e-9)


ALL_GATES = [
    gate_adjoint, gate_intertwining, gate_limits, gate_noise_cov,
    gate_ct_adjoint, gate_intertwining_ct,
    gate_sr_adjoint, gate_intertwining_sr, gate_sr_intertwining_exact,
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
