"""Super-resolution forward operator A = D_s . B_aa  and its LR-grid heat companion L_t.

Forward  A x : HR image (B,C,H,W) -> LR image (B,C,H/s,W/s).
    B_aa   antialias Gaussian blur, std `aa_sigma` HR-pixels, realized in the SAME DCT/Neumann
           basis as the prior's heat K_t (a per-frequency multiply) so B_aa and K_t commute
           EXACTLY -- the intertwining residual is then purely decimation aliasing, nothing else.
    D_s    area (average-pool) decimation by s. avg_pool2d over each s x s block samples the block
           *center*, which is exactly the LR grid point of the DCT-II half-sample (Neumann) basis;
           strided x[..., ::s, ::s] samples the block CORNER (offset 0) instead, a (s-1)/2-pixel
           phase error that breaks the intertwining by ~5-14% for even s -- avg_pool fixes this to
           ~2e-4 for every scale (see gate_intertwining_sr / EXPERIMENTS).
Adjoint  A^T = B_aa . D_s^T (D_s^T = nearest-neighbour broadcast / s) is the exact VJP of the
linear forward, obtained by autograd -- same device-free trick as ops.ct.ParallelBeamRadon and
ops.motion_spatial.SpatialMotionBlur, so <A x, y> = <x, A^T y> to ~1e-14 and any CG data step
needs only A, A^T.

Intertwining (ROADMAP sec. 0 / sec. 1 "Super-res"): decimating an isotropically heat-blurred
image equals heat-blurring the decimated image on the LR grid with the SAME physical Gaussian,
which measured in LR pixels has std sigma_t / s:

        A ( K_t x )  ~=  L_t ( A x ),     L_t = LR-grid heat blur, sigma^LR_t = sigma_t / s.

Because B_aa commutes with K_t, this is exact up to the aliasing that B_aa suppresses. With the
default aa_sigma = scale (transfer ~exp(-pi^2/2) ~ 7e-3 at LR Nyquist) that residual is ~2e-4 --
an order of magnitude tighter than CT (a lighter antialias leaks more: aa=0.5s -> ~3.5e-2, and no
antialias / avg-pool alone -> ~40% since the box is a poor low-pass). So the isotropic HR heat
prior has a natural LR-grid companion built by `lr_heat_schedule(hr_schedule, scale)`; the
scale-matched target is L_t y in LR space.

Exact intertwining (optional) -- Deblur-INR, NeurIPS'24 (Zhang et al., "Cross-Scale Self-Supervised
Blind Image Deblurring via INR"), Prop. 1: for STRIDED dyadic downsampling, blur and downsample do
not commute, and the exact commutator is the aliasing, written as a sum over quadrature-mirror
filters,  4 (x_d2)*(k_d2) = (x*k)_d2 + sum_{d=1..3} (x*g_d)_d2,  g_d = (-1)^m/(-1)^n/(-1)^{m+n} k.
`sr_alias_correction` computes that commutator Delta_i = A(K_i x) - L_i(A x) operator-agnostically
(covers avg-pool too), and `sr_scale_matched_target(..., x_hr=<estimate>)` returns L_i y + Delta_i,
which equals A(K_i x0) exactly when x_hr = x0 and reduces to the plain target when aliasing is
negligible. This makes the SR data step exact for ANY antialias strength (even strided, aa=0), at
the cost of an HR image -- in reconstruction, the current estimate (a plug-in, exact in the limit).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from ops.spectral import SpectralOperator
from ops.transforms import DCTTransform
from ops.deblur import gaussian_blur_transfer
from ops.dct import idct2, dct2
from ops.heat import HeatSchedule


class SuperResolution:
    """A = avg_pool_s . B_aa, an antialias-blur-then-downsample SR forward with exact VJP adjoint.

    aa_sigma (HR pixels) is the antialias std; default `scale` gives ~exp(-pi^2/2) ~ 7e-3 transfer
    at the LR Nyquist, i.e. strong antialiasing -> intertwining residual ~2e-4. Set None to use it.
    """
    def __init__(self, scale: int, image_size: int, channels: int = 3,
                 aa_sigma: float | None = None, decimation: str = "avgpool",
                 device="cpu", dtype=torch.float32):
        assert image_size % scale == 0, f"image_size {image_size} not divisible by scale {scale}"
        assert decimation in ("avgpool", "stride"), decimation
        self.s = int(scale)
        self.H = self.W = int(image_size)
        self.C = channels
        self.decimation = decimation                              # "stride" = Deblur-INR's downarrow
        self.device, self.dtype = device, dtype
        self.aa_sigma = float(scale) if aa_sigma is None else float(aa_sigma)
        g = gaussian_blur_transfer(self.H, self.W, self.aa_sigma, device, dtype)
        self.B = SpectralOperator(g, DCTTransform())              # HR antialias blur (DCT-diagonal)

    def _decimate(self, u: torch.Tensor) -> torch.Tensor:
        return u[..., ::self.s, ::self.s] if self.decimation == "stride" else F.avg_pool2d(u, self.s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B,C,H,W) -> (B,C,H/s,W/s): antialias blur then decimation (avg-pool center, or strided)."""
        return self._decimate(self.B.forward(x))

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """A^T y via the VJP of the linear forward (exact). enable_grad so it also works inside a
        torch.no_grad() restoration loop."""
        with torch.enable_grad():
            x = torch.zeros(y.shape[0], self.C, self.H, self.W,
                            device=y.device, dtype=y.dtype, requires_grad=True)
            out = self.forward(x)
            (g,) = torch.autograd.grad(out, x, grad_outputs=y, retain_graph=False, create_graph=False)
        return g.detach()

    @torch.no_grad()
    def transfer_profile(self) -> torch.Tensor:
        """Per-DCT-frequency normalized gain  â_n(k,l) = |A e_{k,l}| / |A e_{0,0}| in [0,1] (1 at DC,
        →0 in A's null/stop band). A is separable (avg-pool ⊗ Gaussian), so â_n(k,l)=n(k)·n(l) with
        the 1-D axis gain n(k) probed once. Used to build a frequency-aware data regularizer that
        boosts the prior precision where A can't measure (transition/null band) -- the DDRM/Wiener
        'trust the prior where singular values are small' behaviour, without over-smoothing the
        well-measured low band."""
        H = self.H
        idx = torch.arange(H, device=self.device)
        onehot = torch.zeros(H, 1, H, H, device=self.device, dtype=self.dtype)
        onehot[idx, 0, idx, 0] = 1.0                       # sample k: DCT unit at (k,0)
        gains = self.forward(idct2(onehot)).flatten(1).norm(dim=1)   # ∝ c(k)·c(0)
        n = gains / gains[0].clamp_min(1e-12)              # c(k)/c(0)
        return torch.outer(n, n)                           # â_n(k,l) (H,W)


def lr_heat_schedule(hr: HeatSchedule, scale: int) -> HeatSchedule:
    """LR-grid heat companion of an HR HeatSchedule: the SAME physical Gaussians, but measured in
    LR pixels their std is sigma_t / s and they live on the H/s x W/s grid. `apply_K(y_lr, i)` is
    then the measurement-side L_t (scale-matched target L_t y for SR)."""
    s = int(scale)
    assert hr.H % s == 0 and hr.W % s == 0, "HR grid not divisible by scale"
    sig = (hr.sigmas / s).detach().cpu().numpy()               # HeatSchedule takes host sigmas
    return HeatSchedule(hr.H // s, hr.W // s, sig,
                        transform=hr.transform, device=hr.device, dtype=hr.dtype)


def sr_alias_correction(A: SuperResolution, hr: HeatSchedule, lr: HeatSchedule, i: int,
                        x_hr: torch.Tensor) -> torch.Tensor:
    """Exact commutator  Delta_i = A(K_i x) - L_i(A x)  between (HR heat-blur then decimate) and
    (decimate then LR heat-blur). This is the aliasing that the plain intertwining A(K_i x)~=L_i(A x)
    drops; for strided decimation it equals the QMF term sum_d (x*g_d)_down of Deblur-INR (Prop. 1),
    here written operator-agnostically so it also covers avg-pool. Needs an HR image x_hr."""
    return A.forward(hr.apply_K(x_hr, i)) - lr.apply_K(A.forward(x_hr), i)


def sr_scale_matched_target(A: SuperResolution, hr: HeatSchedule, lr: HeatSchedule, i: int,
                            y_lr: torch.Tensor, x_hr: torch.Tensor | None = None) -> torch.Tensor:
    """Scale-matched SR data target b ~= A(K_i x0).  Plain (x_hr=None): b = L_i y  (aliased; fine
    when B_aa is strong).  Exact-corrected (x_hr given): b = L_i y + Delta_i(x_hr) with Delta from
    `sr_alias_correction`; equals A(K_i x0) exactly when x_hr = x0, and stays valid for ANY antialias
    strength (strided / aa=0). In reconstruction pass the current HR estimate for x_hr (a plug-in,
    exact in the x_hr -> x0 limit)."""
    b = lr.apply_K(y_lr, i)
    if x_hr is not None:
        b = b + sr_alias_correction(A, hr, lr, i, x_hr)
    return b
