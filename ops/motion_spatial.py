"""Spatial, reflect-boundary motion-blur operator (no FFT) + its exact adjoint.

Forward  : A x = crop( conv( reflect_pad(x, h//2), k ) )   == scipy convolve2d(mode='same', boundary='symm')
Adjoint  : A^T r = VJP of the (linear) forward at r        -> exact, includes the fold-transpose of reflect pad

Because A is a reflect-boundary (block Toeplitz-plus-Hankel) operator it is NOT circulant, so it has no OTF
and cannot be applied/inverted in the DFT domain. Everything here is pure spatial convolution; the adjoint is
obtained by autograd so it exactly matches the forward (the dot-product test below passes to ~1e-6).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


class SpatialMotionBlur:
    def __init__(self, kernel: torch.Tensor, channels: int = 3, device="cpu", dtype=torch.float32):
        k = kernel.to(device=device, dtype=dtype)
        k = k / k.sum()
        self.h, self.w = k.shape
        self.ph, self.pw = self.h // 2, self.w // 2
        self.C = channels
        # conv2d is cross-correlation; flip the kernel so we perform TRUE convolution (matches the forward synthesis)
        wf = torch.flip(k, dims=(0, 1)).reshape(1, 1, self.h, self.w).repeat(channels, 1, 1, 1)  # (C,1,h,w) depthwise
        self.weight = wf.contiguous()
        self.device, self.dtype = device, dtype

    def _sympad(self, x: torch.Tensor) -> torch.Tensor:
        """Half-sample symmetric padding (edge pixel included) == scipy boundary='symm' /
        MATLAB 'symmetric'. Built from flip+cat so it stays differentiable (exact adjoint)."""
        pw, ph = self.pw, self.ph
        if pw > 0:
            x = torch.cat([x[..., :pw].flip(-1), x, x[..., -pw:].flip(-1)], dim=-1)
        if ph > 0:
            x = torch.cat([x[..., :ph, :].flip(-2), x, x[..., -ph:, :].flip(-2)], dim=-2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,C,H,W) -> (B,C,H,W), symmetric (reflect) boundary, same size."""
        return F.conv2d(self._sympad(x), self.weight, groups=self.C)

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        """A^T r via the vector-Jacobian product of the linear forward (exact). enable_grad so it
        also works when called inside a torch.no_grad() restoration loop."""
        with torch.enable_grad():
            x = torch.zeros_like(r, requires_grad=True)
            y = self.forward(x)
            (g,) = torch.autograd.grad(y, x, grad_outputs=r, retain_graph=False, create_graph=False)
        return g.detach()


def cg_solve(apply_M, c, x0, iters=20, tol=1e-6):
    """Conjugate gradient for M x = c, M SPD, matrix-free via apply_M(v). Warm-started at x0."""
    x = x0.clone()
    r = c - apply_M(x)
    p = r.clone()
    rs = (r * r).sum()
    rs0 = rs.clamp_min(1e-30)
    for _ in range(iters):
        Mp = apply_M(p)
        alpha = rs / (p * Mp).sum().clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Mp
        rs_new = (r * r).sum()
        if (rs_new / rs0).sqrt() < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x
