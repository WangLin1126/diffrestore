"""Operators diagonal in a spectral basis:  M x = Phi^{-1}( g ⊙ Phi x ).

For an orthonormal real basis (DCT) with real transfer g, M is self-adjoint (M^T = M).
For the unitary FFT basis, the adjoint uses conj(g). Two such operators sharing the same
basis commute automatically -> the intertwining L_t A = A K_t is exact (MATH.md sec. 3.2).
"""
from __future__ import annotations
import torch

from ops.operators import LinearOperator
from ops.transforms import Transform


class SpectralOperator(LinearOperator):
    def __init__(self, transfer: torch.Tensor, transform: Transform):
        self.g = transfer                # (H, W) broadcast over (..., H, W)
        self.tf = transform

    def _to(self, x):
        g = self.g
        if g.device != x.device:
            g = g.to(x.device)
        # transfer is real; match real dtype of x for the multiply
        if not torch.is_complex(g) and not torch.is_complex(x) and g.dtype != x.dtype:
            g = g.to(x.dtype)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self._to(x)
        return self.tf.inv(g * self.tf.fwd(x))

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        g = self._to(y)
        return self.tf.inv(torch.conj(g) * self.tf.fwd(y))

    def gram_transfer(self) -> torch.Tensor:
        """Diagonal of A^T A in the shared basis: |g|^2."""
        return torch.abs(self.g) ** 2
