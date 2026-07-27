"""Spectral transforms used to diagonalize shift-invariant / heat operators.

A Transform pairs a forward analysis `fwd` with its inverse `inv` and declares whether
its coefficients are complex. SpectralOperator (operators/spectral.py) multiplies the
coefficients by a per-frequency transfer function.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from ops.dct import dct2, idct2


class Transform:
    complex_coeffs: bool = False
    def fwd(self, x: torch.Tensor) -> torch.Tensor: ...
    def inv(self, X: torch.Tensor) -> torch.Tensor: ...


@dataclass
class DCTTransform(Transform):
    """Orthonormal DCT-II. Real coefficients; operators with real transfer are self-adjoint."""
    complex_coeffs: bool = False
    def fwd(self, x): return dct2(x)
    def inv(self, X): return idct2(X)


@dataclass
class FFTTransform(Transform):
    """2-D FFT for circular (periodic) convolution. Coefficients complex.

    Real images are kept real on inversion; a real-symmetric kernel gives a real operator.
    """
    complex_coeffs: bool = True
    def fwd(self, x): return torch.fft.fft2(x)
    def inv(self, X): return torch.fft.ifft2(X).real
