"""Non-blind Gaussian deblurring forward operator A, built in the shared spectral basis.

A Gaussian blur of std sigma_A is heat dissipation for time sigma_A^2/2; realizing it in
the same DCT basis as the prior's K_t makes A and K_t commute exactly and A self-adjoint
(A^T = A). See MATH.md eq. (6).
"""
from __future__ import annotations
import torch

from ops.spectral import SpectralOperator
from ops.transforms import Transform, DCTTransform
from ops.dct import neumann_laplacian_eigs


def gaussian_blur_transfer(H, W, sigma, device="cpu", dtype=torch.float64) -> torch.Tensor:
    lmbda = neumann_laplacian_eigs(H, W, device, dtype)
    return torch.exp(-0.5 * sigma * sigma * lmbda)


def build_deblur(H, W, blur_sigma, transform: Transform | None = None,
                 device="cpu", dtype=torch.float64):
    """Return (A, A_transfer) for Gaussian deblurring."""
    transform = transform or DCTTransform()
    g = gaussian_blur_transfer(H, W, blur_sigma, device, dtype)
    return SpectralOperator(g, transform), g
