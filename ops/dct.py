"""Orthonormal 2-D DCT-II / IDCT-II via cached cosine matrices.

The DCT-II with 'ortho' normalization diagonalizes the Neumann-boundary Laplacian,
so any function of that Laplacian (Gaussian heat blur, in particular) is exactly a
per-frequency multiply in this basis. See MATH.md sec. 3.2 (N) and (5).
"""
from __future__ import annotations
import functools
import torch


@functools.lru_cache(maxsize=16)
def dct_matrix(n: int, device: str = "cpu", dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Orthonormal DCT-II matrix C of shape (n, n): X = C @ x, and C @ C.T = I.

    C[k, m] = alpha_k * cos(pi (2m+1) k / (2n)),  alpha_0 = sqrt(1/n), alpha_{k>0} = sqrt(2/n).
    """
    k = torch.arange(n, dtype=dtype, device=device).unsqueeze(1)   # freq index  (n,1)
    m = torch.arange(n, dtype=dtype, device=device).unsqueeze(0)   # spatial idx (1,n)
    C = torch.cos(torch.pi * (2 * m + 1) * k / (2 * n))
    alpha = torch.full((n, 1), (2.0 / n) ** 0.5, dtype=dtype, device=device)
    alpha[0, 0] = (1.0 / n) ** 0.5
    return alpha * C


def dct2(x: torch.Tensor) -> torch.Tensor:
    """2-D orthonormal DCT-II over the last two dims. x: (..., H, W)."""
    H, W = x.shape[-2], x.shape[-1]
    Ch = dct_matrix(H, str(x.device), x.dtype)
    Cw = dct_matrix(W, str(x.device), x.dtype)
    return torch.matmul(torch.matmul(Ch, x), Cw.transpose(-1, -2))


def idct2(X: torch.Tensor) -> torch.Tensor:
    """Inverse 2-D orthonormal DCT (== transpose, since C is orthonormal)."""
    H, W = X.shape[-2], X.shape[-1]
    Ch = dct_matrix(H, str(X.device), X.dtype)
    Cw = dct_matrix(W, str(X.device), X.dtype)
    return torch.matmul(torch.matmul(Ch.transpose(-1, -2), X), Cw)


def neumann_laplacian_eigs(H: int, W: int, device="cpu", dtype=torch.float64) -> torch.Tensor:
    """lambda[k,l] = (pi k / H)^2 + (pi l / W)^2  (IHDM DCTBlur convention, MATH.md eq. 5)."""
    fk = (torch.pi * torch.arange(H, dtype=dtype, device=device) / H) ** 2
    fl = (torch.pi * torch.arange(W, dtype=dtype, device=device) / W) ** 2
    return fk.unsqueeze(1) + fl.unsqueeze(0)   # (H, W)
