"""Linear operator interface with a built-in adjoint consistency test.

The solver only ever sees forward() / adjoint(); it never touches FFT/DCT internals.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply A."""

    @abstractmethod
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply A^T (numerical adjoint of forward). Must pass adjoint_test."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def adjoint_test(self, shape, device="cpu", dtype=torch.float64, seed=0) -> float:
        """Relative adjoint mismatch  |<Ax,z> - <x,A^T z>| / max(1,|<Ax,z>|)  (MATH.md gate 1)."""
        g = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(shape, generator=g, device=device, dtype=dtype)
        z = torch.randn(shape, generator=g, device=device, dtype=dtype)
        lhs = (self.forward(x) * z).sum()
        rhs = (x * self.adjoint(z)).sum()
        return (lhs - rhs).abs().item() / max(1.0, lhs.abs().item())
