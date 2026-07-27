"""Reverse non-hot prior interface. The solver never backpropagates through this."""
from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class ReversePrior(ABC):
    @abstractmethod
    def reverse_step(self, x: torch.Tensor, t: int, t_next: int, rng=None) -> torch.Tensor:
        """Return a state intended to lie at degradation level t_next (< t)."""
