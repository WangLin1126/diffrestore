"""Non-hot degradation path K_t = Gaussian heat blur, diagonal in the DCT basis.

Levels are indexed by an integer i in [0, N]; sigmas[i] is the cumulative blur std at
level i, with sigmas[0] ~ 0 (clean). K_i and its measurement companion L_i coincide for
commuting convolutions (L_i = K_i, MATH.md Lemma 3.1). All transfers are diagonal in the
shared DCT basis, so A and K_i commute exactly.
"""
from __future__ import annotations
import numpy as np
import torch

from ops.spectral import SpectralOperator
from ops.transforms import Transform, DCTTransform
from ops.dct import neumann_laplacian_eigs


class HeatSchedule:
    def __init__(self, H, W, sigmas, transform: Transform | None = None,
                 device="cpu", dtype=torch.float64):
        self.H, self.W = H, W
        self.device, self.dtype = device, dtype
        self.sigmas = torch.as_tensor(np.asarray(sigmas), dtype=dtype, device=device)
        self.lmbda = neumann_laplacian_eigs(H, W, device, dtype)      # (H, W)
        self.transform = transform or DCTTransform()

    # ---- constructors -------------------------------------------------------
    @classmethod
    def ihdm(cls, H, W, K=200, sigma_min=0.5, sigma_max=128.0, **kw):
        """IHDM FFHQ schedule: sigma = exp(linspace(log min, log max, K)), prepend 0."""
        blur = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), K))
        sigmas = np.concatenate([[0.0], blur])                        # len K+1
        return cls(H, W, sigmas, **kw)

    # ---- levels -------------------------------------------------------------
    @property
    def num_levels(self) -> int:
        return int(self.sigmas.numel())

    def transfer(self, i: int) -> torch.Tensor:
        """g for K_i (== L_i): exp(-0.5 sigma_i^2 lambda)."""
        s = self.sigmas[int(i)]
        return torch.exp(-0.5 * s * s * self.lmbda)

    def K(self, i: int) -> SpectralOperator:
        return SpectralOperator(self.transfer(i), self.transform)

    L = K  # measurement companion equals K for commuting convolutions

    def apply_K(self, x, i): return self.K(i).forward(x)
    def apply_L(self, y, i): return self.K(i).forward(y)

    # ---- batched blur (per-sample level, for training) ----------------------
    def transfer_batch(self, t_idx: torch.Tensor) -> torch.Tensor:
        """Per-sample transfer g for levels t_idx (B,) -> (B, H, W)."""
        s = self.sigmas.to(t_idx.device)[t_idx]                  # (B,)
        lmbda = self.lmbda.to(t_idx.device)
        return torch.exp(-0.5 * (s[:, None, None] ** 2) * lmbda[None])

    def apply_K_batch(self, x: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """Apply K_{t_idx[b]} to x[b]. x: (B,C,H,W), t_idx: (B,)."""
        g = self.transfer_batch(t_idx).to(device=x.device, dtype=x.dtype).unsqueeze(1)
        tf = self.transform
        return tf.inv(g * tf.fwd(x))

    # ---- noise covariance (diagonal transfers) ------------------------------
    def sigma_transfer(self, i, sigma_n):
        """Diagonal of Sigma_i = L_i Sigma_n L_i^T = sigma_n^2 |g_i|^2  (MATH.md eq. 8)."""
        g = self.transfer(i)
        return (sigma_n ** 2) * g * g

    def sigma_eff_transfer(self, i, sigma_n, sigma_xi, A_transfer):
        """Diagonal of Sigma_i^eff with stochastic path noise (MATH.md eq. 17)."""
        g = self.transfer(i)
        return (sigma_n ** 2) * g * g + (sigma_xi ** 2) * (A_transfer ** 2)
