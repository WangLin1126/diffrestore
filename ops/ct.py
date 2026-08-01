"""Parallel-beam CT (Radon) forward + its sinogram-domain heat companion L_t.

Forward  R x : image (B,C,H,W) -> sinogram (B,C,n_angles,W).  For each projection angle the
image is rotated (differentiable grid_sample) and summed along the rays; the detector has W
bins.  Adjoint R^T (filtered-free back-projection) is the exact VJP of the linear forward,
obtained by autograd -- same device-free trick as ops.motion_spatial.SpatialMotionBlur, so the
dot-product test <R x, s> = <x, R^T s> holds to ~1e-5 and the CG data step needs only R, R^T.

Fourier-slice intertwining (MATH / ROADMAP sec. 0): projecting an isotropically heat-blurred
image equals 1-D-heat-blurring its sinogram *along the detector axis* with the SAME sigma_t,

        R ( K_t x )  =  L_t ( R x ),      L_t = 1-D Neumann heat blur along the detector.

So the isotropic DCT heat prior has a natural sinogram companion: DetectorHeatBlur applies the
1-D DCT-II heat transfer exp(-1/2 sigma^2 (pi k / W)^2) along the detector bins.  This is the CT
analogue of L_t = K_t for deblur; the scale-matched target is L_t y in sinogram space.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from ops.dct import dct_matrix, neumann_laplacian_eigs


class ParallelBeamRadon:
    def __init__(self, n_angles: int, image_size: int, channels: int = 3,
                 angles=None, device="cpu", dtype=torch.float32):
        H = W = image_size
        self.H, self.W, self.C = H, W, channels
        self.device, self.dtype = device, dtype
        if angles is None:
            angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
        self.angles = torch.as_tensor(np.asarray(angles), dtype=dtype, device=device)
        self.n_angles = int(self.angles.numel())
        # affine grids that rotate the sampling lattice by each angle (one (H,W,2) grid per angle)
        c, s = torch.cos(self.angles), torch.sin(self.angles)
        theta = torch.zeros(self.n_angles, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0], theta[:, 0, 1] = c, -s
        theta[:, 1, 0], theta[:, 1, 1] = s, c
        self.grid = F.affine_grid(theta, (self.n_angles, channels, H, W), align_corners=False)  # (A,H,W,2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B,C,H,W) -> sinogram (B,C,n_angles,W)."""
        B, C, H, W = x.shape
        A = self.n_angles
        xe = x[:, None].expand(B, A, C, H, W).reshape(B * A, C, H, W)
        ge = self.grid[None].expand(B, A, H, W, 2).reshape(B * A, H, W, 2).to(x.dtype)
        rot = F.grid_sample(xe, ge, mode="bilinear", padding_mode="zeros", align_corners=False)
        proj = rot.sum(dim=2)                                   # sum over rays (H) -> (B*A, C, W)
        return proj.reshape(B, A, C, W).permute(0, 2, 1, 3).contiguous()   # (B,C,A,W)

    def adjoint(self, s: torch.Tensor) -> torch.Tensor:
        """R^T s (un-filtered back-projection) via the VJP of the linear forward (exact)."""
        with torch.enable_grad():
            x = torch.zeros(s.shape[0], self.C, self.H, self.W,
                            device=s.device, dtype=s.dtype, requires_grad=True)
            y = self.forward(x)
            (g,) = torch.autograd.grad(y, x, grad_outputs=s, retain_graph=False, create_graph=False)
        return g.detach()


class DetectorHeatBlur:
    """1-D Neumann (DCT-II) heat blur of std `sigma` along the detector axis of a sinogram.

    Companion L_t of the isotropic image heat K_t under the Radon transform (Fourier-slice).
    Applied as a per-frequency multiply in the 1-D DCT of the last (detector) dimension.
    """
    def __init__(self, sigma: float, det_bins: int, device="cpu", dtype=torch.float32):
        W = int(det_bins)
        self.C = dct_matrix(W, str(device), dtype)                    # (W,W) orthonormal DCT-II (cached)
        lmbda = neumann_laplacian_eigs(1, W, device=device, dtype=dtype)[0]   # 1-D (pi l / W)^2
        self.g = torch.exp(-0.5 * (float(sigma) ** 2) * lmbda)        # (W,) heat transfer

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s (...,A,W) -> heat-blurred along the last axis. Self-adjoint (real DCT)."""
        C = self.C.to(s.dtype)                                        # no-op when dtypes match
        X = torch.matmul(s, C.transpose(-1, -2))                      # DCT along detector
        return torch.matmul(self.g.to(s.dtype) * X, C)               # scale + inverse DCT
