"""Motion-blur forward operator A — a general (non-symmetric) convolution.

Unlike Gaussian/heat blur, a motion PSF is NOT diagonal in the DCT basis: it is diagonal
in the DFT (Fourier) basis via the OTF (optical transfer function). So A does NOT commute
with the heat prior's K_t — SMDC's exact scale-matching (which needs a shared eigenbasis)
does not hold here, while the HQS data step stays closed-form as a Wiener filter *in DFT*.

Circular convolution is used (periodic boundaries), matching the psf2otf convention.
"""
from __future__ import annotations
import numpy as np
import torch

from ops.operators import LinearOperator


def load_levin_kernel(idx: int, path: str = "/home/xyli/VEM-NBD-master/data/Set12/kernels/Levin09.mat"):
    """Return Levin09 motion kernel #idx (1-based) as a float32 numpy array, sum-normalized."""
    import scipy.io as sio
    kernels = sio.loadmat(path)["kernels"].reshape(-1)
    k = np.asarray(kernels[idx - 1], dtype=np.float32)
    return k / k.sum()


def psf2otf(kernel: torch.Tensor, shape) -> torch.Tensor:
    """OTF of `kernel` (h,w) for an image of `shape` (H,W): DFT of the centered, zero-padded PSF."""
    H, W = shape
    h, w = kernel.shape
    pad = torch.zeros(H, W, dtype=kernel.dtype, device=kernel.device)
    pad[:h, :w] = kernel
    # shift so the kernel center sits at pixel (0,0) -> no net translation of the blur
    pad = torch.roll(pad, shifts=(-(h // 2), -(w // 2)), dims=(0, 1))
    return torch.fft.fft2(pad)


def edgetaper(img: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """MATLAB-style edgetaper: blend the image toward a circular-blurred copy near the
    borders so the periodic (FFT) deconvolution sees a smooth wrap and stops ringing.

    Same size in/out (no padding). The weight alpha is separable, built from the PSF's
    1-D autocorrelation projected onto each axis (all via FFT):
        beta_k(lag) = ifft( |fft(psf_proj_k)|^2 )  / beta_k(0)     # 1 at the edge, decays inward
        alpha       = (1 - beta_row) ⊗ (1 - beta_col)              # ~0 at borders, ~1 interior
        J           = alpha*img + (1 - alpha)*circular_blur(img)

    img: (...,H,W) real ; psf: (h,w).  Matches numpy/MATLAB edgetaper to ~1e-6.
    """
    H, W = img.shape[-2:]
    psf = (psf / psf.sum()).to(device=img.device, dtype=torch.float32)

    def beta(proj, N):                                   # 1-D circular autocorrelation profile
        z = torch.zeros(N, device=img.device, dtype=torch.float32)
        z[:proj.numel()] = proj
        b = torch.fft.ifft(torch.abs(torch.fft.fft(z)) ** 2).real
        return b / b[0]                                  # zero-lag (edge) normalized to 1

    br = beta(psf.sum(dim=1), H)                          # project onto rows -> (H,)
    bc = beta(psf.sum(dim=0), W)                          # project onto cols -> (W,)
    alpha = (1.0 - br)[:, None] * (1.0 - bc)[None, :]     # (H,W)
    otf = psf2otf(psf, (H, W)).to(img.device)
    blurred = torch.fft.ifft2(otf * torch.fft.fft2(img)).real
    out = alpha * img + (1.0 - alpha) * blurred
    return out.clamp(img.amin(), img.amax())             # MATLAB clamps to input range


class MotionBlurOperator(LinearOperator):
    """A x = circular-conv(x, psf), realized as  ifft2( otf ⊙ fft2(x) )  per channel."""

    def __init__(self, kernel: torch.Tensor, shape, device="cpu", dtype=torch.float32):
        self.shape = shape
        self.dtype = dtype
        k = kernel.to(device=device, dtype=dtype)
        self.otf = psf2otf(k, shape).to(device)          # (H,W) complex
        self.kernel = k

    def _otf(self, x):
        return self.otf.to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        otf = self._otf(x)
        return torch.fft.ifft2(otf * torch.fft.fft2(x)).real.to(x.dtype)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        otf = self._otf(y)
        return torch.fft.ifft2(torch.conj(otf) * torch.fft.fft2(y)).real.to(y.dtype)

    def gram_transfer(self) -> torch.Tensor:
        """|OTF|^2 — diagonal of A^T A in the DFT basis (for the Wiener MAP step)."""
        return torch.abs(self.otf) ** 2
