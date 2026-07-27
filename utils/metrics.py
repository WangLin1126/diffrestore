"""Reconstruction and consistency metrics. Images assumed in [-1, 1]."""
from __future__ import annotations
import torch
import torch.nn.functional as F

_LPIPS = None


def _to01(x):
    return (x.clamp(-1, 1) + 1) / 2


def psnr(x, y):
    x, y = _to01(x), _to01(y)
    mse = F.mse_loss(x, y).item()
    return float("inf") if mse == 0 else -10.0 * torch.log10(torch.tensor(mse)).item()


def _gaussian_window(ch, ws=11, sigma=1.5, device="cpu"):
    coords = torch.arange(ws, dtype=torch.float32, device=device) - ws // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    win = (g @ g.t()).expand(ch, 1, ws, ws).contiguous()
    return win


def ssim(x, y):
    x, y = _to01(x), _to01(y)
    if x.dim() == 3:
        x, y = x.unsqueeze(0), y.unsqueeze(0)
    ch = x.shape[1]
    win = _gaussian_window(ch, device=x.device)
    mu_x = F.conv2d(x, win, groups=ch, padding=5)
    mu_y = F.conv2d(y, win, groups=ch, padding=5)
    mx2, my2, mxy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx = F.conv2d(x * x, win, groups=ch, padding=5) - mx2
    sy = F.conv2d(y * y, win, groups=ch, padding=5) - my2
    sxy = F.conv2d(x * y, win, groups=ch, padding=5) - mxy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mxy + c1) * (2 * sxy + c2)) / ((mx2 + my2 + c1) * (sx + sy + c2))
    return s.mean().item()


def lpips_metric(x, y, device="cuda"):
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex").to(device).eval()
    if x.dim() == 3:
        x, y = x.unsqueeze(0), y.unsqueeze(0)
    with torch.no_grad():
        return _LPIPS(x.clamp(-1, 1).to(device), y.clamp(-1, 1).to(device)).mean().item()


def measurement_consistency(y, A, x_hat):
    return (y - A.forward(x_hat)).norm().item() / (y.norm().item() + 1e-12)
