"""BDM denoiser network f_theta(z_t, t): a pixel-space UNet epsilon-predictor.

Two backbones:
  * "ihdm"  -- the exact IHDM NCSN++ net (model/ihdm_backbone, ~211M at 256px). Blurring diffusion
               is an improvement of IHDM, so we reuse its structure and parameter count verbatim.
  * "unet"  -- the compact repo UNet (model/unet.py), for CIFAR-scale runs and the sanity tests.

BDM uses a *continuous* t in [0,1]; the sinusoidal timestep embedding wants a spread-out range, so
t is rescaled t -> t * `time_scale` before the net (IHDM conditions on integer levels 0..K).
Architecture presets follow Table 5 of Hoogeboom & Salimans (2023) plus an IHDM-matched 256 preset.
"""
from __future__ import annotations
import os
import copy
import importlib
from dataclasses import dataclass, replace
import torch
import torch.nn as nn

from model.unet import UNet


@dataclass
class BDMModelConfig:
    image_size: int = 32
    in_channels: int = 3
    backbone: str = "unet"                 # "unet" (compact) | "ihdm" (NCSN++, ~211M)
    ihdm_config: str = "img_size_256_full"  # configs.ffhq.* module, used when backbone == "ihdm"
    # compact-UNet hyperparameters (ignored when backbone == "ihdm")
    ch: int = 128
    ch_mult: tuple = (1, 1, 1)
    num_res_blocks: int = 3
    attn_resolutions: tuple = (16, 8)
    dropout: float = 0.1
    time_scale: float = 1000.0


# Table 5 presets + an IHDM-matched 256px preset (same net/params as model/ihdm_backbone).
PRESETS = {
    "cifar10":          BDMModelConfig(32, 3, "unet", ch=256, ch_mult=(1, 1, 1),       num_res_blocks=3, attn_resolutions=(16, 8),      dropout=0.2),
    "lsun_churches64":  BDMModelConfig(64, 3, "unet", ch=128, ch_mult=(1, 2, 3, 4),    num_res_blocks=3, attn_resolutions=(32, 16, 8), dropout=0.2),
    "lsun_churches128": BDMModelConfig(128, 3, "unet", ch=64, ch_mult=(1, 2, 4, 6, 8), num_res_blocks=3, attn_resolutions=(32, 16, 8), dropout=0.1),
    "ihdm256":          BDMModelConfig(256, 3, "ihdm", ihdm_config="img_size_256_full"),
}

_IHDM_BACKBONE = os.path.join(os.path.dirname(__file__), "..", "ihdm_backbone")


def _build_ihdm_net(ihdm_config: str) -> nn.Module:
    """Instantiate the IHDM NCSN++ UNetModel from a configs.ffhq.* config (built on CPU)."""
    import sys
    bp = os.path.abspath(_IHDM_BACKBONE)
    if bp not in sys.path:
        sys.path.insert(0, bp)
    from model_code.unet import UNetModel
    cfg = importlib.import_module(f"configs.ffhq.{ihdm_config}").get_config()
    cfg.device = torch.device("cpu")
    return UNetModel(cfg)


class BDMDenoiser(nn.Module):
    """Wraps a UNet so the diffusion module can call `model(z_t, t)` with continuous t in [0,1]."""
    def __init__(self, cfg: BDMModelConfig):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)          # never alias a shared PRESETS entry
        if cfg.backbone == "ihdm":
            self.net = _build_ihdm_net(cfg.ihdm_config)
        elif cfg.backbone == "unet":
            self.net = UNet(
                ch=cfg.ch, out_ch=cfg.in_channels, ch_mult=tuple(cfg.ch_mult),
                num_res_blocks=cfg.num_res_blocks, attn_resolutions=tuple(cfg.attn_resolutions),
                dropout=cfg.dropout, in_channels=cfg.in_channels, resolution=cfg.image_size,
            )
        else:
            raise ValueError(f"unknown backbone {cfg.backbone!r}")

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(z_t.shape[0])
        return self.net(z_t, t * self.cfg.time_scale)


def build_bdm_model(preset: str | None = None, **overrides) -> BDMDenoiser:
    base = PRESETS[preset] if preset else BDMModelConfig()     # BDMDenoiser deep-copies, so no alias
    return BDMDenoiser(replace(base, **overrides) if overrides else base)
