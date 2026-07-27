"""IHDM (Inverse Heat Dissipation) non-hot prior, wrapped as a ReversePrior.

Loads a trained IHDM checkpoint using the vendored backbone in `model/ihdm_backbone/`
(NCSN++ U-Net + DCT-heat code). IHDM's reverse mean is `u + model(u, i)` (the network
predicts the deblurring residual), with images in [0,1] (data.centered=False); SMDC works
in [-1,1], so we convert at the model boundary.
"""
from __future__ import annotations
import sys
import importlib
from pathlib import Path
import numpy as np
import torch

from model.base import ReversePrior

BACKBONE_DIR = str((Path(__file__).resolve().parent / "ihdm_backbone"))
DEFAULT_CKPT = str((Path(__file__).resolve().parent.parent / "checkpoint" / "ihdm" / "ffhq128.pth"))


class IHDMPrior(ReversePrior):
    def __init__(self, model_fn):
        self.model_fn = model_fn

    @torch.no_grad()
    def reverse_step(self, u, t, t_next, rng=None):
        i = int(t)
        steps = torch.full((u.shape[0],), i, device=u.device, dtype=torch.long)
        u01 = (u + 1.0) / 2.0                        # SMDC [-1,1] -> IHDM [0,1]
        u_mean01 = u01 + self.model_fn(u01, steps)   # their reverse mean at scale i-1
        return u_mean01 * 2.0 - 1.0                  # back to [-1,1]


def load_ihdm(ckpt=DEFAULT_CKPT, config_name="img_size_128_maxblur128", device="cuda:0",
              backbone_dir=BACKBONE_DIR):
    """Return (prior, blur_sigmas, config) from a vendored IHDM checkpoint."""
    if backbone_dir not in sys.path:
        sys.path.insert(0, backbone_dir)
    cfgmod = importlib.import_module(f"configs.ffhq.{config_name}")
    config = cfgmod.get_config()
    config.device = torch.device(device)
    from model_code import utils as mutils
    from model_code.ema import ExponentialMovingAverage
    model = mutils.create_model(config)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    ema.load_state_dict(ck["ema"])
    ema.copy_to(model.parameters())                  # inference = EMA weights
    model_fn = mutils.get_model_fn(model, train=False)
    return IHDMPrior(model_fn), np.asarray(config.model.blur_schedule), config
