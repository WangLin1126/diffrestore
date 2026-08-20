"""DPS (hot diffusion) restoration on precomputed heat observations — the spare comparison path.

Uses the vendored DPS repo (model/dps_backbone) + ffhq_10m checkpoint. DPS is a different
paradigm (hot noise prior + x0-guidance with backprop), so it lives outside the prior x solver
skeleton; kept here for head-to-head comparison.

  python scripts/run_dps.py --clean_dir results/dps_vs_smdc/clean \
     --observation_dir results/dps_vs_smdc/observation --save_dir results/dps_out \
     --degradation_sigma 4 --scale 0.3 --num_images 8
"""
import os
import sys
import types
import argparse
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import yaml

REPO = Path(__file__).resolve().parent.parent
DPS_ROOT = REPO / "model" / "dps_backbone"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(DPS_ROOT))

# DPS imports `motionblur` at import time; shim it (unused for heat blur).
mb = types.ModuleType("motionblur"); mbm = types.ModuleType("motionblur.motionblur")
mbm.Kernel = type("K", (), {}); mb.motionblur = mbm
sys.modules.setdefault("motionblur", mb); sys.modules.setdefault("motionblur.motionblur", mbm)

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.unet import create_model
from ops.dct import dct2, idct2, neumann_laplacian_eigs


class HeatBlurOperator:
    def __init__(self, intensity):
        self.intensity = float(intensity)

    def _kernel(self, data):
        lam = neumann_laplacian_eigs(data.shape[-2], data.shape[-1], str(data.device), data.dtype)
        return torch.exp(-0.5 * self.intensity ** 2 * lam)

    def forward(self, data, **kwargs):
        return idct2(dct2(data) * self._kernel(data))

    def transpose(self, data, **kwargs):
        return self.forward(data, **kwargs)


class MotionBlurOperatorDPS:
    """Levin motion blur for DPS (circular conv via OTF); general, non-DCT operator."""
    def __init__(self, kernel_npy, image_size, device):
        from ops.motion import MotionBlurOperator
        k = torch.from_numpy(np.load(kernel_npy))
        self.A = MotionBlurOperator(k, (image_size, image_size), device=device, dtype=torch.float32)

    def forward(self, data, **kwargs):
        return self.A.forward(data)

    def transpose(self, data, **kwargs):
        return self.A.adjoint(data)


class InpaintOperatorDPS:
    """Inpainting: elementwise multiply by a shared 0/1 mask (1=observed). Self-adjoint projection;
    differentiable, so autograd gives the (identical) adjoint. Mask loaded from a .pt file."""
    def __init__(self, mask_file, device):
        self.m = torch.load(mask_file, map_location=device).float()

    def forward(self, data, **kwargs):
        return data * self.m

    def transpose(self, data, **kwargs):
        return data * self.m


class ReflectBlurOperatorDPS:
    """Reflect-boundary blur (motion Levin09 or disk defocus) matching the observation synthesis
    exactly (scipy convolve2d mode='same' boundary='symm'); differentiable forward -> autograd adjoint.
    Same operator our SMDC spatial-CG data step uses, so DPS/PiGDM solve the true reflect problem."""
    def __init__(self, kernel_npy, device):
        from ops.motion_spatial import SpatialMotionBlur
        k = torch.from_numpy(np.load(kernel_npy))
        self.A = SpatialMotionBlur(k, channels=3, device=device, dtype=torch.float32)

    def forward(self, data, **kwargs):
        return self.A.forward(data)

    def transpose(self, data, **kwargs):
        return self.A.adjoint(data)


class GaussianNoiser:
    __name__ = "gaussian"


def _to_model_range(path, image_size, device):
    tf = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size),
                             transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    return tf(Image.open(path).convert("RGB"))[None].to(device)


def _save(path, x):
    a = ((x.detach().cpu().squeeze() + 1) / 2).clamp(0, 1).numpy().transpose(1, 2, 0)
    Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", default=str(DPS_ROOT / "configs/model_config.yaml"))
    ap.add_argument("--diffusion_config", default=str(DPS_ROOT / "configs/diffusion_config.yaml"))
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--observation_dir", required=True)
    ap.add_argument("--save_dir", default="results/dps_out")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num_images", type=int, default=8)
    ap.add_argument("--degradation_sigma", type=float, default=4.0)
    ap.add_argument("--operator", choices=["heat", "motion", "inpaint"], default="heat")
    ap.add_argument("--mask_file", default=None, help="shared 0/1 inpaint mask (.pt) for --operator inpaint")
    ap.add_argument("--boundary", choices=["circular", "reflect"], default="circular",
                    help="motion/defocus forward boundary: circular OTF (stock) or reflect (matches obs)")
    ap.add_argument("--kernel_npy", default="results/motion/kernel.npy")
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--method", choices=["ps", "pigdm"], default="ps",
                    help="ps = DPS posterior-gradient; pigdm = pseudoinverse-guided (same sampler/operator)")
    ap.add_argument("--pigdm_sigma", type=float, default=0.2, help="PiGDM guidance temperature")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    mc = yaml.safe_load(open(args.model_config))
    mc["model_path"] = str(DPS_ROOT / "models" / "ffhq_10m.pt")   # symlink -> checkpoint/dps
    image_size = int(mc["image_size"])
    model = create_model(**mc).to(device).eval()
    if args.operator == "inpaint":
        operator = InpaintOperatorDPS(args.mask_file, device)
    elif args.operator == "motion":
        operator = (ReflectBlurOperatorDPS(args.kernel_npy, device) if args.boundary == "reflect"
                    else MotionBlurOperatorDPS(args.kernel_npy, image_size, device))
    else:
        operator = HeatBlurOperator(args.degradation_sigma)
    sampler = create_sampler(**yaml.safe_load(open(args.diffusion_config)))
    if args.method == "pigdm":                       # pseudoinverse-guided; same sampler+operator as DPS
        cond = get_conditioning_method("pigdm", operator, GaussianNoiser(), scale=args.scale,
                                       sigma_y=args.pigdm_sigma, tfactor="sqrt_abar",
                                       alphas_cumprod=sampler.alphas_cumprod)
    else:
        cond = get_conditioning_method("ps", operator, GaussianNoiser(), scale=args.scale)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=cond.conditioning)

    out = Path(args.save_dir) / "heat_blur"
    for s in ["label", "input", "recon"]:
        (out / s).mkdir(parents=True, exist_ok=True)
    cleans = sorted(Path(args.clean_dir).glob("*.png"))[:args.num_images]
    obs = sorted(Path(args.observation_dir).glob("*.png"))[:args.num_images]
    for i, (c, o) in enumerate(zip(cleans, obs)):
        if args.method == "pigdm":
            cond.reset()                             # PiGDM tracks reverse-step index internally
        clean = _to_model_range(c, image_size, device)
        meas = _to_model_range(o, image_size, device)
        x_start = torch.randn(clean.shape, device=device).requires_grad_()
        rec = sample_fn(x_start=x_start, measurement=meas, record=False, save_root=str(out))
        _save(out / "label" / f"{i:05d}.png", clean)
        _save(out / "input" / f"{i:05d}.png", meas)
        _save(out / "recon" / f"{i:05d}.png", rec)
        print(f"saved {i:05d}.png", flush=True)


if __name__ == "__main__":
    main()
