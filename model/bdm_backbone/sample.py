"""Sample images from a trained Blurring Diffusion Model (inference; Algorithm 1).

  python -m model.bdm_backbone.sample --ckpt checkpoint/bdm/bdm.pth --n 16 --steps 250 \
     --out results/bdm_samples/grid.png
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from torchvision.utils import save_image

from model.bdm_backbone.schedule import BlurringSchedule, BlurringScheduleConfig
from model.bdm_backbone.diffusion import BlurringDiffusion
from model.bdm_backbone.model import BDMDenoiser, BDMModelConfig


def load_bdm(ckpt_path, device="cuda:0", use_ema=True):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BDMDenoiser(BDMModelConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["ema"] if (use_ema and "ema" in ck) else ck["model"])
    model.eval()
    sch = BlurringSchedule(BlurringScheduleConfig(**ck["sch"]), device=device)
    return model, BlurringDiffusion(sch), model.cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bdm_samples/grid.png")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model, diff, cfg = load_bdm(args.ckpt, args.device, use_ema=not args.no_ema)
    xs = diff.sample(model, (args.n, cfg.in_channels, cfg.image_size, cfg.image_size),
                     num_steps=args.steps, device=args.device, progress=True)
    save_image((xs.clamp(-1, 1) + 1) / 2, args.out, nrow=int(args.n ** 0.5) or 1)
    print(f"saved {args.n} samples -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
