"""Export a shared test set for the DPS-vs-SMDC comparison.

Takes held-out FFHQ-256 images from the cache, applies the SAME DCT-heat blur (+noise)
both methods use, and writes clean/ + observation/ PNGs that DPS and SMDC both consume.

  python smdc/scripts/make_shared_obs.py --out compare_dps_smdc --n 8 --blur_sigma 4 --noise 0.05
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from PIL import Image

from ops.transforms import DCTTransform
from ops.deblur import build_deblur
from utils.seed import set_seed


def save_png(path, x):  # x in [-1,1], (3,H,W)
    a = ((x.clamp(-1, 1) + 1) / 2 * 255 + 0.5).byte().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(a).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["ffhq_pt", "celebahq"], default="celebahq",
                    help="celebahq = held-out CelebA-HQ-256 val, unseen by FFHQ priors (fair)")
    ap.add_argument("--cache", default="data/ffhq256/ffhq256_uint8.pt")
    ap.add_argument("--out", default="compare_dps_smdc")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--blur_sigma", type=float, default=4.0)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    H = W = args.image_size
    clean_dir = os.path.join(args.out, "clean")
    obs_dir = os.path.join(args.out, "observation")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(obs_dir, exist_ok=True)

    if args.source == "celebahq":
        from huggingface_hub import hf_hub_download
        from data.loaders import load_parquet_faces
        p = hf_hub_download(repo_id="korexyz/celeba-hq-256x256", repo_type="dataset",
                            filename="data/validation-00000-of-00001.parquet",
                            local_dir=os.path.join(args.out, "hf_cache"))
        x0 = load_parquet_faces([p], image_size=args.image_size, limit=args.n)\
            .float().div(127.5).sub(1.0)                 # held-out CelebA-HQ val, [-1,1]
    else:
        t = torch.load(args.cache, map_location="cpu", mmap=True)
        x0 = t[-args.n:].float().div(127.5).sub(1.0)     # FFHQ tail (NOTE: seen in training)
    A, _ = build_deblur(H, W, args.blur_sigma, transform=DCTTransform(),
                        device="cpu", dtype=torch.float32)
    for i in range(x0.shape[0]):
        xi = x0[i:i + 1]
        y = (A.forward(xi) + args.noise * torch.randn_like(xi)).clamp(-1, 1)
        save_png(os.path.join(clean_dir, f"{i:05d}.png"), xi[0])
        save_png(os.path.join(obs_dir, f"{i:05d}.png"), y[0])
    print(f"wrote {x0.shape[0]} clean+observation pairs to {args.out}/ "
          f"(blur_sigma={args.blur_sigma}, noise={args.noise})")


if __name__ == "__main__":
    main()
