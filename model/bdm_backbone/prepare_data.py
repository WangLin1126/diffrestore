"""Prepare a dataset tensor for BDM training: a uint8 (N,C,H,W) tensor in [0,255] saved via
torch.save, which model/bdm_backbone/train.py loads (mmap) and normalizes to [-1,1].

CIFAR-10 (the reference experiment) is downloaded via torchvision. A generic image folder can
also be packed (all images resized to --image_size).

  python -m model.bdm_backbone.prepare_data --dataset cifar10 --out data/cifar10/cifar10_uint8.pt
  python -m model.bdm_backbone.prepare_data --dataset folder --root path/to/images \
     --image_size 64 --out data/mydata_uint8.pt
"""
from __future__ import annotations
import os, glob, argparse
import numpy as np
import torch


def prepare_cifar10(root: str) -> torch.Tensor:
    import torchvision
    ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    return torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()      # (50000,3,32,32) uint8


def prepare_folder(root: str, image_size: int) -> torch.Tensor:
    from PIL import Image
    paths = sorted(sum([glob.glob(os.path.join(root, f"*.{e}")) for e in ("png", "jpg", "jpeg")], []))
    assert paths, f"no images under {root}"
    xs = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((image_size, image_size), Image.LANCZOS)
        xs.append(torch.from_numpy(np.asarray(im, np.uint8)).permute(2, 0, 1))
    return torch.stack(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "folder"], default="cifar10")
    ap.add_argument("--root", default="data/cifar10/raw", help="download/source dir")
    ap.add_argument("--image_size", type=int, default=32, help="folder mode only")
    ap.add_argument("--out", default="data/cifar10/cifar10_uint8.pt")
    args = ap.parse_args()

    x = prepare_cifar10(args.root) if args.dataset == "cifar10" else prepare_folder(args.root, args.image_size)
    assert x.dtype == torch.uint8 and x.dim() == 4
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(x, args.out)
    print(f"saved {tuple(x.shape)} uint8 [{x.min()},{x.max()}] -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
