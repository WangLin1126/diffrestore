"""Train an ADM-capacity IHDM prior on ImageNet-256 (streaming loader + DDP for remote GPUs).

Same IHDM residual objective as scripts/train_ihdm.py (Rissanen et al.):
    x_k = K_k x0,  x_{k-1} = K_{k-1} x0,  u = x_k + sigma*eps   (images in [0,1])
    predict the deblurring residual so that u + model(u,k) ~= x_{k-1}
    loss = sum_pix ( x_{k-1} - (u + model(u,k)) )^2 ,  k ~ Uniform{1..K-1}

Differs from train_ihdm.py in the DATA PATH only: ImageNet is ~1.28M images (~252 GB as a
single uint8 tensor), so we stream from an ImageFolder tree with a DataLoader instead of
loading one .pt into RAM. Preprocessing matches OpenAI ADM (center_crop_arr -> 256, RGB-safe)
so the IHDM prior sees the same distribution the DDRM/DPS baselines' ADM prior was trained on.

Launch (remote, N GPUs on one node) -- DDP via torchrun (recommended at ADM scale):
    torchrun --nproc_per_node=8 scripts/train_ihdm_imagenet.py \
        --data_root /data/imagenet256/train --batch 16 --steps 500000 \
        --out checkpoint/ihdm/imagenet256.pth

Single GPU / no torchrun (falls back automatically):
    python scripts/train_ihdm_imagenet.py --data_root /data/imagenet256/train --batch 8 --device cuda:0

--batch is PER-GPU; global batch = per_gpu * nproc_per_node. Scale --lr ~linearly with it.
"""
import os
import io
import sys
import glob
import time
import random
import argparse

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import datasets
from torchvision.utils import save_image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKBONE = os.path.join(REPO, "model", "ihdm_backbone")
sys.path.insert(0, REPO)
sys.path.insert(0, BACKBONE)

import importlib
from model_code.unet import UNetModel
from model_code import utils as mutils
from model_code.ema import ExponentialMovingAverage
from utils.seed import set_seed


# ---- ADM preprocessing (RGB-safe center crop -> square -> image_size), from openai/guided-diffusion
def center_crop_arr(pil_image, image_size=256):
    pil_image = pil_image.convert("RGB")                 # RGB-safe: some ImageNet imgs are L/CMYK
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    cy = (arr.shape[0] - image_size) // 2
    cx = (arr.shape[1] - image_size) // 2
    return arr[cy:cy + image_size, cx:cx + image_size]


class Preproc:
    """PIL -> float CHW tensor in [0,1], ADM crop + optional random horizontal flip."""
    def __init__(self, image_size=256, hflip=True):
        self.image_size = image_size
        self.hflip = hflip

    def __call__(self, img):
        arr = center_crop_arr(img, self.image_size)      # HWC uint8
        if self.hflip and np.random.rand() < 0.5:
            arr = arr[:, ::-1, :]
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().div_(255.0)
        return t                                          # (3,H,W) in [0,1]


class ParquetImageStream(IterableDataset):
    """Infinite shuffled stream over HuggingFace-style parquet shards (e.g.
    benjamin-paine/imagenet-1k-256x256): image column is struct{bytes, path}. Reads bytes with
    pyarrow (no `datasets` dependency), decodes with PIL. Shards are split disjointly across
    (DDP rank x DataLoader worker); shuffling is file-order + row-group + a reservoir buffer."""

    def __init__(self, files, transform, image_col="image", rank=0, world=1, seed=0, shuffle_buf=2000):
        super().__init__()
        self.files = sorted(files)
        self.transform = transform
        self.col = image_col
        self.rank = rank
        self.world = world
        self.seed = seed
        self.shuffle_buf = shuffle_buf

    def __iter__(self):
        import pyarrow.parquet as pq
        info = get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        gid = self.rank * nw + wid                       # global stream id
        gnum = self.world * nw                            # total streams
        my_files = self.files[gid::gnum]                 # disjoint shard subset
        rng = random.Random(self.seed + gid)

        def raw_cells():
            while True:                                   # infinite: reshuffle files each pass
                order = my_files[:]; rng.shuffle(order)
                for f in order:
                    pf = pq.ParquetFile(f)
                    rgs = list(range(pf.num_row_groups)); rng.shuffle(rgs)
                    for rg in rgs:
                        cells = pf.read_row_group(rg, columns=[self.col]).column(self.col).to_pylist()
                        rng.shuffle(cells)
                        yield from cells

        if not my_files:                                 # more streams than shards: this one idles
            return
        src = raw_cells()                                 # infinite, so priming always succeeds
        buf = [next(src) for _ in range(self.shuffle_buf)]
        while True:
            j = rng.randrange(len(buf))
            cell = buf[j]; buf[j] = next(src)
            b = cell["bytes"] if isinstance(cell, dict) else cell
            yield self.transform(Image.open(io.BytesIO(b))), 0


def ddp_setup():
    """Returns (is_dist, rank, world_size, local_rank). No-op single-process if not under torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return True, rank, world, local
    return False, 0, 1, 0


def infinite(loader, sampler):
    """Yield batches forever, reshuffling each epoch (DistributedSampler needs set_epoch)."""
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="dir of *.parquet shards (HF benjamin-paine/imagenet-1k-256x256) OR an "
                         "ImageFolder tree (train/<class>/*.jpg). Auto-detected.")
    ap.add_argument("--image_col", default="image", help="parquet image column (struct{bytes,path})")
    ap.add_argument("--shuffle_buf", type=int, default=2000, help="parquet reservoir shuffle buffer (per stream)")
    ap.add_argument("--config", default="img_size_256_imagenet",
                    help="configs/ffhq/*: img_size_256_imagenet (589M, ADM-matched)")
    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--batch", type=int, default=16, help="PER-GPU batch; global = batch*world_size")
    ap.add_argument("--lr", type=float, default=None, help="override config lr (scale ~linearly w/ global batch)")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers per process")
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="gradient checkpointing (recompute activations): big memory saving, ~30%% slower")
    ap.add_argument("--device", default="cuda:0", help="single-process device (ignored under torchrun)")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--sample_every", type=int, default=10000)
    ap.add_argument("--out", default="checkpoint/ihdm/imagenet256.pth")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    is_dist, rank, world, local = ddp_setup()
    is_main = (rank == 0)
    device = f"cuda:{local}" if is_dist else args.device
    set_seed(args.seed + rank)                            # decorrelate per-rank augmentation/noise
    if is_main:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        os.makedirs("results/ihdm_imagenet256", exist_ok=True)

    config = importlib.import_module(f"configs.ffhq.{args.config}").get_config()
    config.device = torch.device(device)
    config.training.batch_size = args.batch
    if args.lr is not None:
        config.optim.lr = args.lr
    K = config.model.K
    imsz = config.data.image_size

    # ---- model / ema / opt / heat forward
    net = UNetModel(config, use_checkpoint=args.grad_ckpt).to(device)
    ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
    model = DDP(net, device_ids=[local]) if is_dist else net
    opt = torch.optim.Adam(net.parameters(), lr=config.optim.lr,
                           betas=(config.optim.beta1, 0.999), eps=config.optim.eps)
    heat = mutils.create_forward_process(config, config.device)   # DCTBlur on device
    if is_main:
        n_params = sum(p.numel() for p in net.parameters()) / 1e6
        print(f"  IHDM-ImageNet [{args.config}]: {n_params:.1f}M params | K={K} | "
              f"world={world} | per_gpu_batch={args.batch} | global_batch={args.batch*world} | "
              f"lr={config.optim.lr:.1e}", flush=True)

    # ---- streaming data (auto-detect: parquet shards vs ImageFolder tree)
    preproc = Preproc(imsz, hflip=config.data.random_flip)
    parquet_files = sorted(glob.glob(os.path.join(args.data_root, "*.parquet")))
    if parquet_files:                                     # HF parquet (benjamin-paine/imagenet-1k-256x256)
        ds = ParquetImageStream(parquet_files, preproc, image_col=args.image_col,
                                rank=rank, world=world, seed=args.seed, shuffle_buf=args.shuffle_buf)
        sampler = None                                    # IterableDataset shards internally
        loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                            pin_memory=True, drop_last=True, persistent_workers=(args.workers > 0))
        batches = iter(loader)                            # already infinite
        if is_main:
            print(f"  data: {len(parquet_files)} parquet shards under {args.data_root} "
                  f"(streaming, shuffle_buf={args.shuffle_buf})", flush=True)
    else:                                                 # ImageFolder tree of image files
        ds = datasets.ImageFolder(args.data_root, transform=preproc)
        sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True) if is_dist else None
        loader = DataLoader(ds, batch_size=args.batch, shuffle=(sampler is None), sampler=sampler,
                            num_workers=args.workers, pin_memory=True, drop_last=True,
                            persistent_workers=(args.workers > 0))
        batches = infinite(loader, sampler)
        if is_main:
            print(f"  data: {len(ds)} ImageNet images under {args.data_root} (ImageFolder)", flush=True)

    # ---- resume
    start = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        start = ck["step"]
        if is_main:
            print(f"  resumed @ {start}", flush=True)

    model.train()
    t0 = time.time(); run = 0.0
    for step in range(start, args.steps):
        lr = config.optim.lr * min((step + 1) / config.optim.warmup, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        x0, _ = next(batches)
        x0 = x0.to(device, non_blocking=True)                    # (B,3,H,W) in [0,1]
        k = torch.randint(1, K, (x0.shape[0],), device=device)
        with torch.no_grad():
            blurred = heat(x0, k).float()
            less_blurred = heat(x0, k - 1).float()
            u = blurred + torch.randn_like(blurred) * config.model.sigma
        pred = u + model(u, k)
        loss = ((less_blurred - pred) ** 2).reshape(x0.shape[0], -1).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.optim.grad_clip)
        opt.step()
        ema.update(net.parameters())

        run += loss.item()
        if is_main and (step + 1) % args.log_every == 0:
            ips = args.log_every * args.batch * world / (time.time() - t0)
            print(f"[{step+1:>7d}/{args.steps}] loss={run/args.log_every:.2f} lr={lr:.2e} "
                  f"{ips:.0f} img/s", flush=True)
            run = 0.0; t0 = time.time()
        if is_main and ((step + 1) % args.ckpt_every == 0 or (step + 1) == args.steps):
            torch.save({"model": net.state_dict(), "ema": ema.state_dict(),
                        "opt": opt.state_dict(), "step": step + 1}, args.out)
        if is_main and (step + 1) % args.sample_every == 0:
            ema.store(net.parameters()); ema.copy_to(net.parameters())
            with torch.no_grad():
                x0v = x0[:4]
                kv = torch.full((x0v.shape[0],), K // 2, device=device)
                bl = heat(x0v, kv).float()
                rec = (bl + config.model.sigma * torch.randn_like(bl)) + net(bl, kv)
            save_image(torch.cat([x0v, bl, rec.clamp(0, 1)]),
                       f"results/ihdm_imagenet256/step{step+1}.png", nrow=4)
            ema.restore(net.parameters()); net.train()

    if is_main:
        print("done.", flush=True)
    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
