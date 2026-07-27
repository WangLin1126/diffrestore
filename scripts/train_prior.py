"""Train a non-hot IHDM-style x0-predictor prior on face data.

Forward:  x_t = K_t x0 + sigma_path * eps   (K_t = DCT-heat blur, level t in [1..N])
Loss:     L1( F_theta(x_t, t), x0 )
Usage:
  python smdc/scripts/train_prior.py --shards 2 --image_size 128 --steps 60000 \
      --out runs/celebahq128
"""
import os
import sys
import glob
import time
import copy
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from huggingface_hub import hf_hub_download

from ops.transforms import DCTTransform
from utils.seed import set_seed
from ops.heat import HeatSchedule
from model.unet import UNet
from data.loaders import load_parquet_faces, to_unit


def get_data(shards, image_size, cache_dir, device):
    paths = []
    for i in range(shards):
        paths.append(hf_hub_download(
            repo_id="korexyz/celeba-hq-256x256", repo_type="dataset",
            filename=f"data/train-0000{i}-of-00006.parquet", local_dir=cache_dir))
    x = load_parquet_faces(paths, image_size=image_size)   # (N,3,H,W) uint8
    print(f"  loaded {x.shape[0]} images at {image_size}px")
    return x.pin_memory() if torch.cuda.is_available() else x   # uint8 on CPU (move per-batch)


@torch.no_grad()
def generate(model, sch, n, device, mean_img):
    """Unconditional sanity samples: reverse from a flat mean image over all levels."""
    from model.cold_diffusion import ColdDiffusionPrior
    prior = ColdDiffusionPrior(model, sch)
    x = mean_img.expand(n, -1, -1, -1).clone()
    times = list(range(sch.num_levels - 1, -1, -1))
    for t, t_next in zip(times[:-1], times[1:]):
        x = prior.reverse_step(x, t, t_next)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--ch_mult", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--num_res_blocks", type=int, default=2)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--sigma_max", type=float, default=128.0)
    ap.add_argument("--sigma_min", type=float, default=0.5)
    ap.add_argument("--path_noise", type=float, default=0.01)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--amp", type=int, default=1, help="1=mixed precision fp16 for the model")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--sample_every", type=int, default=5000)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--out", type=str, default="runs/celebahq128")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--device_ids", type=int, nargs="+", default=None,
                    help="GPU ids for DataParallel; >1 id disables AMP (fp32)")
    ap.add_argument("--data", choices=["celebahq", "ffhq_pt"], default="celebahq")
    ap.add_argument("--data_pt", type=str, default="data/ffhq256/ffhq256_uint8.pt")
    ap.add_argument("--attn_res", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    multi = args.device_ids is not None and len(args.device_ids) > 1
    if multi:
        args.device = f"cuda:{args.device_ids[0]}"
        if args.amp:
            print("  [note] AMP disabled under DataParallel (autocast+DP unsafe); using fp32")
            args.amp = 0

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    device = args.device
    H = W = args.image_size

    if args.data == "ffhq_pt":
        data = torch.load(args.data_pt).contiguous()          # (N,3,256,256) uint8 on CPU
        if torch.cuda.is_available():
            data = data.pin_memory()
        print(f"  loaded {data.shape[0]} FFHQ images from {args.data_pt}")
    else:
        data = get_data(args.shards, args.image_size, os.path.join(args.out, "hf_cache"), device)
    N = data.shape[0]
    mean_img = (data.float().div(127.5).sub(1.0)).mean(dim=0, keepdim=True).to(device)  # (1,3,H,W)

    sch = HeatSchedule.ihdm(H, W, K=args.K, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                            transform=DCTTransform(), device=device, dtype=torch.float32)
    n_levels = sch.num_levels

    model = UNet(ch=args.ch, out_ch=3, ch_mult=tuple(args.ch_mult),
                 num_res_blocks=args.num_res_blocks, attn_resolutions=(args.attn_res,),
                 in_channels=3, resolution=H).to(device)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    if multi:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)
    net = model.module if isinstance(model, torch.nn.DataParallel) else model
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"  model {n_params:.1f}M params | levels={n_levels} | images={N} | "
          f"devices={args.device_ids if multi else args.device}")

    start = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); start = ck["step"]
        print(f"  resumed from {args.resume} @ step {start}")

    def ema_update():
        d = args.ema
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), net.parameters()):
                pe.mul_(d).add_(pm, alpha=1 - d)
            for be, bm in zip(ema.buffers(), net.buffers()):
                be.copy_(bm)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
    model.train()
    t0 = time.time()
    run_loss = 0.0
    for step in range(start, args.steps):
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        idx = torch.randint(0, N, (args.batch,))
        x0 = data[idx].to(device, non_blocking=True).float().div(127.5).sub(1.0)  # (B,3,H,W)
        t = torch.randint(1, n_levels, (args.batch,), device=device)
        with torch.no_grad():                                   # blur in fp32 for accuracy
            x_t = sch.apply_K_batch(x0, t)
            if args.path_noise > 0:
                x_t = x_t + args.path_noise * torch.randn_like(x_t)

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(args.amp)):
            pred = model(x_t, t.float())
            loss = F.l1_loss(pred, x0)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step >= args.warmup:
            ema_update()

        run_loss += loss.item()
        if (step + 1) % args.log_every == 0:
            ips = args.log_every * args.batch / (time.time() - t0)
            print(f"[{step+1:>7d}/{args.steps}] loss={run_loss/args.log_every:.4f} "
                  f"lr={lr:.2e} {ips:.0f} img/s", flush=True)
            run_loss = 0.0; t0 = time.time()

        if (step + 1) % args.sample_every == 0:
            ema.eval()
            g = generate(ema, sch, 16, device, mean_img)
            save_image(to_unit(g), os.path.join(args.out, "samples", f"gen_{step+1}.png"), nrow=4)
            model.train()

        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.steps:
            torch.save({"model": net.state_dict(), "ema": ema.state_dict(),
                        "opt": opt.state_dict(), "step": step + 1, "args": vars(args)},
                       os.path.join(args.out, "checkpoint.pth"))

    print("done.")


if __name__ == "__main__":
    main()
