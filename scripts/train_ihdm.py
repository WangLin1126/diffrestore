"""Train a faithful IHDM prior on FFHQ-256 (vendored NCSN++ U-Net + exact IHDM residual loss).

Loss (Rissanen et al.): x_k=K_k x0, x_{k-1}=K_{k-1} x0, u=x_k+sigma*eps (images in [0,1]);
predict the deblurring residual so that u + model(u,k) ~= x_{k-1}.
    loss = sum_pix ( x_{k-1} - (u + model(u,k)) )^2 ,  k ~ Uniform{1..K-1}

  python scripts/train_ihdm.py --steps 200000 --batch 16 --out checkpoint/ihdm/ffhq256_train.pth
"""
import os
import sys
import time
import argparse

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKBONE = os.path.join(REPO, "model", "ihdm_backbone")
sys.path.insert(0, REPO)
sys.path.insert(0, BACKBONE)

import importlib
import numpy as np
import torch
from torchvision.utils import save_image

from model_code import utils as mutils
from model_code.ema import ExponentialMovingAverage
from utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_pt", default="data/ffhq256/ffhq256_uint8.pt")
    ap.add_argument("--config", default="img_size_256_full",
                    help="configs/ffhq/*: img_size_256_full (211M, for 80GB) | img_size_256_train (160M, <=24GB)")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=32, help="total (DataParallel splits); ~14-16 per 80GB GPU")
    ap.add_argument("--lr", type=float, default=None, help="override lr (default from config; scale ~linearly with batch)")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--sample_every", type=int, default=10000)
    ap.add_argument("--out", default="checkpoint/ihdm/ffhq256_train.pth")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs("results/ihdm256_train", exist_ok=True)
    primary = f"cuda:{args.gpus[0]}"

    config = importlib.import_module(f"configs.ffhq.{args.config}").get_config()
    config.device = torch.device(primary)
    config.training.batch_size = args.batch
    if args.lr is not None:
        config.optim.lr = args.lr
    K = config.model.K

    model = mutils.create_model(config, device_ids=args.gpus)     # NCSN++ (DataParallel)
    net = model.module if isinstance(model, torch.nn.DataParallel) else model
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    opt = torch.optim.Adam(model.parameters(), lr=config.optim.lr,
                           betas=(config.optim.beta1, 0.999), eps=config.optim.eps)
    heat = mutils.create_forward_process(config, config.device)   # DCTBlur
    model_fn = mutils.get_model_fn(model, train=True)
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"  IHDM-256 [{args.config}]: {n_params:.1f}M params | K={K} | gpus={args.gpus} | "
          f"batch={args.batch} | lr={config.optim.lr:.1e}", flush=True)

    data = torch.load(args.data_pt, map_location="cpu", mmap=True)   # (N,3,256,256) uint8, [0,255]
    N = data.shape[0]
    print(f"  data: {N} FFHQ-256 images", flush=True)

    start = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=primary, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        start = ck["step"]; print(f"  resumed @ {start}", flush=True)

    model.train()
    t0 = time.time(); run = 0.0
    for step in range(start, args.steps):
        lr = config.optim.lr * min((step + 1) / config.optim.warmup, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, N, (args.batch,))
        x0 = data[idx].to(primary, non_blocking=True).float().div(255.0)   # [0,1]
        k = torch.randint(1, K, (args.batch,), device=primary)
        with torch.no_grad():
            blurred = heat(x0, k).float()
            less_blurred = heat(x0, k - 1).float()
            u = blurred + torch.randn_like(blurred) * config.model.sigma
        pred = u + model_fn(u, k)
        loss = ((less_blurred - pred) ** 2).reshape(args.batch, -1).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
        opt.step()
        ema.update(model.parameters())

        run += loss.item()
        if (step + 1) % args.log_every == 0:
            ips = args.log_every * args.batch / (time.time() - t0)
            print(f"[{step+1:>7d}/{args.steps}] loss={run/args.log_every:.2f} lr={lr:.2e} {ips:.0f} img/s",
                  flush=True)
            run = 0.0; t0 = time.time()
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.steps:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                        "opt": opt.state_dict(), "step": step + 1}, args.out)
        if (step + 1) % args.sample_every == 0:
            ema.store(model.parameters()); ema.copy_to(model.parameters())
            with torch.no_grad():
                x0v = data[:4].to(primary).float().div(255.0)
                kv = torch.full((4,), K // 2, device=primary)
                bl = heat(x0v, kv).float()
                rec = (bl + config.model.sigma * torch.randn_like(bl)) + model_fn(bl, kv)
            save_image(torch.cat([x0v, bl, rec.clamp(0, 1)]),
                       f"results/ihdm256_train/step{step+1}.png", nrow=4)
            ema.restore(model.parameters()); model.train()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
