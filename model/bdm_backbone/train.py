"""Train a Blurring Diffusion Model (epsilon-prediction, unweighted MSE, Eq. 24).

BDM is an improvement of IHDM, so the 256px path reuses IHDM's net (~211M) and its training
tricks/settings from model/ihdm_backbone: linear LR warmup, EMA with num_updates ramp (decay
0.9999), grad-clip 1.0, blur scaled with resolution (blur_sigma_max=128 @256px), DataParallel.

Data is a uint8 tensor (N,C,H,W) in [0,255] saved via torch.save (build it with
model/bdm_backbone/prepare_data.py); images are normalized to [-1,1].  Use --smoke for a
random-data plumbing test.  See model/bdm_backbone/RUNNING.md for the full remote recipe.

  # CIFAR-10 (paper Table-5 arch); scale --batch to the GPUs, --lr ~sqrt with batch:
  python -m model.bdm_backbone.prepare_data --dataset cifar10
  python -m model.bdm_backbone.train --data_pt data/cifar10/cifar10_uint8.pt \
     --preset cifar10 --batch 512 --lr 4e-4 --gpus 0 1 2 3
  # FFHQ-256, IHDM-matched 211M net + settings (80GB or multi-GPU):
  python -m model.bdm_backbone.train --data_pt data/ffhq256/ffhq256_uint8.pt \
     --preset ihdm256 --steps 200000 --batch 32 --gpus 0 1 2 3
"""
from __future__ import annotations
import os, sys, time, copy, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ihdm_backbone")))
import torch
from torchvision.utils import save_image

from model.bdm_backbone.schedule import BlurringSchedule, BlurringScheduleConfig
from model.bdm_backbone.diffusion import BlurringDiffusion
from model.bdm_backbone.model import build_bdm_model, PRESETS
from model_code.ema import ExponentialMovingAverage    # IHDM's EMA (num_updates ramp)


def default_blur(image_size: int) -> float:
    """Blur scaled with resolution: 20 @ <128px (the paper's CIFAR/LSUN value) up to 128 @ 256px,
    mirroring how IHDM raises blur_sigma_max with resolution."""
    return 128.0 if image_size >= 256 else (64.0 if image_size >= 128 else 20.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_pt", default=None, help="uint8 tensor (N,C,H,W) in [0,255]")
    ap.add_argument("--preset", default=None, choices=list(PRESETS.keys()))
    ap.add_argument("--image_size", type=int, default=32, help="ignored when --preset is set")
    ap.add_argument("--in_channels", type=int, default=3)
    ap.add_argument("--sigma_blur_max", type=float, default=None, help="default scales with resolution")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default 2e-5 for ihdm256 else 2e-4")
    ap.add_argument("--warmup", type=int, default=5000, help="linear LR warmup steps (IHDM: 5000)")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--sample_every", type=int, default=10000)
    ap.add_argument("--sample_steps", type=int, default=250)
    ap.add_argument("--out", default="checkpoint/bdm/bdm.pth")
    ap.add_argument("--sample_dir", default="results/bdm_train")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", action="store_true", help="random data, 5 steps: plumbing test")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    primary = f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # model (single source of truth for image_size/in_channels is the built cfg) ---------------
    over = {} if args.preset else dict(image_size=args.image_size, in_channels=args.in_channels)
    base = build_bdm_model(args.preset, **over).to(primary)
    H, C = base.cfg.image_size, base.cfg.in_channels
    lr = args.lr if args.lr is not None else (2e-5 if base.cfg.backbone == "ihdm" else 2e-4)
    blur = args.sigma_blur_max if args.sigma_blur_max is not None else default_blur(H)

    model = torch.nn.DataParallel(base, device_ids=args.gpus) if len(args.gpus) > 1 else base
    sch = BlurringSchedule(BlurringScheduleConfig(image_size=H, sigma_blur_max=blur), device=primary)
    diff = BlurringDiffusion(sch)
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay, use_num_updates=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(args.beta1, 0.999))
    n_params = sum(p.numel() for p in base.parameters()) / 1e6
    print(f"  BDM[{base.cfg.backbone}]: {n_params:.1f}M params | {H}px | sigma_blur_max={blur} | "
          f"batch={args.batch} | lr={lr:.1e} | warmup={args.warmup} | gpus={args.gpus}", flush=True)

    # data ------------------------------------------------------------------------------------
    if args.smoke:
        data = torch.randint(0, 256, (64, C, H, H), dtype=torch.uint8)
        args.steps, args.log_every, args.sample_every, args.sample_steps, args.warmup = 5, 1, 5, 4, 2
    else:
        assert args.data_pt, "--data_pt required (or use --smoke)"
        data = torch.load(args.data_pt, map_location="cpu", mmap=True)
        assert data.shape[-1] == H, f"data is {data.shape[-1]}px but model/preset is {H}px"
    N = data.shape[0]
    print(f"  data: {N} images {tuple(data.shape[1:])}", flush=True)

    start = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=primary, weights_only=False)
        base.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema_state"]); opt.load_state_dict(ck["opt"])
        start = ck["step"]; print(f"  resumed @ {start}", flush=True)

    def save_ckpt(step):
        ema.store(model.parameters()); ema.copy_to(model.parameters())
        ema_model_sd = copy.deepcopy(base.state_dict())         # EMA weights as a plain model state_dict
        ema.restore(model.parameters())
        torch.save({"model": base.state_dict(), "ema": ema_model_sd, "ema_state": ema.state_dict(),
                    "opt": opt.state_dict(), "cfg": base.cfg.__dict__, "sch": sch.cfg.__dict__,
                    "step": step}, args.out)

    model.train()
    t0 = time.time(); run = torch.zeros((), device=primary)
    for step in range(start, args.steps):
        cur_lr = lr * min((step + 1) / max(1, args.warmup), 1.0)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        idx = torch.randint(0, N, (args.batch,))
        x = data[idx].to(primary).float().div(127.5).sub(1.0)          # [-1,1]
        loss = diff.loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        ema.update(model.parameters())

        run += loss.detach()                                            # accumulate on-device (no sync)
        if (step + 1) % args.log_every == 0:
            ips = args.log_every * args.batch / (time.time() - t0)
            print(f"[{step+1:>7d}/{args.steps}] loss={run.item()/args.log_every:.4f} lr={cur_lr:.2e} "
                  f"{ips:.0f} img/s", flush=True)
            run.zero_(); t0 = time.time()
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.steps:
            save_ckpt(step + 1)
        if (step + 1) % args.sample_every == 0:
            ema.store(model.parameters()); ema.copy_to(model.parameters())
            with torch.no_grad():
                xs = diff.sample(base, (min(8, args.batch), C, H, H), num_steps=args.sample_steps,
                                 device=primary)
            ema.restore(model.parameters())
            save_image((xs.clamp(-1, 1) + 1) / 2, f"{args.sample_dir}/step{step+1}.png", nrow=4)
            model.train()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
