# Running BDM experiments (remote): CIFAR-10 & FFHQ-256

Train a Blurring Diffusion Model on CIFAR-10 with the paper's architecture (Hoogeboom &
Salimans 2023, Table 5) and generate samples. 

## 0. Environment

- Python + PyTorch (CUDA build) + torchvision, `pip install lpips` (only for `sanity_check.py`).
- No install step for the module: `model/bdm_backbone/` imports `ops.dct`, `model.unet`, and the
  IHDM `model_code.*` (auto-added to `sys.path` by the scripts). Run everything as
  `python -m model.bdm_backbone.<script>` from the repo root.
- Optional smoke test (random data, ~5 steps, no data needed):
  `python -m model.bdm_backbone.sanity_check && python -m model.bdm_backbone.train --smoke`

## 1. Prepare data (once)

```bash
python -m model.bdm_backbone.prepare_data --dataset cifar10 --out data/cifar10/cifar10_uint8.pt
```
Downloads CIFAR-10 and writes a `(50000,3,32,32)` uint8 tensor (~154 MB). Idempotent.

## 2. Launch training

Paper-faithful CIFAR-10 (batch 128, lr 2e-4), with mixed precision for speed:
```bash
python -m model.bdm_backbone.train \
  --data_pt data/cifar10/cifar10_uint8.pt --preset cifar10 \
  --batch 128 --lr 2e-4 --gpus 0 1 2 3 --steps 2000000 --amp auto \
  --ckpt_every 5000 --sample_every 10000 --sample_steps 250 \
  --out checkpoint/bdm/cifar10.pth --sample_dir results/bdm_cifar10 \
  > results/bdm_cifar10/train.log 2>&1 &
```

**Original paper settings** (Hoogeboom & Salimans, Table 5 / A.2), all fixed by `--preset cifar10`:
net = 49.5M UNet (ch 256, ch_mult (1,1,1), attn @16&8, 3 resblocks, dropout 0.2); `sigma_blur_max=20`
(their best CIFAR value); sin² blur schedule; VP-cosine noise logsnr∈[−10,10]; d_min=1e-3;
**Adam lr 2e-4, batch 128, EMA 0.9999**; unweighted ε-MSE; 2M steps. (The 5k LR warmup is an
implementation add-on, not in the paper — set `--warmup 0` for strict adherence.)

### Mixed precision (`--amp`)
`--amp auto` = **bf16 on Ampere+ (cc≥8, e.g. A100/H100)**, **fp16 on Turing/Volta** — ~2× faster
than fp32 at 256px, and does **not** change any paper hyperparameter (fp32 master weights, same
schedule/lr/EMA). Force with `--amp bf16|fp16`, or `--amp off` for exact fp32. fp16 uses a
GradScaler (saved in the checkpoint); bf16 needs none. Works with the 211M net's gradient
checkpointing (the checkpoint recompute is autocast-aware).

### If you want to fill the GPUs instead (deviates from the paper)
`--gpus` uses DataParallel (batch split evenly). Raising the batch fills memory and improves
throughput but changes the training regime; scale lr ~√(batch): e.g. `--batch 512 --lr 4e-4`
(~18 GB/24 GB card). Total images are batch-invariant, so keep `steps × batch ≈ 2.5e8` (= the
paper's 2M × 128) for an equivalent run. For real multi-GPU speedup, DDP would beat DataParallel.

## 3. Monitor

- Log: `tail -f results/bdm_cifar10/train.log` — lines `[step/total] loss=… lr=… img/s`.
- Samples: `results/bdm_cifar10/step<N>.png` every `--sample_every` steps (EMA weights, 8-image grid).
  CIFAR needs tens of thousands of steps before samples look like objects; early grids are noisy.
- GPUs: `nvidia-smi`.

## 4. Resume after a preemption

Checkpoints (every `--ckpt_every`) hold **model + EMA + optimizer + GradScaler + step**. Point
`--resume` at the checkpoint; it prints `resumed @ <step>` and continues (LR warmup/schedule pick up
from that step). Re-pass the **same flags** as the original run (`--preset`, `--batch`, `--lr`, ...);
keep `--out` = `--resume` to keep writing the same file.

```bash
python -m model.bdm_backbone.train --data_pt data/cifar10/cifar10_uint8.pt --preset cifar10 \
  --batch 128 --lr 2e-4 --gpus 0 1 2 3 --steps 2000000 --amp auto \
  --out checkpoint/bdm/cifar10.pth --resume checkpoint/bdm/cifar10.pth \
  >> results/bdm_cifar10/train.log 2>&1 &
```

Notes: a checkpoint written **before** the AMP change (no scaler key) still resumes fine — the
scaler just starts fresh. Resuming an fp32 run with `--amp` (or vice-versa) is safe (master weights
and optimizer state are fp32); use `--amp off` if you want a bit-for-bit fp32 continuation.

## 5. Sample from a checkpoint (inference)

```bash
python -m model.bdm_backbone.sample --ckpt checkpoint/bdm/cifar10.pth \
  --n 64 --steps 1000 --out results/bdm_cifar10/samples.png
```
Uses the EMA weights; more `--steps` (e.g. 1000) = higher quality than the in-training previews.

## 6. FFHQ-256 (211M, IHDM-matched backbone)

Blurring diffusion on the **exact IHDM NCSN++ net** (`--preset ihdm256` → 210.9M params,
channel_mult (1,2,3,4,5), attn @16&8, 256px). Sizes 1–5 above all apply; only the data, preset,
and defaults change. This is an 80 GB / multi-GPU config.

**Data** — already built in this repo: `data/ffhq256/ffhq256_uint8.pt`, `(70000,3,256,256)` uint8
(~13.7 GB). To rebuild elsewhere from a folder of FFHQ PNGs:
```bash
python -m model.bdm_backbone.prepare_data --dataset folder --root <ffhq_png_dir> \
  --image_size 256 --out data/ffhq256/ffhq256_uint8.pt
```

**Launch** — `--preset ihdm256` pulls the IHDM defaults automatically: **lr 2e-5**,
**sigma_blur_max 128** (blur scaled to 256px), EMA 0.9999, 5k warmup, grad-clip 1.0.
```bash
python -m model.bdm_backbone.train \
  --data_pt data/ffhq256/ffhq256_uint8.pt --preset ihdm256 \
  --batch 32 --gpus 0 1 2 3 --steps 300000 --amp auto \
  --ckpt_every 5000 --sample_every 10000 --sample_steps 250 \
  --out checkpoint/bdm/ffhq256.pth --sample_dir results/bdm_ffhq256 \
  > results/bdm_ffhq256/train.log 2>&1 &
```

**Sizing** — `img_size_256_full` is the 80 GB config (~14–16 img/GPU). Set `--batch` to
`per_gpu × #GPUs` and scale `--lr` ~√(batch) from the 2e-5 base:

| GPU memory | per-GPU batch | e.g. `--batch` on 4 GPUs |
|---|---|---|
| 80 GB | 12–14 | 48 (lr ~2.4e-5) |
| 40 GB | 6 | 24 (lr ~1.7e-5) |
| 24 GB | 2–3 | 8–12 (tight; if OOM, lower `--batch`) |

**Resume** — re-run with `--resume` pointing at the checkpoint and the same flags (drop `--amp`
to use `auto` = bf16 on Ampere+). `>>` appends to the existing log:
```bash
python -m model.bdm_backbone.train --data_pt data/ffhq256/ffhq256_uint8.pt --preset ihdm256 \
  --batch 128 --lr 5e-5 --gpus 0 1 2 3 --steps 300000 \
  --out checkpoint/bdm/ffhq256.pth --sample_dir results/bdm_ffhq256 \
  --resume checkpoint/bdm/ffhq256.pth \
  >> results/bdm_ffhq256/train.log 2>&1 &
```
Prints `resumed @ <step>` and continues (EMA + optimizer + GradScaler + step restored). Starting
fresh (no checkpoint yet) is the same command without `--resume`.

256px sampling is slow — keep `--sample_every` large (10k) and `--sample_steps 250` for previews;
use 500–1000 steps for final samples (§5). FFHQ needs on the order of 10⁵ steps before faces
sharpen; early grids are blurry.

## Notes / knobs
- `--sigma_blur_max 0` recovers a plain DDPM baseline (ablation), for either dataset.
- Single GPU: `--gpus 0` and a batch that fits (CIFAR 128–256; FFHQ 2–4).
