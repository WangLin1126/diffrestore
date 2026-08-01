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

```bash
python -m model.bdm_backbone.train \
  --data_pt data/cifar10/cifar10_uint8.pt --preset cifar10 \
  --batch 512 --lr 4e-4 --gpus 0 1 2 3 --steps 500000 \
  --ckpt_every 5000 --sample_every 2000 --sample_steps 250 \
  --out checkpoint/bdm/cifar10.pth --sample_dir results/bdm_cifar10 \
  > results/bdm_cifar10/train.log 2>&1 &
```

Fixed paper settings (from `--preset cifar10`): net = 49.5M-param UNet (ch 256, ch_mult (1,1,1),
attn @16&8, 3 resblocks, dropout 0.2); `sigma_blur_max=20`; VP-cosine noise logsnr∈[−10,10];
Adam β1=0.9, EMA 0.9999, grad-clip 1.0, 5k linear LR warmup.

### Sizing `--batch` / `--lr` to the GPUs
`--gpus` uses DataParallel (batch split evenly). Pick the batch to fill memory; scale lr ~√(batch):

| `--batch` (÷ #GPUs) | mem/GPU (24 GB card) | throughput | `--lr` |
|---|---|---|---|
| 128 | ~5 GB (under-utilized) | ~130 img/s | 2e-4 (paper) |
| **512** | **~18–19 GB** | **~520+ img/s** | **4e-4** |
| 768 | ~22 GB (near full) | higher | 5e-4 |

Total training is batch-invariant in images: **500k × 512 = 256M images = the paper's 2M × 128.**
On other hardware, keep `steps × batch ≈ 2.5e8` for a paper-equivalent run.

## 3. Monitor

- Log: `tail -f results/bdm_cifar10/train.log` — lines `[step/total] loss=… lr=… img/s`.
- Samples: `results/bdm_cifar10/step<N>.png` every `--sample_every` steps (EMA weights, 8-image grid).
  CIFAR needs tens of thousands of steps before samples look like objects; early grids are noisy.
- GPUs: `nvidia-smi`.

## 4. Resume after a preemption

Checkpoints (`checkpoint/bdm/cifar10.pth`, every `--ckpt_every`) hold model+EMA+optimizer+step.

```bash
python -m model.bdm_backbone.train --data_pt data/cifar10/cifar10_uint8.pt --preset cifar10 \
  --batch 512 --lr 4e-4 --gpus 0 1 2 3 --steps 500000 \
  --out checkpoint/bdm/cifar10.pth --resume checkpoint/bdm/cifar10.pth \
  >> results/bdm_cifar10/train.log 2>&1 &
```

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
  --batch 32 --gpus 0 1 2 3 --steps 300000 \
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

256px sampling is slow — keep `--sample_every` large (10k) and `--sample_steps 250` for previews;
use 500–1000 steps for final samples (§5). Resume with `--resume` exactly as §4. FFHQ needs on the
order of 10⁵ steps before faces sharpen; early grids are blurry.

## Notes / knobs
- `--sigma_blur_max 0` recovers a plain DDPM baseline (ablation), for either dataset.
- Single GPU: `--gpus 0` and a batch that fits (CIFAR 128–256; FFHQ 2–4).
