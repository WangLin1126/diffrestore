# Running the BDM CIFAR-10 experiment (remote)

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

## Notes / knobs
- `sigma_blur_max=0` recovers a plain DDPM baseline (ablation).
- Single GPU: `--gpus 0` and a batch that fits (e.g. 128–256).
- FFHQ-256 (IHDM-matched 211M net): `--preset ihdm256` after building a 256px `--data_pt`; needs
  80 GB or multi-GPU, defaults lr 2e-5 / blur 128.
