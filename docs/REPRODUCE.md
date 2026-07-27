# Reproduction: data to download & scripts to run

This repo ships **code only** — datasets, model checkpoints, and generated results are git-ignored
(`data/`, `checkpoint/`, `results/`). This guide lists what to fetch and which scripts to run to train
the priors and reproduce every result. Run everything from the repo root inside the Python env.

## 0. Environment
```bash
conda create -n nhdr python=3.11 -y && conda activate nhdr      # or use your existing env
pip install -r requirements.txt
```
GPU: multi-GPU recommended for 256² training (we used 4×24 GB). CPU works for the numerical gates and small evals.

Sanity check (no data/model needed):
```bash
python scripts/run_tests.py          # 6 numerical gates: adjoint / intertwining / limits / gradient / noise-cov / plumbing
```

## 1. Datasets to download

| Dataset | Purpose | Source | How |
|---|---|---|---|
| **FFHQ-256** (70k) | train the priors | HF `pravsels/FFHQ_256` (14 zip shards) | `python scripts/fetch_ffhq256.py 14` → `data/ffhq256/ffhq256_uint8.pt` (~13.8 GB) |
| **CelebA-HQ-256** (val) | held-out eval / observations | HF `korexyz/celeba-hq-256x256` (parquet) | auto-downloaded on demand by `scripts/make_shared_obs.py` |

CelebA-HQ is used for evaluation because it is **unseen by the FFHQ-trained priors** (fair, uncontaminated).

## 2. Checkpoints (produced by training, or optional downloads)

Place under `checkpoint/` (git-ignored):

| File | What | Get it |
|---|---|---|
| `checkpoint/cold_diffusion/{celebahq128,ffhq256}.pth` | our cold-diffusion x0-priors | produced by `scripts/train_prior.py` |
| `checkpoint/ihdm/ffhq256_train.pth` | **the IHDM-256 prior (this repo's goal)** | produced by `scripts/train_ihdm.py` |
| `checkpoint/ihdm/ffhq128.pth` | pretrained IHDM FFHQ-128 (optional) | train with the AaltoML IHDM repo, or reuse an existing checkpoint |
| `checkpoint/dps/ffhq_10m.pt` | DPS FFHQ-256 hot model (optional, baseline) | DPS repo's Google Drive (`chung/diffusion-posterior-sampling`) |

## 3. Train a prior

**IHDM prior on FFHQ-256** (faithful residual loss; the main target):
```bash
python scripts/train_ihdm.py --steps 200000 --batch 24 --gpus 0 1 2 3 \
    --out checkpoint/ihdm/ffhq256_train.pth
# config: model/ihdm_backbone/configs/ffhq/img_size_256_train.py
#   compact net (num_res_blocks=2, dropout=0.1) → 160M params, fits 24 GB at batch 24 (~10 img/s).
#   For the ORIGINAL 211M net (num_res_blocks=3, dropout=0.3): make a config with model.use_checkpoint=True
#   (gradient checkpointing) and relaunch — fits 24 GB, ~20-30% slower.
# checkpoints every 5k -> checkpoint/ihdm/ffhq256_train.pth ; previews every 10k -> results/ihdm256_train/
```
**Cold-diffusion prior** (our x0-predictor; CelebA-128 or FFHQ-256):
```bash
python scripts/train_prior.py --data ffhq_pt --data_pt data/ffhq256/ffhq256_uint8.pt --image_size 256 \
    --batch 24 --ch 128 --ch_mult 1 1 2 2 4 --device_ids 1 2 3 --out runs/ffhq256
```

## 4. Restore (prior × solver) & evaluate

Build a shared held-out test set, then run any prior × solver on the identical observations:
```bash
python scripts/make_shared_obs.py --source celebahq --out results/eval --n 16 \
    --image_size 256 --blur_sigma 4 --noise 0.05

# IHDM prior + full-MAP (HQS) solver
python scripts/restore.py --prior ihdm --ihdm_config img_size_256_train \
    --ckpt checkpoint/ihdm/ffhq256_train.pth --solver hqs --image_size 256 \
    --clean_dir results/eval/clean --observation_dir results/eval/observation --out results/eval/ihdm_hqs

# IHDM prior + 1-step SMDC gradient
python scripts/restore.py --prior ihdm --ihdm_config img_size_256_train \
    --ckpt checkpoint/ihdm/ffhq256_train.pth --solver smdc --image_size 256 \
    --clean_dir results/eval/clean --observation_dir results/eval/observation --out results/eval/ihdm_smdc

# cold-diffusion prior + SMDC
python scripts/restore.py --prior cold_diffusion --ckpt checkpoint/cold_diffusion/ffhq256.pth \
    --image_size 256 --ch 128 --ch_mult 1 1 2 2 4 --solver smdc \
    --clean_dir results/eval/clean --observation_dir results/eval/observation --out results/eval/cold_smdc
```

**DPS baseline (optional, spare hot path).** The DPS backbone code is vendored under `model/dps_backbone/`,
but its sample data/figures/weights are git-ignored — download `ffhq_10m.pt` (above) into
`checkpoint/dps/`, then:
```bash
python scripts/run_dps.py --clean_dir results/eval/clean --observation_dir results/eval/observation \
    --save_dir results/eval/dps --degradation_sigma 4 --scale 0.3 --num_images 16
```

**Figures:**
```bash
python scripts/make_figure.py --clean results/eval/clean --obs results/eval/observation \
    --recon "IHDM+SMDC"=results/eval/ihdm_smdc/recon --recon "IHDM+HQS"=results/eval/ihdm_hqs/recon \
    --out results/eval/figure.png
```

## 5. Documents
See [docs/README.md](README.md) — `methods_comparison.md` (methods/models + derivation), `MATH.md` (theory),
`TASK.md`, `EXPERIMENTS.md`.

## Attribution
- `model/ihdm_backbone/` — vendored from **AaltoML / generative-inverse-heat-dissipation** (IHDM, ICLR 2023; MIT).
- `model/dps_backbone/` — vendored from the **DPS** repo (Diffusion Posterior Sampling, ICLR 2023).
Datasets FFHQ and CelebA-HQ retain their original licenses.
