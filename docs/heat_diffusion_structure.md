# heat_diffusion — Repository Structure & Logic

This repo = the **Inverse Heat Dissipation (IHDM)** codebase (forked from AaltoML) **+**
scale-matched posterior-sampling inverse solvers **+** DPS/TV baselines **+** a two-stage
(noise→heat) generative-prior study. Below: what each part does, grouped by role.

## 1. Method docs (read these first)
| File | What it describes |
|---|---|
| `file.md` | **Core method** — *Scale-Matched Bayesian Posterior Sampling*: define the likelihood on the heat latent via `A xₜ = G̃ₜ y`. (Same idea as the SMDC project.) |
| `hqs_posterior_sampling_summary.md` | **IHDM-HQS** algorithm: per-step MAP correction `min ‖x−μθ‖²/δ² + ‖Ax−G̃y‖²/σ_y²` solved by Half-Quadratic Splitting. Includes tuned settings + CelebA-128 results vs TV-HQS. |
| `two_stage_frequency_controlled_diffusion.md` | **Generative-prior design** — two-stage forward: VP noise → fixed-noise-floor heat; frequency-SNR-controlled coarse-to-fine. |
| `TRAINING_COMMANDS.md` | Train commands for the two-stage variants (constant_noise / light_prefix / slow_decay / bridge / default). |
| `MAP_heat_diffusion/README.md`, `MAP_heat_diffusion_128_repro/SUMMARY.md` | MAP-method write-ups + reproduction notes. |

> Note: `MAP_heat_diffusion/summary.md` is a **duplicate** of `hqs_posterior_sampling_summary.md`.

## 2. Generative IHDM infrastructure (from the fork)
- `train.py` — train IHDM / two-stage priors from a config.
- `sample.py` — unconditional sampling.
- `evaluate.py` — FID / NLL(ELBO) evaluation.
- `model_code/` — `unet.py`, `torch_dct.py` (DCT heat ops), `utils.py`, `nn.py`, `ema.py`.
- `scripts/` — `losses.py`, `sampling.py`, `datasets.py`, `utils.py`, `cleanfid_alternatives.py`.
- `configs/` — per-dataset configs (`mnist`, `cifar10`, `ffhq`, `afhq`, `lsun_church`), incl. two-stage variants.

## 3. Inverse-problem solvers (the contribution)
- `sample_posterior_heat.py` — **the scale-matched MAP/HQS posterior sampler** (main method; `--workdir --batch_size` + internal flags for δ, σ_y, map_prior_weight, map_data_weight, data_weight_schedule, hqs_iters/rho, solver).
- `MAP_heat_diffusion/` — scripts + results + figures for the MAP method.
- `MAP_heat_diffusion_128_repro/` — 128-res reproduction (configs, grids, gifs, results, SUMMARY).

## 4. Baselines
- `run_dps_precomputed_heat.py` — **DPS** on precomputed heat observations (uses `diffusion-posterior-sampling/models/ffhq_10m.pt`; same DCT `HeatBlurOperator` as SMDC → fair). Args: `--model_config --diffusion_config --clean_dir --observation_dir --save_dir --degradation_sigma --scale --num_images`.
- `run_tv_hqs_from_observation.py` — **TV-HQS** (total variation + half-quadratic splitting).
- `run_tv_pg_from_observation.py` — **TV-PG** (total variation + proximal gradient).
- `diffusion-posterior-sampling/` — the DPS repo (`guided_diffusion/`, `configs/`, `models/ffhq_10m.pt`).

## 5. Shared evaluation harness
- `make_common_heat_observations.py` — build a **common** degraded test set: `--input_dir --output_dir --num_images --image_size --degradation_sigma --noise_sigma --seed`. Feeds all methods identical `clean_dir` + `observation_dir`.
- `make_hqs_metric_grid.py` — assemble side-by-side PSNR/SSIM comparison grids.

## 6. Trained models
- `runs/ffhq/default_128/checkpoints*/` — **trained IHDM FFHQ-128 prior** (used by the posterior sampler). `checkpoints-meta/checkpoint.pth` = latest; `checkpoints/checkpoint_{1,2}.pth` = older snapshots (~2.4 GB each).
- `diffusion-posterior-sampling/models/ffhq_10m.pt` — **DPS FFHQ model** (357 MB).
- `runs/mnist/two_stage_*` — two-stage MNIST priors (exploratory, ~0.6 GB each).

## 7. Data (`data/`, ~43 GB)
- `ffhq-dataset/` (~39 GB) — FFHQ images (IHDM training data).
- `celeba-hq-resized/` (3.8 GB parquet) — CelebA-HQ.
- `root_celebA_128_test_new/` (177 MB) — **CelebA-128 test set** (used for inverse-problem eval).
- `MNIST/` (64 MB).

## 8. Results / outputs
- `tmp/` (~273 MB) — comparison grids + summaries (`celeba128_noise_sweep_grids/`, `common_celeba128_heat_noise01/`, `fair_common_celeba256_heat_noise01/`). Some `.md` summaries here duplicate `MAP_heat_diffusion_128_repro/results/`.
- `runs/ffhq/two_stage_*_demo/*.gif` — large demo animations (~70–130 MB each).

## 9. End-to-end comparison flow
```
make_common_heat_observations.py   ->  clean_dir/ + observation_dir/  (shared inputs)
        │                                    │                 │
        ▼                                    ▼                 ▼
sample_posterior_heat.py (IHDM-HQS)   run_dps_precomputed_heat.py   run_tv_*_from_observation.py
        └──────────────► make_hqs_metric_grid.py  ◄──────────────┘   (PSNR/SSIM/LPIPS grid)
```
