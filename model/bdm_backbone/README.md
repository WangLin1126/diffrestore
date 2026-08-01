# Blurring Diffusion Models (BDM) backbone

Implementation of **Blurring Diffusion Models** (Hoogeboom & Salimans, ICLR 2023, `paper.pdf`):
a Gaussian diffusion defined in the **DCT frequency space** with a *per-frequency* signal
schedule `α_t = a_t · d_t` (scalar VP-cosine noise `a_t` × heat-dissipation blur `d_t`) and a
*scalar* noise level `σ_t`. This unifies DDPM (`σ_blur_max = 0`) and inverse heat dissipation.

## Layout

| File | Role |
|---|---|
| `schedule.py` | **Reparameterization (schedules).** `BlurringSchedule.get_alpha_sigma(t)` → per-frequency `α` (B,1,H,W) and scalar `σ` (B,1,1,1), from the blur schedule `d_t` (Eq. 25-26) and VP cosine noise `a_t,σ_t`. |
| `diffusion.py` | **Reparameterization (process).** `diffuse` (forward `q(z_t\|x)`), `loss` (ε-MSE, Eq. 24), `denoise` (one reverse step, Eq. 18/23), `sample` (ancestral, Alg. 1). All coefficients diagonal in DCT → DCT · per-freq multiply · IDCT (reuses `ops.dct`). |
| `model.py` | **Model structure.** `BDMDenoiser` is the pixel-space ε-predictor `f_θ(z_t,t)` (continuous `t∈[0,1]` rescaled for the timestep embedding). Two backbones: `ihdm` = the IHDM NCSN++ net (~211M @256px, same structure/params as `model/ihdm_backbone`), `unet` = the compact repo UNet. `PRESETS` = Table 5 archs + `ihdm256`. |
| `train.py` | Training loop reusing IHDM's tricks/settings: linear LR warmup, EMA with num-updates ramp (decay 0.9999), grad-clip 1.0, resolution-scaled blur, DataParallel. ε-MSE loss; checkpoints (raw + EMA weights) + periodic samples. |
| `sample.py` | Inference: load a checkpoint and draw samples (uses the EMA weights). |
| `sanity_check.py` | Correctness gates (VP, `t=0` boundary, **oracle reverse-chain reconstruction, gate > 45 dB — measured ~56**, model/loss/sample smoke). |

## Usage

```bash
# correctness checks
python -m model.bdm_backbone.sanity_check

# plumbing test (random data, 5 steps)
python -m model.bdm_backbone.train --smoke

# train FFHQ-256 with the IHDM-matched 211M net + IHDM settings (80GB or multi-GPU)
python -m model.bdm_backbone.train --data_pt data/ffhq256/ffhq256_uint8.pt \
    --preset ihdm256 --steps 200000 --batch 32 --gpus 0 1 2 3

# sample from a trained checkpoint (uses EMA weights)
python -m model.bdm_backbone.sample --ckpt checkpoint/bdm/bdm.pth --n 16 --steps 250
```

`σ_blur_max` controls the max blur; the default scales with resolution (128 @256px, matching
IHDM's `blur_sigma_max`; the paper uses 20 for CIFAR/LSUN). `σ_blur_max=0` recovers plain DDPM.
The ε-predictor operates in **pixel space** and images are normalized to `[-1,1]`.
