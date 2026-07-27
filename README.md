# Non-Hot Diffusion Image Restoration — a `prior × solver` framework

Linear inverse problems (heat / Gaussian deblurring) solved with a **non-hot (cold/heat) diffusion
prior** + **scale-matched data consistency**. The measurement is transformed to the current blur scale
(`Lₜ A = A Kₜ` ⇒ `Axₜ = Lₜy`), giving an exact likelihood on the reverse state — no clean-image estimate,
no backprop through the prior. Two priors and two solvers plug into **one restoration loop**; DPS (hot) is
kept as a spare baseline.

```
prior  ∈ { cold_diffusion (our x0-predictor),  ihdm (heat, 160M) }      + dps (hot, spare)
solver ∈ { smdc (1-step scale-matched gradient),  hqs (full per-step MAP) }
```
SMDC and HQS are two solvers of the **same** per-step objective — one gradient step vs. the converged
MAP (see `docs/methods_comparison.md` for the full derivation).

## Layout
```
model/       cold_diffusion.py · ihdm.py · dps.py · unet.py · base.py   + ihdm_backbone/ dps_backbone/
solver/      base.py (restoration loop) · smdc↔base · hqs.py · weighting.py · step.py · init.py · dps.py
ops/         dct.py · transforms.py · operators.py · spectral.py · heat.py (Kₜ,Lₜ) · deblur.py (A)
utils/       metrics.py · logging.py · seed.py
data/        celebahq256/ · ffhq256/ · celeba_hq_resized/ · celeba128_test/ + loaders.py
checkpoint/  cold_diffusion/{celebahq128,ffhq256}.pth · ihdm/ffhq128.pth · dps/ffhq_10m.pt
results/     dps_vs_smdc/ · smdc_vs_ihdm_hqs/ (recons, figures, logs)
docs/        MATH.md · TASK.md · methods_comparison.md · EXPERIMENTS.md · ...
scripts/     restore.py · run_dps.py · make_shared_obs.py · make_figure.py · compare_grid.py · train_prior.py · run_tests.py
tests/       gates.py (adjoint / intertwining / limits / gradient / noise-cov / plumbing)
configs/
```

## Quickstart
```bash
conda activate base
# 1. numerical gates (no model)
python scripts/run_tests.py
# 2. make a shared held-out test set (clean + heat-blur observation)
python scripts/make_shared_obs.py --source celebahq --out results/demo --n 8 --image_size 128 --blur_sigma 4 --noise 0.05
# 3. restore:  prior x solver
python scripts/restore.py --prior ihdm --solver hqs  --clean_dir results/demo/clean --observation_dir results/demo/observation
python scripts/restore.py --prior ihdm --solver smdc --clean_dir results/demo/clean --observation_dir results/demo/observation
python scripts/restore.py --prior cold_diffusion --ckpt checkpoint/cold_diffusion/ffhq256.pth --image_size 256 \
       --ch 128 --ch_mult 1 1 2 2 4 --clean_dir results/demo/clean --observation_dir results/demo/observation
# 4. DPS baseline (hot, spare)
python scripts/run_dps.py --clean_dir results/demo/clean --observation_dir results/demo/observation --save_dir results/demo/dps
```

## Headline results (held-out CelebA, heat blur σ=4, noise 0.05)
- **SMDC vs IHDM-HQS** (same IHDM prior): HQS/MAP wins — 25.4 / 0.818 / 0.182 vs 24.7 / 0.787 / 0.225 (PSNR/SSIM/LPIPS). For DCT-diagonal `A` the MAP is closed-form ⇒ same cost, strictly better; SMDC's 1-step gradient is for non-diagonalizable `A`.
- **SMDC vs DPS** (256): perception–distortion tradeoff — SMDC higher PSNR/SSIM, DPS lower LPIPS.

Full logs in `docs/EXPERIMENTS.md`; theory in `docs/MATH.md`; method/model comparison + derivation in `docs/methods_comparison.md`.
