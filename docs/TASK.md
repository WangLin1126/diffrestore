# Implementation Specification & Task Plan
## Scale-Matched Data Consistency for Non-Hot Diffusion Inverse Problems — FFHQ-256

*Companion to `MATH.md` (all derivations and assumptions live there). This document is the
implementation contract: problem, method, concrete instantiation, prior sourcing, architecture,
a short task list, and acceptance gates. It replaces the sprawling original `task.md`.*

---

## 1. Problem, in one paragraph

We solve non-blind linear image inverse problems $\mathbf y=\mathbf A\mathbf x_0+\mathbf n$ using a
**non-hot (cold) diffusion prior** whose forward path is a *deterministic, known* image degradation
$\mathbf x_t=\mathbf K_t\mathbf x_0$ (Gaussian heat blur), instead of the usual noisy hot-diffusion path.
The one idea: because $\mathbf K_t$ is known and linear, we can transform the measurement to the *same
degradation scale* as the current state via a companion operator $\mathbf L_t$ ($\mathbf L_t\mathbf A=\mathbf A\mathbf K_t$),
giving $\mathbf y_t=\mathbf L_t\mathbf y=\mathbf A\mathbf x_t+\mathbf n_t$ — an **exact linear-Gaussian
likelihood directly on $\mathbf x_t$**. This removes DPS's clean-image estimate and denoiser
backpropagation. We alternate a pretrained reverse-degradation step with this data-consistency
correction (Eq. 18 of `MATH.md`), coarse-to-fine.

**First deployment target:** non-blind **Gaussian deblurring on FFHQ-256**, where $\mathbf A$ and
$\mathbf K_t$ are made to **commute exactly** (§3), so every algebraic assumption is verifiable to
machine precision.

---

## 2. Method summary (implementable form)

```
Inputs:  y, A (+ Aᵀ), {K_t}, {L_t}, prior R_θ, schedule T=t_N>…>t_0=0, mode, η-schedule
x ← terminal_init(y)                                  # x_T = L_T y  (matched-measurement, MATH §8)
for (t, t') in zip(times[:-1], times[1:]):            # t' < t, marching to 0
    x̃  ← R_θ(x, t, t')                                # 1) learned reverse-degradation step
    y'  ← L_{t'} y                                    # 2) scale-matched measurement
    for _ in range(inner_steps):                      # 3) data-consistency correction(s)
        r  ← y' − A x̃
        x̃ ← x̃ + η(t') · Aᵀ · W_{t'}(r)               #    W: exact | regularized | surrogate(=I)
    x ← clamp(x̃, −1, 1).detach()                      # no grad through R_θ by default
return x
```

Three guidance modes (weights are diagonal in the shared spectral basis; `MATH.md` §5):

| mode | $\hat W_t[k]$ | use |
|---|---|---|
| `surrogate_l2` | $1$ | default; robust coarse-to-fine (frequency-annealed likelihood) |
| `regularized` | $1/(\sigma_n^2\lvert\hat K_t[k]\rvert^2+\lambda_t)$ | principled + stable; recommended for quality |
| `exact` | $1/(\sigma_n^2\lvert\hat K_t[k]\rvert^2)$, support-thresholded | statistically exact transformed likelihood |

---

## 3. Concrete FFHQ-256 instantiation (exact-commuting deblur)

- **Prior $\mathbf R_\theta$:** IHDM (Inverse Heat Dissipation Model) FFHQ config — `image_size=256`,
  `K=200`, blur schedule $\sigma_B=\exp(\text{linspace}(\log0.5,\log128,K))$ (prepend $0$),
  additive `sigma=0.01`, NCSN++-style U-Net (`model_channels=128`, `channel_mult=(1,2,3,4,5)`,
  `num_res_blocks=3`, `attn_levels=(2,3,4)`). Degradation $\mathbf K_t$ = **Neumann/DCT-II heat**, Eq. (5).
- **Forward $\mathbf A$ (Gaussian deblur):** Gaussian blur std $\sigma_A$ (e.g. $2.0$ or $4.0$ px)
  realized in the **same DCT basis**, Eq. (6): $\hat A[k]=\exp(-\tfrac{\sigma_A^2}{2}\lambda_k)$,
  so $\mathbf A^\top=\mathbf A$ and $\mathbf A\mathbf K_t=\mathbf K_t\mathbf A$ **exactly**.
- **Companion:** $\mathbf L_t=\mathbf K_t$ (Lemma 3.1). **Noise:** $\boldsymbol\Sigma_n=\sigma_n^2\mathbf I$, $\sigma_n=0.01$–$0.05$.
- **Init:** $\mathbf x_T=\mathbf L_T\mathbf y$. **Range:** images in $[-1,1]$, RGB, matching IHDM.

This is the setting where the adjoint, intertwining, and limiting tests all pass to ~$10^{-6}$.
(A circular/FFT variant is also provided for problems posed with periodic boundary — `MATH.md` §3.2(F).)

---

## 4. Prior sourcing — the one real blocker

**Verified fact:** there is **no publicly downloadable non-hot / cold-diffusion prior for FFHQ-256.**
IHDM ships configs (FFHQ default *is* 256) but **no weights** and needs ~1.3M iters; its project page
shows FFHQ/AFHQ-256 samples but hosts nothing. HuggingFace has only *hot* `ddpm-ffhq-256`. The **only
downloadable non-hot face prior** is Cold Diffusion's blur model. We therefore use a **two-phase**
roadmap: a ready stand-in prior first, then the FFHQ-256 target.

**Ready stand-in — Cold Diffusion CelebA-128 (blur)** — [Google Drive](https://drive.google.com/drive/folders/1R7CKUrkiIDsDYh2__Yi1iLvRR6wNxVFF):
deterministic Gaussian-blur cold diffusion, `image_size=128`, `blur_size=27`, `blur_std=0.01`,
`Exponential` (reflect) blur routine, `time_steps=300`, `x0_step_down` sampling, lucidrains DDPM U-Net.
Its path is **fully deterministic** ($\boldsymbol\xi_t=0$), so `MATH.md` §7 is not even needed — a
cleaner fit than IHDM. Boundary = reflect ⇒ pair it with a reflect/DCT Gaussian `A` (they commute up
to a small, measured boundary term, `MATH.md` §3.3/B4).

| Phase | Prior | Data / res | Honors "FFHQ-256"? | Time to first result |
|---|---|---|---|---|
| **2a (first)** | **Cold Diffusion CelebA-128 (blur)** — download | CelebA · 128 | ⚠️ stand-in | **hours** (download + wire) |
| **2b (target)** | **compact IHDM trained on FFHQ-256** | FFHQ · 256 | ✅ exact | ~1–2 GPU-days on the 4×TITAN RTX |

Alternative for 2b if training is undesirable: **FFHQ-128** (IHDM's released 128 config) is cheaper to
train than 256. **Framework correctness does not depend on prior quality** — all numerical gates (§8)
pass with an untrained model — so Phase 0 proceeds regardless, and the prior only affects the
image-quality run (Phase 2).

---

## 5. Code architecture

Package `smdc/` (scale-matched data consistency); maps 1:1 to the original note's §5.4 modules.

```
smdc/
  operators/
    base.py         # LinearOperator: forward(), adjoint(), adjoint_test()
    spectral.py     # SpectralOperator (DCT- or FFT-diagonal): GaussianBlur A, and heat K_t/L_t
  scales/
    heat.py         # HeatSchedule: IHDM blur σ_B(t) schedule; K_t, L_t transfer fns; Σ_t
  priors/
    base.py         # ReversePrior protocol: reverse_step(x, t, t', rng)
    ihdm.py         # wraps pretrained IHDM U-Net + one official reverse step  (Interface A/B/C)
    identity.py     # trivial prior (K_{t'} K_t⁻¹) for testing the loop with NO trained model
  guidance/
    weighting.py    # W_t: exact | regularized | surrogate  (all diagonal in Φ)
    step.py         # η schedules: fixed | spectral-safe (Eq.15) | residual-normalized
  init/terminal.py  # x_T: matched-measurement (default) | prior-sample | coarse-LS
  solvers/scale_matched.py   # the splitting loop (§2); per-step logging
  tasks/deblur.py   # build Gaussian-blur A in the shared basis
  data/ffhq.py      # load/normalize FFHQ-256 val images to [-1,1]
  metrics/quality.py# PSNR, SSIM, LPIPS; measurement- & state-consistency
  utils/{dct.py, fft.py, logging.py, config.py, seed.py}
  configs/deblur_ffhq256.yaml
  tests/            # test_adjoint, test_intertwining, test_limits, test_gradient,
                    #   test_noise_cov, test_state_semantics
  scripts/{fetch_ihdm.py, fetch_ffhq256.py, train_prior.py, run_deblur.py, run_tests.py}
```

Design rules: (1) operators expose only `forward`/`adjoint`; the solver never sees FFT/DCT internals.
(2) prior hidden behind `reverse_step`; **no autograd through it** by default. (3) every mode/schedule is
config-selected; runs are seeded and reproducible. (4) fail loudly on violated assumptions (bad dims,
missing adjoint, boundary mismatch).

---

## 6. Task list (short, phased)

**Phase 0 — framework + verification (no trained model required)**
1. `utils/dct.py`, `utils/fft.py`: orthonormal 2-D DCT-II/IDCT and rFFT wrappers (+ round-trip tests).
2. `operators/` + `tasks/deblur.py`: `SpectralOperator`, Gaussian-blur `A`, heat `K_t/L_t`; `adjoint_test`.
3. `scales/heat.py`: IHDM-exact blur schedule and transfer functions; `Σ_t`, `Σ_t^eff` (Eq. 17).
4. `guidance/` + `init/` + `solvers/scale_matched.py`: three `W_t` modes, η schedules, terminal init, splitting loop, logging.
5. `priors/identity.py` + `tests/`: run the full loop with the analytic identity prior; pass **all six** numerical gates (§8).

**Phase 1 — stand-in prior (Cold Diffusion CelebA-128) + first real run**
6. `scripts/fetch_colddiff.py` + `data/faces.py`: download CelebA-128 blur weights (gdown) + a small CelebA-128 val set (≥100 imgs); normalize to $[-1,1]$.
7. `priors/colddiff.py`: port the lucidrains DDPM U-Net + Cold-Diffusion blur schedule; wire `reverse_step` (`x0_step_down`); point `scales/heat.py` at the **same** blur so $\mathbf K_t=\mathbf L_t$ matches the prior's own forward path.
8. `scripts/run_deblur.py` + `configs/deblur_celeba128.yaml`: end-to-end non-blind Gaussian deblur on CelebA-128; per-step logging; save reconstructions. Sweep modes and $\sigma_n$; report PSNR/SSIM/LPIPS + measurement/state consistency.

**Phase 2 — FFHQ-256 target (train the prior)**
9. `scripts/fetch_ffhq256.py` + `priors/ihdm.py`: FFHQ-256 val set; port IHDM `model_code` + one-step sampler; verify its DCT-blur convention matches `scales/heat.py` **exactly** (state-semantics gate).
10. `scripts/train_prior.py`: train compact IHDM on FFHQ-256 (optionally FFHQ-128 first); rerun the *same* solver at 256 via `configs/deblur_ffhq256.yaml` — only the `ReversePrior` and schedule change.

**Phase 3 — evidence (optional, on request)**
11. Ablations: prior-first vs guidance-first; inner_steps 1 vs k; $\mathbf y_t$ vs untransformed $\mathbf y$; correct vs mismatched $\mathbf L_t$.
12. Baselines: unconditional prior; final-scale-only data consistency; (if time) a DPS reference.

*Gate:* Phase 0 must be green before any image-quality run. Phase 1 (stand-in) gives the first real
reconstructions; Phase 2 reaches the FFHQ-256 target with the identical solver.

---

## 7. Configuration schema (single source of truth)

```yaml
problem:      { type: gaussian_deblur, noise_std: 0.02 }
operator:     { basis: dct, blur_sigma: 2.0 }            # A = Gaussian blur in shared DCT basis
scale_process:{ type: dct_heat, K: 200, blur_sigma_min: 0.5, blur_sigma_max: 128.0 }
prior:        { kind: ihdm, checkpoint: runs/ffhq/.../checkpoint.pth,
                image_range: [-1.0, 1.0], sigma_path: 0.01 }
guidance:     { mode: regularized, inner_steps: 1, step: residual_normalized,
                base_step: 0.1, covariance_regularizer: 1.0e-3 }   # 0.1 stable; 0.3 overshoots (MATH sec.6)
init:         { mode: matched_measurement }             # x_T = L_T y
runtime:      { seed: 1234, device: cuda, precision: float32, subset: 100 }
```

---

## 8. Numerical gates (must pass before image-quality claims)

For random $\mathbf x,\mathbf z$ and tolerance $\varepsilon\approx10^{-5}$:

1. **Adjoint:** $\dfrac{|\langle\mathbf A\mathbf x,\mathbf z\rangle-\langle\mathbf x,\mathbf A^\top\mathbf z\rangle|}{\max(1,|\langle\mathbf A\mathbf x,\mathbf z\rangle|)}<\varepsilon$.
2. **Intertwining:** $\dfrac{\lVert\mathbf L_t\mathbf A\mathbf x-\mathbf A\mathbf K_t\mathbf x\rVert}{\max(1,\lVert\mathbf A\mathbf K_t\mathbf x\rVert)}<\varepsilon\ \forall t$.
3. **Limiting scale:** $\mathbf K_0\approx\mathbf I,\ \mathbf L_0\approx\mathbf I,\ \mathbf y_0\approx\mathbf y$.
4. **Gradient:** analytic $\mathbf A^\top\mathbf W_t\mathbf r_t$ matches autograd of the selected data term.
5. **Noise covariance:** empirical $\operatorname{Cov}(\mathbf L_t\mathbf n)$ (and $\tilde{\mathbf n}_t$, Eq. 17) match the implemented $\boldsymbol\Sigma_t$.
6. **State semantics:** on held-out images, $\mathbf R_\theta$'s level-$t'$ output $\approx\mathbf K_{t'}\mathbf x_0$ (Phase 1).

## 9. Acceptance criteria (deployment complete when…)

Solver runs end-to-end for FFHQ-256 Gaussian deblur; gates 1–5 pass (6 after Phase 1); the three
modes are implemented and labeled; **no autograd through the prior** in the default path; every run logs
per step $\{t,\ \lVert r\rVert,\ \lVert\text{correction}\rVert,\ \lVert\text{prior step}\rVert,\ \text{state range},\ \text{state-consistency}\}$; the
**final** estimate is scored under the *original* model $\mathbf y=\mathbf A\mathbf x_0+\mathbf n$ (not only transformed objectives);
results reproduce from `config + seed`; assumption violations raise clear errors.

## 10. What I need from you

1. **Sign-off (or edits) on `MATH.md` + this doc** — you asked to double-check before I build.
2. **Confirm the two-phase prior roadmap** (§4): Cold Diffusion **CelebA-128** stand-in first, then a
   trained **IHDM FFHQ-256** for the target. (If you'd rather skip the stand-in and go straight to
   training FFHQ-256, or start at FFHQ-128, say so.)

Once you sign off, I implement **Phase 0** (framework + the six numerical gates — no model needed),
then **Phase 1** (wire the CelebA-128 prior and produce the first real deblurring results).
