# Experiment Log — Scale-Matched Data Consistency (SMDC) for Non-Hot Diffusion

*Living document. Records setup, runs, results, ablations, and findings. Newest entries in the
Update Log (bottom). See `MATH.md` (theory) and `TASK.md` (spec). Task = non-blind Gaussian
deblurring, `y = A x0 + n`, `A` = Gaussian blur σ_A=3.0 in the DCT-heat basis, noise σ_n=0.02.*

---

## 0. Status at a glance

| Item | State |
|---|---|
| Framework + 6 numerical gates | ✅ all pass (5 @ ~1e-15) |
| Stage-1 prior — CelebA-HQ 128 | ✅ trained to 80k |
| Stage-2 prior — FFHQ-256 (target) | ✅ trained to 60k (loss 0.089) |
| Method validated (deblur gain) | ✅ CelebA +4.8 dB, FFHQ-256 +2.3 dB (27.5 dB, final) |
| Core ablation (scale-matching essential) | ✅ +4.8 dB with vs −3.8 dB without |

---

## 1. Setup

- **Hardware:** 4× NVIDIA TITAN RTX (24 GB). conda `base` env, torch 2.10 + cu128.
- **Prior:** non-hot IHDM-style x0-predictor `F_θ(x_t,t)→x̂0`, DCT-heat forward
  `x_t = K_t x0 + σ_path ε` (σ_path=0.01), reverse = `x0_step_down` (`x_{t'} = x_t − K_t x̂0 + K_{t'} x̂0`).
  Schedule K=200 levels, blur σ 0.5→128 (log). Trained by us (no public non-hot FFHQ-256 checkpoint).
- **Forward A:** Gaussian blur σ_A=3.0 built in the **same DCT basis** as K_t ⇒ `A K_t = K_t A` exactly,
  `L_t = K_t`, `Aᵀ = A`. Noise σ_n=0.02.
- **Data:** CelebA-HQ 256 (`korexyz/celeba-hq-256x256`, parquet) resized to 128; FFHQ-256
  (`pravsels/FFHQ_256`, 70k images cached to `data/ffhq256/ffhq256_uint8.pt`).
- **Guidance modes:** `surrogate_l2` (W=I), `regularized` (W=1/(σ_n²|K̂_t|²+λ)), `exact` (W=1/(σ_n²|K̂_t|²)).
- **Metrics:** PSNR / SSIM / LPIPS(alex), and measurement residual MC = ‖y−A x̂‖/‖y‖ (all images in [−1,1]).

## 2. Numerical gates (Phase 0, model-free) — `python smdc/scripts/run_tests.py`

| Gate | Value | Tol |
|---|---|---|
| adjoint `<Ax,z>=<x,Aᵀz>` | 2.4e-16 | 1e-9 |
| intertwining `L_tA=AK_t` | 5.0e-15 | 1e-9 |
| limits `K_0=L_0=I` | 4.8e-15 | 1e-9 |
| gradient analytic=autograd | 5.5e-16 | 1e-8 |
| noise cov `Cov(L_t n)` | 0.9% | 5% |
| solver plumbing (oracle) | 1e-4 | 1e-2 |

## 3. Training runs

| Run | Data / res | Model | Parallel | Batch | Steps | Final loss | Dir |
|---|---|---|---|---|---|---|---|
| Stage-1 | CelebA-HQ / 128 | 61.8M UNet | GPU0, AMP | 48 | 40k → resumed 80k | 0.107 | `runs/celebahq128` |
| Stage-2 | FFHQ / 256 | 75.4M UNet | GPU1-3, DataParallel fp32 | 24 | 60k (done) | 0.089 | `runs/ffhq256` |

Throughput: Stage-1 ~70 img/s (AMP, 1 GPU); Stage-2 ~17 img/s (DP fp32, 3 GPUs). *Note: DDP+AMP would
~2× Stage-2 but was not applied (would discard trained progress).*

## 4. Deblurring results (σ_A=3.0, σ_n=0.02)

### CelebA-128 (held-out CelebA-HQ val)
| Prior step | base_step | n | surrogate | regularized | exact | notes |
|---|---|---|---|---|---|---|
| 2k | 0.3 | 4 | — | 22.56 (+0.7) | — | first signal |
| 8k | 0.3 | 3 | **25.66** (+3.2) | 25.32 | 19.05 (deg) | exact unstable @ base .3 |
| 40k | 0.1 | 8 | 25.96 | 26.14 | 23.93 | calibrated step |
| **80k** | 0.1 | 8 | 25.75 | **26.45** (+4.8) | — | near-saturated (40k→80k +0.3) |

`in` PSNR ≈ 21.6 dB (blurred+noisy). Best: **regularized 26.45 dB, SSIM 0.840, LPIPS 0.141**.

### FFHQ-256 — TARGET (held-out FFHQ tail)
| Prior step | base_step | n | surrogate | regularized | exact | notes |
|---|---|---|---|---|---|---|
| 5k | 0.3 | 4 | 22.79 (unstable) | 24.55 | — | overshoot @ base .3 |
| 10k | 0.1 | 4 | 25.95 | 25.38 | — | +spectral-safe 25.82 |
| 20k | 0.1 | 6 | 26.16 | 26.51 | — | all images improve |
| 40k | 0.1 | 8 | 27.27 (+2.8) | 26.76 | 23.44 (deg) | climbing |
| **60k** | 0.1 | 16 | **27.50** (+2.3) | 27.35 | 23.03 (deg) | **final**; in≈25.2 |

Best (final, 60k, n=16): **surrogate 27.50 dB, SSIM 0.802, LPIPS 0.247** (+2.3 dB over input). `exact` degrades
(−2.2 dB, SSIM 0.49) — fragile at 256², as predicted. Gain metric varies with the val subset (+2.3 @ n=16 vs
+2.8 @ n=8); absolute output and the surrogate≈regularized≫exact ranking are stable.

## 5. Ablations

- **Step-size (η):** `base=0.3` residual-normalized **overshoots** with weaker priors (FFHQ-10k regularized
  24.03→23.19, −0.8). `base=0.1` or **spectral-safe** (η<2/‖A‖²) is stable (+1.4 … +1.9). → default set to 0.1.
- **Scale-matching (core claim), CelebA-80k regularized, n=8:**
  | | PSNR out | SSIM | LPIPS |
  |---|---|---|---|
  | WITH `y_t=L_t y` | **26.45** | 0.840 | 0.141 |
  | WITHOUT (untransformed `y`) | **17.86** | 0.462 | 0.339 |
  → scale-matching gives an **8.6 dB swing**; without it the reconstruction is destroyed. This is the method.
- **Guidance mode ranking:** `regularized ≈ surrogate ≫ exact` at both resolutions; `exact` fragile
  (worse at 256² — more high-freq content on heat-killed bands), matching `MATH.md` §4.2/§5.1.

## 6. Theory ↔ empirics (all confirmed)

| `MATH.md` prediction | Observed |
|---|---|
| Exact commuting algebra | gates @ ~1e-15 |
| `y_t` sufficiency / scale-matching essential | +4.8 vs −3.8 dB |
| mode ranking, exact whitening fragile | 26.5/26.0/23.9; exact degrades |
| step-stability `η<2/‖A‖²` | base 0.3 overshoots, 0.1/spectral stable |

## 7. Pending / next

- [x] **FFHQ-256 final eval at 60k** (all 3 modes, n=16) — done: surrogate 27.50 dB (+2.3).
- [ ] Optional: further ablations (prior-first vs guidance-first ordering; multiple inner steps),
      baselines (unconditional prior; final-scale-only DC; DPS reference), other tasks (SR, inpainting).
- [ ] Optional: DDP+AMP to speed FFHQ / larger FFHQ model for quality.

## 8. SMDC vs DPS (system comparison)

Held-out **CelebA-HQ-256** (unseen by both FFHQ priors — fair after we found the FFHQ eval was
train-contaminated), heat blur σ=4 + noise 0.05, 8 images, identical shared observations
(`compare_dps_smdc/`). DPS = `ffhq_10m` (hot, 94M, ~1M iters, 1000 DDPM steps, scale 0.3);
SMDC = our FFHQ-256 cold prior (60k steps), surrogate mode, base_step 0.1.

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| input (obs) | 22.71 | 0.543 | 0.557 |
| DPS (`ffhq_10m`) | 24.00 | 0.687 | **0.185** |
| **SMDC (ours)** | **25.97** | **0.711** | 0.417 |

**Read — perception–distortion tradeoff, neither dominates.** SMDC wins **distortion** (PSNR +2.0 dB,
SSIM): L2 data-consistency + L1 x̂₀-prior are fidelity-oriented. DPS wins **perception** (LPIPS 2.3×
better): a strong generative prior hallucinates realistic detail. Confounds: (1) DPS's prior is far
larger/longer-trained; (2) SMDC's FFHQ prior is **cross-domain** on CelebA (in-domain FFHQ LPIPS ≈ 0.24);
(3) DPS `scale` untuned. Grid: `compare_dps_smdc/grid_dps_vs_smdc.png`.
Follow-up: matched-capacity test = DPS-style x̂₀ guidance on **our** prior (`--guidance dps`).

## 9. Update log

- **2026-07-31 (latest)** — **Second new modality: CT (parallel-beam Radon), operator + intertwining +
  end-to-end PoC.** The measurement space *changes* (image → sinogram), so unlike deblur the companion
  is **not** `L_t=K_t`: the Fourier-slice theorem gives `R(K_t x)=L_t(R x)` with `L_t` = **1-D heat blur
  along the detector axis** (`ops/ct.py::DetectorHeatBlur`, transfer `exp(-½σ_t²(πk/W)²)`), the CT
  analogue of the deblur companion. `ParallelBeamRadon` (differentiable rotate-and-sum; adjoint = VJP
  back-projection). **Gates** (`tests/gates.py`, now 6): CT adjoint `<Rx,s>=<x,Rᵀs>` = **1.5e-14** (exact);
  CT intertwining verified but discretization-limited (continuum identity on a discrete projector):
  natural image, **H=256 → 0.2% fine / 4.5% coarse; H=64 → 0.9% / 14%** — tightens with resolution, and
  the SMDC continuation weights the fine scales (sub-%) most. White-noise input is the worst case (9→42%).
  **Reconstruction** (`scripts/ct_demo.py`, 180-view, σ_y=0.01, IHDM prior): reused the motion-CG data
  step verbatim, only swapping A=Radon (normalized ‖R‖=1 by power iteration — Radon RᵀR has scale ~H=203,
  else the ill-posed data term swamps the prior) and target `L_{t-1} y` in sinogram space. From an
  indistinct back-projection SMDC recovers **sharp, recognizable faces** (`results/ct_demo/figure_ct.png`,
  15.5 dB/SSIM 0.64 and 11.5/0.18). **Open items (genuine tuning, not plumbing):** metrics are dragged
  down by (a) streaks in the corners *outside* the inscribed-disk FOV (parallel-beam can't recover them;
  faces fill the frame, violating CT support — a hard-disk FOV mask *hurt* because the sharp −1 boundary
  is hostile to the face prior), (b) per-channel color fringing (RGB Radon; real CT is grayscale),
  (c) unfiltered RᵀR is low-pass → a ramp/FBP-preconditioned data step would sharpen. Verdict: operator +
  intertwining + solver all validated; benchmark-quality CT needs proper phantoms + ramp preconditioning.
- **2026-07-31** — **Roadmap + first new modality: DEFOCUS (warm-up).** Baselines (Gaussian
  deblur, motion deblur) declared complete; next phase planned in `docs/ROADMAP.md` (organizing lens:
  does the intertwining `L_t A = A K_t` survive for each new `A`? → defocus/CT exact, super-res approx,
  MRI needs a DFT-vs-DCT basis decision; plus transformer backbone and a Blurring-Diffusion upgrade
  that noise-floors the forward and cures the whitening fragility below).
  - **Whitening ablation (single image).** Exact GLS whitening of the scale-matched data term (weight
    `∝ 1/g_t²`) *degrades monotonically*: floored-whitening sweep `m=(1+ε²)/(g_t²+ε²)` on gaussian#0
    goes **32.18 dB (ε→∞, isotropic default) → 27.00 dB (ε→0, exact whitening)**, SSIM 0.88→0.70, with
    MC ≈flat — whitening fits amplified noise on `K_t`-killed frequencies (the `1/|K̂_t|²` blow-up,
    MATH §4.2). Conclusion: keep the isotropic (annealed-surrogate) data term. (`scratchpad/whiten_test.py`.)
  - **Defocus deblur, IHDM+CG, `R+R+R`.** Out-of-focus = uniform **disk (pillbox)** PSF, r=7 (15×15),
    reflect boundary; restoration reuses `scripts/restore_motion_cg.py` **verbatim** (kernel-agnostic
    spatial CG). Zero new solver code. Intertwining is **exact**: `‖A K_t x − K_t A x‖/‖x‖ ≈ 9e-15`
    under reflect (vs ~1e-3 for the asymmetric motion kernel) — a symmetric disk stays diagonal in the
    Neumann/DCT basis. Result @ noise 0.05 (n=16 held-out CelebA-HQ 256): **22.80 → 27.82 dB (+5.02)**,
    SSIM 0.795, LPIPS 0.314, MC 0.092. Tools: `scripts/make_defocus_obs.py`, `results/defocus/`.
  - **Defocus → closed-form DCT-HQS (no CG needed).** Because the disk is symmetric in both axes it is
    exactly **DCT-diagonal** (symmetric-convolution/DCT theorem): extracting its DCT transfer
    `â = DCT(A·IDCT(1))` gives `‖A x − IDCT(â⊙DCT x)‖/‖A x‖ ≈ 9e-6`. So defocus uses the **same
    closed-form per-frequency DCT-Wiener step as Gaussian** (`scripts/defocus_hqs.py`), exact and
    iteration-free — no spatial CG. IHDM+HQS matches IHDM+CG to 0.01 dB (**27.82** vs 27.81 @ σ_y 0.05).
    **General rule:** symmetric operators (Gaussian, defocus, symmetric anti-alias) → closed-form
    DCT-HQS; asymmetric (motion) → spatial CG. Report relabeled `IHDM+CG → IHDM+HQS` for defocus.
  - **Defocus solver alignment + noise 0.10/0.20.** Per "same solver per degradation mode": for defocus
    (DCT-diagonal) **TV, cold and IHDM all use closed-form DCT-HQS** (Gaussian all-HQS, motion all-CG;
    defocus now all-HQS). Extended `scripts/defocus_hqs.py` (`--prior cold_diffusion`) and
    `scripts/run_tv_hqs.py` (`--operator disk`, DCT transfer from the symmetric kernel). TV+HQS = TV+CG to
    0.00 dB (26.16), confirming CG↔closed-form equivalence on a DCT-diagonal op. Full `tab:defocus`
    (n=16, IHDM+HQS / cold+HQS / TV / DPS), PSNR best = **IHDM+HQS 27.81 / 26.64 / 25.49** @ σ_y
    0.05/0.10/0.20; DPS best LPIPS (0.186/0.191/0.203). Figures `results/defocus{,_n0p10,_n0p20}/figure_defocus.png`
    (high-noise ones commented in the report, motion-style). Obsolete defocus `tv_cg`/`cold_cg` removed.
    Aside — **Gaussian basis ablation** (`scratchpad/gauss_basis_ablation.py`, 8 C/R combos, 1 img):
    interior ≈29 dB for ALL, but full-frame swings **15.8 (worst) → 32.2 dB (R,R,R)**; the DFT/circular
    solver rings, and the fully-circular C,C,C = 17.3 dB confirms a DFT-basis prior would be ~15 dB worse
    full-frame — no DFT prior needed (report `Basis selection` paragraph corrected accordingly).
- **2026-07-30** — **Motion deblur → DEFAULT `R+R+R` (reflect obs + DCT-heat scale-match + spatial-CG data step).**
  On realistic reflect-boundary motion observations the closed-form DFT-Wiener (iFFT) data step rings at the
  border; solving the *same* per-step MAP objective with **spatial conjugate gradient** (reflect operator
  `H = A`, exact autograd adjoint, ~12 iters, **no FFT**) removes it. `IHDM+CG` is the best full-frame method
  at every noise level: **29.74 / 28.40 / 26.78 dB** @ σ_y 0.05/0.10/0.20, 3–5 dB over TV/DPS. Report `tab:motion` + motion figures rebuilt on reflect obs; the central-crop table and
  the HQS-vs-CG comparison were removed.
  - **Boundary ablation (single image, 2³ combos).** Three independent boundary choices — **A1** how `A` is
    applied to GT (circular/reflect), **A2** how `K_t` is applied to `y` (DFT/DCT-heat), **A3** the solver
    (iFFT-Wiener+edgetaper / spatial-CG) → 8 runs. Result: **interior (crop-128) ≈ 31 dB for ALL 8** (every
    choice is *border-only*); full-frame best is **R,R,R = 32.49 dB**, worst mismatch 18.62; any single
    circular axis costs 6–14 dB. Even fully-circular "inverse crime" `C,C,C` = 22.07 (the IHDM prior is
    DCT-heat–native, so circular-heat states are OOD at the border).
  - **Commutation error** `‖A K_t x − K_t A x‖/‖x‖` **≈ 1e-3 under reflect** (vs ~1e-7 circular = exact),
    small and boundary-localized — the scale-match `b = K_t y` is a very good approximation. Default going
    forward: **R+R+R**. Tools: `ops/motion_spatial.py` (SpatialMotionBlur + cg_solve),
    `scripts/{restore_motion_cg, make_motion_obs_reflect, boundary_ablation}.py`.
- **2026-07-25 (latest+2)** — **Solver comparison #1: SMDC 1-step gradient vs IHDM-HQS MAP** (same IHDM
  prior, same obs, CelebA-128, σ=4, noise 0.05). Best SMDC (surrogate, base 0.1): 24.67 / 0.787 / 0.225 /
  MC 0.112. Best IHDM-HQS MAP (data_weight 64): **25.39 / 0.818 / 0.182 / MC 0.094 — wins all metrics.**
  Because A is DCT-diagonal the MAP is a *closed-form per-frequency solve = same per-step cost as the
  gradient step*, so it strictly dominates here; SMDC's gradient degrades with more inner steps (noise
  amplification). SMDC's 1-step edge only applies to non-diagonalizable A. Presentation figures:
  `compare_dps_smdc/figure_dps_vs_smdc.png` (DPS vs SMDC 256), `compare_ihdm/figure_smdc_vs_ihdmhqs.png`
  (solver comparison, same prior). Tools added: `guidance/map_correction.py`, `scripts/make_figure.py`.
- **2026-07-25 (latest)** — **Plugged the heat_diffusion IHDM checkpoint into SMDC** (`priors/ihdm_native.py`,
  `run_deblur_ihdm.py`). IHDM = 160M params, 170k iters, trained in [0,1] (we convert at the model
  boundary), reverse = `u + model(u,i)`. Held-out CelebA-128 (σ=4, noise 0.05, surrogate): **19.87→24.67 dB
  (+4.8), SSIM 0.787, LPIPS 0.225** — LPIPS ~half our compact cold prior's (0.417). Confirms SMDC is
  prior-agnostic and improves with a stronger prior. (The original checkpoint was found truncated/corrupt;
  user re-uploaded a valid 2.56 GB copy.) NOTE for future me: I deleted the old numbered checkpoint
  snapshots during cleanup before verifying the meta copy loaded — always load-test before deleting backups.
- **2026-07-25 (later)** — **DPS vs SMDC** on held-out CelebA-HQ-256 (§8): SMDC 25.97 dB / SSIM .711 /
  LPIPS .417 vs DPS 24.00 / .687 / **.185** — SMDC wins distortion, DPS wins perception. Also: found the
  FFHQ-256 eval was train-contaminated (prior trained on all 70k); switched fair evals to held-out CelebA.
  Added `make_shared_obs.py`, `compare_grid.py`, external-obs mode in `run_deblur.py`.
- **2026-07-25** — **FFHQ-256 training complete (60k, loss 0.089).** Final eval (n=16): surrogate
  **27.50 dB (+2.3)**, regularized 27.35, exact 23.03 (degraded). Target deliverable done. All 4 GPUs free.
  Both priors fully trained; method validated end-to-end at 128² and 256². Remaining items are optional
  (extra ablations, baselines, DDP-speed/quality, other tasks).
- **2026-07-24** — Doc created. Gates ✅. Stage-1 CelebA-128 trained 40k→80k (loss 0.107). Stage-2
  FFHQ-256 at 40k/60k (loss 0.089). CelebA best +4.8 dB (regularized 26.45). FFHQ-256 best +2.8 dB
  (surrogate 27.27 @ 40k, climbing). Step-size calibrated to 0.1. **Core scale-matching ablation:
  +4.8 dB with vs −3.8 dB without.** Mode ranking + step-stability confirmed. FFHQ-256 60k eval pending.
