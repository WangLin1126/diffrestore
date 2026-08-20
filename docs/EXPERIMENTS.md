# Experiment Log — Scale-Matched Data Consistency (SMDC) for Non-Hot Diffusion

*Living document: setup, headline results, ablations, findings. §0–8 are the standing reference;
§9 is a compact chronological log (newest first). Theory in `MATH.md`, spec in `TASK.md`, next
phase in `ROADMAP.md`.*

---

## 0. Status at a glance

| Item | State |
|---|---|
| Framework + numerical gates | ✅ 8 gates pass (deblur/CT/SR; 6 @ ~1e-15) |
| Priors (CelebA-HQ 128 / FFHQ-256) | ✅ 80k / 60k (loss 0.089) |
| Modalities | ✅ Gaussian · motion · defocus · **super-res**; ◐ CT (PoC, tuning open) |
| Modern baselines (A1-3) | ✅ DPS · DDRM · DiffPIR (all deblur + SR) · DDNM (box-SR); ImageNet/1k pending |
| Core ablation (scale-matching essential) | ✅ +4.8 dB with vs −3.8 dB without (8.6 dB swing) |

## 1. Setup

- **Hardware/env:** 4× TITAN RTX (24 GB), conda `base`, torch 2.10 + cu128.
- **Prior:** non-hot IHDM x0-predictor, DCT-heat forward `x_t = K_t x0 + σ_path ε` (σ_path=0.01),
  K=200 levels, blur σ 0.5→128 (log). Reverse = `u + model(u,i)` (mean, deterministic).
- **Forward A:** built in the **same DCT basis** as K_t ⇒ `A K_t = K_t A`, `L_t=K_t`, `Aᵀ=A` for
  symmetric convolutions (Gaussian/defocus). Non-symmetric/other-space A get their own companion (below).
- **Data:** CelebA-HQ 256 (resized 128) + FFHQ-256 (`data/ffhq256/…`). Held-out evals only.
- **Metrics:** PSNR / SSIM / LPIPS(alex), MC = ‖y−A x̂‖/‖y‖, all in [−1,1].

## 2. Numerical gates (model-free) — `python scripts/run_tests.py`

| Gate | Value (tol 1e-9 unless noted) | | Gate | Value |
|---|---|---|---|---|
| adjoint `<Ax,z>=<x,Aᵀz>` | 2.4e-16 | | CT adjoint | 1.5e-14 |
| intertwining `L_tA=AK_t` | 5.0e-15 | | CT intertwining (fine) | 4.3e-2 (tol 6e-2) |
| limits `K_0=L_0=I` | 4.8e-15 | | SR adjoint | 2.2e-15 |
| noise cov `Cov(L_t n)` | 0.9% (tol 5%) | | SR intertwining | 2.5e-4 (tol 1e-3) |

## 3. Priors (training)

| Run | Data / res | Model | Steps | Loss | Throughput |
|---|---|---|---|---|---|
| Stage-1 | CelebA-HQ 128 | 61.8M UNet | 80k | 0.107 | ~70 img/s (AMP, 1 GPU) |
| Stage-2 | FFHQ 256 (target) | 75.4M UNet | 60k | 0.089 | ~17 img/s (DP fp32, 3 GPU) |

## 4. Deblur results (headline)

- **CelebA-128** (80k, regularized, n=8): **26.45 dB / SSIM 0.840 / LPIPS 0.141** (in ≈21.6, +4.8).
- **FFHQ-256** (60k, surrogate, n=16): **27.50 dB / 0.802 / 0.247** (in ≈25.2, +2.3). `exact` degrades
  (−2.2 dB) — fragile at 256². Ranking `regularized ≈ surrogate ≫ exact` stable at both resolutions.

## 5. Ablations

- **Scale-matching (the method):** WITH `y_t=L_t y` 26.45 dB vs WITHOUT (untransformed y) 17.86 dB —
  **8.6 dB swing** (CelebA-80k regularized, n=8).
- **Step-size:** `base=0.3` residual-norm overshoots weak priors; `base=0.1` / spectral-safe
  (η<2/‖A‖²) stable → default 0.1.
- **Whitening:** exact GLS whitening (`∝1/g_t²`) degrades monotonically (32.18→27.00 dB as ε→0) — it
  amplifies noise on K_t-killed bands (MATH §4.2). Keep the isotropic data term. → motivates BDM (§9).

## 6. Theory ↔ empirics

Exact commuting algebra (gates ~1e-15); scale-matching essential (+4.8 vs −3.8); mode ranking with
`exact` fragile; step-stability `η<2/‖A‖²` — all confirmed.

## 7. SMDC vs DPS (deblur, held-out CelebA-HQ-256, σ=4 + noise 0.05, n=8)

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| input | 22.71 | 0.543 | 0.557 |
| DPS (`ffhq_10m`) | 24.00 | 0.687 | **0.185** |
| **SMDC (ours)** | **25.97** | **0.711** | 0.417 |

Perception–distortion split, neither dominates: SMDC wins distortion (fidelity-oriented), DPS wins
perception (generative prior). Confounds: DPS prior larger; SMDC FFHQ prior is cross-domain on CelebA
(in-domain LPIPS ≈0.24); DPS `scale` untuned.

---

## 9. Chronological log (newest first)

- **2026-08-20 — DPS added to SR table + inpainting baselines (completes the CelebA method matrix).**
  - **DPS-SR** (4 cells, n=200): added `--abs_noise` to `sr.py demo` so DPS's obs matches the SMDC row
    (validated 28.54 vs freqreg 28.59); generated matched obs (no-CG), ran DPS via `sr.py baselines
    --methods dps` (scale 0.3). Results (P/S/L): box 24.47/.688/.194, 24.09/.675/.205; aa 22.63/.626/.213,
    22.22/.613/.223 — **dominated everywhere** (below bicubic; DPS's posterior-gradient without a
    range/null split struggles on aggressive x4). Added to `tab:sr` + caption.
  - **Inpainting head-to-head** (fixed 50% mask, n=16, composite scoring = observed pinned): minted one
    shared mask (`results/inpaint_mask50.pt` + ddrm `.npy` + ddnm `.npy` + diffpir mask.png), ran ALL 6
    on it. IHDM reproduces the old table (32.96) → mask equivalent. Wiring per backbone: DPS/PiGDM via new
    `InpaintOperatorDPS` + `--operator inpaint --mask_file`; DDRM `DDRM_MASK_NPY`; DDNM `--deg inpainting
    --simplified` (exp/inp_masks/mask.npy); DiffPIR `main_ddpir_inpainting` env overrides + `load_mask`.
    **Results (full/hole/SSIM/LPIPS):** DDNM **33.03/30.00**/.937/.032, IHDM 32.96/29.93/**.941**/.028,
    DiffPIR 32.96/29.94/.933/**.024**, PiGDM 32.29/29.26/.930/.026, DDRM 30.75/27.72/.903/.066, DPS
    30.57/27.54/.895/.066. **Honest reframe:** top cluster (IHDM/DDNM/DiffPIR/PiGDM ~33/30) — parity, not
    a win; DDRM/DPS trail ~2.4 dB. Table+caption+prose updated (was "IHDM beats DDRM").
  - **Gotchas:** PiGDM inpaint collapsed at scale 1.0 (18 dB) like motion → re-tuned to **scale 0.02,
    sig 0.5** (32.29). DDNM inpaint recon index is **+1 offset** from orig (time-travel naming) — score
    recon `i` vs `orig_{i+1}`. Report 23 pp, 0 undefined. [[experiments-framework]] [[smdc-project]]


- **2026-08-19 — ΠGDM added (9 deblur cells) + DPS/ΠGDM motion/defocus re-run under a reflect operator;
  DDNM-Gaussian marked; boundary appendix.** Completes the A1-3 method matrix (DPS·ΠGDM·DDRM·DDNM·DiffPIR).
  - **Feasibility** (report App.~A, `tab:boundary`): SVD-free methods (DPS, ΠGDM, DiffPIR) run on every
    operator; DDRM/DDNM need a tractable SVD → separable-only (no motion/defocus; DDNM also no aa-SR).
  - **ΠGDM**: implemented via the DPS backbone's `pigdm` conditioning (`get_conditioning_method`, needs
    only `A`,`Aᵀ` via autograd) through `run_dps.py` (added `--method pigdm --pigdm_sigma`). n=100 (1000-NFE
    sampler, like DPS). **Operating point is delicate**: SR's scale=1.0 *diverges* on deblur (full-size
    residual over-guides → 7 dB noise). gaussian/defocus (DCT-diagonal, AAᵀ≈I holds) stable at **scale=0.1**,
    σ noise-scaled 0.3/0.4/0.5. **Motion (deep notches) needed re-tuning**: 0.1/noise-σ collapsed at high
    noise (s20→17.05, std 0.6 = uniform garbage; isolated to ΠGDM since DPS-reflect on the *same* operator
    was healthy) → **scale=0.02, σ=1.0** (very gentle + high temp damps null-space amplification): s20
    recovers to 25.37. ΠGDM final (P/S/L): gaussian 25.67/.708/.160·24.98/.665/.209·23.42/.575/.338;
    motion 25.70/.728/.188·25.65/.720/.187·25.37/.703/.184 (noise-robust!); defocus 26.56/.725/.158·
    25.23/.653/.221·22.91/.520/.385.
  - **Reflect operator for DPS & ΠGDM motion/defocus (replaces circular).** Added `ReflectBlurOperatorDPS`
    (wraps `ops.motion_spatial.SpatialMotionBlur` = the exact reflect conv that made the obs and that SMDC-CG
    uses; adjoint exact 2e-5) + `--boundary reflect` to `run_dps.py`. Matching the obs boundary **helps a lot**
    (the circular residual miscalibrates guidance globally, not just at the edge): **DPS-reflect** motion
    25.83/25.08/24.10 (was 24.90/24.49/23.74) and defocus 24.26/23.72/23.01 (was 23.23/22.98/22.46) — +0.4–1.0 dB
    + better SSIM/LPIPS. Saved to `dps_reflect/` (circular `dps/` preserved). MC computed vs the reflect
    operator (≈noise level). Appendix flipped: DPS/ΠGDM now **all-reflect** for deblur (no boundary ring;
    only DiffPIR stays circular).
  - **DDNM-Gaussian failure marked** in `tab:gauss` (`-- (SVD unstable: NaN, σ=4 singulars ~1e-30)`; DDRM
    covers the slot). Report recompiled 22 pp, 0 undefined; aux cleaned. [[experiments-framework]] [[smdc-project]]

- **2026-08-08 — Modern baselines DiffPIR + DDNM (A1-3), shared `ffhq_10m` prior, our metrics.** Both
  reuse our `ffhq_10m.pt`: DiffPIR loads it directly (`diffusion_ffhq_10m`); DDNM via a new `openai`-type
  config `configs/ffhq_gd.yml` (canonical `celeba_hq.ckpt` URL is dead/403), loading 0 missing keys — so
  all methods share one face prior. All 100 steps, on the 16 CelebA-HQ, operators + noise matched to ours
  (noise via the `σ_y/2` [0,1]→[−1,1] doubling); metrics recomputed with `utils.metrics` (PSNR reproduces
  each tool's self-report). Added to `tab:{gauss,motion,defocus,sr}`.

  | task (σ_y) | DiffPIR  P/S/**L** | matched operator |
  |---|---|---|
  | Motion 0.05/0.10/0.20   | 28.87/.784/**.078** · 27.53/.761/**.104** · 26.25/.740/**.140** | exact Levin09 #0 |
  | Gaussian 0.05/0.10/0.20 | 26.21/.744/**.166** · 25.56/.727/**.181** · 24.72/.704/**.195** | σ=4 kernel |
  | Defocus 0.05/0.10/0.20  | 26.88/.761/**.147** · 25.94/.734/**.170** · 24.80/.704/**.182** | our disk kernel |
  | SR box 0.01/0.05        | 27.43/.789/**.111** · 26.81/.760/**.148** | 4×4 box |
  | SR aa 0.01/0.05         | 26.14/.740/**.170** · 24.61/.698/**.192** | Gauss(σ4)⊛box₄ |

  DDNM SR (`sr_averagepooling` = exact box, svd_based path, η=0.85): 0.01 **28.38/.828/.167**, 0.05
  **27.78/.809/.190** — best SSIM@0.05, ≈SMDC PSNR. **Finding (everywhere):** SMDC/IHDM leads distortion
  (PSNR+SSIM); **DiffPIR wins LPIPS in every block** (beats DPS on both metrics; beats DDRM/DDNM on SR).
  Unlike DDRM, DiffPIR's FFT solver applies to non-separable motion/defocus *and* to aa-SR (DDNM's
  avgpool-SVD cannot). **Left out:** DDNM Gaussian — its SVD `Deblurring` uses unzeroed singulars in the
  noisy `Lambda`; our σ=4 kernel's ~1e-30 singulars → NaN (DDRM covers the separable-Gaussian slot).
  **Gotchas (recipes in memory):** DiffPIR SR script sweeps λ over `range(2,13)` and loops `k_num=8`
  (state degrades; `_k0` is clean) with a kernel-blind output dir → added `DIFFPIR_{LAMBDA_MULT(×5),TAG}`,
  score `_k0`; env fixes: guarded `hdf5storage`/`motionblur`/`tensorboard`, `interp2d`→`RegularGridInterpolator`
  (SciPy 1.14). DiffPIR uses circular boundary + native `st=0` SR grid (1-px offset) — noted in captions;
  DDNM scored vs its own `Apy/orig_{i}.png` (its loader reorders). Abstract/Setup name DPS/DDRM/DiffPIR/DDNM.
  [[smdc-project]] [[ddrm-baseline-repro]]

- **2026-08-05 — Per-noise reg γ for the deblur tables (fixes the uniform γ=0.5 that violated the noise rule).**
  Swept γ∈{0,0.25,0.5,1,2} × σ_y∈{0.05,0.10,0.20} for Gaussian(+HQS) and motion(+CG), n=16, on the
  pre-generated `results/{gaussian,motion_reflect}[_n0p10/_n0p20]` obs (`deblur.py freqreg`, sharded 1
  noise/GPU). γ=0.5 reproduced the old table cells **exactly** (validates the pipeline). PSNR-optimal γ*
  (used to update `tab:gauss`/`tab:motion`, + new noise×γ Table `tab:regablation2`):

  | | σ=0.05 | 0.10 | 0.20 |
  |---|---|---|---|
  | Gaussian γ* | 0.25 (27.14) | **0** (26.47) | **0** (25.41) |
  | Motion γ*   | 0.5 (30.43) | **0** (28.41) | **0** (26.79) |

  Full sweep, PSNR/SSIM/LPIPS (**bold** = best PSNR per row):

  *Gaussian (+HQS):*
  | σ_y | γ=0 | γ=0.25 | γ=0.5 | γ=1 | γ=2 |
  |---|---|---|---|---|---|
  | 0.05 | 27.13/.778/.360 | **27.14**/.779/.355 | 27.13/.779/.353 | 27.08/.779/.352 | 26.98/.776/.355 |
  | 0.10 | **26.47**/.756/.380 | 26.45/.756/.377 | 26.41/.755/.375 | 26.34/.754/.375 | 26.18/.750/.377 |
  | 0.20 | **25.41**/.724/.396 | 25.32/.722/.396 | 25.24/.720/.397 | 25.08/.716/.398 | 24.79/.709/.402 |

  *Motion (+CG, reflect):*
  | σ_y | γ=0 | γ=0.25 | γ=0.5 | γ=1 | γ=2 |
  |---|---|---|---|---|---|
  | 0.05 | 29.74/.809/.119 | 30.30/.838/**.113** | **30.43**/.847/.124 | 30.35/**.850**/.150 | 29.96/.845/.189 |
  | 0.10 | **28.41**/.796/**.246** | 28.38/.799/.256 | 28.32/.800/.264 | 28.19/.800/.277 | 27.96/.796/.294 |
  | 0.20 | **26.79**/.761/**.328** | 26.74/.761/.332 | 26.70/.761/.335 | 26.60/.759/.341 | 26.42/.755/.349 |

  (MC ~flat in γ: Gaussian .093/.182/.342, motion .081–.089 / .172 / .330 across the sweep.) Note the
  perception/distortion split *within* the sweep: motion@0.05 best LPIPS is γ=0.25 (.113) and best SSIM
  γ=1 (.850), while PSNR peaks at γ=0.5 — a Pareto pick would shift motion@0.05 to γ=0.25.

  **γ* decreases with noise for both** — the mirror image of box-SR (γ* rises 0→4). Gaussian is full-rank/
  speckle-free so reg only ever costs (γ*=0.25 at 0.05 is a trivial +0.01 dB; strictly harmful above);
  motion's near-zero notches inject most speckle when data is trusted most (low noise), so reg pays only at
  0.05 (+0.69 dB) and γ*→0 by 0.10. All IHDM numbers/bolding in both tables updated to the per-noise γ*
  (report compiles, 13 pp). Same-day `[[transfer-profiles]]` mechanism (σ*∝ε) predicts the sign of dγ*/dσ_y.

- **2026-08-05 — Why optimal γ moves with noise in *opposite* directions for SR vs motion (transfer-profile
  mechanism).** The plain data step's noise gain `g_n[j]=w_y|â|σ_y/(w_p+w_y|â|²)` peaks over frequency at
  `|â|*=√(w_p/w_y)∝σ_y`; for a *fixed* weak frequency of gain ε it peaks over noise at **σ*∝ε** — each
  near-null frequency rings loudest at a noise level set by its own gain. So the sign of dγ*/dσ_y is set by
  the *shape* of `|â|` near its zeros (fig `docs/figures/transfer_profiles.png`, `transfer_profile` +
  |OTF| of the saved kernels): **box-SR** is a contiguous cutoff — Dirichlet sinc with active sidelobes,
  zeros at f=0.25/0.50/0.75 (verified); its true null is inert (prior-filled), the smallest *active* gain
  is the moderate transition floor, so it goes noise-dominated only as σ_y grows → **γ* grows with σ_y**
  (0→4 as 0.01→0.05). **Motion** (Levin09 #1) stays O(1) to high freq but has *isolated deep notches*
  (f≈0.20/0.47/0.64), ε→0 → σ*→0, and w_y∝1/σ_y² is largest at low noise → **γ* grows as σ_y↓**.
  **Gaussian/defocus/anti-alias-SR** are smooth roll-offs with no active near-zero band → g_n peak has no
  spectral mass → reg useless at every noise (the anti-alias Gaussian is exactly what smooths box-SR's
  sidelobes away). Unifies the reg-ablation table under one law: optimal γ tracks how much *active*
  transition/notch mass the operator has where the noise-gain peak currently sits.

- **2026-08-05 — Pure box-downsample SR + DDRM head-to-head (added to `hqs_report.tex` Table `tab:sr`).**
  Added the *pure* box downsample (avg-pool, `aa_sigma=0`) — DDRM's native `sr4` — alongside the
  anti-alias operator, both methods, σ_y∈{0.01,0.05}, n=16. New `--aa_sigma`/`--decimation` flags on
  `scripts/sr.py freqreg`. **DDRM repro gotcha: `--timesteps 20` is load-bearing** — main.py defaults
  to 1000, at which eta=0.85 injects noise every step and PSNR collapses (sr_aa control 23.21 vs the
  table's 26.66); at 20 steps the sr_aa control reproduces the published row *exactly*
  (26.66/0.773/0.237). DDRM's `ood_celeba` faces are md5-identical to `results/gaussian/clean` (same 16);
  `ffhq_256.yml` (openai, class_cond false) loads `imagenet/256x256_diffusion_uncond.pt`; SSIM/LPIPS
  recomputed from saved `{id}_-1.png` vs `orig_{id}.png` via `utils.metrics` (`[[ddrm-baseline-repro]]`).

  | operator (σ_y) | SMDC+IHDM (best γ) | DDRM (`sr4`/`sr_aa`, 20 steps) |
  |---|---|---|
  | pure down (0.01) | **28.60/0.834**/0.252 (γ=0) | 27.84/0.811/**0.186** |
  | pure down (0.05) | **27.80/0.806**/0.316 (γ=4) | 27.31/0.795/**0.204** |

  Same distortion/perception split as everywhere: SMDC leads PSNR/SSIM (+0.76/+0.49 dB), DDRM wins
  LPIPS. Dropping the anti-alias pre-blur destroys less HF → sharper bicubic anchor (27.16 vs 23.30 dB
  @0.01) and milder null space. **Reg peak flips with the operator:** low-noise prefers γ=0 (flat,
  28.60→28.57 as γ:0→1), but 0.05 is strongly speckle-limited on the box's poor transition band — a
  *unimodal* sweep peaking at **γ=4** (27.44→27.80 dB, LPIPS .353→.316, SSIM .783→.806; γ=8 27.76, γ=16
  27.54), a far larger payoff than the anti-alias operator's flat γ-plateau. Consistent with the
  2026-08-03 rule that optimal γ tracks null-space/transition-band dominance.

- **2026-08-03 — SR speckle → frequency-aware (DDRM/Wiener) data regularization (diagnosis + fix + productionized).**
  SMDC reconstructions show high-freq speckle. A **σ_y=0 test proved it is noise, not the prior**:
  the noiseless recon is clean (SR ×4: 26.80→**28.76 dB**, LPIPS .472→.321) and the near-cutoff band
  [32,64) holds 34% of the *noisy* residual but only 6% noiseless. Mechanism: the plain data step's
  per-freq noise gain `w_y|Â|σ_y/(w_p+w_y|Â|²)` peaks at `|Â|=√(w_p/w_y)` with amplitude
  ~σ_y-independent (since `w_y∝1/σ_y²`) → fixed stippling. Fix (DDRM: trust the prior where singular
  values vanish): a DCT-diagonal prior-precision boost `r(f)=γ·w_p·(1−|â|/|â₀|)₊`, i.e. `w_p→w_p+r(f)`
  in the MAP — zero in the passband, large in A's weak/null band; flattens the peak, leaves the
  well-measured low band untouched. **Productionized:** `MAPCorrection` (`solver/hqs.py`, default
  γ=0.5), `--freq_reg` in `deblur.py restore`; sweeps `scripts/sr.py freqreg` / `scripts/deblur.py freqreg`
  (â via analytic transfer / `SuperResolution.transfer_profile` / Hutchinson diag(AᵀA)); report
  §"Frequency-aware regularization" (`hqs_report.tex`, eqs. noisegain/reg/wienerreg, compiles).
  **Optimal γ tracks null-space dominance:**

  | operator (σ_y) | best γ | at optimum vs γ=0 |
  |---|---|---|
  | SR ×4 (0.01) | ≈4 | 26.82→**28.08/.806/.346** (closes ~82% of the noiseless gap) |
  | Gaussian (0.05) | 0.5 | strict Pareto win (27.35→27.37, LPIPS .382→.373) |
  | motion (0.05) | 0.5–1 | +0.78 dB PSNR, +0.037 SSIM (LPIPS +0.012) |
  | defocus (0.05) | ~0 | already clean (LPIPS .333); reg ≈neutral / slightly hurts |

  Large payoff for SR (hard cutoff → high-freq is *only* noise+prior), small for full-rank deblur,
  ~none for defocus. γ should ideally scale with σ_y (proper Wiener `r∝σ_y²`). `[[deblur-inr-crossscale]]`.

- **2026-08-03 — Exact SR intertwining via QMF alias correction** (ref: Deblur-INR, NeurIPS'24, Prop.1
  — strided downsample & blur don't commute; commutator = QMF aliasing `Σ_d(x⊗g_d)↓`, verified in our
  framework to 3e-16). Added operator-agnostic `Δ_i = A(K_i x) − L_i(A x)` and
  `sr_scale_matched_target(x_hr=…)` (`ops/superres.py`) + `decimation="stride"` option; gate
  `gate_sr_intertwining_exact` (strided/aa=0: plain ~8% → corrected **9.7e-17** with oracle x). Makes
  the data target exact for **any** antialias strength — but needs a high-freq HR plug-in. Recon test
  (×4, n=4): avg-pool aa=2 plain 29.33 → exact 29.34 (+0.01); stride aa=0 plain 25.85 → exact **26.01**
  (+0.16). Marginal, because (a) the plug-in (bicubic/prior-mean) lacks true high-freq — the info in
  A's null space we're solving for — and (b) `Δ_t→0` at t→0 but peaks at coarse t where the plug-in is
  worst. Unlike Deblur-INR (fits full-res y=k⊗x, which constrains high-freq → plug-in self-consistent),
  SR's LR obs gives no high-freq. Takeaway: our **antialias+avg-pool** design is the right default
  (clean 2e-4 *and* higher PSNR: 26.80 vs strided 25.85); the QMF correction is a niche tool for when an
  aliased forward model is forced. Ref logged in memory `[[deblur-inr-crossscale]]`.

- **2026-08-03 — Noise-injection test (deblur, IHDM-256).** Optional per-step noise *after* the MAP
  data step, skipped on the last step (`solver/base.py`, `scripts/deblur.py noise`; default off,
  `ns=0` ≡ deterministic 27.65 dB). **Annealed** (std=ns·t/N): benign for ns≲0.05 (LPIPS 0.358→0.349,
  PSNR flat), hard cliff to ~9 dB above. **Const**: collapses at ns=0.05, but **const ns=0.01 (=σ_path)
  is stable and inert** (27.65 dB). Crux: the in-distribution level (≈σ_path) does nothing, anything
  larger is OOD → the deterministic mean mispredicts → chain diverges. A stochastic reverse needs a
  noise-*trained* prior (BDM). Also little room in-band: Gaussian A has no hard null space.

- **2026-08-03 — Super-resolution modality (complete).** `ops/superres.py`: `A = D_s ∘ B_aa`
  (antialias Gaussian in the shared DCT basis, commutes with K_t; then **avg-pool decimation** — lands
  on the LR half-sample grid center, vs strided `::s` which hits the corner and breaks intertwining
  5–14% for even s). Companion `L_t` = LR-grid heat blur, `σ_t/s` (`lr_heat_schedule`). Adjoint = VJP.
  Gates: SR adjoint 2.2e-15, **SR intertwining 2.5e-4** (tighter than CT; controlled by aa: σ=s→2e-4,
  0.5s→3.5e-2, none→~40%). PoC `scripts/sr.py demo` (×4, reuses motion-CG, ‖A‖=1): **bicubic 23.17 →
  SMDC 26.80 dB**. Baselines (same op+obs, `scripts/sr.py baselines`): DPS 22.14 / LPIPS **0.228**;
  TV+CG **26.80** / SSIM **0.771**; SMDC 26.80 / 0.720 / 0.472. SMDC ties (not beats) TV here — ×4 SR
  w/ strong antialias is a mild, TV-friendly problem; SMDC's residual grain (transition-band *noise* amplification — see the freq-reg entry above;
  removed by γ≈4) costs SSIM/LPIPS at γ=0.

- **2026-07-31 — CT modality (parallel-beam Radon; PoC done, tuning open).** Companion is **not**
  `L_t=K_t`: Fourier-slice gives `R(K_t x)=L_t(R x)`, `L_t` = 1-D detector-axis heat blur
  (`ops/ct.py`). Gates: CT adjoint 1.5e-14; intertwining discretization-limited (0.2% fine / 4.5%
  coarse @256, tighter with resolution). `scripts/ct.py` (180-view, ‖R‖=1) recovers recognizable
  faces (15.5 dB / SSIM 0.64). Open (tuning): out-of-FOV corner streaks, RGB color fringing, ramp/FBP
  preconditioning of the low-pass RᵀR.

- **2026-07-31 — Defocus modality + roadmap.** Disk/pillbox PSF, symmetric ⇒ DCT-diagonal (`â` residual
  9e-6), so **closed-form DCT-HQS** like Gaussian (no CG). n=16 held-out: **27.81 / 26.64 / 25.49 dB**
  @ σ_y 0.05/0.10/0.20, beats TV/cold; DPS best LPIPS. Rule: symmetric op → DCT-HQS, asymmetric → CG.
  Solver-alignment: TV/cold/IHDM share one solver per mode. Gaussian basis ablation: full-frame R,R,R
  32.2 dB vs circular C,C,C 17.3 → no DFT prior needed.

- **2026-07-30 — Motion deblur → default `R+R+R`.** Reflect-boundary obs; DFT-Wiener rings at the
  border, so use spatial CG (reflect A, exact autograd adjoint, no FFT). Best full-frame at every noise:
  **29.74 / 28.40 / 26.78 dB** @ σ_y 0.05/0.10/0.20, 3–5 dB over TV/DPS. Commutation ≈1e-3 under reflect
  (boundary-localized). Boundary ablation: interior ≈31 dB for all 8 combos (border-only effect).
  Tools: `ops/motion_spatial.py`, `scripts/deblur.py restore --operator motion`.

- **2026-07-25 — SMDC 1-step gradient vs IHDM-HQS MAP** (CelebA-128): MAP **25.39 / 0.818 / 0.182** wins
  all (DCT-diagonal A ⇒ closed-form MAP = same per-step cost; gradient degrades with inner steps).
  SMDC's 1-step edge only applies to non-diagonalizable A.

- **2026-07-25 — IHDM checkpoint plugged into SMDC** (160M, prior-agnostic): CelebA-128 **+4.8 dB**,
  LPIPS 0.225. Found FFHQ eval train-contaminated → switched fair evals to held-out CelebA. *(Lesson:
  load-test before deleting checkpoint backups — [[verify-before-deleting-backups]].)*

- **2026-07-24/25 — Priors trained + method validated.** Gates ✅; CelebA-128 → 80k, FFHQ-256 → 60k
  (loss 0.089). Core scale-matching ablation +4.8 vs −3.8 dB. End-to-end validated at 128² and 256².
