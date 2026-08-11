# Roadmap — Scale-Matched Data Consistency (SMDC)

Two tracks run in parallel. **Track A (paper submission) is the active priority**; Track B is
the longer-horizon research agenda. Items are listed in priority order within each track.

---

# Track A — Submission-grade experiments (target: top CV/ML conf)

Goal: turn the current technical report (`docs/hqs_report.tex`) into a submittable experiments
section for CVPR / NeurIPS / ICLR. Every claim in the title — **scale-matched DC**, **non-hot
prior**, **freq-aware reg** — must be defended by a measured experiment.

**Where we stand (gaps that block submission):** larger eval set **in progress** (A1-1, `n=200`);
one dataset (CelebA-HQ-256 faces only); the title claim (scale-matched DC) has **no ablation**.
Baselines are DPS · DDRM · **DiffPIR** (all deblur + SR) · **DDNM** (box-SR) — validated at `n=16`,
being re-run at `n=200` (A1-1); still to run on ImageNet (A1-2).

## A1 — Must-have (desk-reject risk without these)

1. ✅ **DONE — Larger test set (n=200) + mean±std.** *[dependency for everything else — do FIRST]* **Scope
   change (2026-08-09): use `n=200`, not 1k** (1k too heavy on 2–4 shared GPUs), **our own indices =
   first 200 of the korexyz CelebA-HQ-256 validation split** (superset of the current 16 → each
   method's first-16 must reproduce the published cell, a built-in guard). Re-run all main tables on
   this set, report mean±std. **DONE (2026-08-11):** all methods run + validated to the published
   first-16 (built-in guard), scored at n=200 (DPS n=100), and the **4 main tables in `hqs_report.tex`
   (gauss/motion/defocus/sr) rewritten to n=200 with PSNR mean±std** (compiles, 14pp). Numbers in
   `results200/scores_n200.md`. Bugs fixed en route: motion-CG `--freq_reg`, SR `--abs_noise`, TV
   recipe (per-noise β / reflect-CG / disk), DDRM-gaussian operator (`deblur_gauss`→`deblur_aa`; proved
   our σ=4 ≡ spatial std-4 Gaussian exactly). Honest shifts at high noise: operator-matched DDRM edges
   IHDM on gaussian-0.20 & SR; DiffPIR wins LPIPS everywhere; IHDM leads distortion on motion/defocus.
   Prose (defocus/SR narratives + abstract) synced to n=200. Benchmark now driven by the declarative
   `experiments/` package (registry→run→score, guard auto-runs); pitfalls in `docs/PITFALLS.md`.
   **Deferred (not blocking A1-1):** reg-ablation tables `tab:regablation`/`regablation2` still n=16
   (IHDM-only γ sweeps — re-run at n=200 when needed); figures still show old runs.
2. **Second dataset — ImageNet-256.** Port method + all baselines. This is where the *non-hot
   prior* claim is actually on trial (faces are low-entropy; natural images are the honest test).
   Largest compute block — freeze the eval pipeline and baseline harness before starting, or redo.
3. **Modern baselines under identical operator/noise/test-set.** ✅ *Done at n=16 (2026-08-08):*
   **DiffPIR** (novelty-defining, also HQS+diffusion — all deblur + both SR operators; wins LPIPS
   everywhere) and **DDNM** (box-SR; NaN on our strong-Gaussian, DDRM covers that slot), both on the
   shared `ffhq_10m` prior, our metrics. **Remaining:** **ΠGDM** (closest to our closed-form Wiener
   DC); RED-diff optional. Then re-run the whole table at 1k / ImageNet (A1-1/2). DDRM/DPS = floor.

## A2 — Strongly expected (borderline → clear accept)

4. **Scale-matched vs naive DC ablation** (same prior). The *title* contribution; currently zero
   ablation. Highest-value single experiment missing.
5. **Non-hot (IHDM) vs hot/VP prior** ablation, same framework — isolates the prior claim.
6. **NFE + runtime columns** on every method × table. The compute–quality tradeoff is the selling
   point of this method class; a cheaper per-step closed form is a headline only if it's tabulated.
7. **NFE sweep** (PSNR/LPIPS vs number of HQS steps) — shows graceful degradation, justifies the
   chosen operating point.
8. **Robustness to misspecification:** assumed σ_y ≠ true σ_y, and kernel mismatch (estimated PSF).
   Extends the existing γ-vs-noise mechanism from a tuning detail into a contribution.

## A3 — Breadth (widens acceptance)

9. **Inpainting operator** (box / random mask) — canonical DDNM/DDRM benchmark, cheap to add;
   omitting it reads as avoidance. (Note nonlinear tasks as out-of-scope: framework is linear.)
10. **Qualitative grids** per operator vs baselines, plus the **speckle-without-reg failure case**
    (motivates the freq-reg contribution visually). Keep the transfer-profile figure.
11. Keep the existing **freq-aware γ ablation** (`tab:regablation`, `tab:regablation2`).

## Execution discipline (carry into every re-run — a broken baseline makes our win look fake)

- **DDRM `--timesteps 20` is load-bearing.** `main.py` defaults to `1000`, which silently collapses
  DDRM (~23 dB vs correct ~26.7 — no error, just a bad number). Control (`sr_aa`, σ_y=0.05) must
  reproduce **26.66 / 0.773 / 0.237** before trusting *any* DDRM cell. Put "20 steps" in **every**
  DDRM caption (currently only the SR caption states it). General rule: each baseline must reproduce
  its own *published* number on a standard operator before we trust it on ours (DiffPIR/DDNM/ΠGDM
  each have their own load-bearing settings — step count, guidance scale, schedule).
- **DiffPIR / DDNM load-bearing settings (learned 2026-08-08, carry into the 1k/ImageNet re-run):**
  DiffPIR SR `main_ddpir_sisr.py` *sweeps* λ (`range(2,13)`) and loops `k_num=8` (later k-indices
  corrupt; only `_k0` is clean) — use `DIFFPIR_LAMBDA_MULT` (×5) + `DIFFPIR_TAG`, score `_k0`. All
  baselines: noise via the `σ_y/2` [0,1]→[−1,1] doubling; DiffPIR uses circular boundary + native
  `st=0` SR grid; score DDNM against its own reordered `Apy/orig_{i}.png`. Env patches (guarded
  `hdf5storage`/`motionblur`/`tensorboard`, `interp2d`→`RegularGridInterpolator`) live in the backbones.
- **γ / hyperparameter tuning on a held-out tuning subset, never the 1k eval set** (else it reads as
  test-set tuning). Same discipline for every baseline: tune all methods on the same split, or none.
- Report the current tables' numbers as they stand (already run at 20 steps, so the *existing*
  report is correct) — the risk above is forward-looking, about the 1k re-run.

---

# Track B — Method extension (research agenda)

Status baseline (2026-08-03): **Gaussian** (IHDM+HQS, DCT-exact), **motion** (IHDM+CG,
reflect-boundary), **defocus** (IHDM+HQS, closed-form DCT-Wiener, 3 noise levels), **super-res**
(IHDM+CG, LR-grid heat companion, ×4) complete; **CT** operator+gates+PoC done (tuning open).

**Solver-alignment principle (2026-07-31):** TV, cold, and IHDM share one data-consistency solver,
chosen by operator symmetry. Both-axes-symmetric kernel under reflect boundary → exactly diagonalized
by DCT-II (Neumann heat basis) → **closed-form DCT-Wiener HQS** (residual ~9e-6). Asymmetric motion
kernel → **spatial reflect-CG**. CG and closed-form agree to ~0.01 dB where both apply (efficiency
choice, not accuracy).

## The load-bearing assumption: intertwining `L_t A = A K_t`

`K_t` = heat prior's forward blur; `A` = measurement operator; `L_t` = **measurement-side companion**
carrying the scale-t blur onto the observation so the per-step likelihood lives on `x_t` directly
(no DPS-style `x̂₀`/backprop). For each new `A` the first question is **"does an intertwining `L_t`
exist?"** (gate: `tests/gates.py::gate_intertwining`).

| Modality | Forward `A` | Companion `L_t` | Intertwining | Status |
|---|---|---|---|---|
| Gaussian ✅ | isotropic conv | `K_t` | exact / ~1e-3 reflect | done (DCT-HQS) |
| Motion ✅ | motion conv | `K_t` | ~1e-3 reflect | done (reflect-CG) |
| Defocus ✅ | disk conv | `K_t` | ~1e-15 (DCT-diag) | done (closed-form DCT-HQS) |
| Super-res ✅ | blur ∘ downsample | LR-grid heat (`σ_t/s`) | ~2e-4 (avg-pool) | done (operator+gate+PoC) |
| CT ◐ | Radon `R` | 1-D detector-axis heat | ~0.2% fine / 4.5% coarse @256 | op+gate+PoC done; **tuning open** |
| MRI | mask ∘ `F` | k-space multiply by `g_t` | exact **iff** prior uses DFT/periodic heat | **basis decision first (§B4)** |

- *Why CT is exact:* Fourier-slice — blur-then-project = project-then-1-D-blur-sinogram. Genuine
  result, not a hack. Open work is CT-specific tuning (inscribed-disk FOV / phantoms, grayscale vs
  per-channel, ramp/FBP-preconditioned data step), not plumbing.
- *Why MRI needs a decision:* masked-Fourier commutes with **DFT/periodic** heat but only
  approximately with the current **Neumann/DCT** heat. Pin the basis before starting (§B4).

## B4 — The one early decision: prior heat eigenbasis

DCT/Neumann (current) serves deblur/defocus/CT/super-res; DFT/periodic is required for exact MRI.
Both a transformer backbone and a BDM upgrade force a retrain — that retrain is the moment to switch
to periodic heat or make the basis a config flag. **Pin before MRI.**

## B5 — Blurring-Diffusion (BDM) upgrade

BDM (Hoogeboom & Salimans 2022) = heat dissipation + added Gaussian noise as a proper VP diffusion
in frequency space. **Fixes what we measured:** exact GLS whitening degrades (32.18 → 27.00 dB,
`scratchpad/whiten_test.py`) because forward covariance `σ_n² g_t²` is singular on damped
frequencies; BDM's floored covariance `g_t²σ_blur² + σ_noise² > 0` is never singular → principled
noise-floored schedule + stochastic reverse. Highest value, highest effort; re-derive scale-matching
for the noisy forward + retrain. Do **after ≥1 non-deblur modality** validates generalization.

## B6 — Transformer backbone

Backbone is orthogonal to all SMDC math (solver only calls `prior.reverse_step`). Candidates:
Restormer / Uformer / NAFNet (restoration-oriented, exploit IHDM's coarse-to-fine ladder), not a
generation DiT. Cost = full retrain of the ~211M prior (no equation changes). **Fold into the BDM
retrain (§B5)** so the prior trains once.

## Recommended Track-B sequence

1. Defocus ✅ · 2. CT ◐ (core validated, tuning open) · 3. Super-res ✅ ·
4. **MRI** (after §B4 eigenbasis decision; Cartesian → radial) ·
5. **BDM (§B5)** — retrain here, subsumes the whitening fix ·
6. **Transformer (§B6)** — fold into the BDM retrain.
