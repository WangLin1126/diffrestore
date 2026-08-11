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

4. ✅ **DONE (2026-08-11) — Scale-matched vs naive DC ablation** (same prior). The *title*
   contribution. **One-line toggle** (`solver/base.py` `scale_match`, exposed as
   `scripts/deblur.py restore --naive_dc`): identical prior/solver/schedule/γ, only `L_{t-1}` dropped
   so the data step compares `A x_{t-1}` against the raw sharp `y` instead of `K_{t-1} y`. Gaussian
   deblur, n=200, 3 noise levels: **naive collapses to ≈12 dB** (11.97/11.79/12.08) vs scale-matched
   **27.30/26.60/25.47** — a **13–15 dB gap, and 6–11 dB *below* the blurry input**. Structural, not a
   tuning artifact: a full `w_data∈[0.25,64] × γ∈[0,2]` sweep on the tuning split tops out at 13.01 dB
   (13.26 at n=200), still ≈14 dB below ours, with *worse* MC (0.45 vs 0.09). Scale-matching makes the
   per-step target `A x_{t-1}=K_{t-1}y` feasible (A,K commute); naive chases an infeasible sharp target
   and diverges. New `\tab:dcablation` + subsection in `hqs_report.tex` (15pp, compiles clean). The
   x̂₀-based DC alternative (DPS/DiffPIR) is a *different* method, already a baseline in the main tables.
   **Deferred (optional breadth):** replicate the ablation on motion/defocus/SR (flag already works for
   all operators) — gaussian alone is decisive for the title claim.
5. ✅ **RESOLVED by existing results (2026-08-11) — Non-hot (IHDM) vs hot/VP prior.** User decision:
   the main tables already answer this; **no new experiment, BDM deferred.** Rationale (now stated in
   the report, "On the prior: why non-hot" paragraph after Table~\ref{tab:dcablation}): scale-matched
   DC is *structurally only defined* for a non-hot prior — the exact target `A x_{t-1}=K_{t-1}y` exists
   because the heat blur `K_t` commutes with `A`; a hot VP prior injects noise (no companion `L_t`) and
   reduces to x̂₀-guidance. Every *learned* baseline (DPS/DDRM/DiffPIR/DDNM) is exactly a hot VP prior
   (shared FFHQ `ffhq_10m`) and IHDM+SMDC leads them on distortion throughout; A2-4 (Table 6) shows what
   a per-step data step does once the scale match is removed. A fully controlled swap = a *hot heat*
   (blurring-diffusion/BDM) prior, which needs retraining → Track-B B5, future work.
6. ✅ **DONE (2026-08-11) — NFE + runtime (compute table).** Presented as one dedicated compute table
   (`\tab:compute`, report Table 7) + "Computational cost" paragraph, not columns bloating all 4 tables
   (NFE is ~operator-independent). **NFE (exact, from configs):** ours/cold 200, DPS 1000(+backprop),
   DDRM 20, DiffPIR 100, DDNM 100, TV 0. **Runtime (1× TITAN RTX, restoration only, measured):** IHDM
   gaussian 24.8 / defocus 31.3 / motion 36.4, cold 13.2, TV 1.6–2.7, DDRM 0.99, DDNM 4.8, DiffPIR 6.6,
   **DPS 95.5 s/img** (steady-state n=16; 2-point wall-clock is corrupted by TITAN-RTX thermal throttle
   → use total/n over ≥16 imgs). **Honest framing (not raw-speed superiority):** our scale-matched DC
   is closed-form/CG → **0 extra NFE** (runtime = a plain 200-step sampling pass); we're ~3× faster &
   5× fewer NFE than DPS; slower than DDRM/DiffPIR/DDNM (smaller 94M VP prior, fewer steps) but they're
   SVD-separable-only → inapplicable to motion/defocus. Two orthogonal cost levers noted: 211M backbone,
   200-step default (→ A2-7 sweep). Bench scripts in scratchpad (`bench_inrepo.py`, `bench_ext.sh`).
7. ✅ **DONE (2026-08-11) — NFE sweep.** Respace the K=200 IHDM blur-level grid to M steps
   (strictly-decreasing subset ending at 0; NFE=M); the closed-form MAP is still applied at every
   retained level so the data step carries consistency while the prior budget shrinks. **Gaussian
   n=200 (report Table 8 `tab:nfe`):** M=200→27.30, 100→26.21, 50→25.69, 25→25.24, 12→24.73, 6→24.12;
   **monotone, graceful, MC flat 0.091→0.102** (loss is perceptual/LPIPS, not fidelity). M=200 row
   reproduces the guard (27.14 first-16) exactly. **Headline:** IHDM at **M=12 (~2s/img) = 24.73 dB
   BEATS DPS's 24.12 dB** (1000 NFE, 95.5s/img) — matches/beats the guidance baseline at a few % of
   compute. Trend confirmed across both solver types (first-16: motion CG 30.43→27.12, defocus 27.82→
   25.21 from M=200→25). 200-step default = a quality ceiling, not a requirement. Sweep driver
   `scratchpad/nfe_sweep.py` (respace + score, any op/cell/n).
8. ◐ **PARTIAL (2026-08-11) — Robustness to noise-level misspecification DONE; kernel/PSF mismatch
   deferred.** Report Table 9 `tab:robust`. Origin: user noted IHDM's prior is **noise-unconditioned**
   (σ_y enters ONLY as the soft data weight `w_y=64/σ_y²` in the MAP; `model/ihdm.py` takes no noise
   input), unlike DDRM where σ_0 sets the SVD spectral-shrinkage divisor. Fed both the TRUE noise but a
   WRONG assumed σ_y (×0.5..×4), gaussian, γ fixed at true-noise value. **Finding (robust ≠ free, and
   asymmetric in OPPOSITE directions):** correct σ_y is always best; IHDM tolerates under-estimation
   (≤1.5dB) and never diverges — worst cell (×4 over @0.20) falls back gracefully to the ~18dB
   blurry-input level (prior takes over as data term starves); **DDRM COLLAPSES to 4.98dB (94% saturated
   noise, SVD blow-up) under ANY ×0.5 under-estimate, at every noise level.** → IHDM has no catastrophic
   regime, the safer choice under σ_y uncertainty. IHDM n=50 / DDRM n=200; ×1 col = guard (DDRM
   26.88/26.40/25.73 ✓). DDRM patch: guarded env hook `DDRM_TRUE_SIGMA` in `runners/diffusion.py`
   (decouples added-noise from assumed σ_0; unset = stock, guard-safe). Scripts: `scratchpad/a28_*.sh`.
   **DEFERRED:** kernel/PSF mismatch (true σ_blur=4 vs assumed 3/5) — the second misspecification axis.

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
