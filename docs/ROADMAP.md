# Roadmap — extending Scale-Matched Data Consistency (SMDC)

Status baseline (2026-08-03): Gaussian deblur (IHDM+HQS, DCT-exact), motion deblur
(IHDM+CG, reflect-boundary `R+R+R`), **defocus deblur** (IHDM+HQS, closed-form DCT-Wiener,
three noise levels), and **super-resolution** (IHDM+CG, LR-grid heat companion, ×4 PoC)
complete; CT operator+gates+PoC done (tuning open). This document plans the next phase: more
degradation models, a stronger backbone, and a Blurring-Diffusion upgrade of the heat framework.

**Solver-alignment principle (established 2026-07-31).** For each degradation mode, TV,
cold, and IHDM all use the *same* data-consistency solver, chosen by the operator's symmetry:
a both-axes-symmetric kernel under reflect (half-sample-symmetric) boundary is exactly
diagonalized by DCT-II — the Neumann heat basis — so **Gaussian and defocus get a closed-form
DCT-Wiener HQS step** (`â = DCT(A·IDCT(𝟙))`, diagonalization residual ~9e-6), while the
**asymmetric motion kernel needs spatial reflect-CG**. CG and the closed-form DCT solve agree
to ~0.01 dB where both apply, so the split is an efficiency choice, not an accuracy one.

---

## 0. The load-bearing assumption: intertwining

Every SMDC result rests on the exact relation (verified by `tests/gates.py::gate_intertwining`)

```
    L_t A = A K_t
```

`K_t` is the heat prior's forward degradation; `A` is the measurement operator; `L_t` is the
**measurement-side companion** that carries the scale-t blur onto the observation so the
per-step likelihood lives on `x_t` directly (no DPS-style `x̂₀`/backprop). For each new `A`
the first question is *not* "can we code it" but **"does an intertwining `L_t` exist?"** That
single question sorts the modality list:

| Modality | Forward `A` | Companion `L_t` | Intertwining | Fit |
|---|---|---|---|---|
| Gaussian deblur ✅ | isotropic conv | `L_t = K_t` | exact (circular) / ~1e-3 (reflect) | done (DCT-HQS) |
| Motion deblur ✅ | motion conv | `L_t = K_t` | ~1e-3 (reflect) | done (reflect-CG) |
| Defocus deblur ✅ | disk/pillbox conv | `L_t = K_t` | exact (circular) / ~1e-15 (DCT-diagonal) | **done (closed-form DCT-HQS)** |
| CT ◐ | Radon `R` | **1D heat blur along detector axis** | continuum-exact; ~0.2% fine / 4.5% coarse @256 (discrete) | **operator+gate+PoC done; tuning open** |
| Super-res ✅ | blur ∘ downsample | heat blur on the **LR grid** (`σ_t/s`) | ~2e-4 (avg-pool decimation) | **done (operator+gate+PoC)** |
| **MRI** | mask ∘ `F` | k-space multiply by `g_t` on sampled lines | **exact iff prior uses DFT/periodic heat** (not DCT) | basis decision first |

**Why CT is exact:** blurring the image by an isotropic Gaussian then projecting equals
projecting then 1-D-Gaussian-blurring the sinogram along the detector (Fourier-slice theorem).
So the isotropic heat prior has a *natural* sinogram-space companion `L_t` = 1-D detector-axis
heat blur. This is a genuine result, not a hack.

**Why MRI needs a decision:** masked-Fourier intertwines perfectly with **periodic (DFT)**
heat (`M`, `g_t` both diagonal in the DFT basis, so they commute), but the current IHDM prior
uses **Neumann/DCT** heat. DCT-heat ≠ DFT-heat → the commutation is only approximate. See §4.

---

## 1. New degradation models (item 1)

The universal enabler is already built: the **spatial CG data step** needs only `A·v` and
`Aᵀ·v`, and the **autograd-VJP adjoint** (`ops/motion_spatial.py::SpatialMotionBlur.adjoint`)
gives the *exact* adjoint of any differentiable forward for free. So each new modality needs
only a forward operator + a scale-matched target `L_t y`; the solver (`restore_motion_cg.py`
loop) is reused verbatim.

Per-modality plan and the acceptance gate each needs:

- **Defocus** ✅ *(done, 2026-07-31)* — disk/pillbox PSF. `L_t = K_t` (same as deblur), and
  because the disk is symmetric it is DCT-diagonal, so TV/cold/IHDM all use the **closed-form
  DCT-Wiener HQS** data step (no CG needed; `scripts/defocus_hqs.py`, `run_tv_hqs.py --operator
  disk`). Results at σ_y ∈ {0.05, 0.10, 0.20}, n=16 (`tab:defocus`): IHDM+HQS
  **27.81 / 26.64 / 25.49 dB** beats TV (26.16 / 25.16 / 23.91) and cold (24.10 / 24.50 / 24.00);
  DPS is the circulant baseline (best LPIPS). Intertwining residual ~1e-15.
- **CT** ◐ *(operator + gate + PoC done, 2026-07-31)* — `ops/ct.py`: `ParallelBeamRadon`
  (pure-torch differentiable rotate-and-sum; adjoint = VJP back-projection, exact to 1e-14) +
  `DetectorHeatBlur` (1-D detector-axis heat companion `L_t`). Gates `gate_ct_adjoint` and
  `gate_intertwining_ct` in `tests/gates.py`. The Fourier-slice intertwining `R(K_t x)=L_t(R x)`
  is a *continuum* identity → on the discrete projector it holds to ~0.2% (fine) / 4.5% (coarse)
  at 256, tightening with resolution — looser than deblur's 1e-3 but fine for SMDC (the
  continuation weights fine scales most). `scripts/ct_demo.py` reconstructs recognizable faces
  from 180-view sinograms (reuse motion-CG, normalize `‖R‖=1`, target `L_{t-1} y`).
  **Open (tuning, not plumbing):** inscribed-disk FOV / proper phantoms (faces fill the frame),
  grayscale vs per-channel color, and a ramp/FBP-preconditioned data step (unfiltered `RᵀR` is
  low-pass). See EXPERIMENTS 2026-07-31.
- **Super-res** ✅ *(done, 2026-08-03)* — `ops/superres.py`: `SuperResolution` (`A = D_s ∘ B_aa`,
  antialias Gaussian blur built in the shared DCT basis so it commutes with `K_t` *exactly*, then
  **area/avg-pool decimation** — avg-pool samples the LR half-sample-symmetric grid *center*, so it
  is the DCT-II-consistent decimation; plain strided `::s` samples the corner and injects a
  `(s-1)/2`-px phase error that breaks the intertwining by 5–14% for even `s`). Adjoint = exact
  autograd VJP (`<Ax,y>=<x,Aᵀy>` to ~2e-15). Companion `L_t` = LR-grid heat blur (`σ_t/s`,
  `lr_heat_schedule`). Gates `gate_sr_adjoint` + `gate_intertwining_sr` in `tests/gates.py`: the
  decimation intertwining `A(K_t x)=L_t(A x)` holds to **~2e-4** (an order of magnitude tighter than
  CT — the antialias `B_aa` at `σ=s` suppresses the aliasing; a light `σ=0.5s` leaks ~3.5e-2 and
  avg-pool alone ~40%). `scripts/sr_demo.py` reconstructs FFHQ-256 faces from ×4 LR (64px) reusing
  the motion-CG data step verbatim (target `L_{t-1} y` in LR space, `‖A‖=1` normalized): n=4,
  σ_y=0.01 → **bicubic 23.17 → SMDC+IHDM 26.80 dB** (SSIM 0.72). Baselines on the *same* operator +
  observation (`scripts/sr_baselines.py`, `figure_sr_compare.png`): **DPS 22.14 dB / LPIPS 0.228**
  (wins perception, hallucinates off-GT), **TV+CG 26.80 / SSIM 0.771** (ties SMDC on PSNR). Unlike
  deblur, SMDC does not lead here — ×4 SR w/ strong antialias is a mild, TV-friendly problem; SR
  validates the framework rather than showcasing the learned prior (de-speckling SMDC is the open lever).
- **MRI** — `A = M ∘ F`; Cartesian first (`L_t` = multiply sampled k-space by `g_t`), then
  radial via NUFFT (adjoint = gridding, VJP). Resolve the eigenbasis (§4) before starting.

Each modality is a self-contained `results/<modality>/` dir + a `make_<modality>_obs.py`
generator; restoration reuses `scripts/restore_motion_cg.py` with the new operator.

---

## 2. Backbone upgrade — transformers (item 2)

The backbone is **orthogonal to all SMDC math**: the solver only calls `prior.reverse_step`,
so the network behind it is a black box. Candidates, for a *restoration* prior with progressive
coarse-to-fine structure: **Restormer / Uformer / NAFNet** (conv-transformer hybrids) rather
than a generation-oriented DiT. Exploit IHDM's coarse-to-fine ladder to keep attention cheap
at fine scales.

Cost: a full retrain of the ~211M prior (no equation changes). **Sequence last**, or fold it
into the Blurring-Diffusion retrain (§3) so the prior is trained only once.

---

## 3. Blurring Diffusion upgrade of the heat framework (item 3)

Blurring Diffusion Models (Hoogeboom & Salimans, 2022) = heat dissipation **+ added Gaussian
noise**, framed as a proper VP diffusion in frequency space, preserving coarse-to-fine.

**Why it is the deepest and most aligned direction — it fixes what we measured.** The
whitened-data-consistency test (`scratchpad/whiten_test.py`, 2026-07-31) showed exact GLS
whitening degrades monotonically (32.18 → 27.00 dB) because the forward covariance
`σ_n² g_t²` is singular on damped frequencies (`1/g_t²` blow-up, MATH §4.2). BDM makes the
forward covariance

```
    g_t² σ_blur²  +  σ_noise²   >  0     (never singular)
```

i.e. it turns the fragile whitening into a **principled noise-floored schedule**, and gives a
stochastic reverse (diversity / uncertainty) while keeping progressive restoration.

Cost/scope: re-derive scale-matching for the noisy forward (the companion `L_t` becomes the
*floored* whitening, not plain `K_t`), and retrain the prior. Highest value, highest effort.
Do **after ≥1 non-deblur modality** (CT or SR) validates that the framework generalizes.

---

## 4. The one early decision: the prior's heat eigenbasis

- **DCT / Neumann** (current): serves deblur, defocus, CT, super-res.
- **DFT / periodic**: required for exact MRI intertwining.

Since both §2 (transformer) and §3 (BDM) force a retrain, that retrain is the moment to either
(a) switch to periodic heat, or (b) make the basis a config flag. **Pin this before MRI (item
1d), not during it.**

---

## Recommended sequence

1. **Defocus** ✅ — warm-up; proved "any convolution" + closed-form DCT-HQS for symmetric
   kernels; three noise levels done. *(complete)*
2. **CT** ◐ — operator, both gates, and an end-to-end PoC done (recognizable faces from 180
   views); remaining work is CT-specific tuning (disk FOV / phantoms, grayscale, ramp-preconditioned
   data step). *(core validated)*
3. **Super-res** ✅ — `L_t` = LR-grid heat blur (`σ_t/s`); avg-pool decimation makes the
   intertwining ~2e-4; ×4 FFHQ PoC beats bicubic by +3.5 dB. *(complete)*
4. **MRI** — after the eigenbasis decision (§4); Cartesian → radial.
5. **BDM upgrade (§3)** — retrain here; subsumes the whitening fix.
6. **Transformer backbone (§2)** — fold into the BDM retrain.
