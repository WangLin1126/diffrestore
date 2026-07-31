# Roadmap — extending Scale-Matched Data Consistency (SMDC)

Status baseline (2026-07-31): Gaussian deblur (IHDM+HQS, DCT-exact) and motion deblur
(IHDM+CG, reflect-boundary `R+R+R`) complete and in `docs/hqs_report.tex`. This document
plans the next phase: more degradation models, a stronger backbone, and a Blurring-Diffusion
upgrade of the heat framework.

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
| Gaussian deblur ✅ | isotropic conv | `L_t = K_t` | exact (circular) / ~1e-3 (reflect) | done |
| Motion deblur ✅ | motion conv | `L_t = K_t` | ~1e-3 (reflect) | done |
| **Defocus** | disk/pillbox conv | `L_t = K_t` | exact (circular) / ~1e-3 (reflect) | **trivial — new kernel only** |
| **CT** | Radon `R` | **1D heat blur along detector axis** | **exact** (Fourier-slice theorem) | clean & novel |
| **Super-res** ×s | blur ∘ downsample | heat blur on the **LR grid** (coarse `K_t`) | exact up to aliasing (small) | easy + aliasing gate |
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

- **Defocus** *(in progress)* — disk/pillbox PSF into the existing motion-CG pipeline.
  `L_t = K_t` (same as deblur). Gate: reuse `gate_intertwining` with the disk kernel.
- **CT** — add a Radon forward (`torch-radon` or a differentiable projector; adjoint =
  backprojection via VJP). `L_t` = 1-D heat blur of the sinogram along the detector axis.
  New gate `gate_intertwining_ct`: `R(K_t x)` vs `L̃_t(R x)` < 1e-3.
- **Super-res** — `A = B_antialias ∘ D_s`; adjoint = `Dᵀ ∘ Bᵀ` (upsample-zero then blurᵀ),
  free via VJP. `L_t` = heat blur on the LR grid. New gate `gate_intertwining_sr` to quantify
  the aliasing residual `D(K_t x)` vs `K_t^LR(D x)`.
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

1. **Defocus** — warm-up; proves "any convolution", zero new math. *(in progress, noise 0.05)*
2. **CT** — highest novelty-per-effort; exact Fourier-slice intertwining.
3. **Super-res** — `L_t` = LR-grid heat blur + aliasing gate.
4. **MRI** — after the eigenbasis decision (§4); Cartesian → radial.
5. **BDM upgrade (§3)** — retrain here; subsumes the whitening fix.
6. **Transformer backbone (§2)** — fold into the BDM retrain.
