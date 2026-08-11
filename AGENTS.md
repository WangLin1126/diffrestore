# AGENTS.md — working in this repo

Guidance for coding agents. The `README.md` explains the *idea*; this file explains how to *operate*
here without breaking things. Read `docs/ROADMAP.md` (current priorities) and `docs/EXPERIMENTS.md`
(what's been run) before starting non-trivial work.

## What this project is

**Scale-Matched Data Consistency (SMDC) for non-hot diffusion priors** — linear inverse problems
(deblur / super-res / CT) solved with a **`prior × solver`** framework. The measurement is transformed
to the current blur scale via the intertwining `Lₜ A = A Kₜ` ⇒ `A xₜ = Lₜ y`, giving an exact per-step
likelihood on the reverse state (no clean-image estimate, no backprop through the prior).

- **priors**: `ihdm` (non-hot heat diffusion, x0-predictor), `cold_diffusion`; `dps` is a hot spare.
- **solvers**: `smdc` (1-step scale-matched gradient) and `hqs` (full per-step MAP, DCT-Wiener closed
  form for symmetric ops, spatial reflect-CG for asymmetric ops). Same objective, different accuracy.
- Deliverable: `docs/hqs_report.tex` (the paper). Theory in `docs/` + `MATH.md` history.

## Environment

- **Always run inside the activated conda `base` env** (`source activate base`). Torch 2.x + cu12x,
  LPIPS, pyarrow, huggingface_hub. Don't create new envs.
- 4× TITAN RTX (24 GB). **Check `nvidia-smi` before launching — GPUs may be shared with other users;
  never preempt someone else's job.**
- Scratch/temp files go in the session scratchpad, not `/tmp` or the repo.

## Repo layout

```
model/     ihdm.py · cold_diffusion.py · dps.py + backbones: ihdm_backbone/ dps_backbone/
           ddrm_backbone/ ddnm_backbone/ DiffPIR_backbone/ bdm_backbone/   (external baselines)
solver/    base.py (scale_matched_solver = the restoration loop) · hqs.py (MAPCorrection) · init.py
ops/       transforms.py (DCT) · deblur.py · spectral.py · motion_spatial.py · superres.py · heat.py
utils/     pipeline.py (shared IO / build_prior / ‖A‖=1 / CG / freq-reg / metrics) · metrics.py · seed.py
scripts/   REUSABLE pipeline only: deblur.py · sr.py · ct.py · run_dps.py · run_tv_hqs.py ·
           run_tv_cg.py · run_tests.py · train_ihdm*.py · make_figure.py · crop_score.py ·
           score_dir.py (generic dir-vs-dir scorer)
scratchpad/ RETIRED one-off scripts: task-specific eval drivers (a1_*.sh), old rerun_*.sh, ablations
data/      loaders.py + celebahq256/ ffhq256/ …          tests/  gates.py (numerical gates)
checkpoint/ ihdm/ihdm_ffhq256_full.pth (211M) · cold_diffusion/ffhq256.pth · dps/ffhq_10m.pt (94M)
docs/      hqs_report.tex · ROADMAP.md · EXPERIMENTS.md      results/ (n=16) · results200/ (n=200 eval)
```

`ffhq_10m.pt` (94M FFHQ face prior) is the **shared** prior for DPS **and** DDRM/DiffPIR/DDNM — so all
learned baselines use one prior. It is *not* the 552M ImageNet ADM.

## The benchmark: `experiments/` (use this to run/score the tables at any n)

**`experiments/` is the single source of truth for the CelebA-HQ restoration benchmark.** All method
recipes (flags, per-cell γ/β, the σ_y/2 noise convention), applicability (DDRM is separable-only),
per-tool scoring quirks, and the published first-16 guard values live in `experiments/registry.py`.
Never re-derive a recipe in a shell script or from memory — edit the registry.

```bash
python -m experiments.run   --n 1000              # print the full plan (obs + every cell×method), from the registry
python -m experiments.run   --n 1000 --gpus 0,1,2,3 --exec   # run it (stage external datasets first)
python -m experiments.score --n 1000 --metrics    # uniform mean±std; the first-16 GUARD auto-runs, prints !! GUARD on drift
```
The guard is load-bearing: any `n≥16` set is a superset of the published n=16 set, so each method's
first-16 must reproduce `registry.PUBLISHED_PSNR`. **Read `docs/PITFALLS.md` before expanding n** — it
records every trap (operator matching, TV solver-per-operator, DDRM `deblur_aa`, DiffPIR panels/λ,
`pkill` self-match, `glob.escape` for `[` in paths, DataParallel pinning, …). Retired one-off drivers
live in `scratchpad/` (git-ignored).

## Core commands (building blocks the registry calls)

```bash
python scripts/run_tests.py                      # numerical gates (model-free): adjoint, intertwining…
# make a shared held-out test set (clean + observation):
python scripts/deblur.py obs --operator gaussian --source celebahq --n 200 --image_size 256 --blur_sigma 4 --noise 0.05 --out results200/gaussian_s05
# restore (prior × per-step MAP); --operator ∈ {gaussian, motion, defocus}:
python scripts/deblur.py restore --operator gaussian --prior ihdm --ihdm_config img_size_256_full \
  --ckpt checkpoint/ihdm/ihdm_ffhq256_full.pth --image_size 256 --n 200 --sigma_y 0.05 --freq_reg 0.25 \
  --clean_dir results200/gaussian_s05/clean --observation_dir results200/gaussian_s05/observation --out results200/gaussian_s05/ihdm
python scripts/sr.py freqreg --scale 4 --abs_noise --data_weight 64 --sigmas 0.01 --regs 0 --n 200 --clean_dir <clean> --out <dir>
python scripts/score_dir.py --clean <clean> --recon <recon> --tag <name>   # generic mean±std + first-16 guard
```
Metrics: PSNR / SSIM / LPIPS(alex) / MC (`‖y−Ax̂‖/‖y‖`), all computed on **[−1, 1]** (`utils/metrics.py`).

## Invariants — do not break these

- **GPU / DataParallel:** the priors are wrapped in `DataParallel(device_ids[0]=cuda:0)`. To pin a job
  to GPU N, use `CUDA_VISIBLE_DEVICES=N python … --device cuda:0`. **Never** pass `--device cuda:1`
  (crashes). Shard sweeps one job per GPU; keep all free GPUs busy.
- **Determinism / data source:** eval sets are seeded (`seed=0`) and share **one** clean set (first-N of
  the korexyz CelebA-HQ-256 *validation* split). Every method must read the identical images — verify by
  md5 when in doubt.
- **Solver ↔ operator pairing:** symmetric conv (gaussian/defocus, reflect boundary) → closed-form
  DCT-Wiener HQS; asymmetric (motion) → spatial reflect-CG. Motion/defocus are **not** DCT-separable, so
  DDRM (SVD, separable-only) is N/A for them.
- **Frequency-aware reg `γ` (`--freq_reg`)** is per-cell and noise-dependent; it is applied in *both* the
  closed-form and the CG data steps. Don't assume a single γ.

## Baseline discipline (a broken baseline makes our win look fake)

- **Reproduce the baseline's own published/known number on a control before trusting any new cell.** The
  n=200 eval uses a built-in guard: since the 200-set ⊃ the old 16-set, each method's **first-16 must
  reproduce the published table cell** (see `experiments/score.py`, which auto-runs this guard).
- **Noise convention:** our `σ_y` is std on [−1, 1]; DPS/DDRM/DiffPIR/DDNM define noise on [0, 1] then
  double it → **pass `σ_y/2`** (`--sigma_0`, `DIFFPIR_NOISE`, `--sigma_y`).
- **Load-bearing settings:** DDRM `--timesteps 20` (default 1000 silently collapses it). DiffPIR SR
  sweeps λ / loops `k_num` → use `DIFFPIR_LAMBDA_MULT`/`DIFFPIR_TAG`, score `_k0`. DPS = 1000 NFE (slow).
  SR IHDM must use `--abs_noise --data_weight 64`. Recipes live in `docs/EXPERIMENTS.md` + agent memory.
- External baselines run **from their own backbone dir** with their own dataset staged under it
  (`testsets/`, `exp/datasets/…`); re-score their recon PNGs with `utils/metrics.py` (their self-reported
  LPIPS differs — only PSNR matches self-report).

## Conventions

- **Commit/push only when asked.** Branch off `main` first; never commit directly to `main`.
- **After compiling `docs/hqs_report.tex`, delete `*.aux *.fls *.out *.fdb_latexmk *.log`** — keep only
  `.tex` + `.pdf`. Build: `cd docs && pdflatex hqs_report.tex && pdflatex hqs_report.tex`.
- **Verify before deleting/overwriting** any backup or result — load-test the kept copy first.
- Long runs: launch detached, log to `results*/logs/`, and validate the first cell's first-16 early to
  catch recipe errors before spending hours of compute.
- Prefer extending the existing `scripts/*.py` + `utils/pipeline.py` scaffolding over new one-offs.
- **Keep `scripts/` reusable.** Parameterized, general-purpose tools live in `scripts/`; task-specific
  one-off drivers (hardcoded cells/params, a single run's orchestration) go in `scratchpad/`.
