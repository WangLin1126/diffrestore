# Benchmark pitfalls & lessons — expanding the eval set (n=16 → n=200 → 1k → ImageNet)

Re-running the CelebA-HQ restoration benchmark at a larger `n` *looks* trivial ("bump a flag") but
took many wrong turns the first time. The fix is architectural — a declarative **`experiments/`**
package (`registry.py` = every recipe + guard + scoring adapter; `run.py` = commands derived from it;
`score.py` = uniform scoring with the guard auto-run). This file records *why*, so the traps don't
recur. **Golden rule: no method setting lives in a shell script or your head — it lives in
`registry.py`, and the first-16 guard catches drift before you trust full-n numbers.**

## The root cause
Every method's exact recipe (flags, per-cell hyper-parameters, noise convention, *and* how to load
its recons) was scattered across ad-hoc shell commands, the chat transcript, and memory. Expanding `n`
meant re-deriving each recipe by archaeology, and mistakes only surfaced *after* burning GPU-hours.
Centralizing the recipes + auto-running the guard turns a day of debugging into `run --n N; score --n N`.

## The guard that makes it safe
Because the test set is **the first-`n` of a fixed split with seeded noise**, any `n ≥ 16` set is a
**superset** of the published n=16 set. So **each method's first-16 metrics must reproduce the
published cell** (`registry.PUBLISHED_PSNR`, tol 0.4 dB). `experiments.score` runs this automatically
and prints `!! GUARD` on failure — *that* is a wrong recipe, caught for free. Validate the first cell's
first-16 EARLY (before the full sweep finishes) to avoid wasting compute.

## Inevitable pitfalls (each cost real time)

1. **Per-cell / per-noise hyper-parameters, not one value.** IHDM `--freq_reg γ` and TV `--beta`
   change with operator AND noise (γ: gaussian .25/0/0, motion .5/0/0; TV-gaussian β 2/3/6). A single
   value silently mis-scores the high-noise cells. → `IHDM_GAMMA_*`, `TV_RECIPE` in the registry.

2. **Solver ↔ operator pairing.** TV/deblur uses a DIFFERENT script per operator: reflect-boundary
   ops (motion, defocus) → `run_tv_cg.py` (spatial CG); DCT ops (gaussian, disk) → `run_tv_hqs.py`.
   Using the circular `run_tv_hqs` on motion rings → 22.66 vs 28.34. Same trap for the learned priors
   (motion = per-step CG, gaussian/defocus = closed-form DCT-Wiener).

3. **A code path silently ignored a flag.** `deblur.py restore`'s motion (CG) branch didn't apply
   `--freq_reg` (only the closed-form branch did) → motion_s05 came out at γ=0 (29.74) not γ=0.5
   (30.43). Fixed in-repo. Lesson: when a "shared" flag feeds only one of several code paths, the
   guard is what catches it.

4. **Operator matching for baselines is subtle.** DDRM's default `--deg deblur_gauss` is a near-uniform
   5-tap kernel — NOT our operator. Our DCT-heat **σ_blur = 4 is *exactly* a spatial Gaussian of std 4
   px** (`transfer = exp(-½σ²|ξ|²)`, separable; empirically matches a scipy std-4 reflect Gaussian to
   106 dB). The matched DDRM degradation is **`--deg deblur_aa`** (loads `deblur_gauss_G_s4_N256.pt`,
   our exact 1D matrix). DiffPIR matches it via `gauss_sigma4.npy`. **Always confirm a baseline solves
   OUR A** (compare its degraded input's PSNR to ours) before trusting its recon.

5. **Noise convention.** DPS/DDRM/DiffPIR/DDNM define σ on [0,1] then double to [-1,1] → **pass
   σ_y/2** (`--sigma_0`, `DIFFPIR_NOISE`, `--sigma_y`). DDRM *also* doubles internally — you still pass
   σ_y/2. → `registry.half()`.

6. **Load-bearing baseline settings.** DDRM `--timesteps 20` (default 1000 collapses it: ~23 vs ~26.7).
   IHDM-SR needs `--abs_noise --data_weight 64` (relative noise mis-scores the null-space aa cells:
   LPIPS .55 vs .40). DPS = ~1000 NFE (slow → we cap it at **n=100**, a subset of the 200, footnoted).

7. **Per-tool scoring quirks (encode once, in `adapters.py`).**
   - **DDRM** reorders its dataset → score recon `{id}_-1.png` vs its OWN `orig_{id}.png` (numeric id),
     never vs our clean.
   - **DiffPIR SR** saves a **2816×256 process panel** (recon = last 256×256 block) and sweeps 8 λ
     (one file/id) → take the last block, pick best-PSNR λ per cell.
   - **DiffPIR deblur** recon = `{idx}_diffusion_ffhq_10m.png` (clean 256); its self-reported LPIPS ≠
     `utils.metrics` (only PSNR matches) → always re-score from PNGs.
   - **DDNM** recon = `{i}_*.png` vs `Apy/orig_{i}.png`.
   - **DPS** scored on the first-100 subset.

8. **Stale outputs collide.** Old n=16 external dirs (`ddnm_g005`, wrong-operator `ddrm_g05`) sit next
   to the new ones (`ddrm_gaa*`) — exclude by name at scoring, don't blindly glob "newest".

9. **Shell footguns (cost several aborted commands).**
   - `pkill -f "foo"` / `pgrep -f "foo"` **match their own shell** if "foo" appears in the command line
     → the shell gets killed (exit 143/144). Kill by explicit PID instead.
   - A `[` in a path (DiffPIR dir `...blur[4]...`) is a **glob character class** → `*.png` matches
     nothing. Use `glob.escape(dir)`.

10. **GPU / DataParallel.** The priors are `DataParallel(device_ids[0]=cuda:0)`; to pin a job to GPU N
    use `CUDA_VISIBLE_DEVICES=N … --device cuda:0`. `--device cuda:1` crashes. Shard one job/GPU;
    check `nvidia-smi` for other users first.

11. **Orchestration hygiene.** Chained `waitpid` lanes keep GPUs fed, but: killing a chain's parent
    bash orphans its running child (no follow-on); relaunching a shorter run over a longer one leaves a
    stale-file MIX unless you clear the dir first; "200/200 files" ≠ "this run finished" when an older
    run pre-filled the dir (check newest mtime / the printed summary line).

## Expanding n now (the payoff)
```bash
python -m experiments.run   --n 1000            # print plan (obs + every cell×method), derived from registry
# stage external datasets once: testsets/our1000, ood_celeba/faces (1000), our1000_deblur/cls
python -m experiments.run   --n 1000 --gpus 0,1,2,3 --exec
python -m experiments.score --n 1000 --metrics  # mean±std + guard; fix any `!! GUARD` row, then trust
```
For a NEW dataset (ImageNet-256): add its `TESTSET`/paths + published guards to the registry, add any
new operator's γ/β, and the run/score layer is unchanged. The registry is the only thing that grows.
