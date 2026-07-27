# Methods & Models: DPS vs SMDC vs IHDM-HQS

Three ways to solve linear image inverse problems (heat / Gaussian deblurring) with a diffusion prior,
evaluated on held-out face images. **SMDC and IHDM-HQS share the same core idea** (a likelihood defined
directly on the blurred latent via scale matching) and differ only in the *data-consistency solver*;
**DPS is a different paradigm** (a hot noise-diffusion prior guided through a clean-image estimate).

## At a glance

| | **DPS** | **SMDC** (this repo) | **IHDM-HQS** (heat_diffusion) |
|---|---|---|---|
| Paradigm | hot diffusion + posterior sampling | non-hot + scale-matched data consistency | non-hot + scale-matched MAP |
| Prior type | hot (Gaussian-noise, VP) | non-hot (heat blur) | non-hot (heat blur, IHDM) |
| Likelihood defined on | `x̂₀` (Tweedie clean estimate) | `xₜ` (current blur scale) | `xₜ` (current blur scale) |
| Data-consistency step | `∇‖y − A x̂₀‖²`, backprop through net | 1-step gradient `Aᵀ Wₜ(Lₜy − A xₜ)` | full per-step **MAP** via HQS |
| Needs `x̂₀` in the likelihood? | **yes** | no | no |
| Backprop through the prior? | **yes** | no | no |
| Model / architecture | ADM U-Net, **94M**, learn_sigma | our IHDM-style x₀-U-Net, **62M / 75M** | NCSN++ IHDM U-Net, **160M** |
| Training data / resolution | **FFHQ 256²** | CelebA-HQ **128²** / FFHQ **256²** | **FFHQ 128²** |
| Reverse steps | 1000 (DDPM) | 200 (heat) | 200 (heat) |
| Sampling | stochastic (posterior) | deterministic (point estimate) | stochastic (posterior) |

## Methods

**DPS — Diffusion Posterior Sampling.** Standard variance-preserving (noise) diffusion prior. Each reverse
step forms a clean estimate `x̂₀(xₜ)` (Tweedie) and adds the likelihood gradient `∇_{xₜ}(1/2σ²)‖y − A x̂₀‖²`,
differentiating through the denoiser. Model: `ffhq_10m.pt` (guided-diffusion ADM U-Net, `learn_sigma`, 256²) —
the DPS paper's FFHQ checkpoint. *Cost:* denoiser-Jacobian backprop; early-step `x̂₀` bias. *Strength:* strong
generative prior → perceptually realistic output.

**SMDC — Scale-Matched Data Consistency (ours).** Non-hot (heat/cold) prior with state `xₜ = Kₜ x₀`. A
companion operator `Lₜ` (with `Lₜ A = A Kₜ`) transforms the measurement to the current scale:
`yₜ = Lₜ y = A xₜ + nₜ` — an **exact likelihood on `xₜ`, with no `x̂₀` and no backprop through the prior.**
Reverse = IHDM-style x₀-predictor + re-degrade (`x0_step_down`); correction = one preconditioned gradient step
`Aᵀ Wₜ(yₜ − A xₜ)` in three weighting modes (surrogate / regularized / exact). Priors trained in this repo
(CelebA-128 62M, FFHQ-256 75M) — but **any heat prior can be plugged in** (verified with the IHDM checkpoint).
*Strength:* cheap, no denoiser backprop, and works for general (even non-diagonalizable) `A`.

**IHDM-HQS (heat_diffusion).** Same scale-matching (`A xₜ = G̃ₜ y`) as SMDC, but replaces the 1-step gradient
with a **full per-step MAP solve** balancing the IHDM reverse mean `μ_θ` and the scale-matched measurement:
`min_x (λ_p/2δ²)‖x − μ_θ‖² + (λ_y/2σ_y²)‖A x − G̃ₜ y‖²`, solved by Half-Quadratic Splitting (closed-form in the
DCT domain), keeping IHDM's stochastic reverse. Model: their trained IHDM (NCSN++ U-Net, 160M, 170k steps,
FFHQ 128², `[0,1]` range).

## Results (held-out faces, heat blur σ=4, noise 0.05)

**DPS vs SMDC** — CelebA-HQ **256**; DPS(`ffhq_10m`) vs SMDC(FFHQ-256 prior). *System* comparison (priors
differ in size/training):

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| SMDC | **25.97** | **0.711** | 0.417 |
| DPS | 24.00 | 0.687 | **0.185** |

→ perception–distortion tradeoff: **SMDC wins fidelity** (PSNR/SSIM), **DPS wins realism** (LPIPS).

**SMDC vs IHDM-HQS** — CelebA **128**, **same IHDM prior** (isolates the *solver*):

| | PSNR | SSIM | LPIPS | Meas.resid |
|---|---|---|---|---|
| SMDC (1-step gradient) | 24.67 | 0.787 | 0.225 | 0.112 |
| IHDM-HQS (MAP) | **25.39** | **0.818** | **0.182** | **0.094** |

→ **MAP wins every metric.** Since `A` is DCT-diagonal, the MAP is a *closed-form per-frequency solve at the
same per-step cost* as the gradient step, so it dominates here. SMDC's 1-step gradient (which amplifies noise,
worse with more inner steps) is preferable only for **non-diagonalizable `A`** (super-res, inpainting, MRI),
where the MAP would need expensive iterations.

## Takeaways

1. **SMDC ≈ IHDM-HQS in idea, differ in solver.** Scale-matched likelihood on the heat latent is the shared
   insight; 1-step gradient (cheap, general) vs full MAP (better when `A` is diagonalizable).
2. **DPS is orthogonal** — hot prior + `x̂₀` guidance: perceptually strong but needs a large prior and
   backprop through the denoiser.
3. **Prior-agnostic.** SMDC drives our own priors *and* the heat_diffusion IHDM checkpoint unchanged.

---

## Derivation: one-step gradient (SMDC) vs full per-step MAP (IHDM-HQS)

SMDC and IHDM-HQS optimize the **same per-step objective**; SMDC truncates it at a single gradient
iteration while IHDM-HQS solves it. This section makes that precise.

### Notation and reverse time $t$

Let the forward blur **time** be $t\ge 0$ ($t=0$ = clean, larger $t$ = more blur), discretized as
$t_0=0<t_1<\dots<t_K$ with blur std $\sigma_B(t_k)$. The degraded state is

$$\mathbf x^{(k)}=\mathbf K_{t_k}\mathbf x_0,\qquad
\mathbf K_{t}= \Phi^{\!\top}\operatorname{diag}\!\big(\hat g_t\big)\Phi,\quad
\hat g_t(\omega)=\exp\!\big(-\tfrac12\sigma_B(t)^2\,\omega^2\big),$$

with $\Phi$ the orthonormal DCT (heat operators are DCT-diagonal). The forward blur $\mathbf A$ is likewise
DCT-diagonal with transfer $\hat a(\omega)$, and the companion $\mathbf L_t=\mathbf K_t$ satisfies
$\mathbf L_t\mathbf A=\mathbf A\mathbf K_t$.

**Restoration runs in reverse time:** the index decreases $k=K\to 0$, i.e. we traverse $t$ downward
$t_K\to t_{K-1}\to\dots\to t_0=0$. One reverse step maps level $k$ to level $k-1$ (from $t_k$ to
$t_{k-1}<t_k$). Write $\mathbf u^{(k)}$ for the running reverse iterate at level $k$.

The learned prior supplies a **Gaussian reverse kernel** with the network's predicted less-blurred mean,

$$p_\theta\big(\mathbf x\mid\mathbf u^{(k)}\big)=\mathcal N\!\big(\mathbf x;\ \boldsymbol\mu_k,\ \delta^2\mathbf I\big),
\qquad \boldsymbol\mu_k:=\boldsymbol\mu_\theta\big(\mathbf u^{(k)}\big),$$

where $\delta$ is the reverse sampling std, and (IHDM) $\boldsymbol\mu_k=\mathbf u^{(k)}+F_\theta(\mathbf u^{(k)},k)$,
or (SMDC Interface B) $\boldsymbol\mu_k=\mathbf u^{(k)}-\mathbf K_{t_k}\hat{\mathbf x}_0+\mathbf K_{t_{k-1}}\hat{\mathbf x}_0$
with $\hat{\mathbf x}_0=F_\theta(\mathbf u^{(k)},k)$.

**Scale-matched measurement.** Blur the observation to level $k-1$:

$$\mathbf b_k:=\mathbf L_{t_{k-1}}\mathbf y=\mathbf A\,\mathbf x^{(k-1)}+\mathbf n_{k-1},\qquad
\mathbf n_{k-1}=\mathbf L_{t_{k-1}}\mathbf n,$$

giving a Gaussian likelihood **directly on the level-$(k{-}1)$ state**,
$p(\mathbf b_k\mid\mathbf x)=\mathcal N(\mathbf b_k;\mathbf A\mathbf x,\boldsymbol\Sigma_{k-1})$,
$\boldsymbol\Sigma_{k-1}=\mathbf L_{t_{k-1}}\boldsymbol\Sigma_n\mathbf L_{t_{k-1}}^{\!\top}$
($\approx\sigma_y^2\mathbf I$ in the isotropic/surrogate model).

### The restoration procedure (shared skeleton)

$$
\begin{aligned}
&\mathbf u^{(K)}\leftarrow \text{terminal init } \mathbf L_{t_K}\mathbf y\\
&\textbf{for } k=K,\dots,1:\\
&\quad \boldsymbol\mu_k=\boldsymbol\mu_\theta(\mathbf u^{(k)}) &&\text{(1) prior reverse mean at level }k{-}1\\
&\quad \mathbf b_k=\mathbf L_{t_{k-1}}\mathbf y &&\text{(2) scale-matched measurement}\\
&\quad \mathbf x^{(k-1)}=\textsc{Correct}(\boldsymbol\mu_k,\mathbf b_k) &&\text{(3) data consistency — methods differ here}\\
&\quad \mathbf u^{(k-1)}=\mathbf x^{(k-1)}\ (+\,\delta\boldsymbol\epsilon_k \text{ if stochastic}) &&\text{(4) optional reverse noise}\\
&\textbf{return } \mathbf u^{(0)}
\end{aligned}
$$

### The per-step objective

By Bayes, the per-step posterior at level $k-1$ combines the prior kernel and the scale-matched likelihood,
$p(\mathbf x\mid\mathbf u^{(k)},\mathbf b_k)\propto \mathcal N(\mathbf x;\boldsymbol\mu_k,\delta^2\mathbf I)\,
\mathcal N(\mathbf b_k;\mathbf A\mathbf x,\boldsymbol\Sigma_{k-1})$, whose negative log is the convex quadratic

$$\boxed{\ J_k(\mathbf x)=\underbrace{\tfrac{1}{2\delta^2}\lVert\mathbf x-\boldsymbol\mu_k\rVert^2}_{\text{prior-proximal}}
+\underbrace{\tfrac12\lVert\mathbf A\mathbf x-\mathbf b_k\rVert^2_{\boldsymbol\Sigma_{k-1}^{-1}}}_{\text{data}}\ }\tag{$\dagger$}$$

### IHDM-HQS $=$ exact minimizer of $(\dagger)$ — full per-step MAP

Setting $\nabla J_k=\mathbf 0$:

$$\tfrac1{\delta^2}(\mathbf x-\boldsymbol\mu_k)+\mathbf A^{\!\top}\boldsymbol\Sigma_{k-1}^{-1}(\mathbf A\mathbf x-\mathbf b_k)=\mathbf 0
\ \Rightarrow\
\boxed{\ \mathbf x^\star_{k-1}=\Big[\tfrac1{\delta^2}\mathbf I+\mathbf A^{\!\top}\boldsymbol\Sigma_{k-1}^{-1}\mathbf A\Big]^{-1}
\Big[\tfrac1{\delta^2}\boldsymbol\mu_k+\mathbf A^{\!\top}\boldsymbol\Sigma_{k-1}^{-1}\mathbf b_k\Big]\ }\tag{$\ddagger$}$$

This is the **full per-step MAP**. For general $\mathbf A$ the inverse is expensive, so it is realized by
**Half-Quadratic Splitting** (auxiliary $\mathbf z$, penalty $\rho\uparrow$):
$\mathbf x\leftarrow[\tfrac1{\delta^2}\boldsymbol\mu_k+\rho\mathbf z]/[\tfrac1{\delta^2}+\rho]$,
$\mathbf z\leftarrow[\mathbf A^{\!\top}\boldsymbol\Sigma^{-1}\mathbf A+\rho\mathbf I]^{-1}[\mathbf A^{\!\top}\boldsymbol\Sigma^{-1}\mathbf b_k+\rho\mathbf x]$,
which converges to $(\ddagger)$. Because $\mathbf A,\boldsymbol\Sigma_{k-1}$ are DCT-diagonal here, $(\ddagger)$ is
**closed-form per frequency** (write $\sigma_y^2$ for the noise, $\lambda_y$ the data weight):

$$\hat{\mathbf x}^\star_{k-1}(\omega)=
\frac{\tfrac1{\delta^2}\,\hat{\boldsymbol\mu}_k(\omega)+\tfrac{\lambda_y}{\sigma_y^2}\,\hat a(\omega)\,\hat{\mathbf b}_k(\omega)}
{\tfrac1{\delta^2}+\tfrac{\lambda_y}{\sigma_y^2}\,\lvert\hat a(\omega)\rvert^2}.$$

### SMDC $=$ one gradient step of $(\dagger)$ from $\boldsymbol\mu_k$ — one-step gradient

SMDC initializes at the prior mean, $\mathbf x^{(0)}=\boldsymbol\mu_k$, and takes **one** gradient-descent step
$\mathbf x^{(1)}=\mathbf x^{(0)}-\eta\,\nabla J_k(\mathbf x^{(0)})$. The gradient at $\boldsymbol\mu_k$ is

$$\nabla J_k(\boldsymbol\mu_k)=\underbrace{\tfrac1{\delta^2}(\boldsymbol\mu_k-\boldsymbol\mu_k)}_{=\,\mathbf 0}
+\mathbf A^{\!\top}\boldsymbol\Sigma_{k-1}^{-1}(\mathbf A\boldsymbol\mu_k-\mathbf b_k)
=\mathbf A^{\!\top}\boldsymbol\Sigma_{k-1}^{-1}(\mathbf A\boldsymbol\mu_k-\mathbf b_k).$$

The **prior-proximal gradient vanishes at $\boldsymbol\mu_k$** (the iterate starts exactly at the proximal
center), so the first step uses only the data gradient:

$$\boxed{\ \mathbf x_{k-1}=\boldsymbol\mu_k+\eta\,\mathbf A^{\!\top}\mathbf W_{k-1}\big(\mathbf b_k-\mathbf A\boldsymbol\mu_k\big)\ }
\qquad \mathbf W\in\{\boldsymbol\Sigma^{-1}_{k-1}\ (\text{exact}),\ (\boldsymbol\Sigma_{k-1}+\lambda\mathbf I)^{-1}\ (\text{reg}),\ \mathbf I\ (\text{surrogate})\}.$$

This is exactly SMDC's update — one preconditioned gradient-ascent step on the log-likelihood, evaluated at
the prior mean, with no explicit prior term because it is inactive at the initialization.

### The precise relationship and its consequences

- **SMDC is the first gradient iteration of the full MAP objective $(\dagger)$ started from $\boldsymbol\mu_k$;
  IHDM-HQS is its converged minimizer $(\ddagger)$.** Same objective — SMDC truncates at one step, IHDM-HQS solves it.
- **Why more SMDC inner steps $\neq$ MAP.** SMDC's inner loop iterates the *data-only* gradient
  $\mathbf x\!\leftarrow\!\mathbf x+\eta\mathbf A^{\!\top}\mathbf W(\mathbf b_k-\mathbf A\mathbf x)$ (it drops the proximal
  term after step 1). Its fixed point is the weighted least-squares / pseudo-inverse of $\lVert\mathbf A\mathbf x-\mathbf b_k\rVert^2_{\mathbf W}$,
  which **overfits the measurement noise** — nothing pulls back toward $\boldsymbol\mu_k$. (Empirically, inner$\times3$
  degrades PSNR/LPIPS.) The MAP keeps $\tfrac1{\delta^2}\lVert\mathbf x-\boldsymbol\mu_k\rVert^2$, which regularizes
  toward the prior and suppresses noise.
- **Cost.** For DCT-diagonalizable $\mathbf A$, $(\ddagger)$ is one closed-form per-frequency solve — the *same*
  per-step cost as SMDC's single gradient step — so IHDM-HQS strictly dominates (our result: 25.39 vs 24.67 dB,
  LPIPS 0.182 vs 0.225). For **non-diagonalizable $\mathbf A$** (super-res, inpainting, MRI), $(\ddagger)$/HQS needs
  several inner solves per step while SMDC keeps a single cheap $\mathbf A^{\!\top}(\cdot)$ — that is SMDC's regime.

---

*Details: `EXPERIMENTS.md` (full logs), `MATH.md` (SMDC theory), `heat_diffusion/{file.md, STRUCTURE.md}`;
figures `compare_dps_smdc/figure_dps_vs_smdc.png`, `compare_ihdm/figure_smdc_vs_ihdmhqs.png`.*
