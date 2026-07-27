# Mathematical Theory
## Scale-Matched Data Consistency for Non-Hot Diffusion Priors

*Companion to `TASK.md`. This document states the model, gives the derivations in full, fills the
gaps left in the original note, and makes every assumption explicit. Equations that the
implementation must satisfy numerically are marked **[TEST]**.*

---

## 0. Notation and conventions

| Symbol | Meaning |
|---|---|
| $\mathbf x_0\in\mathbb R^d$ | unknown clean image, $d=C\cdot H\cdot W$ (for FFHQ-256: $3\cdot256\cdot256$) |
| $\mathbf y\in\mathbb R^m$ | measurement |
| $\mathbf A\in\mathbb R^{m\times d}$ | known linear forward operator |
| $\mathbf A^\top$ | its adjoint (real transpose); for our operators it is available in closed form |
| $\mathbf n\in\mathbb R^m$ | measurement noise, $\mathbf n\sim\mathcal N(\mathbf 0,\boldsymbol\Sigma_n)$ |
| $\{\mathbf K_t\}_{t\in[0,T]}$ | known linear **image-space** degradation family, $\mathbf K_t:\mathbb R^d\to\mathbb R^d$ |
| $\mathbf x_t=\mathbf K_t\mathbf x_0$ | degraded state at level $t$ (deterministic path; stochastic variant in §7) |
| $\mathbf L_t:\mathbb R^m\to\mathbb R^m$ | **measurement-space** companion operator |
| $\mathbf R_\theta$ | pretrained non-hot reverse prior (one reverse-degradation step) |
| $\Phi$ | orthonormal spectral basis (DFT for circular, DCT-II for Neumann) |
| $\hat{(\cdot)}=\Phi(\cdot)$ | spectral coefficients; $\hat{\mathbf A}=\Phi\mathbf A\Phi^{-1}$ is diagonal when $\mathbf A$ is $\Phi$-diagonal |

"Non-hot" (a.k.a. *cold*) diffusion means the forward path adds **no thermal Gaussian noise**; it
applies a deterministic, structured, and usually physically interpretable degradation
$\mathbf K_t$ (here: Gaussian heat smoothing). Contrast with "hot" diffusion
$\mathbf x_t=\alpha_t\mathbf x_0+\sigma_t\boldsymbol\epsilon_t$.

Convention: $t$ **decreases** from $T$ (most degraded) to $0$ (clean). Discrete schedule
$T=t_N>\dots>t_0=0$.

---

## 1. Problem statement

**Given** the measurement $\mathbf y$, the known operator $\mathbf A$, the noise law $\boldsymbol\Sigma_n$,
and a pretrained non-hot prior over natural images accessed through its degradation family $\{\mathbf K_t\}$,
**recover** $\mathbf x_0$ from

$$\mathbf y=\mathbf A\mathbf x_0+\mathbf n,\qquad \mathbf n\sim\mathcal N(\mathbf 0,\boldsymbol\Sigma_n).\tag{1}$$

Two equivalent targets:

- **Bayesian:** sample / estimate $p(\mathbf x_0\mid\mathbf y)\propto p(\mathbf y\mid\mathbf x_0)\,p(\mathbf x_0)$.
- **MAP / variational:** minimize $\;\mathcal J(\mathbf x_0)=\tfrac12\lVert\mathbf y-\mathbf A\mathbf x_0\rVert^2_{\boldsymbol\Sigma_n^{\dagger}}+\mathcal R(\mathbf x_0)$, with $\mathcal R$ an implicit learned prior.

The prior enters **only** through the reverse degradation model $\mathbf R_\theta$; it is never
required to know $\mathbf A$. This is the plug-and-play property.

### 1.1 Why intermediate likelihoods are hard for hot diffusion

For a hot state $\mathbf x_t=\alpha_t\mathbf x_0+\sigma_t\boldsymbol\epsilon_t$,

$$\mathbf A\mathbf x_t=\alpha_t\mathbf A\mathbf x_0+\sigma_t\mathbf A\boldsymbol\epsilon_t,$$

so comparing $\mathbf A\mathbf x_t$ against $\mathbf y$ is contaminated by the diffusion noise
$\sigma_t\mathbf A\boldsymbol\epsilon_t$, which is unrelated to the physical model (1). The exact
time-$t$ likelihood

$$p_t(\mathbf y\mid\mathbf x_t)=\int p(\mathbf y\mid\mathbf x_0)\,p(\mathbf x_0\mid\mathbf x_t)\,d\mathbf x_0\tag{2}$$

is intractable, which is why DPS-type methods approximate it through a single clean estimate
$\widehat{\mathbf x}_0(\mathbf x_t)$ and must backpropagate through the denoiser Jacobian.

### 1.2 The opening created by non-hot diffusion

If instead $\mathbf x_t=\mathbf K_t\mathbf x_0$ with $\mathbf K_t$ **known and linear**, the state is
not a random noisy latent — it is *the same image at a known degradation scale*. This lets us push
the measurement to the same scale and obtain a likelihood **directly on $\mathbf x_t$**, with no
$\widehat{\mathbf x}_0$ and no denoiser backprop. The rest of this document makes that precise.

---

## 2. Core construction: scale matching by intertwining

**Definition 2.1 (companion operator).** $\mathbf L_t$ is a *companion* of $\mathbf K_t$ under $\mathbf A$ if it satisfies the **intertwining relation**

$$\boxed{\;\mathbf L_t\,\mathbf A=\mathbf A\,\mathbf K_t\;}\tag{3}$$

**Proposition 2.2 (scale-matched observation).** *Define $\mathbf y_t:=\mathbf L_t\mathbf y$. If (3) holds and $\mathbf x_t=\mathbf K_t\mathbf x_0$, then*

$$\boxed{\;\mathbf y_t=\mathbf A\mathbf x_t+\mathbf n_t,\qquad \mathbf n_t=\mathbf L_t\mathbf n.\;}\tag{4}$$

*Proof.* $\mathbf y_t=\mathbf L_t(\mathbf A\mathbf x_0+\mathbf n)=(\mathbf L_t\mathbf A)\mathbf x_0+\mathbf L_t\mathbf n
\overset{(3)}{=}\mathbf A(\mathbf K_t\mathbf x_0)+\mathbf L_t\mathbf n=\mathbf A\mathbf x_t+\mathbf n_t.$ ∎

Equation (4) is a **bona fide linear-Gaussian observation model for the current state $\mathbf x_t$**.
It replaces the intractable route (2) with exact algebra. Everything downstream is a consequence.

---

## 3. Existence and construction of $\mathbf L_t$

Equation (3) does not always have a solution; when it does, it may be non-unique. Three regimes.

### 3.1 Commuting operators (the clean case, $m=d$)

**Lemma 3.1.** *If $\mathbf A$ and $\mathbf K_t$ act on the same space and commute, $\mathbf A\mathbf K_t=\mathbf K_t\mathbf A$, then $\mathbf L_t=\mathbf K_t$ solves (3).*

*Proof.* $\mathbf L_t\mathbf A=\mathbf K_t\mathbf A=\mathbf A\mathbf K_t$. ∎

### 3.2 Simultaneous diagonalization (how commuting is guaranteed in practice)

**Lemma 3.2.** *Let $\Phi$ be an orthonormal basis with $\mathbf A=\Phi^{-1}\hat{\mathbf A}\,\Phi$ and
$\mathbf K_t=\Phi^{-1}\hat{\mathbf K}_t\,\Phi$, where $\hat{\mathbf A},\hat{\mathbf K}_t$ are diagonal.
Then $\mathbf A\mathbf K_t=\mathbf K_t\mathbf A$ and $\mathbf L_t=\mathbf K_t$ works. Moreover $\mathbf A^\top=\Phi^{-1}\hat{\mathbf A}^{*}\Phi$.*

Two image-domain instances, both realized exactly by an FFT-family transform:

- **(F) Circular convolution — $\Phi=$ DFT.** $\mathbf A=\mathcal F^{-1}\mathrm{diag}(H)\mathcal F$,
  $\mathbf K_t=\mathcal F^{-1}\mathrm{diag}(\hat K_t)\mathcal F$. All shift-invariant circular convolutions
  commute. This is the original note's "FFT specialization."

- **(N) Neumann-boundary heat — $\Phi=$ DCT-II.**  Let $\boldsymbol\Delta$ be the discrete Laplacian
  with Neumann (reflecting) boundary conditions; it is diagonalized by the orthonormal 2-D DCT-II,
  with eigenvalues $\lambda_{k\ell}$. Any function of $\boldsymbol\Delta$ is DCT-diagonal, so **all
  such operators mutually commute**. Heat smoothing to blur std $\sigma$ is
  $$\hat K[\,k,\ell\,]=\exp\!\Big(-\tfrac{\sigma^2}{2}\,\lambda_{k\ell}\Big),\qquad
    \lambda_{k\ell}=\big(\pi k/H\big)^2+\big(\pi \ell/W\big)^2\;\text{(IHDM convention).}\tag{5}$$

> **Design consequence (this is the key practical choice).** *Instantiate the deblurring operator
> $\mathbf A$ in the **same spectral basis $\Phi$** as the prior's degradation $\mathbf K_t$.* Then
> $\mathbf A$ and $\mathbf K_t$ commute **exactly** (to machine precision), $\mathbf L_t=\mathbf K_t$
> is exact, $\mathbf A^\top$ is exact, and all algebraic assumptions are verifiable — the setting the
> original note (§4.5) asks for. Since IHDM's $\mathbf K_t$ is Neumann/DCT heat (5), we take $\mathbf A$
> to be a **Gaussian blur of std $\sigma_A$ realized in the same DCT basis**:
> $$\hat A[k,\ell]=\exp\!\big(-\tfrac{\sigma_A^2}{2}\lambda_{k\ell}\big),\qquad \mathbf A^\top=\mathbf A.\tag{6}$$
> Gaussian deblurring is a standard inverse-problem benchmark, so this is not a toy: it is the exact,
> testable realization of the method for a real task.

### 3.3 General linear operators (when $\mathbf A,\mathbf K_t$ do not commute)

**Proposition 3.3.** *A solution $\mathbf L_t$ of (3) exists iff $\operatorname{range}(\mathbf A\mathbf K_t)\subseteq\operatorname{range}(\mathbf A)$, equivalently $\mathbf A\mathbf K_t=\mathbf A\mathbf K_t\mathbf A^{\dagger}\mathbf A$. When $\mathbf A$ has full row rank (surjective), the choice*
$$\mathbf L_t=\mathbf A\,\mathbf K_t\,\mathbf A^{\dagger}\tag{7}$$
*is a valid companion; if $\mathbf A$ is also injective it is unique.*

*Proof.* If $\mathbf L_t$ exists then $\mathbf A\mathbf K_t=\mathbf L_t\mathbf A$ has columns in
$\operatorname{range}(\mathbf A)$. Conversely, under the range condition, $\mathbf A\mathbf A^\dagger$ is
the orthogonal projector onto $\operatorname{range}(\mathbf A)$, so (7) gives
$\mathbf L_t\mathbf A=\mathbf A\mathbf K_t\mathbf A^\dagger\mathbf A=\mathbf A\mathbf K_t$. ∎

This formalizes the "potentially supported with task-specific construction" tasks (super-resolution,
Fourier/MRI/CT sensing): one must supply $\mathbf A^\dagger$ (or an approximate right inverse) and
verify the range condition. Super-resolution in particular has $m<d$, so $\mathbf L_t=\mathbf K_t$ is
**dimensionally invalid** and (7) (or a task-specific low-pass companion) is required. **Approximate
companions** $\mathbf L_t\approx\mathbf A\mathbf K_t\mathbf A^\dagger$ produce a biased-but-usable
guidance term; the bias is $\lVert\mathbf L_t\mathbf A-\mathbf A\mathbf K_t\rVert$ **[TEST]**.

---

## 4. Exact Gaussian state-matched likelihood

From (4), $\mathbf n_t=\mathbf L_t\mathbf n$ is a linear image of a Gaussian, hence Gaussian:

$$\mathbf n_t\sim\mathcal N(\mathbf 0,\boldsymbol\Sigma_t),\qquad
\boxed{\;\boldsymbol\Sigma_t=\mathbf L_t\boldsymbol\Sigma_n\mathbf L_t^\top.\;}\tag{8}$$

### 4.1 Non-degenerate case ($\boldsymbol\Sigma_t\succ0$)

$$p(\mathbf y_t\mid\mathbf x_t)=\mathcal N(\mathbf y_t;\mathbf A\mathbf x_t,\boldsymbol\Sigma_t),\qquad
\mathcal D_t^{\mathrm{exact}}(\mathbf x_t)=\tfrac12(\mathbf y_t-\mathbf A\mathbf x_t)^\top\boldsymbol\Sigma_t^{-1}(\mathbf y_t-\mathbf A\mathbf x_t),$$

$$\boxed{\;\nabla_{\mathbf x_t}\log p(\mathbf y_t\mid\mathbf x_t)=\mathbf A^\top\boldsymbol\Sigma_t^{-1}(\mathbf y_t-\mathbf A\mathbf x_t).\;}\tag{9}$$

No clean-image predictor and **no backpropagation through $\mathbf R_\theta$** appear in (9).

### 4.2 Degenerate case — the gap the original note glosses over

Heat operators are **low-pass**: $\hat K_t[k]\to0$ at high frequency, so $\mathbf L_t=\mathbf K_t$ (and
hence $\boldsymbol\Sigma_t$) is **singular / severely ill-conditioned**. Then:

1. $\mathbf n_t$ is supported on $V_t:=\operatorname{range}(\mathbf L_t)$; the density (9) is defined only
   on $V_t$, and $\mathbf y_t-\mathbf A\mathbf x_t\in V_t$ must be enforced (it holds automatically on the
   scale manifold, see below). The correct object is the pseudo-inverse quadratic
   $$\mathcal D_t^{\mathrm{exact}}=\tfrac12(\mathbf y_t-\mathbf A\mathbf x_t)^\top\boldsymbol\Sigma_t^{\dagger}(\mathbf y_t-\mathbf A\mathbf x_t),\qquad
   \nabla_{\mathbf x_t}=\mathbf A^\top\boldsymbol\Sigma_t^{\dagger}(\mathbf y_t-\mathbf A\mathbf x_t).\tag{10}$$
2. **Numerically $\boldsymbol\Sigma_t^{\dagger}$ must be support-thresholded**: invert only where
   $\hat K_t[k]$ exceeds a floor; otherwise the $1/|\hat K_t|^2$ weight explodes on the noise. This is
   exactly what the **regularized mode** $(\boldsymbol\Sigma_t+\lambda_t\mathbf I)^{-1}$ tames.
3. **Sufficiency caveat (resolves the note's Open Question 6.1).** When $\mathbf L_t$ is non-invertible,
   conditioning on $\mathbf y_t=\mathbf L_t\mathbf y$ *discards* the components of $\mathbf y$ in
   $\ker(\mathbf L_t)$. Hence in general
   $$p(\mathbf x_t\mid\mathbf y_t)\neq p(\mathbf x_t\mid\mathbf y).$$
   $\mathbf y_t$ is a **sufficient statistic** for $\mathbf x_0$ *within the transformed model (4)*, but
   not for the original $\mathbf y$ unless $\mathbf L_t$ is invertible. Therefore the sequence
   $\{p(\mathbf x_t\mid\mathbf y_t)\}_t$ is a **homotopy of progressively stronger objectives**
   (a continuation), **not** the marginals of one exact posterior diffusion. This is why the honest
   description of the full algorithm is *continuation-MAP / plug-and-play splitting* (see §9, §11).

---

## 5. Convolutional specialization (spectral, diagonal)

Work in the shared basis $\Phi$ (write $\hat{}$ for coefficients). With $\mathbf L_t=\mathbf K_t$ and (6):

$$\hat y_t[k]=\hat K_t[k]\,\hat y[k],\qquad
\text{residual }\;\hat r_t[k]=\hat K_t[k]\hat y[k]-\hat A[k]\hat x_t[k].\tag{11}$$

**On the scale manifold** $\hat x_t[k]=\hat K_t[k]\hat x_0[k]$ (state genuinely at level $t$), (11) becomes

$$\hat r_t[k]=\hat K_t[k]\big(\hat y[k]-\hat A[k]\hat x_0[k]\big)=\hat K_t[k]\,\hat N[k],\qquad
\hat N:=\hat{\mathbf y}-\hat{\mathbf A}\hat{\mathbf x}_0=\hat{\mathbf n}.\tag{12}$$

i.e. the scale-matched residual is the **original measurement residual filtered by $\hat K_t$**. All
three guidance modes are diagonal:

| Mode | Weight $\hat W_t[k]$ | Objective in terms of original residual $\hat N$ | Interpretation |
|---|---|---|---|
| surrogate `L2` | $1$ | $\tfrac12\sum_k \lvert\hat K_t[k]\rvert^2\,\lvert\hat N[k]\rvert^2$ | **frequency-annealed** likelihood (low-pass emphasis) |
| regularized | $\dfrac{1}{\sigma_n^2\lvert\hat K_t[k]\rvert^2+\lambda_t}$ | interpolates | tempered whitening |
| exact | $\dfrac{1}{\sigma_n^2\lvert\hat K_t[k]\rvert^2}$ (on $\hat K_t\neq0$) | $\tfrac{1}{2\sigma_n^2}\!\!\sum\limits_{k:\hat K_t[k]\neq0}\!\!\lvert\hat N[k]\rvert^2$ | **original** likelihood, restricted to the surviving band |

### 5.1 Exact whitening = original likelihood on a growing band (proof)

Put $\boldsymbol\Sigma_n=\sigma_n^2\mathbf I$, so $\boldsymbol\Sigma_t=\sigma_n^2\mathbf K_t\mathbf K_t^\top$ has
spectrum $\sigma_n^2|\hat K_t[k]|^2$. Using (12),

$$\mathcal D_t^{\mathrm{exact}}=\tfrac12\sum_{k:\hat K_t\neq0}\frac{|\hat r_t[k]|^2}{\sigma_n^2|\hat K_t[k]|^2}
=\tfrac12\sum_{k:\hat K_t\neq0}\frac{|\hat K_t[k]|^2|\hat N[k]|^2}{\sigma_n^2|\hat K_t[k]|^2}
=\frac{1}{2\sigma_n^2}\!\!\sum_{k:\hat K_t[k]\neq0}\!\!|\hat y[k]-\hat A[k]\hat x_0[k]|^2.\tag{13}$$

**Reading of (13).** Exact whitening recovers the *original* full-resolution likelihood, but summed
**only over frequencies $\hat K_t$ has not annihilated**. Coarse-to-fine therefore emerges from the
**support** $\{k:\hat K_t[k]\neq0\}$ **growing** as $t\to0$ — *not* from spectral reweighting. This
reconciles the note's two claims: exact mode "removes the scale-selective weighting" (true — the
weights inside the band cancel) yet is still coarse-to-fine (true — via the band). The surrogate keeps
an explicit $|\hat K_t|^2$ **reweighting** on top of the whole band; it is a genuinely different
(tempered) objective, and must be labeled as such. As $t\to0$, $\hat K_t\to\hat K_0$ and both objectives
approach the full measurement term.

---

## 6. Guidance step and its stability

One preconditioned gradient-ascent correction of the data term at level $t$:

$$\boxed{\;\mathbf x_t^{+}=\mathbf x_t+\eta_t\,\mathbf A^\top\mathbf W_t(\mathbf y_t-\mathbf A\mathbf x_t),\qquad
\mathbf W_t\in\{\boldsymbol\Sigma_t^{\dagger},\,(\boldsymbol\Sigma_t+\lambda_t\mathbf I)^{-1},\,\mathbf I\}.\;}\tag{14}$$

**Stability (replenishes the note's Open Question 6.7).** The inner iteration
$\mathbf x\mapsto\mathbf x+\eta\mathbf A^\top\mathbf W(\mathbf y_t-\mathbf A\mathbf x)$ has Jacobian
$\mathbf I-\eta\,\mathbf A^\top\mathbf W\mathbf A$. Non-divergence requires the spectral radius $\le1$:

$$0<\eta_t<\frac{2}{\rho(\mathbf A^\top\mathbf W_t\mathbf A)}
=\frac{2}{\max_k \hat W_t[k]\,|\hat A[k]|^2}.\tag{15}$$

- surrogate ($\hat W=1$): $\;0<\eta_t<2/\max_k|\hat A[k]|^2=2/\lVert\mathbf A\rVert_2^2$ (matches the note);
- regularized: $\;0<\eta_t<2\big/\max_k\dfrac{|\hat A[k]|^2}{\sigma_n^2|\hat K_t[k]|^2+\lambda_t}$.

A convenient **residual-normalized** schedule, robust across $t$:
$\eta_t=\eta_0\,\dfrac{\lVert\mathbf x_t\rVert_2}{\lVert\mathbf A^\top\mathbf W_t(\mathbf y_t-\mathbf A\mathbf x_t)\rVert_2+\varepsilon}$.
Interaction with the learned prior step may require a stricter constant than (15); treat $\eta_0$ as the one tuned scalar.

---

## 7. Stochastic non-hot path (needed because IHDM adds noise)

IHDM's forward process is **not** purely deterministic: it is
$\mathbf x_t=\mathbf K_t\mathbf x_0+\boldsymbol\xi_t,\ \boldsymbol\xi_t\sim\mathcal N(\mathbf 0,\sigma_\xi^2\mathbf I)$
with a small $\sigma_\xi$ (config `model.sigma = 0.01`). We must account for it (Open Question 6.3).

Substituting $\mathbf K_t\mathbf x_0=\mathbf x_t-\boldsymbol\xi_t$ into $\mathbf y_t=\mathbf L_t\mathbf y=\mathbf A\mathbf K_t\mathbf x_0+\mathbf L_t\mathbf n$:

$$\mathbf y_t=\mathbf A\mathbf x_t+\underbrace{(\mathbf L_t\mathbf n-\mathbf A\boldsymbol\xi_t)}_{\tilde{\mathbf n}_t}.\tag{16}$$

Treating $\mathbf x_t$ as the given point produced by the prior (so $\boldsymbol\xi_t$ is exogenous) and
$\boldsymbol\xi_t\perp\mathbf n$, the **effective covariance** is

$$\boxed{\;\boldsymbol\Sigma_t^{\mathrm{eff}}=\mathbf L_t\boldsymbol\Sigma_n\mathbf L_t^\top+\sigma_\xi^2\,\mathbf A\mathbf A^\top.\;}\tag{17}$$

In the shared basis (white $\mathbf n$): $\hat\Sigma_t^{\mathrm{eff}}[k]=\sigma_n^2|\hat K_t[k]|^2+\sigma_\xi^2|\hat A[k]|^2$.
Two useful readings:

- The $\sigma_\xi^2|\hat A[k]|^2$ term **floors** the covariance where $\hat K_t[k]\to0$, so (17) is a
  *physically motivated* version of the regularizer $\lambda_t$ — it prevents the exact-mode blow-up of §4.2 automatically.
- **Caveat (honest):** $\boldsymbol\xi_t$ is literally part of $\mathbf x_t$, so conditioning on $\mathbf x_t$
  correlates $\tilde{\mathbf n}_t$ with the state; (17) uses an independence approximation, exact only in the
  limit where the prior's $\mathbf x_t$ is taken as an external anchor. For $\sigma_\xi=0.01$ the correction
  is small; the surrogate and regularized modes are insensitive to it. We therefore (i) default to
  surrogate/regularized modes, and (ii) offer (17) as the "exact-stochastic" covariance. **[TEST]** empirical
  $\operatorname{Cov}(\tilde{\mathbf n}_t)$ vs (17).

---

## 8. Terminal initialization

A deterministic blur has **no universal Gaussian terminal**; $\mathbf x_T$ must be set explicitly.

**Matched-measurement initializer (recommended).** Take $\mathbf x_T=\mathbf y_T=\mathbf L_T\mathbf y$.
Justification in the shared basis: for the surviving low band, a normalized blur has $\hat A[k]\approx1$
(DC gain $1$), and $\hat n_T[k]=\hat K_T[k]\hat n[k]\to0$, so

$$\hat y_T[k]=\hat A[k]\hat K_T[k]\hat x_0[k]+\hat K_T[k]\hat n[k]\;\approx\;\hat K_T[k]\hat x_0[k]=\hat x_T[k].$$

Thus $\mathbf x_T\approx\mathbf y_T$ up to the tiny blur-vs-identity gap on the few surviving modes — a
consistent, data-informed start requiring no pseudo-inverse. Alternatives: **prior sampling** from the
model's terminal distribution; **coarse least-squares** $\min_{\mathbf x}\lVert\mathbf y_T-\mathbf A\mathbf x\rVert^2$;
**data-informed** $\mathbf K_T\mathbf A^\dagger\mathbf y$ when a stable $\mathbf A^\dagger$ exists.
Initialization sensitivity is an explicit experiment (Open Question 6.5).

**Residual-$\mathbf K_0$ note.** IHDM's least-blurred state uses `blur_sigma_min = 0.5`, so
$\mathbf K_0=\text{heat}(\sigma{=}0.5)\neq\mathbf I$ exactly. For most frequencies $\hat K_0[k]\approx1$
(sub-pixel blur), so we treat $\mathbf K_0\approx\mathbf I$ and report $\widehat{\mathbf x}_0$ as IHDM's
$\mathbf x_0$. An optional final micro-deconvolution by $\mathbf K_0^{-1}$ (thresholded) is available but
usually negligible. This is an **assumption**, listed in §10.

---

## 9. The algorithm (operator splitting) and what it targets

Prior-first splitting, one outer step $t_i\to t_{i-1}$:

$$
\begin{aligned}
\textbf{(prior)}\quad & \widetilde{\mathbf x}_{t_{i-1}}=\mathbf R_\theta(\mathbf x_{t_i},t_i,t_{i-1}),\\
\textbf{(match)}\quad & \mathbf y_{t_{i-1}}=\mathbf L_{t_{i-1}}\mathbf y,\\
\textbf{(correct)}\quad & \mathbf x_{t_{i-1}}=\widetilde{\mathbf x}_{t_{i-1}}
      +\eta_{t_{i-1}}\mathbf A^\top\mathbf W_{t_{i-1}}\big(\mathbf y_{t_{i-1}}-\mathbf A\widetilde{\mathbf x}_{t_{i-1}}\big)\ \ (\times\,\text{inner\_steps}),
\end{aligned}
$$

$$\boxed{\;\mathbf x_{t'}\leftarrow\underbrace{\mathbf R_\theta(\mathbf x_t,t,t')}_{\text{learned reverse prior}}
+\underbrace{\eta_{t'}\mathbf A^\top\mathbf W_{t'}\big(\mathbf L_{t'}\mathbf y-\mathbf A\,\mathbf R_\theta(\mathbf x_t,t,t')\big)}_{\text{scale-matched data consistency}},\qquad t'<t.\;}\tag{18}$$

Prior-first is preferred so the correction is evaluated on a state explicitly tagged to scale $t'$.

**What (18) is.** A **plug-and-play / continuation-MAP** iteration: a learned reverse-degradation
(prior/proximal-like) step followed by a preconditioned gradient step on the scale-$t'$ data term
$\mathcal D_{t'}$. Each $\mathcal D_{t'}$ is a valid negative log-likelihood *of the transformed model (4)*.

**What (18) is not (claim boundary, per note §3.9, §6.1).** It is **not**, without further conditions,
an exact posterior sampler for $p(\mathbf x_0\mid\mathbf y)$, because (i) $\mathbf y_t$ is not a sufficient
statistic for $\mathbf y$ when $\mathbf L_t$ is singular (§4.2), and (ii) the deterministic
$\mathbf R_\theta$ + deterministic correction yields a *point estimate conditioned on initialization*,
not a calibrated posterior. Exact/asymptotic posterior sampling would additionally require: a calibrated
stochastic reverse kernel/score for $p_t(\mathbf x_t)$; a consistent family of conditional marginals;
correctly scaled stochastic terms; and a proof that the splitting targets the posterior. We do **not**
claim these; we claim a principled, testable data-consistency scheme with an exact per-step likelihood.

---

## 10. Assumptions — stated explicitly

**Exactly required (verified numerically):**

1. **A1** $\mathbf A$ known, linear, fixed at inference; $\mathbf A^\top$ available. **[TEST: adjoint]**
2. **A2** Known linear path $\mathbf x_t=\mathbf K_t\mathbf x_0$ with $\mathbf K_0\approx\mathbf I$.
3. **A3** An implementable $\mathbf L_t$ satisfies $\mathbf L_t\mathbf A=\mathbf A\mathbf K_t$. **[TEST: intertwining]**
   *(Guaranteed exactly by the shared-basis construction of §3.2 / (6).)*
4. **A4** Consistent discretization: padding/boundary, spectral convention, and image normalization
   identical between prior training and inference. **[TEST: limiting-scale $\mathbf K_0,\mathbf L_0\approx\mathbf I$, $\mathbf y_0\approx\mathbf y$]**
5. **A5** Known/approximated noise law: $\boldsymbol\Sigma_t$ (or $\boldsymbol\Sigma_t^{\mathrm{eff}}$)
   computable, or a surrogate residual explicitly selected. **[TEST: noise covariance]**

**Approximations accepted for the IHDM-FFHQ-256 instantiation (documented, bounded, not hidden):**

6. **B1 (residual $\mathbf K_0$)** $\mathbf K_0=\text{heat}(0.5)\approx\mathbf I$; final report ignores sub-pixel blur (§8).
7. **B2 (stochastic path)** $\sigma_\xi=0.01$ path noise handled via (17) or absorbed into $\lambda_t$; default modes are insensitive.
8. **B3 (state consistency)** the learned reverse output is assumed to lie near the scale manifold,
   $\widetilde{\mathbf x}_t\approx\mathbf K_t\widetilde{\mathbf x}_0$. If the model drifts off-manifold the
   state-matched likelihood is misspecified. **[TEST: state semantics]** on held-out images.
9. **B4 (approximate companion, only for non-commuting tasks §3.3)** guidance is biased by
   $\lVert\mathbf L_t\mathbf A-\mathbf A\mathbf K_t\rVert$; not used in the deblurring PoC (there it is $0$).

---

## 11. One-screen summary

$$
\boxed{\;
\begin{aligned}
\mathbf y&=\mathbf A\mathbf x_0+\mathbf n, &
\mathbf x_t&=\mathbf K_t\mathbf x_0,&
\mathbf L_t\mathbf A&=\mathbf A\mathbf K_t,\\
\mathbf y_t&=\mathbf L_t\mathbf y=\mathbf A\mathbf x_t+\mathbf L_t\mathbf n,&
\boldsymbol\Sigma_t&=\mathbf L_t\boldsymbol\Sigma_n\mathbf L_t^\top,&
\nabla_{\mathbf x_t}\log p(\mathbf y_t\!\mid\!\mathbf x_t)&=\mathbf A^\top\boldsymbol\Sigma_t^{\dagger}(\mathbf y_t-\mathbf A\mathbf x_t),\\
\end{aligned}\;}
$$

with update (18). Exactly realizable for commuting convolutions via the **shared DCT basis** of §3.2;
extends to general linear tasks via the companion (7). Honest status: **learned reverse-degradation
prior + scale-matched data consistency** = continuation-MAP / plug-and-play, with an exact per-step
transformed likelihood — not (yet) a proven exact posterior sampler.
