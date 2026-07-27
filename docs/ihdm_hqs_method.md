# Scale-Matched Bayesian Posterior Sampling for Inverse Heat Dissipation

## Motivation

The original Inverse Heat Dissipation Model (IHDM) is an unconditional generative model that progressively reverses the heat equation.

For an inverse problem

$$
y=Ax_0,
$$

the objective is to incorporate measurement information into the reverse generation process.

The central question is **not**

> How can MAP be introduced?

Instead, the more fundamental question is

> **On which latent variable can the measurement likelihood be naturally defined?**

Once such a likelihood exists, Bayesian posterior inference follows directly from Bayes' rule.

The proposed framework answers this question by exploiting the scale correspondence induced by heat diffusion, which enables the observation to be propagated to the same diffusion scale as the reverse latent state.

---

# 1. Reverse Transition in IHDM

At reverse step $k$, IHDM defines the probabilistic transition

$$
p_\theta(u_{k-1}\mid u_k)
=
\mathcal N
\left(
\mu_\theta(u_k),
\delta_k^2I
\right).
$$

Here,

- $u_k$ denotes the current reverse sample;
- $\mu_\theta(u_k)$ is the conditional mean predicted by the neural network;
- $\delta_k$ controls the reverse sampling variance.

Sampling is performed as

$$
u_{k-1}
=
\mu_\theta(u_k)
+
\delta_k\epsilon_k,
\qquad
\epsilon_k\sim\mathcal N(0,I).
$$

Therefore, the network prediction naturally provides the prior mean for the next reverse state.

---

# 2. Scale Correspondence Through Heat Diffusion

Assume the degradation model

$$
y=Ax_0,
$$

and let $G_t$ denote the heat diffusion operator.

More generally, rather than requiring literal commutativity, we assume the existence of a measurement-domain propagation operator satisfying

$$
AG_t
=
\widetilde G_tA,
$$

or approximately,

$$
AG_t
\approx
\widetilde G_tA.
$$

Since the reverse latent state satisfies

$$
x_t=G_tx_0,
$$

we immediately obtain

$$
Ax_t
=
\widetilde G_ty.
$$

Therefore every heat-diffused latent state possesses a deterministic observation counterpart at the same diffusion scale.

Unlike conventional diffusion, the reverse latent variable is therefore **measurement-compatible**.

More importantly, this scale correspondence establishes an observation model directly on the reverse latent state,

$$
Ax_t
=
\widetilde G_ty,
$$

making the likelihood

$$
p(y\mid x_t)
$$

naturally well-defined without introducing an auxiliary clean-image estimate.

---

### Examples of Scale Correspondence

| Inverse problem | Scale correspondence | Treatment |
|-----------------|---------------------|-----------|
| Gaussian / Defocus / Motion Deblurring | $AG_t=G_tA$ | Exact commutativity (FFT/DCT diagonalization) |
| MRI | Frequency-domain propagation | Direct propagation in $k$-space |
| CT | $AG_t=\widetilde G_tA$ | Heat propagation in the sinogram domain |
| Super-resolution | $AG_t\approx\widetilde G_tA$ | Approximate measurement-domain propagation |
| Image inpainting | Approximate correspondence | Approximate likelihood |
| General linear inverse problems | $AG_t\approx\widetilde G_tA$ | Approximate scale correspondence with modeling error absorbed into the likelihood variance |

The proposed framework therefore does not require exact commutativity for every inverse problem. The essential requirement is the existence of a known or sufficiently accurate measurement-domain propagation operator that propagates the observation to the same diffusion scale as the latent state.

---

# 3. Scale-Matched Measurement Likelihood

At reverse step $k-1$, the desired structured latent state satisfies

$$
x_{k-1}
\approx
G_{t_{k-1}}x_0.
$$

Using the scale correspondence,

$$
Ax_{k-1}
\approx
\widetilde G_{t_{k-1}}y.
$$

The measurement likelihood is therefore defined as

$$
p(y\mid x_{k-1})
\propto
\exp
\left(
-
\frac{1}{2\sigma_y^2}
\|
Ax_{k-1}
-
\widetilde G_{t_{k-1}}y
\|^2
\right).
$$

Unlike conventional diffusion, the data-consistency term compares two quantities defined at exactly the same diffusion scale,

$$
Ax_{k-1}
\qquad\text{and}\qquad
\widetilde G_{t_{k-1}}y.
$$

Consequently, Bayesian inference can be performed directly on the reverse latent variable itself.

---

# 4. Bayesian Posterior and MAP Estimation

The reverse heat process provides the Gaussian prior

$$
p_\theta(x_{k-1}\mid u_k)
=
\mathcal N
\left(
\mu_\theta(u_k),
\delta_k^2I
\right).
$$

Combining this prior with the scale-matched likelihood yields the posterior

$$
p(x_{k-1}\mid u_k,y)
\propto
p(y\mid x_{k-1})
\,p_\theta(x_{k-1}\mid u_k).
$$

The corresponding MAP estimator is

$$
x_{k-1}^{\mathrm{MAP}}
=
\arg\max_x
p(x\mid u_k,y),
$$

or equivalently,

$$
\boxed{
\min_x
\;
\frac1{2\delta_k^2}
\|
x-\mu_\theta(u_k)
\|^2
+
\frac1{2\sigma_y^2}
\|
Ax-
\widetilde G_{t_{k-1}}y
\|^2.
}
$$

The objective naturally consists of two complementary terms.

- **Prior term**, which keeps the corrected latent state close to the reverse prediction of IHDM.

- **Likelihood term**, which enforces consistency with the propagated observation at the corresponding diffusion scale.

For linear forward operators, this is a standard quadratic MAP problem. When the operators are diagonalizable in the Fourier or DCT domain, the solution can be obtained efficiently frequency-by-frequency. For more general forward operators, the same objective may be optimized using standard iterative inverse-problem solvers.

It is worth emphasizing that **MAP itself is not the contribution of the proposed framework**.

Once the measurement likelihood is naturally defined on the reverse latent state, the Bayesian posterior and its MAP estimator follow immediately from Bayes' rule.

---

# 5. MAP Correction in Practice

The MAP objective may be solved either exactly or approximately, depending on the forward operator.

For convolutional degradations and other operators diagonalizable in the Fourier or DCT domain, the solution can be obtained efficiently by solving the quadratic objective frequency-by-frequency.

For more general inverse problems, one may instead perform a small number of gradient-based updates initialized from the reverse prediction,

$$
x^{(0)}
=
\mu_\theta(u_k).
$$

The numerical solver itself is independent of the probabilistic formulation.

The essential point is that the optimization variable is the reverse latent state predicted by IHDM, while the data term compares it directly with the observation propagated to the same diffusion scale.

---

# 6. Reverse Sampling

After posterior correction, reverse sampling proceeds as

$$
u_{k-1}
=
x_{k-1}^{\mathrm{MAP}}
+
\delta_k\epsilon_k,
\qquad
\epsilon_k\sim\mathcal N(0,I).
$$

Therefore, the original Gaussian reverse transition remains unchanged.

Only the conditional mean is corrected using measurement information, while the stochastic nature of reverse diffusion is fully preserved.

The reverse pipeline becomes

```text
Current sample u_k
        │
        ▼
Reverse prior prediction
μθ(u_k)
        │
        ▼
Scale-matched MAP correction
        │
        ▼
Gaussian sampling
        │
        ▼
Next sample u_{k-1}
```

The proposed method therefore modifies only the reverse mean, while leaving the probabilistic sampling mechanism of IHDM intact.

---

# 7. Why Heat Diffusion Naturally Supports Bayesian Posterior Inference

Both conventional denoising diffusion and heat diffusion ultimately perform Bayesian posterior sampling by combining a learned reverse prior with measurement information.

The fundamental difference is **not** Bayesian inference itself, but the latent variable on which the likelihood is naturally defined.

## 7.1 Conventional Diffusion

In DDPM, the reverse latent state takes the form

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
$$

which contains an unknown realization of diffusion noise.

Although the observation satisfies

$$
y=Ax_0,
$$

there is generally no deterministic relationship between

$$
Ax_t
$$

and

$$
y.
$$

Consequently, the likelihood

$$
p(y\mid x_t)
$$

cannot be naturally formulated on the reverse latent variable.

Modern diffusion methods therefore estimate auxiliary variables such as

$$
x_0,\qquad
\epsilon,\qquad
\text{or}\qquad
v,
$$

perform posterior correction on these variables, and then reconstruct the reverse transition.

Thus Bayesian inference and reverse sampling are performed on different variables.

---

## 7.2 Heat Diffusion

In contrast, heat diffusion propagates the structured latent state

$$
x_t
=
G_tx_0.
$$

Through the scale correspondence

$$
AG_t
=
\widetilde G_tA,
$$

the reverse latent state admits the deterministic observation model

$$
Ax_t
=
\widetilde G_ty.
$$

The measurement likelihood can therefore be defined directly on the latent variable propagated by the reverse process,

$$
p(y\mid x_t).
$$

Bayesian inference is consequently performed on exactly the same latent variable that is sampled during reverse generation.

The overall inference pipeline becomes

```text
Reverse prior
        │
        ▼
Scale-matched likelihood
        │
        ▼
Bayesian posterior
        │
        ▼
MAP correction
        │
        ▼
Gaussian sampling
```

Unlike conventional denoising diffusion, no auxiliary clean-image estimate is required.

The reverse prior, the measurement model, and the stochastic reverse transition are all defined on the same structured latent representation.

---

# 8. Key Insight

The proposed framework is built upon three aligned components.

1. **A probabilistic reverse prior**

$$
p_\theta(x_{k-1}\mid u_k),
$$

provided by the reverse heat diffusion process.

2. **A structured latent representation**

$$
x_{k-1}
\approx
G_{t_{k-1}}x_0,
$$

whose physical meaning is preserved throughout the reverse process.

3. **A scale-matched observation model**

$$
AG_t
=
\widetilde G_tA,
$$

which propagates the measurement to the same diffusion scale as the latent state.

Together, these properties establish a deterministic observation model directly on the reverse latent variable,

$$
Ax_t
=
\widetilde G_ty,
$$

making the likelihood

$$
p(y\mid x_t)
$$

naturally well-defined.

The contribution is therefore **not** introducing a new MAP formulation.

Instead, it identifies a measurement-compatible latent representation on which the learned reverse prior, the physical measurement model, and the reverse Markov process are intrinsically aligned.

Once this latent representation is established, Bayesian posterior inference follows directly from Bayes' rule, while the original stochastic reverse sampling process remains completely unchanged.