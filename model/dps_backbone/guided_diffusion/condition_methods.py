from abc import ABC, abstractmethod
import torch

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm
        
@register_conditioning_method(name='pigdm')
class PseudoinverseGuidedDiffusion(ConditioningMethod):
    """PiGDM (Song et al., ICLR 2023): Pseudoinverse-Guided Diffusion Models.

    Replaces the DPS gradient of ``||y - A x0_hat||`` with the *pseudoinverse-guided* gradient
    ``(d x0_hat / d x_t)^T A^T (r_t^2 A A^T + sigma_y^2 I)^{-1} (y - A x0_hat)``. Under PiGDM's
    orthonormal-rows approximation ``A A^T ~= I`` the preconditioner collapses to the scalar
    ``1 / (r_t^2 + sigma_y^2)``, which needs only the operator's forward/adjoint (adjoint supplied by
    autograd through ``operator.forward``), so it applies to every operator we use.

    ``r_t^2 = (1 - abar_t) / abar_t`` is the standard PiGDM estimate variance. The sampler does not
    pass ``t``, so we track the reverse step internally: ``conditioning`` is called once per step in
    strictly decreasing ``t`` (num_timesteps-1 -> 0). Pass ``alphas_cumprod`` (numpy/torch, len =
    num_timesteps) at construction; call ``reset()`` before each new image.

    NOTE (unvalidated): ``scale`` still needs tuning to reproduce the DPS-backbone's *published* SR
    number on a standard operator before this baseline is trusted (guard discipline in ROADMAP).
    """
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.sigma_y = float(kwargs.get('sigma_y', 0.0))
        ac = kwargs.get('alphas_cumprod', None)
        self.abar = None if ac is None else torch.as_tensor(ac, dtype=torch.float32)
        self.reset()

    def reset(self):
        self._step = None                      # set on first call to (num_timesteps - 1)

    def _r2(self, device):
        if self.abar is None:                  # fall back to a plain unit variance if not supplied
            return 1.0
        if self._step is None:
            self._step = self.abar.numel() - 1
        a = self.abar[self._step].to(device).clamp_min(1e-8)
        self._step = max(self._step - 1, 0)
        return ((1.0 - a) / a)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        r2 = self._r2(x_prev.device)
        residual = measurement - self.operator.forward(x_0_hat, **kwargs)
        norm = torch.linalg.norm(residual)
        # scalar-preconditioned residual (A A^T ~= I):  v = (y - A x0) / (r_t^2 + sigma_y^2)
        v = (residual / (r2 + self.sigma_y ** 2)).detach()
        # vector-Jacobian product: g = (d A x0_hat / d x_prev)^T v  = (d x0_hat/d x_prev)^T A^T v
        s = (self.operator.forward(x_0_hat, **kwargs) * v).sum()
        grad = torch.autograd.grad(outputs=s, inputs=x_prev)[0]
        # PiGDM guidance = (d x0_hat/d x_t)^T A^T (r^2 A A^T + s^2 I)^{-1} (y - A x0). The 1/(r^2+s^2)
        # preconditioner already carries the r_t scaling, so NO extra r_t^2 factor here.
        x_t = x_t + self.scale * grad
        return x_t, norm

@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
