"""All functions and modules related to model definition."""
import torch
import torch.nn as nn
import logging
import numpy as np
from model_code.unet import UNetModel
from model_code import torch_dct


class DCTBlur(nn.Module):

    def __init__(self, blur_sigmas, image_size, device):
        super(DCTBlur, self).__init__()
        self.blur_sigmas = torch.tensor(blur_sigmas).to(device)
        freqs = np.pi*torch.linspace(0, image_size-1,
                                     image_size).to(device)/image_size
        self.frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2

    def forward(self, x, fwd_steps):
        if len(x.shape) == 4:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None, None]
        elif len(x.shape) == 3:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None]
        t = sigmas**2/2
        dct_coefs = torch_dct.dct_2d(x, norm='ortho')
        dct_coefs = dct_coefs * torch.exp(- self.frequencies_squared * t)
        return torch_dct.idct_2d(dct_coefs, norm='ortho')


class TwoStageForwardProcess(nn.Module):
    """Two-stage marginal corruption: DDPM-style noise then heat blur."""

    def __init__(self, noise_sigmas, blur_sigmas, image_size, device,
                 blur_phase_noise_sigma=0.0, blur_phase_signal_scale=1.0,
                 transition_steps=0, blur_noise_schedule=None,
                 sequential_blur=False):
        super(TwoStageForwardProcess, self).__init__()
        noise_sigmas = torch.as_tensor(
            noise_sigmas, dtype=torch.float32, device=device)
        blur_sigmas = torch.as_tensor(
            blur_sigmas, dtype=torch.float32, device=device)
        if noise_sigmas.ndim != 1 or blur_sigmas.ndim != 1:
            raise ValueError("noise_sigmas and blur_sigmas must be 1D schedules")
        if len(noise_sigmas) < 2 or len(blur_sigmas) < 2:
            raise ValueError("Both schedules must include k=0 and at least one positive step")
        if not torch.isclose(noise_sigmas[0], torch.zeros((), device=device)):
            raise ValueError("noise_sigmas[0] must be 0")
        if not torch.isclose(blur_sigmas[0], torch.zeros((), device=device)):
            raise ValueError("blur_sigmas[0] must be 0")
        if torch.any(noise_sigmas < 0) or torch.any(noise_sigmas > 1):
            raise ValueError("noise_sigmas must be in [0, 1]")
        if torch.any(blur_sigmas < 0):
            raise ValueError("blur_sigmas must be non-negative")

        self.K_noise = len(noise_sigmas) - 1
        self.K_blur = len(blur_sigmas) - 1
        self.K = self.K_noise + self.K_blur
        self.register_buffer("noise_sigmas", noise_sigmas)
        self.register_buffer("blur_sigmas", blur_sigmas)
        self.register_buffer(
            "a_boundary", torch.sqrt(1 - noise_sigmas[-1]**2))
        self.register_buffer(
            "blur_phase_noise_sigma",
            torch.as_tensor(blur_phase_noise_sigma, dtype=torch.float32, device=device))
        self.register_buffer(
            "blur_phase_signal_scale",
            torch.as_tensor(blur_phase_signal_scale, dtype=torch.float32, device=device))
        if blur_noise_schedule is None:
            blur_noise_schedule = torch.full(
                (self.K_blur + 1,),
                float(blur_phase_noise_sigma),
                dtype=torch.float32,
                device=device,
            )
            blur_noise_schedule[0] = float(noise_sigmas[-1])
        else:
            blur_noise_schedule = torch.as_tensor(
                blur_noise_schedule, dtype=torch.float32, device=device)
            if blur_noise_schedule.ndim != 1 or len(blur_noise_schedule) != self.K_blur + 1:
                raise ValueError("blur_noise_schedule must have shape (K_blur + 1,)")
        self.register_buffer("blur_noise_schedule", blur_noise_schedule)
        self.transition_steps = int(transition_steps)
        self.sequential_blur = bool(sequential_blur)
        if self.transition_steps < 0 or self.transition_steps >= self.K_noise:
            raise ValueError("transition_steps must be in [0, K_noise)")
        freqs = np.pi*torch.linspace(0, image_size-1,
                                     image_size, device=device)/image_size
        self.register_buffer(
            "frequencies_squared", freqs[:, None]**2 + freqs[None, :]**2)

    def _prepare_steps(self, steps, batch_size, device):
        steps = torch.as_tensor(steps, device=device)
        if steps.ndim == 0:
            steps = steps.expand(batch_size)
        steps = steps.long()
        if steps.shape != (batch_size,):
            raise ValueError(
                "steps must be a scalar or a tensor with shape (batch_size,)")
        if torch.any(steps < 0) or torch.any(steps > self.K):
            raise ValueError(f"steps must be in [0, {self.K}]")
        return steps

    def _view(self, values, ndim):
        return values.reshape(values.shape[0], *([1] * (ndim - 1)))

    def _heat_blur(self, x, blur_indices):
        sigmas = self.blur_sigmas[blur_indices].to(dtype=x.dtype)
        t = self._view(sigmas, x.ndim)**2/2
        frequencies_squared = self.frequencies_squared.to(dtype=x.dtype)
        dct_coefs = torch_dct.dct_2d(x, norm='ortho')
        dct_coefs = dct_coefs * torch.exp(-frequencies_squared * t)
        return torch_dct.idct_2d(dct_coefs, norm='ortho')

    def _bridge_start_step(self):
        return self.K_noise - self.transition_steps + 1

    def _bridge_progress(self, steps, dtype):
        start_step = self._bridge_start_step()
        numer = (steps - start_step + 1).to(dtype=dtype)
        denom = torch.tensor(float(self.transition_steps),
                             device=steps.device, dtype=dtype)
        return numer / denom

    def _boundary_sample(self, x0, noise):
        boundary_sigma = self.noise_sigmas[-1].to(dtype=x0.dtype)
        boundary_scale = self.a_boundary.to(dtype=x0.dtype)
        return boundary_scale * x0 + boundary_sigma * noise

    def mean(self, x0, steps):
        """Return E[x_k | x_0] for mixed noise and blur timesteps."""
        if x0.ndim not in (3, 4):
            raise ValueError("x0 must have shape (B,C,H,W) or (B,H,W)")
        if x0.device != self.noise_sigmas.device:
            raise ValueError(
                "x0 and TwoStageForwardProcess buffers must be on the same device")
        steps = self._prepare_steps(steps, x0.shape[0], x0.device)
        out = torch.empty_like(x0)

        bridge_start_step = self._bridge_start_step()
        pure_noise_mask = steps <= self.K_noise
        if self.transition_steps > 0:
            pure_noise_mask = pure_noise_mask & (steps < bridge_start_step)
        if torch.any(pure_noise_mask):
            noise_steps = steps[pure_noise_mask]
            sigmas = self.noise_sigmas[noise_steps].to(dtype=x0.dtype)
            amplitudes = torch.sqrt(1 - sigmas**2)
            out[pure_noise_mask] = self._view(amplitudes, x0.ndim) * x0[pure_noise_mask]

        if self.transition_steps > 0:
            bridge_mask = (steps >= bridge_start_step) & (steps <= self.K_noise)
            if torch.any(bridge_mask):
                bridge_steps = steps[bridge_mask]
                start_sigma = self.noise_sigmas[bridge_start_step - 1].to(dtype=x0.dtype)
                w = self._bridge_progress(bridge_steps, x0.dtype)
                target_blur_sigma = self.blur_sigmas[1].to(dtype=x0.dtype)
                blur_sigmas = w * target_blur_sigma
                blur_indices = torch.zeros_like(bridge_steps)
                blurred = self._heat_blur(x0[bridge_mask], blur_indices)
                if torch.any(blur_sigmas > 0):
                    t = self._view(blur_sigmas, x0.ndim)**2 / 2
                    frequencies_squared = self.frequencies_squared.to(dtype=x0.dtype)
                    dct_coefs = torch_dct.dct_2d(x0[bridge_mask], norm='ortho')
                    dct_coefs = dct_coefs * torch.exp(-frequencies_squared * t)
                    blurred = torch_dct.idct_2d(dct_coefs, norm='ortho')
                start_scale = torch.sqrt(1 - start_sigma**2)
                end_scale = self.blur_phase_signal_scale.to(dtype=x0.dtype)
                scales = (1 - w) * start_scale + w * end_scale
                out[bridge_mask] = self._view(scales, x0.ndim) * blurred

        blur_mask = steps > self.K_noise
        if torch.any(blur_mask):
            blur_indices = steps[blur_mask] - self.K_noise
            blurred = self._heat_blur(x0[blur_mask], blur_indices)
            out[blur_mask] = self.blur_phase_signal_scale.to(dtype=x0.dtype) * blurred

        return out

    def noise_std(self, steps, ndim=4):
        """Return marginal noise std for each timestep, broadcast-ready."""
        steps = torch.as_tensor(steps, device=self.noise_sigmas.device)
        if steps.ndim == 0:
            steps = steps[None]
        steps = steps.long()
        if torch.any(steps < 0) or torch.any(steps > self.K):
            raise ValueError(f"steps must be in [0, {self.K}]")
        std = torch.empty(steps.shape[0], device=steps.device,
                          dtype=self.noise_sigmas.dtype)
        bridge_start_step = self._bridge_start_step()
        pure_noise_mask = steps <= self.K_noise
        if self.transition_steps > 0:
            pure_noise_mask = pure_noise_mask & (steps < bridge_start_step)
        if torch.any(pure_noise_mask):
            std[pure_noise_mask] = self.noise_sigmas[steps[pure_noise_mask]]
        if self.transition_steps > 0:
            bridge_mask = (steps >= bridge_start_step) & (steps <= self.K_noise)
            if torch.any(bridge_mask):
                bridge_steps = steps[bridge_mask]
                start_sigma = self.noise_sigmas[bridge_start_step - 1]
                end_sigma = self.blur_phase_noise_sigma
                w = self._bridge_progress(bridge_steps, std.dtype)
                std[bridge_mask] = (1 - w) * start_sigma + w * end_sigma
        blur_mask = steps > self.K_noise
        if torch.any(blur_mask):
            blur_indices = steps[blur_mask] - self.K_noise
            std[blur_mask] = self.blur_noise_schedule[blur_indices]
        return std.reshape(std.shape[0], *([1] * (ndim - 1)))

    def sample(self, x0, steps, noise=None):
        """Draw x_k from the configured two-stage marginal."""
        if noise is None:
            noise = torch.randn_like(x0)
        steps = self._prepare_steps(steps, x0.shape[0], x0.device)
        out = torch.empty_like(x0)

        stage1_mask = steps <= self.K_noise
        if torch.any(stage1_mask):
            mean = self.mean(x0[stage1_mask], steps[stage1_mask])
            std = self.noise_std(steps[stage1_mask], x0.ndim).to(dtype=x0.dtype)
            out[stage1_mask] = mean + std * noise[stage1_mask]

        stage2_mask = steps > self.K_noise
        if torch.any(stage2_mask):
            blur_indices = steps[stage2_mask] - self.K_noise
            if self.sequential_blur:
                boundary = self._boundary_sample(x0[stage2_mask], noise[stage2_mask])
                out[stage2_mask] = self._heat_blur(boundary, blur_indices)
            else:
                mean = self.mean(x0[stage2_mask], steps[stage2_mask])
                std = self.noise_std(steps[stage2_mask], x0.ndim).to(dtype=x0.dtype)
                out[stage2_mask] = mean + std * noise[stage2_mask]
        return out

    def sample_pair(self, x0, steps, noise=None):
        """Sample x_k together with a consistent training target for x_{k-1}."""
        if noise is None:
            noise = torch.randn_like(x0)
        steps = self._prepare_steps(steps, x0.shape[0], x0.device)
        current = torch.empty_like(x0)
        previous = torch.empty_like(x0)

        stage1_mask = steps <= self.K_noise
        if torch.any(stage1_mask):
            stage1_steps = steps[stage1_mask]
            current_mean = self.mean(x0[stage1_mask], stage1_steps)
            current_std = self.noise_std(stage1_steps, x0.ndim).to(dtype=x0.dtype)
            current[stage1_mask] = current_mean + current_std * noise[stage1_mask]
            previous[stage1_mask] = self.mean(x0[stage1_mask], stage1_steps - 1)

        stage2_mask = steps > self.K_noise
        if torch.any(stage2_mask):
            blur_indices = steps[stage2_mask] - self.K_noise
            if self.sequential_blur:
                boundary = self._boundary_sample(x0[stage2_mask], noise[stage2_mask])
                current[stage2_mask] = self._heat_blur(boundary, blur_indices)
                previous[stage2_mask] = self._heat_blur(boundary, blur_indices - 1)
            else:
                current_mean = self.mean(x0[stage2_mask], steps[stage2_mask])
                current_std = self.noise_std(
                    steps[stage2_mask], x0.ndim).to(dtype=x0.dtype)
                current[stage2_mask] = current_mean + current_std * noise[stage2_mask]
                previous[stage2_mask] = self.mean(x0[stage2_mask], steps[stage2_mask] - 1)

        return current, previous

    def forward(self, x0, steps):
        return self.mean(x0, steps)


class VEForwardProcess(nn.Module):
    """Discrete variance-exploding forward chain with VE marginals."""

    def __init__(self, noise_sigmas, device):
        super(VEForwardProcess, self).__init__()
        noise_sigmas = torch.as_tensor(
            noise_sigmas, dtype=torch.float32, device=device)
        if noise_sigmas.ndim != 1 or len(noise_sigmas) < 2:
            raise ValueError(
                "noise_sigmas must be a 1D schedule with k=0 and at least one positive step")
        if not torch.isclose(noise_sigmas[0], torch.zeros((), device=device)):
            raise ValueError("noise_sigmas[0] must be 0")
        if torch.any(noise_sigmas < 0):
            raise ValueError("noise_sigmas must be non-negative")
        if torch.any(noise_sigmas[1:] < noise_sigmas[:-1]):
            raise ValueError("noise_sigmas must be non-decreasing")
        self.K = len(noise_sigmas) - 1
        self.register_buffer("noise_sigmas", noise_sigmas)
        deltas = torch.zeros_like(noise_sigmas)
        deltas[1:] = torch.sqrt(torch.clamp(
            noise_sigmas[1:]**2 - noise_sigmas[:-1]**2, min=0.0))
        self.register_buffer("step_noise_stds", deltas)

    def _prepare_steps(self, steps, batch_size, device):
        steps = torch.as_tensor(steps, device=device)
        if steps.ndim == 0:
            steps = steps.expand(batch_size)
        steps = steps.long()
        if steps.shape != (batch_size,):
            raise ValueError(
                "steps must be a scalar or a tensor with shape (batch_size,)")
        if torch.any(steps < 0) or torch.any(steps > self.K):
            raise ValueError(f"steps must be in [0, {self.K}]")
        return steps

    def mean(self, x0, steps):
        if x0.device != self.noise_sigmas.device:
            raise ValueError(
                "x0 and VEForwardProcess buffers must be on the same device")
        self._prepare_steps(steps, x0.shape[0], x0.device)
        return x0

    def noise_std(self, steps, ndim=4):
        steps = torch.as_tensor(steps, device=self.noise_sigmas.device)
        if steps.ndim == 0:
            steps = steps[None]
        steps = steps.long()
        if torch.any(steps < 0) or torch.any(steps > self.K):
            raise ValueError(f"steps must be in [0, {self.K}]")
        std = self.noise_sigmas[steps]
        return std.reshape(std.shape[0], *([1] * (ndim - 1)))

    def sample(self, x0, steps, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        steps = self._prepare_steps(steps, x0.shape[0], x0.device)
        std = self.noise_std(steps, x0.ndim).to(dtype=x0.dtype)
        return x0 + std * noise

    def sample_pair(self, x0, steps, noise=None):
        del noise
        steps = self._prepare_steps(steps, x0.shape[0], x0.device)
        previous = torch.empty_like(x0)
        current = torch.empty_like(x0)

        first_step_mask = steps == 1
        if torch.any(first_step_mask):
            previous[first_step_mask] = x0[first_step_mask]
            first_std = self.step_noise_stds[1].to(dtype=x0.dtype)
            current[first_step_mask] = (
                x0[first_step_mask]
                + first_std * torch.randn_like(x0[first_step_mask])
            )

        later_mask = steps > 1
        if torch.any(later_mask):
            previous_steps = steps[later_mask] - 1
            previous_std = self.noise_std(previous_steps, x0.ndim).to(dtype=x0.dtype)
            previous_noise = torch.randn_like(x0[later_mask])
            previous[later_mask] = (
                x0[later_mask] + previous_std * previous_noise
            )
            step_std = self.step_noise_stds[steps[later_mask]].to(dtype=x0.dtype)
            step_std = step_std.reshape(step_std.shape[0], *([1] * (x0.ndim - 1)))
            current[later_mask] = (
                previous[later_mask]
                + step_std * torch.randn_like(x0[later_mask])
            )

        return current, previous

    def step_std(self, steps):
        steps = torch.as_tensor(steps, device=self.noise_sigmas.device)
        if steps.ndim == 0:
            steps = steps[None]
        steps = steps.long()
        if torch.any(steps < 0) or torch.any(steps > self.K):
            raise ValueError(f"steps must be in [0, {self.K}]")
        return self.step_noise_stds[steps]

    def forward(self, x0, steps):
        return self.mean(x0, steps)


def create_forward_process_from_sigmas(config, sigmas, device):
    forward_process_module = DCTBlur(sigmas, config.data.image_size, device)
    return forward_process_module


def create_ve_forward_process(config, device):
    """Create the discrete VE forward process from config schedules."""
    return VEForwardProcess(config.model.noise_schedule, device)


def create_two_stage_forward_process(config, device):
    """Create the two-stage forward process from config schedules."""
    default_blur_phase_noise_sigma = float(config.model.noise_schedule[-1])
    default_blur_phase_signal_scale = float(
        np.sqrt(max(1 - default_blur_phase_noise_sigma**2, 0.0)))
    return TwoStageForwardProcess(config.model.noise_schedule,
                                  config.model.blur_schedule,
                                  config.data.image_size, device,
                                  blur_phase_noise_sigma=getattr(
                                      config.model, "blur_phase_noise_sigma",
                                      default_blur_phase_noise_sigma),
                                  blur_phase_signal_scale=getattr(
                                      config.model, "blur_phase_signal_scale",
                                      default_blur_phase_signal_scale),
                                  blur_noise_schedule=getattr(
                                      config.model, "blur_noise_schedule", None),
                                  sequential_blur=getattr(
                                      config.model, "sequential_blur", False),
                                  transition_steps=getattr(
                                      config.model, "transition_steps", 0))


def create_forward_process(config, device):
    """Create the configured forward process without changing legacy defaults."""
    forward_process = getattr(config.model, "forward_process", "heat")
    if forward_process == "two_stage":
        return create_two_stage_forward_process(config, device)
    if forward_process == "ve":
        return create_ve_forward_process(config, device)
    return create_forward_process_from_sigmas(
        config, config.model.blur_schedule, device)


"""Utilities related to log-likelihood evaluation"""


def KL(dists, sigma0, sigma1, dim):
    # Calculates a matrix of KL divergences between spherical Gaussian distributions
    # with distances dists between their centers, where dists is a matrix
    return 0.5 * ((sigma0**2/sigma1**2)*dim + (dists)**2/sigma1**2 - dim + 2*dim*np.log(sigma1/sigma0))


def L_K_upperbound(K, trainloader, testloader, blur_module, sigma_inf, sigma_prior, dim,
                   train_size, test_size, device='cpu'):

    # Calculates the upper bound for the term E_q[KL[q(x_K|x_0)|p(x_K)]]
    # in a memory-efficient way, that is, calculates the distances between
    # test and training data points in batches, and uses those distances to calculate
    # the upper bound

    KL_div_upper_bound = torch.zeros(test_size, device=device)
    testdata_count = 0
    count = 0
    for testbatch in testloader:
        logging.info("Batch {}".format(count))
        count += 1
        blur_fwd_steps_test = [K] * len(testbatch[0])
        testbatch = blur_module(testbatch[0].to(
            device), blur_fwd_steps_test).reshape(len(testbatch[0]), -1)
        dists = torch.zeros(train_size, len(testbatch), device=device)
        traindata_count = 0
    # Get distances between the test batch and training data
        for trainbatch in trainloader:
            blur_fwd_steps_train = [K] * len(trainbatch[0])
            trainbatch = blur_module(trainbatch[0].to(
                device), blur_fwd_steps_train).reshape(len(trainbatch[0]), -1)
            dists[traindata_count:traindata_count +
                  len(trainbatch), :] = torch.cdist(trainbatch, testbatch)
            traindata_count += len(trainbatch)
        # Calculate the upper bounds on the KL divergence for each test batch element
        kl_divs = KL(dists, sigma_inf, sigma_prior, testbatch.shape[-1])
        inference_entropy = dim*0.5 * \
            torch.log(
                2*np.pi*torch.exp(torch.tensor([1]))*sigma_inf**2).to(device)
        cross_entropies = kl_divs + inference_entropy
        # log-sum-exp trick
        log_phi = -kl_divs - torch.logsumexp(-kl_divs, 0)[None, :]
        phi = torch.exp(log_phi)
        KL_div_upper_bound_batch = -inference_entropy + \
            (phi * (cross_entropies + log_phi + np.log(train_size))).sum(0)
        KL_div_upper_bound[testdata_count:testdata_count +
                           len(testbatch)] = KL_div_upper_bound_batch
        testdata_count += len(testbatch)
    return KL_div_upper_bound


def neg_ELBO(config, trainloader, testloader, blur_module, sigma, delta, image_size,
             train_size, test_size, model, device='cpu', num_epochs=10):
    """Estimates the terms in the negative evidence lower bound for the model
    num_epochs: Used for the estimation of terms L_k: How many epochs through these?"""

    logging.info("Calculating the upper bound for L_K...")
    L_K_upbound = L_K_upperbound(config.model.K, trainloader, testloader, blur_module, sigma,
                               delta, image_size**2, train_size, test_size, device)
    logging.info("... done! Value {}, len {}".format(
      L_K_upbound, len(L_K_upbound)))

    model_fn = get_model_fn(model, train=False)
    num_dims = image_size**2 * next(iter(trainloader))[0].shape[1]

    # There are K - 1 intermediate scales
    L_others = torch.zeros(config.model.K, device=device)
    mse_losses = torch.zeros(config.model.K, device=device)

    logging.info("Calculating the other terms...")
    with torch.no_grad():
        # Go through the set a few times for more accuracy, not just once
        for i in range(num_epochs):
            for testbatch in testloader:
                testbatch = testbatch[0].to(device).float()
                batch_size = len(testbatch)
                fwd_steps = torch.randint(
                    1, config.model.K, (batch_size,), device=device)
                blurred_batch = blur_module(testbatch, fwd_steps).float()
                less_blurred_batch = blur_module(testbatch, fwd_steps-1).float()
                noise = torch.randn_like(blurred_batch) * sigma
                perturbed_data = noise + blurred_batch
                diff = model_fn(perturbed_data, fwd_steps)
                prediction = perturbed_data + diff
                mse_loss = ((less_blurred_batch - prediction)
                            ** 2).sum((1, 2, 3))
                loss = mse_loss / delta**2
                loss += 2*num_dims*np.log(delta/sigma)
                loss += sigma**2/delta**2*num_dims
                loss -= num_dims
                loss /= 2
                # Normalize so that the significance of these terms matches with L_K and L_0
                # This way, we only go through once for each data point
                loss *= (config.model.K-1)
                mse_loss *= (config.model.K-1)
                L_others.scatter_add_(0, fwd_steps, loss)
                mse_losses.scatter_add_(0, fwd_steps, mse_loss)

        L_others = L_others / (test_size*num_epochs)
        mse_losses = mse_losses / (test_size*num_epochs)

        # Calculate L_0
        for testbatch in testloader:
            testbatch = testbatch[0].to(device).float()
            batch_size = len(testbatch)
            blurred_batch = blur_module(testbatch, [1]).float()
            non_blurred_batch = testbatch
            fwd_steps = torch.ones(batch_size, device=device)
            noise = torch.randn_like(blurred_batch) * sigma
            perturbed_data = noise + blurred_batch
            diff = model_fn(perturbed_data, fwd_steps)
            prediction = perturbed_data + diff
            mse_loss = ((non_blurred_batch - prediction)**2).sum((1, 2, 3))
            loss = 0.5*mse_loss/delta**2
            # Normalization constant
            loss += num_dims*np.log(delta*np.sqrt(2*np.pi))
            L_others[0] += loss.sum()
            mse_losses[0] += mse_loss.sum()
        L_others[0] = L_others[0] / test_size
        mse_losses[0] = mse_losses[0] / test_size

    logging.info("... Done! Values {}".format(L_others))
    return L_K_upbound.detach().cpu(), L_others.detach().cpu(), mse_losses.detach().cpu()


"""The next two functions based on https://github.com/yang-song/score_sde"""


def create_model(config, device_ids=None):
    """Create the model."""
    model = UNetModel(config)
    model = model.to(config.device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model


def get_model_fn(model, train=False):
    """A wrapper for using the model in eval or train mode"""
    def model_fn(x, fwd_steps):
        """Args:
                x: A mini-batch of input data.
                fwd_steps: A mini-batch of conditioning variables for different levels.
        """
        if not train:
            model.eval()
            return model(x, fwd_steps)
        else:
            model.train()
            return model(x, fwd_steps)
    return model_fn
