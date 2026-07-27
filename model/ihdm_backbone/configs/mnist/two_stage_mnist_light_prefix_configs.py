from configs.mnist import default_mnist_configs
import numpy as np


def _denoise_sampling_stds(noise_schedule):
    eps = 1e-12
    alpha_bar = 1 - noise_schedule**2
    beta = np.zeros_like(noise_schedule)
    beta[1:] = 1 - alpha_bar[1:] / np.maximum(alpha_bar[:-1], eps)
    posterior_variance = np.zeros_like(noise_schedule)
    posterior_variance[1:] = (
        beta[1:] * (1 - alpha_bar[:-1]) / np.maximum(1 - alpha_bar[1:], eps)
    )
    posterior_variance = np.maximum(posterior_variance, 0)
    posterior_stds = np.sqrt(posterior_variance)
    posterior_stds[0] = 0
    posterior_stds[1] = 0
    return posterior_stds


def _ddpm_linear_noise_schedule(boundary_noise_sigma, num_steps, beta_start=1e-6):
    alpha_bar_end = max(1 - boundary_noise_sigma**2, 1e-12)

    def alpha_bar_from_beta_end(beta_end):
        betas = np.linspace(beta_start, beta_end, num_steps)
        betas = np.clip(betas, 1e-8, 0.999)
        return np.prod(1 - betas)

    low = beta_start
    high = 0.999
    for _ in range(100):
        mid = 0.5 * (low + high)
        if alpha_bar_from_beta_end(mid) > alpha_bar_end:
            low = mid
        else:
            high = mid

    betas = np.linspace(beta_start, high, num_steps)
    betas = np.clip(betas, 1e-8, 0.999)
    alpha_bar = np.concatenate([[1.0], np.cumprod(1 - betas)])
    noise_schedule = np.sqrt(np.maximum(1 - alpha_bar, 0))
    noise_schedule[-1] = boundary_noise_sigma
    return noise_schedule


def get_config():
    config = default_mnist_configs.get_default_configs()
    default_model = default_mnist_configs.get_default_configs().model
    model = config.model

    model.forward_process = "two_stage"
    model.K_noise = 10
    model.K_blur = 90
    model.K = model.K_noise + model.K_blur
    model.transition_steps = 0

    model.boundary_noise_sigma = default_model.sigma
    model.sigma = default_model.sigma
    model.stage1_prediction = "mean"

    model.blur_sigma_min = default_model.blur_sigma_min
    model.blur_sigma_max = default_model.blur_sigma_max
    model.blur_phase_noise_sigma = default_model.sigma
    model.blur_phase_signal_scale = 1.0
    model.blur_sampling_noise = 1.25 * model.blur_phase_noise_sigma

    model.noise_schedule = _ddpm_linear_noise_schedule(
        model.boundary_noise_sigma, model.K_noise)
    positive_blur_schedule = np.exp(
        np.linspace(
            np.log(model.blur_sigma_min),
            np.log(model.blur_sigma_max),
            model.K_blur
        )
    )
    model.blur_schedule = np.concatenate([[0.0], positive_blur_schedule])
    model.denoise_sampling_stds = _denoise_sampling_stds(model.noise_schedule)

    return config
