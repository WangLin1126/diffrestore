from configs.mnist import default_mnist_configs
import numpy as np


def _ve_noise_schedule(sigma_min, sigma_max, num_steps):
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must be positive")
    if sigma_max < sigma_min:
        raise ValueError("sigma_max must be >= sigma_min")
    positive_sigmas = np.exp(
        np.linspace(np.log(sigma_min), np.log(sigma_max), num_steps)
    )
    return np.concatenate([[0.0], positive_sigmas])


def get_config():
    config = default_mnist_configs.get_default_configs()
    model = config.model

    model.forward_process = "ve"
    model.prediction_target = "previous_mean"
    model.K = 100
    model.sigma_min = 0.01
    model.sigma_max = 1.0
    model.noise_schedule = _ve_noise_schedule(
        model.sigma_min, model.sigma_max, model.K)
    model.sampling_eta = 1.25

    # Keep legacy fields populated so the existing training scripts continue to work.
    model.sigma = model.sigma_max
    model.blur_schedule = model.noise_schedule

    return config
