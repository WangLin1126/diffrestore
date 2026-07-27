from configs.ffhq import img_size_128_maxblur128
import numpy as np


def get_config():
    """256x256 inference config for the FFHQ default_128 checkpoint.

    The checkpoint was trained at 128x128. This config keeps the same network
    hyperparameters and blur schedule family, but builds 256x256 DCT heat
    operators and resizes inputs to 256 for exploratory evaluation.
    """
    config = img_size_128_maxblur128.get_config()
    config.data.image_size = 256
    config.eval.batch_size = 4
    config.model.blur_sigma_max = 256
    config.model.blur_sigma_min = 0.5
    config.model.blur_schedule = np.exp(np.linspace(
        np.log(config.model.blur_sigma_min),
        np.log(config.model.blur_sigma_max),
        config.model.K,
    ))
    config.model.blur_schedule = np.array([0] + list(config.model.blur_schedule))
    return config
