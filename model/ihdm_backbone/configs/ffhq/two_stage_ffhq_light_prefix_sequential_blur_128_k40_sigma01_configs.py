from configs.ffhq import two_stage_ffhq_light_prefix_sequential_blur_k40_sigma01_configs
import numpy as np


def get_config():
    config = two_stage_ffhq_light_prefix_sequential_blur_k40_sigma01_configs.get_config()
    model = config.model

    config.data.image_size = 128
    model.model_channels = 128
    model.num_heads = 1
    model.num_res_blocks = 2
    model.channel_mult = (1, 2, 3, 4, 5)
    model.attention_levels = (2, 3, 4)
    model.dropout = 0.1
    config.training.batch_size = 32
    config.eval.batch_size = 9
    model.blur_sigma_max = 128
    model.blur_sigma_min = 0.5
    positive_blur_schedule = np.exp(
        np.linspace(np.log(model.blur_sigma_min),
                    np.log(model.blur_sigma_max),
                    model.K_blur)
    )
    model.blur_schedule = np.concatenate([[0.0], positive_blur_schedule])
    model.blur_rate = 'custom'
    config.optim.lr = 2e-4
    config.eval.num_samples = 10000
    return config
