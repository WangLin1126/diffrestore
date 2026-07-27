"""FFHQ 256x256 IHDM training config (compact: 128-config net at 256 res, num_res_blocks=2).

Same NCSN++ hyperparameters and DCT-heat blur family as the released 128 model, but builds
256x256 operators. blur_schedule already has 0 prepended (K_0 = identity).
"""
from configs.ffhq import img_size_128_maxblur128


def get_config():
    config = img_size_128_maxblur128.get_config()
    config.data.image_size = 256
    config.training.batch_size = 16
    return config
