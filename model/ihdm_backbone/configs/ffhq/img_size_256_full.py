"""FFHQ 256x256 IHDM training config — ORIGINAL full net (num_res_blocks=3, dropout=0.3).

Same architecture as the paper's default_ffhq_configs (256, channels 128, ch_mult (1,2,3,4,5),
attn (2,3,4), K=200, blur 0.5->128), giving ~211M params — but built on the 128-config's
[0]-prepended blur schedule for consistency with this repo's HeatSchedule / inference.

Use on 80 GB GPUs. Rough capacity: ~14-16 images per 80 GB GPU (fp32, DataParallel), so set
--batch = per_gpu * num_gpus. batch_size=32 matches the original lr=2e-5; scale lr ~linearly for
larger batches. On <=24 GB use img_size_256_train.py (compact, num_res_blocks=2).
"""
from configs.ffhq import img_size_128_maxblur128


def get_config():
    config = img_size_128_maxblur128.get_config()
    config.data.image_size = 256
    config.model.num_res_blocks = 3      # original (compact uses 2)
    config.model.dropout = 0.3           # original (compact uses 0.1)
    config.training.batch_size = 32       # original reference; scale to your GPU count
    return config
