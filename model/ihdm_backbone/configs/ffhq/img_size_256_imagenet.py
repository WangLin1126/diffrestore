"""ImageNet-256 IHDM training config — ADM-matched capacity (~589M params).

Sized to be comparable to OpenAI's `256x256_diffusion_uncond.pt` (552.8M params), so a
SMDC-vs-DDRM/DPS/DDNM comparison on ImageNet is *prior-capacity-matched* and any gap is
attributable to the data-consistency scheme, not model size.

Knobs vs the 211M FFHQ full config (img_size_256_full):
  model_channels 128 -> 192   (GroupNorm needs mc divisible by 32; 192 is the tightest match)
  num_res_blocks   3 -> 4      (192/nrb=4 -> 588.9M; 192/nrb=3 -> 474M is the cheaper fallback)
  dropout        0.3 -> 0.1    (1.28M ImageNet images: far less overfitting risk than 70k FFHQ)
  random_flip  False -> True   (standard ImageNet augmentation; applied in the dataloader)

Same DCT-heat blur family (0.5 -> 128, K=200, [0]-prepended) as the released models for
inference compatibility with this repo's HeatSchedule. batch_size here is the *reference* for
the config's lr; the training script sets the real per-GPU batch. Scale lr ~linearly with the
global batch (world_size * per_gpu_batch).
"""
from configs.ffhq import img_size_128_maxblur128


def get_config():
    config = img_size_128_maxblur128.get_config()
    config.data.image_size = 256
    config.data.dataset = "ImageNet"
    config.data.random_flip = True
    config.model.model_channels = 192      # 128 -> 192  (mc must be divisible by 32)
    config.model.num_res_blocks = 4        # 3 -> 4      => 588.9M params (~ADM 552.8M)
    config.model.dropout = 0.1             # 0.3 -> 0.1
    config.training.batch_size = 32        # reference for lr=2e-5; script overrides per-GPU
    return config
