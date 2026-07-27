# IHDM backbone (vendored)

`model_code/` (NCSN++ U-Net, DCT-heat ops, EMA) and `configs/` are vendored from
**AaltoML / generative-inverse-heat-dissipation** (Rissanen, Heinonen, Solin, ICLR 2023), MIT licensed.
Only the pieces needed to build/train/run the IHDM prior are kept; `configs/ffhq/img_size_256_train.py`
was added by this repo. See the upstream project for the original training pipeline and license.
