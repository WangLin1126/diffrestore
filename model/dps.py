"""DPS hot-diffusion prior — spare comparison path (NOT a non-hot prior).

DPS uses a variance-preserving (Gaussian-noise) diffusion model, `ffhq_10m.pt`, which does
NOT fit the non-hot `prior x solver` restoration loop (different forward process, model I/O,
architecture). It is kept for head-to-head comparison and driven by its own runner.

Model:      model/dps_backbone/  (guided-diffusion ADM U-Net) + checkpoint/dps/ffhq_10m.pt
Restoration: scripts/run_dps.py   (DPS = x0-estimate guidance with denoiser backprop; see solver/dps.py)
"""
CHECKPOINT = "checkpoint/dps/ffhq_10m.pt"
BACKBONE = "model/dps_backbone"
RUNNER = "scripts/run_dps.py"
