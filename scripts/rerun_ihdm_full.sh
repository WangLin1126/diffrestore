#!/usr/bin/env bash
# Re-run every IHDM+HQS experiment used in the report with the new full 211M checkpoint.
# Gaussian (DCT) x3 noise + motion (DFT) x3 noise. Reuses the identical shared observations.
set -u
cd "$(dirname "$0")/.."
CKPT=checkpoint/ihdm/ihdm_ffhq256_full.pth
CFG=img_size_256_full
COMMON="--prior ihdm --ckpt $CKPT --ihdm_config $CFG --image_size 256 --n 16"
mkdir -p results/rerun_logs

# ---- Wave 1 : 4 jobs on 4 GPUs ----
# Gaussian, DCT, restore.py --solver hqs. sigma_y matched to observation noise.
CUDA_VISIBLE_DEVICES=0 python scripts/restore.py $COMMON --solver hqs \
  --clean_dir results/gaussian/clean --observation_dir results/gaussian/observation \
  --blur_sigma 4 --noise_std 0.05 --sigma_y 0.05 --device cuda:0 \
  --out results/gaussian/ihdm_hqs > results/rerun_logs/g05.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/restore.py $COMMON --solver hqs \
  --clean_dir results/gaussian_n0p10/clean --observation_dir results/gaussian_n0p10/observation \
  --blur_sigma 4 --noise_std 0.10 --sigma_y 0.10 --device cuda:0 \
  --out results/gaussian_n0p10/ihdm_hqs > results/rerun_logs/g10.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/restore.py $COMMON --solver hqs \
  --clean_dir results/gaussian_n0p20/clean --observation_dir results/gaussian_n0p20/observation \
  --blur_sigma 4 --noise_std 0.20 --sigma_y 0.20 --device cuda:0 \
  --out results/gaussian_n0p20/ihdm_hqs > results/rerun_logs/g20.log 2>&1 &
# Motion, DFT, restore_motion_dft.py. sigma_y=0.05.
CUDA_VISIBLE_DEVICES=3 python scripts/restore_motion_dft.py $COMMON \
  --clean_dir results/motion/clean --observation_dir results/motion/observation \
  --kernel_npy results/motion/kernel.npy --sigma_y 0.05 --device cuda:0 \
  --out results/motion/ihdm_hqs_dft > results/rerun_logs/m05.log 2>&1 &
wait
echo "=== wave 1 done ==="

# ---- Wave 2 : remaining 2 motion noise levels ----
CUDA_VISIBLE_DEVICES=0 python scripts/restore_motion_dft.py $COMMON \
  --clean_dir results/motion_n0p10/clean --observation_dir results/motion_n0p10/observation \
  --kernel_npy results/motion_n0p10/kernel.npy --sigma_y 0.10 --device cuda:0 \
  --out results/motion_n0p10/ihdm_hqs_dft > results/rerun_logs/m10.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/restore_motion_dft.py $COMMON \
  --clean_dir results/motion_n0p20/clean --observation_dir results/motion_n0p20/observation \
  --kernel_npy results/motion_n0p20/kernel.npy --sigma_y 0.20 --device cuda:0 \
  --out results/motion_n0p20/ihdm_hqs_dft > results/rerun_logs/m20.log 2>&1 &
wait
echo "=== wave 2 done ==="
echo "=== RESULT LINES ==="
grep -h "ihdm x hqs\|motion-DFT ihdm" results/rerun_logs/*.log
