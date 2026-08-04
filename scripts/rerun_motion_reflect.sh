#!/usr/bin/env bash
# Full motion table on REALISTIC reflect-boundary observations: learned priors solved by spatial CG
# (no FFT), baselines recomputed on the same obs. 3 noise levels.
set -u
cd "$(dirname "$0")/.."
IHDM="checkpoint/ihdm/ihdm_ffhq256_full.pth"
COLD="checkpoint/cold_diffusion/ffhq256.pth"
mkdir -p results/rerun_logs

run_noise () {  # $1 dir  $2 sigma_y  $3 gpu-offset
  D="$1"; SY="$2"
  K="results/$D/kernel.npy"; CL="results/$D/clean"; OB="results/$D/observation"; TAG="${D}"
  # IHDM + CG  (gpu0)
  CUDA_VISIBLE_DEVICES=0 python scripts/deblur.py restore --operator motion --prior ihdm --ihdm_config img_size_256_full \
    --ckpt "$IHDM" --clean_dir "$CL" --observation_dir "$OB" --kernel_npy "$K" --sigma_y "$SY" \
    --cg_iters 12 --n 16 --device cuda:0 --out results/$D/ihdm_cg > results/rerun_logs/${TAG}_ihdmcg.log 2>&1 &
  # cold + CG  (gpu1)
  CUDA_VISIBLE_DEVICES=1 python scripts/deblur.py restore --operator motion --prior cold_diffusion \
    --ckpt "$COLD" --image_size 256 --ch 128 --ch_mult 1 1 2 2 4 --num_res_blocks 2 \
    --clean_dir "$CL" --observation_dir "$OB" --kernel_npy "$K" --sigma_y "$SY" \
    --cg_iters 12 --n 16 --device cuda:0 --out results/$D/cold_cg > results/rerun_logs/${TAG}_coldcg.log 2>&1 &
  # HQS + TV  (gpu2)
  CUDA_VISIBLE_DEVICES=2 python scripts/run_tv_hqs.py --operator motion --kernel_npy "$K" \
    --clean_dir "$CL" --observation_dir "$OB" --beta 4 --sigma_y "$SY" --n 16 \
    --out results/$D/tv_hqs > results/rerun_logs/${TAG}_tv.log 2>&1 &
  # DPS  (gpu3)
  CUDA_VISIBLE_DEVICES=3 python scripts/run_dps.py --operator motion --kernel_npy "$K" \
    --clean_dir "$CL" --observation_dir "$OB" --num_images 16 --scale 0.3 --gpu 0 \
    --save_dir results/$D/dps > results/rerun_logs/${TAG}_dps.log 2>&1 &
  wait
  echo "=== done $D (sigma_y=$SY) ==="
}

run_noise motion_reflect        0.05
run_noise motion_reflect_n0p10  0.10
run_noise motion_reflect_n0p20  0.20

echo "=========== ALL RESULT LINES ==========="
grep -h "motion-CG\|TV+HQS motion\|DPS" results/rerun_logs/motion_reflect*_*.log 2>/dev/null | grep -i "psnr"
