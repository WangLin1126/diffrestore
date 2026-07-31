#!/usr/bin/env bash
# Regenerate every report figure from the (updated) step-200k recon dirs.
# Uses free GPU cuda:1 for LPIPS in the figure headers. Run after rerun_ihdm_full.sh.
set -u
cd "$(dirname "$0")/.."
DEV=cuda:1
MF="python scripts/make_figure.py --device $DEV --rows 4"

# ---- refresh cropped IHDM (+ others already cropped) for the crop figures ----
for pair in "motion:ihdm_dft" "motion_n0p10:ihdm" "motion_n0p20:ihdm"; do
  d="${pair%%:*}"; name="${pair##*:}"
  python scripts/crop_score.py --clean results/$d/clean --recon results/$d/ihdm_hqs_dft/recon \
    --crop 128 --save_dir results/$d/crop128/$name --label "$d $name" --device $DEV >/dev/null 2>&1
done

# ---- Gaussian (DCT), 3 noise levels : Clean|Obs|HQS+TV|cold+HQS|IHDM+HQS|DPS ----
gauss () {  # $1 dir  $2 outname  $3 title
  $MF --clean results/$1/clean --obs results/$1/observation \
    --recon "HQS+TV"=results/$1/tv_hqs/recon \
    --recon "cold+HQS"=results/$1/cold_hqs/recon \
    --recon "IHDM+HQS"=results/$1/ihdm_hqs/recon \
    --recon "DPS"=results/$1/dps/heat_blur/recon \
    --title "$3" --out results/$1/$2
}
gauss gaussian        figure_gaussian.png "Gaussian deblur (sigma=4, noise=0.05) - held-out CelebA-HQ 256"
gauss gaussian_n0p10  figure.png          "Gaussian deblur (sigma=4, noise=0.10) - held-out CelebA-HQ 256"
gauss gaussian_n0p20  figure.png          "Gaussian deblur (sigma=4, noise=0.20) - held-out CelebA-HQ 256"

# ---- Motion (DFT) full frame, higher-noise levels : Clean|Obs|HQS+TV|cold+HQS|IHDM+HQS|DPS ----
motionfull () {
  $MF --clean results/$1/clean --obs results/$1/observation \
    --recon "HQS+TV"=results/$1/tv_hqs/recon \
    --recon "cold+HQS (DFT)"=results/$1/cold_hqs_dft/recon \
    --recon "IHDM+HQS (DFT)"=results/$1/ihdm_hqs_dft/recon \
    --recon "DPS"=results/$1/dps/heat_blur/recon \
    --title "$3" --out results/$1/$2
}
motionfull motion_n0p10 figure_full.png "Motion deblur (Levin09 #1, noise=0.10), full frame - CelebA-HQ 256"
motionfull motion_n0p20 figure_full.png "Motion deblur (Levin09 #1, noise=0.20), full frame - CelebA-HQ 256"

# ---- Motion crop (central 128), higher-noise : Clean|Obs|TV|cold|IHDM|DPS (cropped dirs) ----
motioncrop () {
  $MF --clean results/$1/crop128/clean --obs results/$1/crop128/observation \
    --recon "HQS+TV"=results/$1/crop128/tv \
    --recon "cold+HQS"=results/$1/crop128/cold \
    --recon "IHDM+HQS"=results/$1/crop128/ihdm \
    --recon "DPS"=results/$1/crop128/dps \
    --title "$3" --out results/$1/$2
}
motioncrop motion_n0p10 figure_crop.png "Motion deblur, central 128 crop (noise=0.10)"
motioncrop motion_n0p20 figure_crop.png "Motion deblur, central 128 crop (noise=0.20)"

# ---- Motion base (noise=0.05): crop128 figure + DCT-vs-DFT figure ----
$MF --clean results/motion/crop128/clean --obs results/motion/crop128/observation \
  --recon "HQS+TV"=results/motion/crop128/tv \
  --recon "cold+HQS"=results/motion/crop128/cold_dft \
  --recon "IHDM+HQS"=results/motion/crop128/ihdm_dft \
  --recon "DPS"=results/motion/crop128/dps \
  --title "Motion deblur, central 128 crop (noise=0.05)" --out results/motion/figure_crop128.png

$MF --clean results/motion/clean --obs results/motion/observation \
  --recon "cold+HQS (DCT)"=results/motion/cold_hqs/recon \
  --recon "IHDM+HQS (DCT)"=results/motion/ihdm_hqs/recon \
  --recon "cold+HQS (DFT)"=results/motion/cold_hqs_dft/recon \
  --recon "IHDM+HQS (DFT)"=results/motion/ihdm_hqs_dft/recon \
  --title "Motion deblur: DCT-basis (ringing) vs DFT-basis (correct), noise=0.05" \
  --out results/motion/figure_dct_vs_dft.png

echo "=== figures regenerated ==="
ls -la results/gaussian/figure_gaussian.png results/motion/figure_dct_vs_dft.png results/motion/figure_crop128.png
