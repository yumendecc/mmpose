#!/usr/bin/env bash
    --ntasks=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=1 --kill-on-bad-exit=1 \

# eval
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p mm_human --quotatype=auto\
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=eval python -u tools/test.py  --launcher="slurm" \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_small2_coco_256x192.py \
    work_dirs/coco/hrformer_small2_256/epoch_210.pth

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_small_coco_256x192.py \
    work_dirs/coco/hrformer_small2_256/epoch_210.pth

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_small_coco_256x192.py \
    models/hrformer_small_coco_256x192.pth

# train
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p mm_human --quotatype=auto\
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=coco python -u tools/train.py  --launcher="slurm" \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_base_coco_256x192.py \
    --work-dir=work_dirs/coco/hrformer_base_256

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_base_coco_384x288.py \
    --work-dir=work_dirs/coco/hrformer_base_384

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_small_coco_384x288.py \
    --work-dir=work_dirs/coco/hrformer_small_384

    configs/top_down/hrt/coco/hrt_small_coco_256x192.py \
    --work-dir=work_dirs/coco/hrt_small_256

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_small3_coco_256x192.py \
    --work-dir=work_dirs/coco/hrformer_small3_256

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_small2_coco_256x192.py \
    --work-dir=work_dirs/coco/hrformer_small2_256

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_small_coco_256x192.py \
    --work-dir=work_dirs/coco/hrformer_small_256
