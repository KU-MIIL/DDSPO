#!/usr/bin/env bash
# DDSPO training for SD1.x / SDXL in reference-pair mode (no extra training).
# The target is induced from the frozen reference model conditioned on the
# original vs. the degraded prompt (--only_cfg).
#
#   MODEL_TYPE=sd15 bash scripts/train_ddspo.sh
#   MODEL_TYPE=sdxl MODEL_NAME=stabilityai/stable-diffusion-xl-base-1.0 bash scripts/train_ddspo.sh
set -euo pipefail

MODEL_TYPE=${MODEL_TYPE:-sd15}                     # sd15 | sdxl
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
DATA_DIR=${DATA_DIR:-./data/paired_latents_${MODEL_TYPE}}
OUTPUT_DIR=${OUTPUT_DIR:-./results/ddspo_${MODEL_TYPE}}
NUM_GPUS=${NUM_GPUS:-1}

accelerate launch --num_processes "${NUM_GPUS}" -m ddspo.train \
    --model_type "${MODEL_TYPE}" \
    --pretrained_model_name_or_path "${MODEL_NAME}" \
    --train_data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --cache_dir ./cache \
    --mixed_precision fp16 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --max_train_steps 200 \
    --learning_rate 2.5e-9 --scale_lr \
    --lr_scheduler constant_with_warmup --lr_warmup_steps 100 \
    --beta_dpo 12000 \
    --only_cfg \
    --guidance_scale 1 \
    --checkpointing_steps 50 \
    --dataloader_num_workers 8
