#!/usr/bin/env bash
# DDSPO training for Stable Diffusion 3 (flow matching, full transformer finetune).
#   bash scripts/train_ddspo_sd3.sh
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-stabilityai/stable-diffusion-3-medium-diffusers}
DATA_DIR=${DATA_DIR:-./data/paired_latents_sd3}
OUTPUT_DIR=${OUTPUT_DIR:-./results/ddspo_sd3}
NUM_GPUS=${NUM_GPUS:-1}

accelerate launch --num_processes "${NUM_GPUS}" -m ddspo.train \
    --model_type sd3 \
    --pretrained_model_name_or_path "${MODEL_NAME}" \
    --train_data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --cache_dir ./cache \
    --resolution 1024 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --max_train_steps 150 \
    --learning_rate 1e-8 --scale_lr \
    --lr_scheduler constant_with_warmup --lr_warmup_steps 100 \
    --beta_dpo 1000 \
    --only_cfg \
    --weighting_scheme logit_normal \
    --precondition_outputs 1 \
    --checkpointing_steps 50 \
    --dataloader_num_workers 8
