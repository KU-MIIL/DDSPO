#!/usr/bin/env bash
# DDSPO with DD-CPP (data-driven contrastive policy pair), SD1.x / SDXL:
#   1) pre-train the winning/losing LoRA pair on a preference dataset (MSE);
#   2) run DDSPO using those policies as the target source via --lora_path.
# DATA_DIR should hold a preference dataset: pos_file = preferred sample (x_w),
# neg_file = dispreferred sample (x_l), for each prompt (e.g. Pick-a-Pic).
# Paper DD-CPP defaults: pair LoRA rank 4, lr 1e-4, up to 300 steps; DDSPO beta 8000.
#
#   MODEL_TYPE=sd15 bash scripts/train_ddspo_lora_target.sh
set -euo pipefail

MODEL_TYPE=${MODEL_TYPE:-sd15}                     # sd15 | sdxl
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
DATA_DIR=${DATA_DIR:-./data/paired_latents_${MODEL_TYPE}}
LORA_DIR=${LORA_DIR:-./results/ddspo_lora_target_${MODEL_TYPE}}
OUTPUT_DIR=${OUTPUT_DIR:-./results/ddspo_${MODEL_TYPE}_lora_target}
NUM_GPUS=${NUM_GPUS:-1}
SDXL_FLAG=""; [ "${MODEL_TYPE}" = "sdxl" ] && SDXL_FLAG="--sdxl"

# 1) Pre-train pos_lora_unet / neg_lora_unet (winning / losing models).
accelerate launch --num_processes "${NUM_GPUS}" -m ddspo.train_lora_target \
    --pretrained_model_name_or_path "${MODEL_NAME}" ${SDXL_FLAG} \
    --train_data_dir "${DATA_DIR}" \
    --output_dir "${LORA_DIR}" \
    --cache_dir ./cache \
    --mixed_precision fp16 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 2048 \
    --max_train_steps 300 \
    --lora_rank 4 --lora_alpha 4.0 --lora_lr 1e-4 \
    --checkpointing_steps 50 \
    --dataloader_num_workers 8

# 2) DDSPO with the trained policy pair as the target source (beta 8000).
accelerate launch --num_processes "${NUM_GPUS}" -m ddspo.train \
    --model_type "${MODEL_TYPE}" \
    --pretrained_model_name_or_path "${MODEL_NAME}" \
    --train_data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --cache_dir ./cache \
    --mixed_precision fp16 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 2048 \
    --max_train_steps 100 \
    --learning_rate 3.906e-5 \
    --lr_scheduler constant_with_warmup --lr_warmup_steps 100 \
    --beta_dpo 8000 \
    --cpp \
    --lora_path "${LORA_DIR}" \
    --checkpointing_steps 50 \
    --dataloader_num_workers 8
