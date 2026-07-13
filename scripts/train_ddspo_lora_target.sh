#!/usr/bin/env bash
# DDSPO with DD-CPP (data-driven contrastive policy pair), SD1.x / SDXL:
#   1) pre-train the winning/losing LoRA pair on a preference dataset (MSE);
#   2) run DDSPO using those LoRAs as the target source via --lora_path.
# DATA_DIR should hold preference pairs (pos_file = chosen, neg_file = rejected).
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
    --gradient_accumulation_steps 128 \
    --max_train_steps 200 \
    --lora_rank 4 --lora_alpha 4.0 --lora_lr 1e-4 \
    --checkpointing_steps 100 \
    --dataloader_num_workers 8

# 2) DDSPO with the trained policy pair as the target source.
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
    --lora_path "${LORA_DIR}" \
    --checkpointing_steps 50 \
    --dataloader_num_workers 8
