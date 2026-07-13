#!/usr/bin/env bash
# DDSPO training for SANA (flow matching, LoRA on the transformer).
#   bash scripts/train_ddspo_sana.sh
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-Efficient-Large-Model/Sana_600M_1024px_diffusers}
DATA_DIR=${DATA_DIR:-./data/paired_latents_sana}
OUTPUT_DIR=${OUTPUT_DIR:-./results/ddspo_sana}
NUM_GPUS=${NUM_GPUS:-1}

accelerate launch --num_processes "${NUM_GPUS}" -m ddspo.train \
    --model_type sana \
    --pretrained_model_name_or_path "${MODEL_NAME}" \
    --train_data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --cache_dir ./cache \
    --resolution 1024 \
    --mixed_precision bf16 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --max_train_steps 150 \
    --learning_rate 1e-4 \
    --lr_scheduler constant_with_warmup --lr_warmup_steps 100 \
    --beta_dpo 1000 \
    --rank 128 \
    --only_cfg \
    --weighting_scheme none \
    --checkpointing_steps 50 \
    --dataloader_num_workers 8
