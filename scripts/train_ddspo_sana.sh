#!/usr/bin/env bash
# DDSPO training for SANA (flow matching, LoRA on the transformer), TF-CPP.
# Config follows the paper experiment (rank 128, beta 1000, fp32, effective
# batch 2048, lr 4e-8 with --scale_lr).
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
    --mixed_precision no \
    --train_batch_size 4 \
    --gradient_accumulation_steps 256 \
    --max_train_steps 500 \
    --learning_rate 4e-8 --scale_lr \
    --lr_scheduler constant --lr_warmup_steps 0 \
    --beta_dpo 1000 \
    --rank 128 \
    --cpp \
    --weighting_scheme none \
    --checkpointing_steps 50 \
    --dataloader_num_workers 8
