#!/usr/bin/env bash
# DDSPO training for SD1.x / SDXL with TF-CPP (training-free contrastive policy
# pair). --cpp turns on CPP score targets (DDSPO); with no --lora_path the pair
# is the frozen reference model on the original vs. the degraded prompt (TF-CPP).
#
# Configs follow the paper experiments (effective batch 2048, beta 16000,
# lr 2.5e-9 with --scale_lr). SDXL additionally uses linear loss weighting.
#
#   MODEL_TYPE=sd15 bash scripts/train_ddspo.sh
#   MODEL_TYPE=sdxl MODEL_NAME=stabilityai/stable-diffusion-xl-base-1.0 bash scripts/train_ddspo.sh
set -euo pipefail

MODEL_TYPE=${MODEL_TYPE:-sd15}                     # sd15 | sdxl
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
DATA_DIR=${DATA_DIR:-./data/paired_latents_${MODEL_TYPE}}
OUTPUT_DIR=${OUTPUT_DIR:-./results/ddspo_${MODEL_TYPE}}
NUM_GPUS=${NUM_GPUS:-1}

if [ "${MODEL_TYPE}" = "sdxl" ]; then
    GA=64; STEPS=500; EXTRA="--loss_weighting linear"
else
    GA=16; STEPS=200; EXTRA=""
fi

accelerate launch --num_processes "${NUM_GPUS}" -m ddspo.train \
    --model_type "${MODEL_TYPE}" \
    --pretrained_model_name_or_path "${MODEL_NAME}" \
    --train_data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --cache_dir ./cache \
    --mixed_precision fp16 \
    --train_batch_size 8 \
    --gradient_accumulation_steps "${GA}" \
    --max_train_steps "${STEPS}" \
    --learning_rate 2.5e-9 --scale_lr \
    --lr_scheduler constant_with_warmup --lr_warmup_steps 100 \
    --beta_dpo 16000 \
    --cpp \
    --checkpointing_steps 50 \
    --dataloader_num_workers 8 ${EXTRA}
