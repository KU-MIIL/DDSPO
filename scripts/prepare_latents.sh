#!/usr/bin/env bash
# Pre-compute paired (positive / degraded) latents for DDSPO training.
# Usage: MODEL_TYPE=sd15 MODEL_NAME=CompVis/stable-diffusion-v1-4 bash scripts/prepare_latents.sh
set -euo pipefail

MODEL_TYPE=${MODEL_TYPE:-sd15}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
JSON_FILE=${JSON_FILE:-./data/prompts/diffusiondb_removal.jsonl}
SAVE_DIR=${SAVE_DIR:-./data/paired_latents_${MODEL_TYPE}}
NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
CFG=${CFG:-7.5}

accelerate launch --num_processes "${NUM_GPUS}" -m ddspo.prepare_latents \
    --model_type "${MODEL_TYPE}" \
    --model_name "${MODEL_NAME}" \
    --json_file "${JSON_FILE}" \
    --save_dir "${SAVE_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --cfg "${CFG}" \
    --save_type latent
