# DDSPO: Direct Diffusion Score Preference Optimization

Official implementation of **Direct Diffusion Score Preference Optimization via
Stepwise Contrastive Policy-Pair Supervision** (DDSPO).

> Dohyun Kim, Seungwoo Lyu, Seung Wook Kim, Paul Hongsuck Seo.
> [[paper]](https://arxiv.org/abs/2512.23426) · [[project page]](https://dohyun-as.github.io/DDSPO)

## Overview

Preference-based fine-tuning of diffusion models (e.g. Diffusion-DPO) usually
takes its supervision targets from the forward process `q(x_{t-1} | x_t, x_0)`
derived from terminal samples, which is not aligned with the model's actual
**backward** denoising transitions. DDSPO instead defines stepwise preference
supervision directly over the backward denoising transitions through a
**contrastive policy pair**. We provide two practical instantiations:

| Mode | Flag | Idea |
|------|------|------|
| **Reference-pair** (no extra training) | `--only_cfg` | Induce the policy pair from a *frozen* pretrained model conditioned on the original prompt vs. a semantically **degraded** prompt. Needs no reward model and no annotations. |
| **Trained-pair** | `--lora_path` | Train a separate **winning** and **losing** model (LoRA pair) on preference data, then use them to supply the targets. |

The same Diffusion-DPO objective and training loop are shared across all model
families; only the model-specific pieces live behind an adapter.

## Supported models

| `--model_type` | Backbone | Prediction | Notes |
|----------------|----------|------------|-------|
| `sd15` | SD 1.4 / 1.5 UNet | epsilon | single CLIP text encoder |
| `sdxl` | SDXL UNet | epsilon | dual CLIP encoders + micro-conditioning |
| `sd3`  | SD3 MMDiT | flow matching | full transformer finetune, 2×CLIP + T5 |
| `sana` | SANA transformer | flow matching | LoRA, DC-AE VAE, Gemma2 encoder |

## Repository structure

```
ddspo/
  train.py             # main DDSPO training entry point (--model_type dispatch)
  train_lora_target.py # pre-train the winning/losing LoRA pair (trained-pair mode)
  prepare_latents.py   # pre-compute paired positive/degraded latents
  dpo.py               # shared Diffusion-DPO loss
  data.py              # paired-latent dataset + collate
  args.py              # command-line arguments
  adapters/            # per-model adapters (sd, sd3, sana)
data_prep/             # prompt generation + semantic degradation
scripts/               # runnable example scripts
tests/                 # CPU smoke tests
```

## Installation

```bash
pip install -r requirements.txt
```

SD3 / SANA require a recent `diffusers` (>= 0.32) for `SD3Transformer2DModel`,
`SanaTransformer2DModel` and `AutoencoderDC`.

## Data preparation

Training consumes **pre-encoded VAE latents** for pairs of (positive prompt,
degraded prompt). The pipeline is:

1. **Prompts + degraded variants.** Produce a JSONL where each line has a
   `prompt` and a list of degraded `neg_prompts`. See `data_prep/` (random word
   removal, or LLM-based degradation).
2. **Encode latents.** Generate and cache positive/negative latents:

   ```bash
   MODEL_TYPE=sd15 MODEL_NAME=CompVis/stable-diffusion-v1-4 \
   JSON_FILE=./data/prompts/diffusiondb_removal.jsonl \
   bash scripts/prepare_latents.sh
   ```

   This writes:

   ```
   <save_dir>/
     metadata.jsonl
     latents/<id>.safetensors        # positive
     latents/<id>_neg.safetensors    # degraded
   ```

   `metadata.jsonl` fields: `id`, `prompt`, `neg_prompt`, `pos_file`,
   `neg_file` (see `ddspo/data.py`).

## Training

Reference-pair mode (no extra training), SD1.5 / SDXL:

```bash
MODEL_TYPE=sd15 bash scripts/train_ddspo.sh
MODEL_TYPE=sdxl MODEL_NAME=stabilityai/stable-diffusion-xl-base-1.0 bash scripts/train_ddspo.sh
```

SD3 and SANA:

```bash
bash scripts/train_ddspo_sd3.sh
bash scripts/train_ddspo_sana.sh
```

Trained-pair mode (pre-train the winning/losing LoRA pair, then run DDSPO):

```bash
MODEL_TYPE=sd15 bash scripts/train_ddspo_lora_target.sh
```

All scripts are thin wrappers around `python -m ddspo.train ...`; see them for
the exact flags, and `ddspo/args.py` for the full argument reference.

### Key arguments

- `--model_type {sd15,sdxl,sd3,sana}` — model family.
- `--beta_dpo` — DPO temperature (KL strength).
- `--only_cfg` — reference-pair mode (derive every target from the reference
  model on original vs. degraded prompt).
- `--rand_cond`, `--extra_text_path` — random-condition-removal: mix unpaired
  reference-pair supervision drawn from extra prompt files.
- `--lora_path` — trained-pair mode: directory with `pos_lora_unet/` and
  `neg_lora_unet/` (SD1.x / SDXL).
- `--guidance_scale` — CFG scale for the reference target (SD1.x / SDXL).
- Flow-matching (SD3 / SANA): `--weighting_scheme`, `--precondition_outputs`
  (SD3), `--rank` (SANA LoRA), `--max_sequence_length`.

## Tests

CPU smoke tests (no GPU or pretrained weights required):

```bash
python -m pytest tests/ -q
```

They cover the DPO loss, the dataset/collate, the full import graph and a
tiny-UNet forward/backward for the SD adapter. Full-model runs require the
pretrained weights and are performed on a GPU.

## Citation

```bibtex
@article{kim2025ddspo,
  title   = {Direct Diffusion Score Preference Optimization via Stepwise Contrastive Policy-Pair Supervision},
  author  = {Kim, Dohyun and Lyu, Seungwoo and Kim, Seung Wook and Seo, Paul Hongsuck},
  journal = {arXiv preprint arXiv:2512.23426},
  year    = {2025}
}
```

## Acknowledgements

The training loop derives from the 🤗 Diffusers Diffusion-DPO / DreamBooth
reference scripts (Apache-2.0). See `LICENSE`.
