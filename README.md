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
**contrastive policy pair (CPP)** — a *winning* policy and a *losing* policy
whose denoising scores serve as the preferred / dispreferred targets. We provide
two ways to instantiate the CPP:

| Instantiation | How the winning/losing pair is obtained | Data you need | Flag |
|---------------|------------------------------------------|---------------|------|
| **TF-CPP** (training-free) | A *frozen* reference model conditioned on the **original** prompt (winning) vs. a **semantically degraded** prompt (losing). No extra training, reward model, or annotations. | prompts + degraded variants you create | `--only_cfg` |
| **DD-CPP** (data-driven) | A **winning** and a **losing** model trained on an **existing preference dataset** (e.g. Pick-a-Pic chosen vs. rejected), then used as the pair. | a preference dataset | pre-train with `ddspo.train_lora_target`, then pass `--lora_path` |

Both share the same Diffusion-DPO objective and training loop; only the source
of the contrastive pair (and therefore the data) differs. Model-specific pieces
live behind an adapter.

> Naming: **TF-CPP** = training-free CPP, **DD-CPP** = data-driven CPP.

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

Both modes train on **pre-encoded VAE latent pairs** in the same on-disk layout
(see `ddspo/data.py`) — what differs is where the pair comes from:

```
<data_dir>/
  metadata.jsonl                    # {id, prompt, neg_prompt, pos_file, neg_file}
  latents/<id>.safetensors          # positive / winning latent
  latents/<id>_neg.safetensors      # negative / losing latent
```

## TF-CPP (training-free)

The contrastive pair is induced from a frozen reference model, so the "data" is
just prompts and their degraded variants — no preference labels needed.

1. **Make prompts + degraded variants.** Produce a JSONL where each line has a
   `prompt` and a list of degraded `neg_prompts` (`data_prep/`: random word
   removal or LLM-based degradation):

   ```bash
   python data_prep/random_removal.py --output ./data/prompts/diffusiondb_removal.jsonl
   ```

2. **Encode paired latents** (positive from the original prompt, negative from a
   degraded one):

   ```bash
   MODEL_TYPE=sd15 MODEL_NAME=CompVis/stable-diffusion-v1-4 \
   JSON_FILE=./data/prompts/diffusiondb_removal.jsonl \
   bash scripts/prepare_latents.sh
   ```

3. **Train** with `--only_cfg` (the target is the frozen reference model on the
   original vs. degraded prompt):

   ```bash
   MODEL_TYPE=sd15 bash scripts/train_ddspo.sh                      # SD1.5
   MODEL_TYPE=sdxl MODEL_NAME=stabilityai/stable-diffusion-xl-base-1.0 \
       bash scripts/train_ddspo.sh                                  # SDXL
   bash scripts/train_ddspo_sd3.sh                                  # SD3
   bash scripts/train_ddspo_sana.sh                                 # SANA
   ```

## DD-CPP (data-driven)

The winning / losing pair is *trained* on an existing preference dataset
(e.g. Pick-a-Pic), then used to supply the targets. SD1.x / SDXL.

1. **Encode the preference dataset** into the latent layout above, with the
   **chosen** image as `pos_file` and the **rejected** image as `neg_file` for
   each prompt.

2. **Pre-train the winning/losing model pair** (two LoRA adapters, MSE objective)
   and then **run DDSPO** using them via `--lora_path`. Both steps are wired up
   in one script:

   ```bash
   MODEL_TYPE=sd15 bash scripts/train_ddspo_lora_target.sh
   ```

   Under the hood: `ddspo.train_lora_target` trains `pos_lora_unet/`
   (winning) and `neg_lora_unet/` (losing); then `ddspo.train --lora_path <dir>`
   uses their scores as the contrastive-pair targets.

All scripts are thin wrappers around `python -m ddspo.train ...`; see them for
the exact flags and `ddspo/args.py` for the full argument reference.

### Key arguments

- `--model_type {sd15,sdxl,sd3,sana}` — model family.
- `--beta_dpo` — DPO temperature (KL strength).
- `--only_cfg` — **TF-CPP**: derive every target from the frozen reference model
  on the original vs. degraded prompt.
- `--lora_path` — **DD-CPP**: directory with the trained `pos_lora_unet/` and
  `neg_lora_unet/` pair (SD1.x / SDXL).
- `--rand_cond`, `--extra_text_path` — random-condition-removal: mix in extra
  unpaired TF-CPP supervision from additional prompt files.
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
