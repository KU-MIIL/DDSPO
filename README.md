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
whose per-step denoising scores serve as the preferred / dispreferred targets
(ε\*ʷ, ε\*ˡ). Training is enabled with `--cpp`: every sample then uses CPP score
targets (DDSPO). Without `--cpp`, targets are the forward-process noise, which
reduces to the Diffusion-DPO baseline.

The paper gives two ways to instantiate the CPP:

| Instantiation | How the winning/losing pair is obtained | Data you need | Flags |
|---------------|------------------------------------------|---------------|-------|
| **TF-CPP** (training-free) | A *frozen* reference model, conditioned on the **original** prompt `c` (winning) vs. a **semantically degraded** prompt `c⁻` (losing). No extra training, reward model, or annotations. | prompts + degraded variants you generate | `--cpp` |
| **DD-CPP** (data-driven) | A **winning** model `φʷ` and a **losing** model `φˡ`, each fine-tuned on the preferred / dispreferred samples of a preference-labeled dataset, then used as the pair. | a preference dataset `{(xʷ₀, xˡ₀, c)}` | pre-train the pair with `ddspo.train_lora_target`, then `--cpp --lora_path <dir>` |

Both share the same DDSPO objective and training loop; only the source of the
contrastive pair (and therefore the data) differs. Model-specific pieces live
behind an adapter.

> Naming: **CPP** = contrastive policy pair, **TF-CPP** = training-free CPP,
> **DD-CPP** = data-driven CPP.

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
  latents/<id>.safetensors          # preferred latent  (x_w)
  latents/<id>_neg.safetensors      # dispreferred latent (x_l)
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

3. **Train** with `--cpp` (no `--lora_path`); the target is the frozen reference
   model on the original prompt `c` vs. the degraded prompt `c⁻`:

   ```bash
   MODEL_TYPE=sd15 bash scripts/train_ddspo.sh                      # SD1.5
   MODEL_TYPE=sdxl MODEL_NAME=stabilityai/stable-diffusion-xl-base-1.0 \
       bash scripts/train_ddspo.sh                                  # SDXL
   bash scripts/train_ddspo_sd3.sh                                  # SD3
   bash scripts/train_ddspo_sana.sh                                 # SANA
   ```

## DD-CPP (data-driven)

The winning / losing pair is *trained* on any preference-labeled dataset
`{(xʷ₀, xˡ₀, c)}` (preferred sample, dispreferred sample, prompt), then used to
supply the targets. SD1.x / SDXL. (In the paper, Pick-a-Pic is used for the
aesthetic-quality task.)

1. **Encode your preference dataset** into the latent layout above: for each
   prompt, the **preferred** sample `xʷ₀` becomes `pos_file` and the
   **dispreferred** sample `xˡ₀` becomes `neg_file`.

2. **Pre-train the winning/losing model pair** (two LoRA adapters, standard
   diffusion MSE objective) and then **run DDSPO** using them via `--lora_path`.
   Both steps are wired up in one script:

   ```bash
   MODEL_TYPE=sd15 bash scripts/train_ddspo_lora_target.sh
   ```

   Under the hood: `ddspo.train_lora_target` fine-tunes `pos_lora_unet/` (the
   winning model `φʷ` on preferred samples) and `neg_lora_unet/` (the losing
   model `φˡ` on dispreferred samples); then `ddspo.train --cpp --lora_path <dir>`
   uses their per-step scores as the contrastive-pair targets.

All scripts are thin wrappers around `python -m ddspo.train ...`; see them for
the exact flags and `ddspo/args.py` for the full argument reference.

### Key arguments

- `--model_type {sd15,sdxl,sd3,sana}` — model family.
- `--cpp` — use CPP score targets for every sample (DDSPO). Omit it to fall back
  to forward-process noise targets (Diffusion-DPO baseline). Default instantiation
  is TF-CPP; add `--lora_path` for DD-CPP.
- `--lora_path` — **DD-CPP**: directory with the trained `pos_lora_unet/` (winning)
  and `neg_lora_unet/` (losing) pair (SD1.x / SDXL).
- `--beta_dpo` — divergence-penalty temperature β. Paper defaults: 16000 (TF-CPP,
  SD1.x/SDXL), 8000 (DD-CPP), 2000 (SANA / SD3). lr is scaled as β / 2.048e8.
- `--rand_cond`, `--extra_text_path` — random-condition-removal: mix in extra
  unpaired TF-CPP supervision from additional prompt files.
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
