"""Command-line arguments for DDSPO training.

A single, cleaned argument surface shared by every model family. Model selection
is done with ``--model_type``; family-specific options are grouped below.
"""

import argparse
import os

from .adapters import DEFAULT_COMPLEX_HUMAN_INSTRUCTION

MODEL_TYPES = ["sd15", "sdxl", "sd3", "sana"]


def parse_args():
    p = argparse.ArgumentParser(description="DDSPO training.")

    # --- model / data ---
    p.add_argument("--model_type", type=str, required=True, choices=MODEL_TYPES,
                   help="Which model family to train.")
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                   help="HuggingFace model id or local path.")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--variant", type=str, default=None,
                   help="Model file variant, e.g. 'fp16' (SD3 / SANA).")
    p.add_argument("--train_data_dir", type=str, required=True,
                   help="Directory with metadata.jsonl and latents/ (see ddspo/data.py).")
    p.add_argument("--output_dir", type=str, default="ddspo-output")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--resolution", type=int, default=None,
                   help="Defaults to 512 for sd15 and 1024 otherwise.")
    p.add_argument("--max_train_samples", type=int, default=None)

    # --- optimization ---
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--max_train_steps", type=int, default=None,
                   help="Overrides num_train_epochs when set.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--learning_rate", type=float, default=1e-8)
    p.add_argument("--scale_lr", action="store_true")
    p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--use_adafactor", action="store_true")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # --- logging / checkpointing ---
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--tracker_project_name", type=str, default="ddspo")
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="A checkpoint path or 'latest'.")

    # --- DDSPO method ---
    p.add_argument("--beta_dpo", type=float, default=5000,
                   help="DPO temperature controlling the strength of the KL penalty.")
    p.add_argument("--only_cfg", action="store_true",
                   help="TF-CPP (training-free): derive every target from the frozen reference "
                        "model conditioned on the original vs. degraded prompt (no training needed).")
    p.add_argument("--guidance_scale", type=float, default=1,
                   help="CFG scale for the reference target (SD1.x/SDXL only).")
    p.add_argument("--rand_cond", action="store_true",
                   help="Random-condition-removal: mix in unpaired reference-pair supervision.")
    p.add_argument("--rand_cond_lambda", type=float, default=20)
    p.add_argument("--rand_cond_pt", type=str, default="sigmoid",
                   choices=["sigmoid", "exponential", "constant_1"])
    p.add_argument("--extra_text_path", type=str, nargs="+", default=None,
                   help="Extra unpaired prompt JSONL files for random-condition-removal.")
    p.add_argument("--timestep_sampling", type=str, default=None, choices=["sigmoid", "linear"])
    p.add_argument("--pos_neg_different_noise", action="store_true")
    p.add_argument("--loss_weighting", type=str, default=None, choices=["linear"],
                   help="Optional timestep weighting of the DPO loss (SD1.x/SDXL).")
    p.add_argument("--lora_path", type=str, default=None,
                   help="DD-CPP (data-driven): directory with the pre-trained pos_lora_unet/ and "
                        "neg_lora_unet/ (winning/losing) pair used to build targets. SD1.x/SDXL only.")

    # --- SDXL ---
    p.add_argument("--pretrained_vae_model_name_or_path", type=str, default=None,
                   help="Optional VAE with better numerical stability (SDXL).")

    # --- flow-matching models (SD3 / SANA) ---
    p.add_argument("--weighting_scheme", type=str, default="none",
                   choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)
    p.add_argument("--max_sequence_length", type=int, default=None,
                   help="Text encoder max length. Defaults to 77 (SD3) / 300 (SANA).")
    p.add_argument("--precondition_outputs", type=int, default=1,
                   help="EDM-style output preconditioning (SD3).")

    # --- LoRA (SANA trains via LoRA) ---
    p.add_argument("--rank", type=int, default=128, help="LoRA rank (SANA).")
    p.add_argument("--lora_layers", type=str, default=None,
                   help="Comma-separated target module names for LoRA (SANA).")
    p.add_argument("--complex_human_instruction", type=str, nargs="+",
                   default=DEFAULT_COMPLEX_HUMAN_INSTRUCTION,
                   help="Gemma2 prompt-enhancement instruction (SANA).")

    args = p.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.resolution is None:
        args.resolution = 512 if args.model_type == "sd15" else 1024
    if args.max_sequence_length is None:
        args.max_sequence_length = 300 if args.model_type == "sana" else 77

    return args
