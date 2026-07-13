#!/usr/bin/env python
# coding=utf-8
#
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted for DDSPO (Direct Diffusion Score Preference Optimization).
"""DDSPO training entry point.

A single, model-agnostic training loop. Everything model-specific lives behind a
:class:`~ddspo.adapters.base.ModelAdapter` selected with ``--model_type``; the
Diffusion-DPO loss and the optimization / checkpointing machinery are shared
across all families.
"""

import logging
import math
import os

import accelerate
import datasets
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from tqdm.auto import tqdm

import diffusers
from diffusers.optimization import get_scheduler

from .adapters import get_adapter
from .args import parse_args
from .dpo import dpo_loss

logger = get_logger(__name__, log_level="INFO")


def _sanitize_tracker_config(config):
    safe = {}
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool)):
            safe[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            safe[k] = ",".join(v)
    return safe


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- model (via adapter) ---
    adapter = get_adapter(args.model_type)
    model = adapter.load(args, accelerator)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.scale_lr:
        args.learning_rate *= (
            args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.use_adafactor:
        optimizer = transformers.Adafactor(
            trainable_params, lr=args.learning_rate, weight_decay=args.adam_weight_decay,
            clip_threshold=1.0, scale_parameter=False, relative_step=False)
    else:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    train_dataloader = adapter.make_dataloader(args)

    overrode_max_train_steps = args.max_train_steps is None
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        accelerator.mixed_precision, torch.float32)
    adapter.place_frozen(accelerator, weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, _sanitize_tracker_config(vars(args)))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running DDSPO training *****")
    logger.info(f"  Model type = {args.model_type}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step, first_epoch, resume_step = 0, 0, 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = sorted((d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")),
                          key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting fresh.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = (global_step * args.gradient_accumulation_steps) % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process, desc="Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        implicit_acc_accum = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue
            with accelerator.accumulate(model):
                model_pred, ref_pred, target, timesteps = adapter.training_step(
                    model, batch, args, weight_dtype, accelerator.device)
                loss, metrics = dpo_loss(
                    model_pred, ref_pred, target, args.beta_dpo, timesteps,
                    args.loss_weighting, adapter.num_train_timesteps)

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                avg_acc = accelerator.gather(metrics["implicit_acc"]).mean().item()
                implicit_acc_accum += avg_acc / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients and not args.use_adafactor:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss,
                                 "implicit_acc": implicit_acc_accum}, step=global_step)
                train_loss = 0.0
                implicit_acc_accum = 0.0

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            progress_bar.set_postfix(step_loss=loss.detach().item(),
                                     lr=lr_scheduler.get_last_lr()[0], implicit_acc=avg_acc)
            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        adapter.save(args, accelerator, model)
    accelerator.end_training()


if __name__ == "__main__":
    main()
