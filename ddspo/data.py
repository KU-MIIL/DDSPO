"""Dataset for DDSPO training.

Loads pre-encoded VAE latents for positive/negative sample pairs together with
their prompts and a sampled diffusion timestep. Works for every supported model
family; the only per-model difference is how prompts are turned into tokens,
which is handled by ``collate_fn`` (tokenize now) vs. passing raw strings
through (tokenize inside the training loop, e.g. SD3 / SANA).

Expected on-disk layout::

    <data_dir>/
        metadata.jsonl          # one JSON object per line (see below)
        latents/
            <id>.safetensors        # positive latent, key "latent"
            <id>_neg.safetensors    # negative latent, key "latent"

Each ``metadata.jsonl`` line::

    {"id": "...", "prompt": "<positive prompt>", "neg_prompt": "<negative prompt>",
     "pos_file": "<id>.safetensors", "neg_file": "<id>_neg.safetensors"}
"""

import json
import math
import os
import random

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, Subset


class SelfTrainingDataset(Dataset):
    """Paired positive/negative latents with random-condition-removal support.

    Args:
        data_dir: Directory containing ``metadata.jsonl`` and ``latents/``.
        rand_cond: If True, with a timestep-dependent probability replace the
            (prompt, neg_prompt) pair with an unpaired sample drawn from
            ``extra_text_path`` and flag it as unpaired (``paired == 0``).
        rand_cond_lambda: Steepness of the random-condition-removal schedule.
        rand_cond_pt: Schedule shape, one of {"sigmoid", "exponential", "constant_1"}.
        n_T: Number of diffusion training timesteps.
        extra_text_path: Optional list of JSONL files with extra prompts. Each
            line needs {"prompt": ..., "neg_prompts": [...]}.
        timestep_sampling: One of {"linear", "sigmoid", None}. None samples uniformly.
    """

    def __init__(
        self,
        data_dir,
        rand_cond=False,
        rand_cond_lambda=5.0,
        rand_cond_pt="sigmoid",
        n_T=1000,
        extra_text_path=None,
        timestep_sampling=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.latent_dir = os.path.join(data_dir, "latents")
        self.rand_cond = rand_cond
        self.rand_cond_lambda = rand_cond_lambda
        self.rand_cond_pt = rand_cond_pt
        self.n_T = n_T
        self.timestep_sampling = timestep_sampling

        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        self.entries = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                self.entries.append(json.loads(line.strip()))

        # Extra unpaired prompts used by random-condition-removal.
        existing_prompts = {entry["prompt"] for entry in self.entries}
        self.extratext = []
        if extra_text_path is not None:
            for path in extra_text_path:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line.strip())
                        prompt = item.get("prompt")
                        neg_prompts = item.get("neg_prompts")
                        if (
                            prompt
                            and prompt not in existing_prompts
                            and isinstance(neg_prompts, list)
                            and len(neg_prompts) > 0
                        ):
                            self.extratext.append(item)

    def __len__(self):
        return len(self.entries)

    def _sample_timestep(self):
        if self.timestep_sampling == "linear":
            probs = torch.linspace(1.0, 0.0, self.n_T)
            probs /= probs.sum()
            return torch.multinomial(probs, 1).long()
        if self.timestep_sampling == "sigmoid":
            t_value = torch.arange(0, self.n_T) / self.n_T
            probs = 1.0 / (1.0 + torch.exp(-20 * (t_value - 0.7)))
            probs /= probs.sum()
            return torch.multinomial(probs, 1).long()
        return torch.randint(0, self.n_T, (1,)).long()

    def _rand_cond_prob(self, t_value):
        if self.rand_cond_pt == "sigmoid":
            return 1.0 / (1.0 + math.exp(-self.rand_cond_lambda * (t_value / self.n_T - 0.7)))
        if self.rand_cond_pt == "exponential":
            return math.exp(-self.rand_cond_lambda * (1 - t_value / self.n_T))
        if self.rand_cond_pt == "constant_1":
            return 1.0
        raise ValueError(f"Unknown rand_cond_pt: {self.rand_cond_pt}")

    def __getitem__(self, idx):
        item = self.entries[idx]
        prompt = item["prompt"]
        neg_prompt = item["neg_prompt"]
        pos_file = item["pos_file"]
        neg_file = item["neg_file"]

        timestep = self._sample_timestep()
        paired = torch.tensor(1).unsqueeze(0)

        if self.rand_cond and self.extratext:
            p = self._rand_cond_prob(timestep.item())
            if torch.rand(1).item() < p:
                extra = random.choice(self.extratext)
                prompt = extra["prompt"]
                neg_prompt = random.choice(extra["neg_prompts"])
                paired = torch.tensor(0).unsqueeze(0)

        pos_latent = load_file(os.path.join(self.latent_dir, pos_file))["latent"]
        neg_latent = load_file(os.path.join(self.latent_dir, neg_file))["latent"]
        # Stacked along channel dim -> (2, C, H, W); split back into a batch pair
        # by the training loop.
        latents = torch.cat([pos_latent, neg_latent], dim=0)

        return latents, prompt, neg_prompt, timestep, paired


def collate_fn(tokenizer=None, tokenizer_2=None):
    """Build a collate function.

    If ``tokenizer`` is provided the prompts are tokenized here (SD1.5 / SDXL).
    If ``tokenizer`` is None the raw prompt strings are returned so the model
    family can tokenize inside the training loop (SD3 / SANA).
    """

    def collate(batch):
        latents, pos_prompts, neg_prompts, timesteps, paireds = zip(*batch)
        out = {
            "latents": torch.stack(latents),
            "timesteps": torch.cat(timesteps),
            "paireds": torch.cat(paireds),
            "pos_prompts": list(pos_prompts),
            "neg_prompts": list(neg_prompts),
        }

        if tokenizer is not None:
            out["pos_input_ids"] = tokenizer(
                list(pos_prompts), max_length=tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            ).input_ids
            out["neg_input_ids"] = tokenizer(
                list(neg_prompts), max_length=tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            ).input_ids

        if tokenizer_2 is not None:
            out["pos_input_ids_2"] = tokenizer_2(
                list(pos_prompts), max_length=tokenizer_2.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            ).input_ids
            out["neg_input_ids_2"] = tokenizer_2(
                list(neg_prompts), max_length=tokenizer_2.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            ).input_ids

        return out

    return collate


def make_self_training_dataloader(args, collate, num_train_timesteps):
    """Build the training ``DataLoader`` shared by every adapter.

    ``collate`` is the model-specific collate function (tokenizing or raw).
    """
    dataset = SelfTrainingDataset(
        data_dir=args.train_data_dir,
        rand_cond=args.rand_cond,
        rand_cond_lambda=args.rand_cond_lambda,
        rand_cond_pt=args.rand_cond_pt,
        n_T=num_train_timesteps,
        extra_text_path=args.extra_text_path,
        timestep_sampling=args.timestep_sampling,
    )
    if args.max_train_samples is not None:
        state = random.getstate()
        random.seed(42)  # fixed seed -> reproducible subset
        indices = random.sample(range(len(dataset)), args.max_train_samples)
        random.setstate(state)
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset, shuffle=True, collate_fn=collate,
        batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
