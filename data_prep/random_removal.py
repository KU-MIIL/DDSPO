#!/usr/bin/env python
# coding=utf-8
"""Generate degraded ("negative") prompts by random word removal.

For each prompt, several degraded variants are produced by dropping 40-70% of
the words. These serve as the semantically degraded prompts of the DDSPO
contrastive policy pair. Output is a JSONL consumable by
``ddspo/prepare_latents.py`` (fields: id, tag, prompt, neg_prompts).

Example (uses the DiffusionDB prompt set):
    python data_prep/random_removal.py --output diffusiondb_removal.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from urllib.request import urlretrieve

DIFFUSIONDB_PARQUET = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet"


def generate_negative_prompts(prompt, max_variants=5, max_trials=5):
    words = prompt.split()
    if len(words) < 2:
        return []
    candidates = set()
    for _ in range(max_trials):
        ratio = random.uniform(0.4, 0.7)
        num_to_remove = max(1, int(ratio * len(words)))
        drop = set(random.sample(range(len(words)), num_to_remove))
        neg = " ".join(w for i, w in enumerate(words) if i not in drop)
        if neg != prompt:
            candidates.add(neg)
        if len(candidates) >= max_variants:
            break
    return list(candidates)


def load_prompts(args):
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            return [json.loads(l)["prompt"] for l in f]
    # Default: DiffusionDB prompt set.
    import pandas as pd
    local = args.parquet_path
    if not Path(local).exists():
        print("Downloading DiffusionDB metadata.parquet ...")
        urlretrieve(DIFFUSIONDB_PARQUET, local)
    return pd.read_parquet(local)["prompt"].dropna().unique().tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="diffusiondb_removal.jsonl")
    p.add_argument("--prompts_file", type=str, default=None,
                   help="Optional JSONL with a 'prompt' field per line. "
                        "Defaults to the DiffusionDB prompt set.")
    p.add_argument("--parquet_path", type=str, default="metadata.parquet")
    p.add_argument("--tag", type=str, default="DiffusionDB")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    random.seed(args.seed)

    prompts = load_prompts(args)
    with open(args.output, "w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts):
            entry = {"id": f"{idx + 1:08}", "tag": args.tag, "prompt": prompt,
                     "neg_prompts": generate_negative_prompts(prompt)}
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {len(prompts)} entries to {args.output}")


if __name__ == "__main__":
    main()
