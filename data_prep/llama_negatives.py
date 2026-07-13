#!/usr/bin/env python
# coding=utf-8
"""Generate degraded ("negative") prompts with Llama-3.

An alternative to random word removal: an instruction-tuned LLM rewrites each
prompt into semantically degraded variants. Uses a prompt template
(``template_prompt.txt``) and shards the work across processes when launched
with ``accelerate``. Output is per-rank JSONL (merge with ``concat_ranks.py``).

Authentication: set the ``HF_TOKEN`` environment variable or run
``huggingface-cli login`` first. Never hard-code tokens.

Example:
    accelerate launch data_prep/llama_negatives.py --template template_prompt.txt
"""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.request import urlretrieve

import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DIFFUSIONDB_PARQUET = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet"

PROMPT_TEMPLATE = (
    "Below is the new prompt to process.\n"
    "prompt: {}\n"
    "<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)


def extract_last_response_info(full_text):
    assistant_indices = [m.start() for m in re.finditer(r"\bassistant\b", full_text)]
    if len(assistant_indices) < 2:
        raise ValueError("Second 'assistant' block not found in the text.")
    response_text = full_text[assistant_indices[1]:]

    important_match = re.search(r"Important words: \[(.*?)\]", response_text)
    important_words = ([w.strip().strip('"').strip("'") for w in important_match.group(1).split(",")]
                       if important_match else [])
    final_output_match = re.search(r"Final output:\s*({.*})", response_text, re.DOTALL)
    final_output = json.loads(final_output_match.group(1)) if final_output_match else {}
    return important_words, final_output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--template", type=str, default="template_prompt.txt")
    p.add_argument("--parquet_path", type=str, default="metadata.parquet")
    p.add_argument("--cache_dir", type=str, default="../cache")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    # Read the token from the environment; do NOT hard-code it.
    auth_token = os.environ.get("HF_TOKEN")

    accelerator = Accelerator()
    device = accelerator.device
    rank, world_size = accelerator.process_index, accelerator.num_processes
    output_path = f"output_rank{rank}.jsonl"
    error_log_path = f"error_rank{rank}.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, token=auth_token, trust_remote_code=True, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, token=auth_token,
        trust_remote_code=True, cache_dir=args.cache_dir).to(device)
    model.eval()

    with open(args.template, "r", encoding="utf-8") as f:
        base_context = f.read()

    # Load the DiffusionDB prompt set.
    if not Path(args.parquet_path).exists():
        print("Downloading DiffusionDB metadata.parquet ...")
        urlretrieve(DIFFUSIONDB_PARQUET, args.parquet_path)
    import pandas as pd
    prompts = pd.read_parquet(args.parquet_path)["prompt"].dropna().unique().tolist()

    # Skip prompts already processed by any rank, then shard.
    processed = set()
    for file in Path(".").glob("output_rank*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["prompt"])
                except Exception:
                    pass
    prompts = [p for p in prompts if p not in processed][rank::world_size]

    def query_llm_batched(batch_prompts):
        inputs = tokenizer(batch_prompts, padding=True, truncation=True,
                           max_length=4096, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature, do_sample=True,
                                 pad_token_id=tokenizer.eos_token_id)
        return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    idx_counter = 1
    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    idx_counter = max(idx_counter, int(json.loads(line)["id"]) + 1)
                except Exception:
                    pass

    with open(output_path, "a", encoding="utf-8") as fw, open(error_log_path, "a", encoding="utf-8") as fe:
        for start in tqdm(range(0, len(prompts), args.batch_size), desc="Processing prompts"):
            batch = prompts[start:start + args.batch_size]
            try:
                responses = query_llm_batched([base_context + PROMPT_TEMPLATE.format(p) for p in batch])
            except Exception as e:
                for bp in batch:
                    fe.write(json.dumps({"prompt": bp, "error": str(e)}, ensure_ascii=False) + "\n")
                continue
            accelerator.wait_for_everyone()
            for prompt_text, response_text in zip(batch, responses):
                try:
                    important_words, final_output = extract_last_response_info(response_text)
                    fw.write(json.dumps({
                        "id": f"{idx_counter:08d}", "prompt": prompt_text,
                        "important_words": important_words,
                        "neg_prompts": final_output.get("neg_prompts", []),
                    }, ensure_ascii=False) + "\n")
                except Exception as e:
                    fe.write(json.dumps({"prompt": prompt_text, "response_text": response_text,
                                         "error": str(e)}, ensure_ascii=False) + "\n")
                idx_counter += 1


if __name__ == "__main__":
    main()
