# Prompt preparation

Builds the (prompt, degraded-prompts) JSONL that `ddspo/prepare_latents.py`
turns into paired latents. Each output line looks like:

```json
{"id": "00000001", "tag": "DiffusionDB", "prompt": "...", "neg_prompts": ["...", "..."]}
```

Two ways to produce the degraded ("negative") prompts:

## 1. Random word removal (no model needed)

```bash
python random_removal.py --output diffusiondb_removal.jsonl
```

Drops 40–70% of the words from each prompt. Fast and reproducible.

## 2. Llama-3 rewriting (LLM-based degradation)

Requires access to `meta-llama/Meta-Llama-3-8B-Instruct`. Authenticate first —
**never hard-code a token**:

```bash
export HF_TOKEN=...          # or: huggingface-cli login
accelerate launch llama_negatives.py --template template_prompt.txt
python concat_ranks.py --pattern 'output_rank*.jsonl' --output llama_negatives.jsonl
```

`llama_negatives.py` shards prompts across processes and writes per-rank files;
`concat_ranks.py` merges them.
