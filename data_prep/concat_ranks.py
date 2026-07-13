#!/usr/bin/env python
# coding=utf-8
"""Merge the per-rank JSONL shards produced by ``llama_negatives.py``."""

import argparse
import glob
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", type=str, default="output_rank*.jsonl")
    p.add_argument("--output", type=str, default="llama_negatives.jsonl")
    args = p.parse_args()

    n = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for path in sorted(glob.glob(args.pattern)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        n += 1
    print(f"Merged {n} entries into {args.output}")


if __name__ == "__main__":
    main()
