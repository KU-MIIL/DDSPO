# Evaluation

We evaluate DDSPO checkpoints on standard text-to-image preference / alignment
benchmarks. Each benchmark relies on its own upstream repository and model
weights, which are **not vendored here** (licenses + size). The protocol is:

1. **Generate images** from a trained checkpoint for the benchmark's prompt set.
   A checkpoint saved by `ddspo/train.py` is a standard `diffusers` pipeline
   directory, so it loads with `StableDiffusionPipeline` /
   `StableDiffusionXLPipeline` / `StableDiffusion3Pipeline` / `SanaPipeline`
   (SANA LoRA weights load via `pipeline.load_lora_weights(<output_dir>)`).
2. **Score** the generated images with the benchmark's official scorer.

## Benchmarks and upstream repositories

| Benchmark | Measures | Upstream |
|-----------|----------|----------|
| **GenEval** | compositional alignment | https://github.com/djghosh13/geneval |
| **T2I-CompBench** | attribute/relation binding | https://github.com/Karine-Huang/T2I-CompBench |
| **HPSv2** | human preference score | https://github.com/tgxs002/HPSv2 |
| **PickScore** | human preference score | https://github.com/yuvalkirstain/PickScore |
| **FID** | image quality vs. MS-COCO val2014 | e.g. `clean-fid` |

## Notes

- Install each benchmark following its own instructions (GenEval needs
  `mmdet`/`mmcv` + a Mask2Former detector; T2I-CompBench ships its own
  sub-evaluators; HPSv2/PickScore pull their reward-model weights on first use).
- Point the generator at the benchmark's prompt list and your checkpoint, then
  run the benchmark's scorer over the produced images.
- All benchmark models are inference-only and independent of DDSPO training.
