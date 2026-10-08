# Anima 3.8B v1.1 training

This fork connects the existing Standalone Jobs UI to the native 3.8B trainer from
[GumGum10/sd-scripts](https://github.com/GumGum10/sd-scripts), pinned at
`2d76db07396f2a36cc1dbd484d32d010cc3bcb46`.
The original Standalone code and license remain in this repository.

## Models and conditioning

Set all four files in **Global Settings → Anima 3.8B v1.1**:

| UI field | File |
| --- | --- |
| Bundle | `Anima-3.8B-v1.1.safetensors` |
| Qwen3 0.6B | `qwen_3_06b_base.safetensors` |
| Qwen3.5 4B | `qwen35_4b.safetensors` |
| VAE | `qwen_image_vae.safetensors` |

Get the models from [lylogummy/Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B).
The 3.8B diffusion transformer has 52 blocks. Qwen3.5 provides semantic features
alongside the native Qwen3 quality path; it does not replace Qwen3.

The v1.1 Semantic Connector v2 is extracted from the selected bundle into an
ignored local cache. Its cache identity includes the model path, size, and mtime,
so selecting a different bundle does not reuse a connector from the previous one.
The DiT loader separates only bundled connector tensors and still checks the
remaining model weights. The exact extracted connector is evaluated at each
training and sampling timestep.

The text encoders and connector are frozen. Only the DiT LoRA is trained.
Latents and both text-conditioning streams are cached to disk; text encoders can
leave GPU memory after caching. Caption dropout uses the native cached-output
implementation. Caption shuffling/tag dropout remain subject to the native
backend's cache restrictions.

## Initial style preset

New jobs use this starting point, which was exercised on one RTX 4090 24 GB:

| Setting | Default |
| --- | --- |
| Batch / gradient accumulation | 2 / 1 |
| Resolution / buckets | 1536 × 1536, no upscale, 16-pixel bucket steps |
| Rank / alpha | 32 / 32 |
| Optimizer / learning rate | AdamW / 2e-5 |
| Scheduler | Constant with 100 warmup steps |
| Checkpoint / samples interval | 250 optimizer steps |
| Maximum steps | 12000 |
| Caption dropout | 0.05 |
| Precision / gradient checkpointing | bf16 / enabled |
| Block swapping | 0 |
| Qwen3 / Qwen3.5 / T5 token limits | 512 / 512 / 512 |

These are starting values, not a universal best preset. The adapter preserves
the job's learning rate, rank, alpha, optimizer arguments, gradient accumulation,
step limit, output paths, dataset config, resume state, and sample interval.
Each dataset subset's repeats stay as configured in the Dataset tab.
The token limits can also be set in the job's `anima38_arguments` section as
`qwen3_max_token_length`, `qwen35_max_token_length`, and `t5_token_length`.

For this native mode the adapter enables mandatory conditioning/latent caches,
DiT-only training, 2D Qwen VAE, torch attention, and sigmoid timestep sampling
with no additional loss weighting. Text encoder LR is zero. The native backend
uses `constant_with_warmup` when the UI selects Constant with nonzero warmup.

## Saved trials and comparison

Enter your prompts in **Prompts**, then enable sampling in **Training**. The
sampler uses the supplied positive and negative text, seed, size, CFG, and steps.
It adds no extra prompt words or variants. An empty prompt list skips sampling.

The saved-trial sampler uses Euler flow integration with flow shift 3. Its
timesteps use the model dtype to avoid the FP32/BF16 AdaLN failure, while sigma
and integration arithmetic stay FP32. Sampling restores RNG and training mode.
Model/state checkpoints are saved before step-based sampling, so a failed trial
does not discard the checkpoint already reached.

Example files:

```text
output/mystyle-step00006000.safetensors
output/sample/step_006000/mystyle-step00006000__prompt_01_seed_42.png
output/sample/step_006000/mystyle-step00006000__prompt_02_seed_42.png
```

**Samples** places all versions of Prompt 1 together, Prompt 2 together, and so
on, with newer training steps first. The card shows the step; the filename names
the checkpoint. Existing original Standalone filenames and older cloud filenames
also work. A sample generated without a corresponding saved checkpoint is
labelled `preview_step`, rather than claiming a checkpoint exists.

New PNGs retain the original trainer's `parameters` metadata format and also
record training step, prompt index, and checkpoint. A JSON manifest carries only
file/checkpoint/seed/index metadata. The gallery groups by filenames and directory
metadata; it does not need to read PNG pixels or caption files.

The per-prompt display limit controls how many thumbnails are shown, not how many
files are retained. Choose **All** to see every version in each prompt group.

## Current scope and validation

The native 3.8B adapter supports single-GPU LoRA training and saved trials during
training. Manual Generate, native 3.8B full finetuning, and native 3.8B multi-GPU
training are not implemented by this adapter. Legacy architectures keep their
original entry points. Use the separate `venv38` for 3.8B; use your legacy venv
when running the original architectures.

The cloud training integration produced checkpoints and saved trials past
11000 steps on an RTX 4090. The portable packaging is checked with CPU imports,
configuration-preservation tests, pinned-backend installation/patch checks, and
gallery grouping tests. A fresh Windows GPU training run has not been performed.

Run the checks:

```text
python -m unittest discover -s tests
node --test training-ui/tests/*.test.js
```

Advanced deployments can set `ANIMA38_BACKEND_DIR` and `ANIMA38_CACHE_DIR` to
reuse an existing patched native backend and cache. These paths are local
deployment settings and are never required to match a particular cloud server.
