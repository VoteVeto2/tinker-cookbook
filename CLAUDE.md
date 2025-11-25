# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`tinker-cookbook` is the client-side library for the hosted Tinker service. Training/eval loops run on CPU; Tinker executes GPU work (LoRA fine-tuning, sampling, checkpointing) on synchronized worker pools.

## Build & Test Commands

```bash
# Setup (requires TINKER_API_KEY env var)
pip install tinker
pip install -e .[dev]
# Or with uv: uv pip install -e .[dev]

# Run tests
pytest tinker_cookbook/tests/test_renderers.py
pytest tinker_cookbook/tests/test_utils.py

# Lint
ruff check .

# Type check
pyright
```

## Running Recipes

```bash
# Basic SFT
python -m tinker_cookbook.recipes.sl_basic model_name=meta-llama/Llama-3.2-1B log_path=/tmp/tinker-examples/sl_basic

# Basic RL
python -m tinker_cookbook.recipes.rl_basic model_name=meta-llama/Llama-3.1-8B log_path=/tmp/tinker-examples/rl_basic

# DPO training
python -m tinker_cookbook.recipes.preference.train log_path=/tmp/dpo-run model_name=meta-llama/Llama-3.2-1B

# Inspect eval
python -m tinker_cookbook.eval.run_inspect_evals model_path=tinker://YOUR_MODEL model_name=meta-llama/Llama-3.2-1B tasks=<task_id>
```

## Architecture

### Builder Pattern
Config objects are `chz` dataclasses with `.build()`/`__call__()` methods:
- `SupervisedDatasetBuilder` → `SupervisedDataset`
- `RLDatasetBuilder` → `RLDataset` → `EnvGroupBuilder` → `Env`
- Launch scripts define `CLIConfig` (CLI-facing) that constructs `Config` (training)

### Key Components
- **Completers** (`completers.py`): `TokenCompleter` interface; `TinkerTokenCompleter` wraps `SamplingClient`
- **Renderers** (`renderers.py`): Convert tokens ↔ chat messages; match renderer to tokenizer (`llama3`, `qwen3`, `role_colon`)
- **Training loops**: `supervised/train.py` (SFT), `rl/train.py` (RL), `preference/train_dpo.py` (DPO)
- **Logging**: `ml_log` for metrics (stdout, `metrics.jsonl`, W&B); `logtree` for HTML transcripts

### Dimension Subscripts
- `_P`: Problems/prompts
- `_G`: Groups (rollouts per problem)
- `_T`: Tokens/time
- `_D`: Datums
- Combined: `tokens_P_G_T[p][g][t]` = token for problem p, group g, position t

### Envs
`Env` is single-use (no reset); created by `EnvGroupBuilder`. Groups enable GRPO-style centering or multi-agent comparisons.

## Common Pitfalls

- **LoRA LR**: Use `hyperparam_utils.get_lr(model_name)` - LoRA needs ~10-100x higher LR than full fine-tuning
- **Renderer mismatch**: Match `renderer_name` to tokenizer family; wrong pairing breaks loss weights and stops
- **Async gaps**: Submit `forward_backward_async` and `optim_step_async` back-to-back before awaiting to avoid wasting clock cycles
- **Sampler desync**: Always get new sampling client via `save_weights_and_get_sampling_client` before evals
- **DPO beta/LR**: Start with `dpo_beta=0.1`, LR≈1e-5; too high causes collapse
