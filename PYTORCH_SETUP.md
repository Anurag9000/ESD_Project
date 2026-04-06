# PyTorch Setup

This document covers the current training-side environment and workflow for this repo.

## Scope

The retained training path is now:

- dynamic all-class training from a flat dataset root
- EfficientNet-B0 only
- progressive full-model fine-tuning
- recursive loss refinement
- recursive raw-accuracy refinement

The old fixed-class assumptions are no longer part of the maintained workflow.

## Environment

Target environment:

- Linux
- Python virtual environment in `.venv`
- CUDA-enabled PyTorch wheels
- NVIDIA GPU

Create the venv and install the CUDA builds with:

```bash
chmod +x scripts/setup_venv_cuda.sh
./scripts/setup_venv_cuda.sh .venv
```

The installer uses [requirements-cu128.txt](/home/anurag-basistha/Projects/ESD/requirements-cu128.txt), which points `pip` at the official CUDA 12.8 PyTorch wheel index.

## Current Dataset Contract

The training code no longer expects:

- `Dataset_Final/train/<class>`
- `Dataset_Final/val/<class>`
- `Dataset_Final/test/<class>`

Instead, it expects a flat dataset root:

- `Dataset_Final/<class>/<image files>`

Classes are inferred from folder names at runtime.

The current local dataset classes are:

- `battery`
- `clothes`
- `ewaste`
- `glass`
- `metal`
- `organic`
- `paper`
- `plastic`
- `shoes`
- `trash`

The shared pipeline creates deterministic stratified train/val/test splits automatically from the flat root.

## Core Training Files

- [run_training.sh](/home/anurag-basistha/Projects/ESD/run_training.sh)
- [run_full_training_pipeline.sh](/home/anurag-basistha/Projects/ESD/run_full_training_pipeline.sh)
- [scripts/train_efficientnet_b0_progressive.py](/home/anurag-basistha/Projects/ESD/scripts/train_efficientnet_b0_progressive.py)
- [scripts/metric_learning_pipeline.py](/home/anurag-basistha/Projects/ESD/scripts/metric_learning_pipeline.py)
- [scripts/run_recursive_refinement.py](/home/anurag-basistha/Projects/ESD/scripts/run_recursive_refinement.py)
- [scripts/evaluate_saved_classifier.py](/home/anurag-basistha/Projects/ESD/scripts/evaluate_saved_classifier.py)

## Training Flow

The current master flow is:

1. initialize pretrained `efficientnet_b0`
2. infer classes from the dataset root
3. auto-build deterministic train/val/test splits
4. run progressive fine-tuning from head to deeper backbone slices
5. restore the best progressive checkpoint
6. launch recursive loss refinement with LR halving
7. stop loss recursion when the configured improvement threshold is no longer met
8. launch recursive raw-accuracy refinement with LR halving
9. stop accuracy recursion when its threshold is no longer met

The main entry point is:

```bash
cd /home/anurag-basistha/Projects/ESD
powerprofilesctl set performance
./run_training.sh
```

## Optimization Defaults

Current retained defaults:

- backbone: `efficientnet_b0`
- default batch size: `224`
- optimizer family: `AdamW`
- scheduler: warmup + cosine decay
- classifier loss: cross-entropy
- progressive unfreeze chunking: retained
- recursive refinement: retained

The repo philosophy is now:

- train on every available raw class
- do not hardcode the class taxonomy
- collapse unselected classes into runtime `Other` only at evaluation or app/runtime use

## Outputs

Training runs write local artifacts under `Results/` and `logs/` when you actually run them.

Typical outputs include:

- progressive checkpoints
- recursive refinement checkpoints
- evaluation summaries
- JSONL training logs

These folders are treated as local run artifacts unless explicitly retained.

## Evaluation

Saved-model evaluation is handled by:

- [scripts/evaluate_saved_classifier.py](/home/anurag-basistha/Projects/ESD/scripts/evaluate_saved_classifier.py)

It supports runtime-style class collapsing, so you can keep only selected classes explicit and merge every other trained class into `Other`.

Example:

```bash
cd /home/anurag-basistha/Projects/ESD
.venv/bin/python scripts/evaluate_saved_classifier.py \
  --checkpoint Results/<run>/rawacc_refine/accepted_best.pt \
  --output-dir Results/<run>/runtime_eval_selected \
  --selected-class metal \
  --selected-class organic \
  --selected-class paper
```

## Practical Notes

- If you change the dataset taxonomy again, the training code should continue to infer the class list from the folder tree.
- If you change class counts heavily, revisit class balance, weighted sampling, and refinement thresholds.
- If you want a new stable training contract, update [README.md](/home/anurag-basistha/Projects/ESD/README.md) first and keep this file aligned with it.
