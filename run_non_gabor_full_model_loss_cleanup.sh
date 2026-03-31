#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/train_efficientnet_b0_progressive.py \
  --dataset-root Dataset_Final \
  --weighted-sampling \
  --skip-supcon \
  --optimizer adamw \
  --batch-size 224 \
  --head-epochs 0 \
  --stage-epochs 30 \
  --stage-early-stopping-patience 8 \
  --classifier-train-mode full_model \
  --classifier-early-stopping-metric val_loss \
  --head-lr 1e-4 \
  --backbone-lr 5e-5 \
  --output-dir Results/ce_full_model_loss_cleanup \
  --log-file logs/ce_full_model_loss_cleanup.log.jsonl \
  --resume-checkpoint Results/ce_progressive_no_supcon_no_sam/last.pt \
  --resume-mode global_best \
  --resume-phase-index 1 \
  "$@"
