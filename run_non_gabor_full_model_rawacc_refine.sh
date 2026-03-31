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
  --stage-early-stopping-patience 5 \
  --classifier-train-mode full_model \
  --classifier-early-stopping-metric val_raw_acc \
  --head-lr 5e-5 \
  --backbone-lr 3e-5 \
  --output-dir Results/efficientnet_b0_full_model_rawacc_refine_adamw \
  --log-file logs/efficientnet_b0_full_model_rawacc_refine_adamw.log.jsonl \
  --resume-checkpoint Results/efficientnet_b0_full_model_loss_cleanup_adamw/last.pt \
  --resume-mode global_best \
  --resume-phase-index 1 \
  "$@"
