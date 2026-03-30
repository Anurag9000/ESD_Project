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
  --batch-size 384 \
  --head-epochs 0 \
  --stage-epochs 40 \
  --stage-early-stopping-patience 8 \
  --classifier-train-mode full_model \
  --classifier-early-stopping-metric val_raw_acc \
  --head-lr 3e-4 \
  --backbone-lr 3e-5 \
  --output-dir Results/ce_full_model_rawacc_refine \
  --log-file logs/ce_full_model_rawacc_refine.log.jsonl \
  --resume-checkpoint Results/ce_progressive_no_supcon_no_sam/last.pt \
  --resume-mode global_best \
  --resume-phase-index 1 \
  "$@"
