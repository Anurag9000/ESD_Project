#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR="Results/efficientnet_b0_full_model_metal_focus_adamw"
LOG_FILE="logs/efficientnet_b0_full_model_metal_focus_adamw.log.jsonl"
BASE_CHECKPOINT="Results/efficientnet_b0_full_model_loss_cleanup_adamw/iteration_003/step_last.pt"

python scripts/train_efficientnet_b0_progressive.py \
  --dataset-root Dataset_Final \
  --optimizer adamw \
  --batch-size 224 \
  --head-epochs 0 \
  --stage-epochs 30 \
  --stage-early-stopping-patience 5 \
  --classifier-train-mode full_model \
  --classifier-early-stopping-metric val_loss \
  --head-lr 2.5e-5 \
  --backbone-lr 1.25e-5 \
  --confidence-gap-penalty-weight 0.0 \
  --class-loss-weight metal=2.0 \
  --targeted-confusion-penalty metal:other:0.5 \
  --targeted-confusion-penalty other:metal:0.2 \
  --weighted-sampling \
  --skip-supcon \
  --output-dir "$OUTPUT_DIR" \
  --log-file "$LOG_FILE" \
  --resume-checkpoint "$BASE_CHECKPOINT" \
  --resume-mode global_best \
  --resume-phase-index 1 \
  "$@"
