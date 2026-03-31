#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR="Results/efficientnet_b0_full_model_rawacc_refine_adamw"
LOG_FILE="logs/efficientnet_b0_full_model_rawacc_refine_adamw.log.jsonl"
LOSS_OUTPUT_DIR="Results/efficientnet_b0_full_model_loss_cleanup_adamw"
LOSS_STATE_JSON="$LOSS_OUTPUT_DIR/recursive_state.json"
BASE_CHECKPOINT="$LOSS_OUTPUT_DIR/accepted_best.pt"
INITIAL_HEAD_LR="5e-5"
INITIAL_BACKBONE_LR="3e-5"

if [[ -f "$LOSS_STATE_JSON" ]]; then
  eval "$(
    .venv/bin/python scripts/derive_recursive_bootstrap.py \
      --state-json "$LOSS_STATE_JSON" \
      --fallback-checkpoint "$BASE_CHECKPOINT" \
      --fallback-head-lr 1e-4 \
      --fallback-backbone-lr 5e-5 \
      --halve-lrs \
      --use-half-backbone-for-both
  )"
  BASE_CHECKPOINT="$BOOTSTRAP_CHECKPOINT"
  INITIAL_HEAD_LR="$BOOTSTRAP_HEAD_LR"
  INITIAL_BACKBONE_LR="$BOOTSTRAP_BACKBONE_LR"
fi

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  BASE_CHECKPOINT="$LOSS_OUTPUT_DIR/best.pt"
fi
if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  BASE_CHECKPOINT="$LOSS_OUTPUT_DIR/last.pt"
fi

python scripts/run_recursive_refinement.py \
  --base-output-dir "$OUTPUT_DIR" \
  --base-log-file "$LOG_FILE" \
  --initial-checkpoint "$BASE_CHECKPOINT" \
  --metric val_raw_acc \
  --threshold 0.0005 \
  --initial-head-lr "$INITIAL_HEAD_LR" \
  --initial-backbone-lr "$INITIAL_BACKBONE_LR" \
  --dataset-root Dataset_Final \
  --weighted-sampling \
  --skip-supcon \
  --optimizer adamw \
  --batch-size 224 \
  --stage-epochs 30 \
  --patience 5 \
  --head-epochs 0 \
  --resume-phase-index 1 \
  "$@"
