#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOSS_OUTPUT_DIR="Results/efficientnet_b0_full_model_loss_cleanup_adamw"
LOSS_LOG_FILE="logs/efficientnet_b0_full_model_loss_cleanup_adamw.log.jsonl"
LOSS_BASE_CHECKPOINT="Results/efficientnet_b0_cleaned_dataset_loss_cleanup_iter003_final/efficientnet_b0_cleaned_dataset_loss_cleanup_iter003_final_best_val_loss.pt"

python scripts/run_recursive_refinement.py \
  --base-output-dir "$LOSS_OUTPUT_DIR" \
  --base-log-file "$LOSS_LOG_FILE" \
  --initial-checkpoint "$LOSS_BASE_CHECKPOINT" \
  --metric val_loss \
  --threshold 0.0001 \
  --initial-head-lr 1e-4 \
  --initial-backbone-lr 5e-5 \
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

RAWACC_OUTPUT_DIR="Results/efficientnet_b0_full_model_rawacc_refine_adamw"
RAWACC_LOG_FILE="logs/efficientnet_b0_full_model_rawacc_refine_adamw.log.jsonl"
LOSS_STATE_JSON="$LOSS_OUTPUT_DIR/recursive_state.json"
RAWACC_BASE_CHECKPOINT="$LOSS_OUTPUT_DIR/accepted_best.pt"
RAWACC_HEAD_LR="5e-5"
RAWACC_BACKBONE_LR="3e-5"

if [[ -f "$LOSS_STATE_JSON" ]]; then
  eval "$(
    .venv/bin/python scripts/derive_recursive_bootstrap.py \
      --state-json "$LOSS_STATE_JSON" \
      --fallback-checkpoint "$RAWACC_BASE_CHECKPOINT" \
      --fallback-head-lr 1e-4 \
      --fallback-backbone-lr 5e-5 \
      --halve-lrs \
      --use-half-backbone-for-both
  )"
  RAWACC_BASE_CHECKPOINT="$BOOTSTRAP_CHECKPOINT"
  RAWACC_HEAD_LR="$BOOTSTRAP_HEAD_LR"
  RAWACC_BACKBONE_LR="$BOOTSTRAP_BACKBONE_LR"
fi

if [[ ! -f "$RAWACC_BASE_CHECKPOINT" ]]; then
  RAWACC_BASE_CHECKPOINT="$LOSS_BASE_CHECKPOINT"
fi

python scripts/run_recursive_refinement.py \
  --base-output-dir "$RAWACC_OUTPUT_DIR" \
  --base-log-file "$RAWACC_LOG_FILE" \
  --initial-checkpoint "$RAWACC_BASE_CHECKPOINT" \
  --metric val_raw_acc \
  --threshold 0.0005 \
  --initial-head-lr "$RAWACC_HEAD_LR" \
  --initial-backbone-lr "$RAWACC_BACKBONE_LR" \
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
