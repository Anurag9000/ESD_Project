#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR="Results/efficientnet_b0_full_model_rawacc_refine_adamw"
LOG_FILE="logs/efficientnet_b0_full_model_rawacc_refine_adamw.log.jsonl"
BASE_CHECKPOINT="Results/efficientnet_b0_full_model_loss_cleanup_adamw/accepted_best.pt"

python scripts/run_recursive_refinement.py \
  --base-output-dir "$OUTPUT_DIR" \
  --base-log-file "$LOG_FILE" \
  --initial-checkpoint "$BASE_CHECKPOINT" \
  --metric val_raw_acc \
  --threshold 0.0005 \
  --initial-head-lr 5e-5 \
  --initial-backbone-lr 3e-5 \
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
