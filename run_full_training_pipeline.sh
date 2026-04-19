#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-}"
RUN_ROOT="${RUN_ROOT:-Results/convnextv2_nano_master_run}"
LOG_ROOT="${LOG_ROOT:-logs/convnextv2_nano_master_run}"
DATASET_ROOT="${DATASET_ROOT:-Dataset_Final}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$RUN_ROOT/progressive/best.pt}"
RECURSIVE_ACCEPTANCE_MIN_DELTA="${RECURSIVE_ACCEPTANCE_MIN_DELTA:-0.0}"

if [[ -z "$INITIAL_CHECKPOINT" ]]; then
  echo "INITIAL_CHECKPOINT must point to the progressive best checkpoint." >&2
  exit 1
fi

if [[ ! -f "$INITIAL_CHECKPOINT" ]]; then
  echo "INITIAL_CHECKPOINT does not exist: $INITIAL_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

FILTERED_ARGS=()
IGNORED_ARGS=()
SKIP_NEXT=0
for ARG in "$@"; do
  if [[ "$SKIP_NEXT" -eq 1 ]]; then
    SKIP_NEXT=0
    continue
  fi
  case "$ARG" in
    --dataset-root|--batch-size|--output-dir|--log-file|--resume-checkpoint|--resume-mode|--resume-phase-index|--classifier-train-mode|--classifier-early-stopping-metric|--head-lr|--backbone-lr|--stage-early-stopping-patience|--optimizer)
      IGNORED_ARGS+=("$ARG")
      SKIP_NEXT=1
      ;;
    --dataset-root=*|--batch-size=*|--output-dir=*|--log-file=*|--resume-checkpoint=*|--resume-mode=*|--resume-phase-index=*|--classifier-train-mode=*|--classifier-early-stopping-metric=*|--head-lr=*|--backbone-lr=*|--stage-early-stopping-patience=*|--optimizer=*)
      IGNORED_ARGS+=("${ARG%%=*}")
      ;;
    *)
      FILTERED_ARGS+=("$ARG")
      ;;
  esac
done

if [[ ${#IGNORED_ARGS[@]} -gt 0 ]]; then
  echo "⚠️ Ignoring wrapper-managed CLI options: ${IGNORED_ARGS[*]}" >&2
  echo "   Use RUN_ROOT, LOG_ROOT, DATASET_ROOT, or INITIAL_CHECKPOINT to control wrapper-managed paths." >&2
fi

LOSS_OUTPUT_DIR="$RUN_ROOT/loss_cleanup"
LOSS_LOG_FILE="$LOG_ROOT/loss_cleanup.log.jsonl"
LOSS_BASE_CHECKPOINT="$INITIAL_CHECKPOINT"

python scripts/run_recursive_refinement.py \
  --base-output-dir "$LOSS_OUTPUT_DIR" \
  --base-log-file "$LOSS_LOG_FILE" \
  --initial-checkpoint "$LOSS_BASE_CHECKPOINT" \
  --metric val_loss \
  --threshold "$RECURSIVE_ACCEPTANCE_MIN_DELTA" \
  --initial-head-lr 1e-4 \
  --initial-backbone-lr 5e-5 \
  --dataset-root "$DATASET_ROOT" \
  --sampling-strategy balanced \
  --skip-supcon \
  --optimizer adamw \
  --batch-size 240 \
  --patience 1 \
  --resume-phase-index 1 \
  "${FILTERED_ARGS[@]}"

RAWACC_OUTPUT_DIR="$RUN_ROOT/rawacc_refine"
RAWACC_LOG_FILE="$LOG_ROOT/rawacc_refine.log.jsonl"
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
  --threshold "$RECURSIVE_ACCEPTANCE_MIN_DELTA" \
  --initial-head-lr "$RAWACC_HEAD_LR" \
  --initial-backbone-lr "$RAWACC_BACKBONE_LR" \
  --dataset-root "$DATASET_ROOT" \
  --sampling-strategy balanced \
  --skip-supcon \
  --optimizer adamw \
  --batch-size 240 \
  --patience 1 \
  --resume-phase-index 1 \
  "${FILTERED_ARGS[@]}"

# ─── Final test-set evaluation on the best accepted model ────────────────────
# Runs ONCE at the very end of the entire pipeline (after all recursive
# refinement is complete). Full 16-aug stochastic test pass — no skipping.
FINAL_CHECKPOINT="$RAWACC_OUTPUT_DIR/accepted_best.pt"
if [[ ! -f "$FINAL_CHECKPOINT" ]]; then
  FINAL_CHECKPOINT="$LOSS_OUTPUT_DIR/accepted_best.pt"
fi
if [[ ! -f "$FINAL_CHECKPOINT" ]]; then
  FINAL_CHECKPOINT="$INITIAL_CHECKPOINT"
fi

FINAL_TEST_OUTPUT_DIR="$RUN_ROOT/final_test_evaluation"
echo "" >& 2
echo "=== Pipeline complete. Running final test-set evaluation ==="  >& 2
echo "    Checkpoint : $FINAL_CHECKPOINT"  >& 2
echo "    Output dir : $FINAL_TEST_OUTPUT_DIR"  >& 2

python scripts/evaluate_saved_classifier.py \
  --checkpoint "$FINAL_CHECKPOINT" \
  --output-dir "$FINAL_TEST_OUTPUT_DIR" \
  --dataset-root "$DATASET_ROOT" \
  --batch-size 240 \
  --evaluation-stage final_test_evaluation \
  --phase-name final_test \
  --splits test

echo "=== Final test evaluation written to $FINAL_TEST_OUTPUT_DIR ===" >& 2
