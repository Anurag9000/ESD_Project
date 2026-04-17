#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── AUTO-RESUME LOGIC ──
# If no RUN_STAMP is set in the environment, detect the most recent one automatically
if [[ -z "${RUN_STAMP:-}" ]]; then
  LATEST_RUN=$(ls -td Results/convnextv2_nano_all_classes_* 2>/dev/null | head -1 || true)
  if [[ -n "$LATEST_RUN" ]]; then
    # Extract the timestamp part
    RUN_STAMP=$(basename "$LATEST_RUN" | sed 's/convnextv2_nano_all_classes_//')
    echo "🔄 Found previous run: $RUN_STAMP. Enforcing automatic resume."
  else
    RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
    echo "🚀 Starting new run: $RUN_STAMP"
  fi
fi

RUN_ROOT="${RUN_ROOT:-Results/convnextv2_nano_all_classes_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/convnextv2_nano_all_classes_${RUN_STAMP}}"
PROGRESSIVE_OUTPUT_DIR="$RUN_ROOT/progressive"
PROGRESSIVE_LOG_FILE="$LOG_ROOT/progressive.log.jsonl"
DATASET_ROOT="${DATASET_ROOT:-Dataset_Final}"

FILTERED_ARGS=()
IGNORED_ARGS=()
SKIP_NEXT=0
for ARG in "$@"; do
  if [[ "$SKIP_NEXT" -eq 1 ]]; then
    SKIP_NEXT=0
    continue
  fi
  case "$ARG" in
    --dataset-root|--batch-size|--output-dir|--log-file)
      IGNORED_ARGS+=("$ARG")
      SKIP_NEXT=1
      ;;
    --dataset-root=*|--batch-size=*|--output-dir=*|--log-file=*)
      IGNORED_ARGS+=("${ARG%%=*}")
      ;;
    *)
      FILTERED_ARGS+=("$ARG")
      ;;
  esac
done

if [[ ${#IGNORED_ARGS[@]} -gt 0 ]]; then
  echo "⚠️ Ignoring wrapper-managed CLI options: ${IGNORED_ARGS[*]}" >&2
  echo "   Use RUN_ROOT, LOG_ROOT, or DATASET_ROOT to control the wrapper-managed paths." >&2
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

AUTO_RESUME_ARGS=()
if [[ -f "$PROGRESSIVE_OUTPUT_DIR/step_last.pt" ]]; then
  echo "🔄 Auto-resuming from EXACT LAST STEP: $PROGRESSIVE_OUTPUT_DIR/step_last.pt"
  AUTO_RESUME_ARGS=(--resume-checkpoint "$PROGRESSIVE_OUTPUT_DIR/step_last.pt")
elif [[ -f "$PROGRESSIVE_OUTPUT_DIR/last.pt" ]]; then
  echo "🔄 Auto-resuming from LAST COMPLETED EPOCH: $PROGRESSIVE_OUTPUT_DIR/last.pt"
  AUTO_RESUME_ARGS=(--resume-checkpoint "$PROGRESSIVE_OUTPUT_DIR/last.pt")
fi

python scripts/train_efficientnet_b0_progressive.py \
  --dataset-root "$DATASET_ROOT" \
  --sampling-strategy balanced \
  --output-dir "$PROGRESSIVE_OUTPUT_DIR" \
  --log-file "$PROGRESSIVE_LOG_FILE" \
  "${AUTO_RESUME_ARGS[@]}" \
  "${FILTERED_ARGS[@]}"

PROGRESSIVE_BEST_CHECKPOINT="$PROGRESSIVE_OUTPUT_DIR/best.pt"
if [[ ! -f "$PROGRESSIVE_BEST_CHECKPOINT" ]]; then
  echo "Progressive run did not produce $PROGRESSIVE_BEST_CHECKPOINT" >&2
  exit 1
fi

INITIAL_CHECKPOINT="$PROGRESSIVE_BEST_CHECKPOINT" \
RUN_ROOT="$RUN_ROOT" \
LOG_ROOT="$LOG_ROOT" \
DATASET_ROOT="$DATASET_ROOT" \
./run_full_training_pipeline.sh "${FILTERED_ARGS[@]}"
