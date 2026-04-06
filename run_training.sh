#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-Results/efficientnet_b0_all_classes_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/efficientnet_b0_all_classes_${RUN_STAMP}}"
PROGRESSIVE_OUTPUT_DIR="$RUN_ROOT/progressive"
PROGRESSIVE_LOG_FILE="$LOG_ROOT/progressive.log.jsonl"
DATASET_ROOT="${DATASET_ROOT:-Dataset_Final}"

FILTERED_ARGS=()
SKIP_NEXT=0
for ARG in "$@"; do
  if [[ "$SKIP_NEXT" -eq 1 ]]; then
    SKIP_NEXT=0
    continue
  fi
  case "$ARG" in
    --dataset-root|--batch-size|--output-dir|--log-file)
      SKIP_NEXT=1
      ;;
    --dataset-root=*|--batch-size=*|--output-dir=*|--log-file=*)
      ;;
    *)
      FILTERED_ARGS+=("$ARG")
      ;;
  esac
done

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

python scripts/train_efficientnet_b0_progressive.py \
  --dataset-root "$DATASET_ROOT" \
  --weighted-sampling \
  --batch-size 224 \
  --output-dir "$PROGRESSIVE_OUTPUT_DIR" \
  --log-file "$PROGRESSIVE_LOG_FILE" \
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
