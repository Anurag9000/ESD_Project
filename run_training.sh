#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BACKBONE_NAME="${BACKBONE_NAME:-convnextv2_nano}"
WEIGHTS_MODE="${WEIGHTS_MODE:-default}"
for ((i = 1; i <= $#; i++)); do
  arg="${!i}"
  case "$arg" in
    --backbone)
      next_index=$((i + 1))
      if [[ $next_index -le $# ]]; then
        BACKBONE_NAME="${!next_index}"
      fi
      ;;
    --backbone=*)
      BACKBONE_NAME="${arg#*=}"
      ;;
    --weights)
      next_index=$((i + 1))
      if [[ $next_index -le $# ]]; then
        WEIGHTS_MODE="${!next_index}"
      fi
      ;;
    --weights=*)
      WEIGHTS_MODE="${arg#*=}"
      ;;
  esac
done
export BACKBONE_NAME WEIGHTS_MODE

# ── AUTO-RESUME LOGIC ──
# If no RUN_STAMP is set in the environment, detect the most recent one automatically
RUN_PREFIX="${BACKBONE_NAME}_three_classes"
if [[ -z "${RUN_STAMP:-}" ]]; then
  LATEST_RUN=$(ls -td Results/${RUN_PREFIX}_* 2>/dev/null | head -1 || true)
  if [[ -n "$LATEST_RUN" ]]; then
    # Extract the timestamp part
    RUN_STAMP=$(basename "$LATEST_RUN" | sed "s/^${RUN_PREFIX}_//")
    echo "🔄 Found previous run: $RUN_STAMP. Enforcing automatic resume."
  else
    RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
    echo "🚀 Starting new run: $RUN_STAMP"
  fi
fi

RUN_ROOT="${RUN_ROOT:-Results/${RUN_PREFIX}_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/${RUN_PREFIX}_${RUN_STAMP}}"
PROGRESSIVE_OUTPUT_DIR="$RUN_ROOT/progressive"
PROGRESSIVE_LOG_FILE="$LOG_ROOT/progressive.log.jsonl"
DATASET_ROOT="${DATASET_ROOT:-Dataset_Final}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
SEED="${SEED:-42}"
ENABLE_PHASE0_MIM=0
PHASE0_MIM_EPOCHS="${PHASE0_MIM_EPOCHS:-0}"
PHASE0_MIM_BATCH_SIZE="${PHASE0_MIM_BATCH_SIZE:-8}"
PHASE0_MIM_ACCUM_STEPS="${PHASE0_MIM_ACCUM_STEPS:-40}"
PHASE0_MIM_MASK_RATIO="${PHASE0_MIM_MASK_RATIO:-0.6}"
PHASE0_MIM_PATCH_SIZE="${PHASE0_MIM_PATCH_SIZE:-32}"
PHASE0_MIM_DECODER_DIM="${PHASE0_MIM_DECODER_DIM:-512}"
PHASE0_MIM_LR="${PHASE0_MIM_LR:-1.5e-4}"
PHASE0_MIM_WEIGHT_DECAY="${PHASE0_MIM_WEIGHT_DECAY:-0.05}"
PHASE0_MIM_TRAIN_LOSS_WINDOW="${PHASE0_MIM_TRAIN_LOSS_WINDOW:-5000}"

FILTERED_ARGS=()
IGNORED_ARGS=()
PHASE0_ARGS=()
SKIP_NEXT=0
NEXT_PHASE0_VAR=""
for ARG in "$@"; do
  if [[ "$SKIP_NEXT" -eq 1 ]]; then
    if [[ -n "$NEXT_PHASE0_VAR" ]]; then
      printf -v "$NEXT_PHASE0_VAR" '%s' "$ARG"
      NEXT_PHASE0_VAR=""
    fi
    SKIP_NEXT=0
    continue
  fi
  case "$ARG" in
    --phase0-mim)
      ENABLE_PHASE0_MIM=1
      ;;
    --phase0-mim-epochs|--phase0-mim-batch-size|--phase0-mim-accum-steps|--phase0-mim-mask-ratio|--phase0-mim-patch-size|--phase0-mim-decoder-dim|--phase0-mim-learning-rate|--phase0-mim-weight-decay|--phase0-mim-train-loss-window)
      case "$ARG" in
        --phase0-mim-epochs) NEXT_PHASE0_VAR=PHASE0_MIM_EPOCHS ;;
        --phase0-mim-batch-size) NEXT_PHASE0_VAR=PHASE0_MIM_BATCH_SIZE ;;
        --phase0-mim-accum-steps) NEXT_PHASE0_VAR=PHASE0_MIM_ACCUM_STEPS ;;
        --phase0-mim-mask-ratio) NEXT_PHASE0_VAR=PHASE0_MIM_MASK_RATIO ;;
        --phase0-mim-patch-size) NEXT_PHASE0_VAR=PHASE0_MIM_PATCH_SIZE ;;
        --phase0-mim-decoder-dim) NEXT_PHASE0_VAR=PHASE0_MIM_DECODER_DIM ;;
        --phase0-mim-learning-rate) NEXT_PHASE0_VAR=PHASE0_MIM_LR ;;
        --phase0-mim-weight-decay) NEXT_PHASE0_VAR=PHASE0_MIM_WEIGHT_DECAY ;;
        --phase0-mim-train-loss-window) NEXT_PHASE0_VAR=PHASE0_MIM_TRAIN_LOSS_WINDOW ;;
      esac
      SKIP_NEXT=1
      ;;
    --phase0-encoder-checkpoint|--phase0-encoder-checkpoint=*)
      IGNORED_ARGS+=("${ARG%%=*}")
      if [[ "$ARG" != *=* ]]; then
        SKIP_NEXT=1
      fi
      ;;
    --backbone|--weights|--backbone=*|--weights=*)
      IGNORED_ARGS+=("${ARG%%=*}")
      if [[ "$ARG" != *=* ]]; then
        SKIP_NEXT=1
      fi
      ;;
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

PHASE0_ENCODER_CHECKPOINT=""
if [[ "$ENABLE_PHASE0_MIM" -eq 1 ]]; then
  PHASE0_OUTPUT_DIR="$RUN_ROOT/phase0_mim"
  PHASE0_LOG_FILE="$LOG_ROOT/phase0_mim.log.jsonl"
  PHASE0_COMPLETE_MARKER="$PHASE0_OUTPUT_DIR/.phase0_complete"
  PHASE0_ARGS=()
  PHASE0_ENCODER_CHECKPOINT="$PHASE0_OUTPUT_DIR/phase0_encoder_final.pth"
  if [[ -f "$PHASE0_ENCODER_CHECKPOINT" ]]; then
    echo "✅ Phase 0 MIM already complete; using $PHASE0_ENCODER_CHECKPOINT"
    touch "$PHASE0_COMPLETE_MARKER"
  else
    echo "🧠 Running/resuming Phase 0 MIM pretraining before SupCon/CE..."
    python scripts/train_phase0_mim.py \
      --dataset-root "$DATASET_ROOT" \
      --output-dir "$PHASE0_OUTPUT_DIR" \
      --log-file "$PHASE0_LOG_FILE" \
      --backbone "$BACKBONE_NAME" \
      --weights "$WEIGHTS_MODE" \
      --image-size "$IMAGE_SIZE" \
      --batch-size "$PHASE0_MIM_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --prefetch-factor "$PREFETCH_FACTOR" \
      --epochs "$PHASE0_MIM_EPOCHS" \
      --grad-accum-steps "$PHASE0_MIM_ACCUM_STEPS" \
      --mask-ratio "$PHASE0_MIM_MASK_RATIO" \
      --patch-size "$PHASE0_MIM_PATCH_SIZE" \
      --decoder-dim "$PHASE0_MIM_DECODER_DIM" \
      --learning-rate "$PHASE0_MIM_LR" \
      --weight-decay "$PHASE0_MIM_WEIGHT_DECAY" \
      --train-loss-window "$PHASE0_MIM_TRAIN_LOSS_WINDOW" \
      --seed "$SEED"
    touch "$PHASE0_COMPLETE_MARKER"
  fi
  if [[ ! -f "$PHASE0_ENCODER_CHECKPOINT" ]]; then
    echo "Phase 0 did not produce $PHASE0_ENCODER_CHECKPOINT" >&2
    exit 1
  fi
  PHASE0_ARGS=(--phase0-encoder-checkpoint "$PHASE0_ENCODER_CHECKPOINT")
fi

AUTO_RESUME_ARGS=()
PROGRESSIVE_COMPLETE_MARKER="$PROGRESSIVE_OUTPUT_DIR/.progressive_complete"
if [[ -f "$PROGRESSIVE_COMPLETE_MARKER" && -f "$PROGRESSIVE_OUTPUT_DIR/best.pt" ]]; then
  echo "✅ Progressive SupCon/CE already complete; using $PROGRESSIVE_OUTPUT_DIR/best.pt"
elif [[ -f "$PROGRESSIVE_OUTPUT_DIR/step_last.pt" ]]; then
  echo "🔄 Auto-resuming from EXACT LAST STEP: $PROGRESSIVE_OUTPUT_DIR/step_last.pt"
  AUTO_RESUME_ARGS=(--resume-checkpoint "$PROGRESSIVE_OUTPUT_DIR/step_last.pt")
elif [[ -f "$PROGRESSIVE_OUTPUT_DIR/last.pt" ]]; then
  echo "🔄 Auto-resuming from LAST COMPLETED EPOCH: $PROGRESSIVE_OUTPUT_DIR/last.pt"
  AUTO_RESUME_ARGS=(--resume-checkpoint "$PROGRESSIVE_OUTPUT_DIR/last.pt")
fi

if [[ ! -f "$PROGRESSIVE_COMPLETE_MARKER" || ! -f "$PROGRESSIVE_OUTPUT_DIR/best.pt" ]]; then
  python scripts/train_efficientnet_b0_progressive.py \
    --dataset-root "$DATASET_ROOT" \
    --sampling-strategy balanced \
    --output-dir "$PROGRESSIVE_OUTPUT_DIR" \
    --log-file "$PROGRESSIVE_LOG_FILE" \
    --backbone "$BACKBONE_NAME" \
    --weights "$WEIGHTS_MODE" \
    "${PHASE0_ARGS[@]}" \
    "${AUTO_RESUME_ARGS[@]}" \
    "${FILTERED_ARGS[@]}"
  touch "$PROGRESSIVE_COMPLETE_MARKER"
fi

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
