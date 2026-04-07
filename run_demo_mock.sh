#!/usr/bin/env bash
set -euo pipefail

# Demo Mock Run Script
# Objective: Verify end-to-end training pipeline with minimal data and rapid transitions.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# 1. Create Mock Dataset (1 train, 1 val, 1 test per class)
echo "[*] Creating Mock Dataset..."
MOCK_ROOT="Dataset_Mock"
mkdir -p "$MOCK_ROOT"/{train,val,test}/{plastic,organic,glass,metal}
.venv/bin/python3 -c "
from PIL import Image
import os
for cls in ['plastic', 'organic', 'glass', 'metal']:
    for split in ['train', 'val', 'test']:
        img = Image.new('RGB', (224, 224), color=(255, 0, 0))
        img.save(f'$MOCK_ROOT/{split}/{cls}/mock.jpg')
"

# 2. Define Demo Paths
RUN_ID="demo_run_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="Results/$RUN_ID"
LOG_ROOT="logs/$RUN_ID"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

# 3. Run Progressive Training (2 epochs per phase, mid-epoch val)
echo "[*] Phase 1: Progressive Training (SupCon + Progressive CE)..."
.venv/bin/python3 scripts/train_efficientnet_b0_progressive.py \
  --dataset-root "$MOCK_ROOT" \
  --batch-size 4 \
  --supcon-epochs 2 \
  --head-epochs 1 \
  --stage-epochs 2 \
  --unfreeze-chunk-size 50 \
  --optimizer adamw \
  --eval-every-epochs 0.5 \
  --log-every-steps 1 \
  --output-dir "$RUN_ROOT/progressive" \
  --log-file "$LOG_ROOT/progressive.log.jsonl" \
  --max-train-batches 2 \
  --max-eval-batches 2

# 4. Transition to Recursive Refinement
PROGRESSIVE_BEST="$RUN_ROOT/progressive/best.pt"
if [[ ! -f "$PROGRESSIVE_BEST" ]]; then
  echo "[!] Progressive training failed to produce best.pt"
  exit 1
fi

# 5. Run Recursive Loss Refinement
echo "[*] Phase 2: Recursive Loss Refinement (LR Halving)..."
.venv/bin/python3 scripts/run_recursive_refinement.py \
  --base-output-dir "$RUN_ROOT/loss_cleanup" \
  --base-log-file "$LOG_ROOT/loss_cleanup.log.jsonl" \
  --initial-checkpoint "$PROGRESSIVE_BEST" \
  --metric val_loss \
  --threshold 0.0001 \
  --initial-head-lr 1e-4 \
  --initial-backbone-lr 5e-5 \
  --dataset-root "$MOCK_ROOT" \
  --batch-size 4 \
  --stage-epochs 2 \
  --patience 2 \
  --skip-supcon \
  -- \
  --eval-every-epochs 0.5 --log-every-steps 1 --max-train-batches 2 --max-eval-batches 2

# 6. Run Recursive Accuracy Refinement
echo "[*] Phase 3: Recursive Accuracy Refinement (LR Halving)..."
LOSS_ACCEPTED="$RUN_ROOT/loss_cleanup/accepted_best.pt"
.venv/bin/python3 scripts/run_recursive_refinement.py \
  --base-output-dir "$RUN_ROOT/rawacc_refine" \
  --base-log-file "$LOG_ROOT/rawacc_refine.log.jsonl" \
  --initial-checkpoint "${LOSS_ACCEPTED:-$PROGRESSIVE_BEST}" \
  --metric val_raw_acc \
  --threshold 0.0005 \
  --initial-head-lr 5e-5 \
  --initial-backbone-lr 3e-5 \
  --dataset-root "$MOCK_ROOT" \
  --batch-size 4 \
  --stage-epochs 2 \
  --patience 2 \
  --skip-supcon \
  -- \
  --eval-every-epochs 0.5 --log-every-steps 1 --max-train-batches 2 --max-eval-batches 2

echo "[*] Demo Mock Run Complete."
echo "Results stored in: $RUN_ROOT"
echo "Logs stored in: $LOG_ROOT"
echo "Metrics CSVs and Confusion Matrices generated in subfolders."
