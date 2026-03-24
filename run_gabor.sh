#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

python scripts/train_efficientnet_b0_gabor_progressive.py \
  --dataset-root Dataset_Final \
  --weighted-sampling \
  "$@"
