#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

python scripts/compare_head_only_ce_vs_arcface.py \
  --dataset-root Dataset_Final \
  --weighted-sampling \
  "$@"
