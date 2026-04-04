#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/anurag-basistha/Projects/ESD"
ZIP_PATH="$ROOT/external_datasets/zerowaste/zerowaste-f-final.zip"
LOG_PATH="$ROOT/.tooling/zerowaste_watchdog.log"
EXPECTED_SIZE=7518242799
WGET_PID="${1:-}"

cd "$ROOT"

echo "[$(date --iso-8601=seconds)] zerowaste watchdog started pid=${WGET_PID:-none}" >>"$LOG_PATH"

if [[ -n "$WGET_PID" ]]; then
  while kill -0 "$WGET_PID" 2>/dev/null; do
    if [[ -f "$ZIP_PATH" ]]; then
      SIZE=$(stat -c '%s' "$ZIP_PATH" 2>/dev/null || echo 0)
      echo "[$(date --iso-8601=seconds)] download_progress bytes=$SIZE expected=$EXPECTED_SIZE" >>"$LOG_PATH"
    else
      echo "[$(date --iso-8601=seconds)] download_progress bytes=0 expected=$EXPECTED_SIZE" >>"$LOG_PATH"
    fi
    sleep 120
  done
fi

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "[$(date --iso-8601=seconds)] zerowaste watchdog abort missing_zip" >>"$LOG_PATH"
  exit 1
fi

SIZE=$(stat -c '%s' "$ZIP_PATH")
echo "[$(date --iso-8601=seconds)] download_finished bytes=$SIZE expected=$EXPECTED_SIZE" >>"$LOG_PATH"

if [[ "$SIZE" -lt "$EXPECTED_SIZE" ]]; then
  echo "[$(date --iso-8601=seconds)] zerowaste watchdog abort incomplete_zip" >>"$LOG_PATH"
  exit 1
fi

python3 scripts/integrate_external_dataset.py \
  --source zerowaste \
  --dataset-root Dataset_Final \
  --workspace-root external_datasets >>"$LOG_PATH" 2>&1

echo "[$(date --iso-8601=seconds)] zerowaste watchdog import_finished" >>"$LOG_PATH"
