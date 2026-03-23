#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-cu128.txt

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("torch_cuda", torch.version.cuda)
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    print("device_name", torch.cuda.get_device_name(0))
PY
