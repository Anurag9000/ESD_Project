#!/usr/bin/env python3
"""Public ESD metric-learning entrypoint with restart-addressable data semantics."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "training_control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

from esd_deterministic_data_v1 import install  # noqa: E402


def _load_pipeline():
    path = ROOT / "scripts" / "metric_learning_pipeline.py"
    name = "_esd_metric_learning_pipeline_exact_v1"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import ESD metric-learning pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    install(module)
    return module


def main() -> int:
    pipeline = _load_pipeline()
    parser = pipeline.build_parser()
    args = parser.parse_args()
    return int(pipeline.run_experiment(args))


if __name__ == "__main__":
    raise SystemExit(main())
