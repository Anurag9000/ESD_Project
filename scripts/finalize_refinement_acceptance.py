#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {path} is not a dict checkpoint.")
    return checkpoint


def metric_value(checkpoint: dict[str, Any], metric: str, role: str) -> float:
    if role == "candidate":
        resume = checkpoint.get("resume")
        if isinstance(resume, dict):
            if metric == "val_loss" and resume.get("phase_best_loss") is not None:
                return float(resume["phase_best_loss"])
            if metric == "val_raw_acc" and resume.get("phase_best_raw_acc") is not None:
                return float(resume["phase_best_raw_acc"])
    key = "best_val_loss" if metric == "val_loss" else "best_val_raw_acc"
    value = checkpoint.get(key)
    if value is None:
        raise ValueError(f"Checkpoint is missing required metric {key!r}.")
    return float(value)


def improved(metric: str, baseline: float, candidate: float, min_delta: float) -> bool:
    if metric == "val_loss":
        return candidate < baseline - min_delta
    return candidate > baseline + min_delta


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize whether a refinement run is accepted over its baseline.")
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--decision-json", required=True)
    parser.add_argument("--metric", choices=("val_loss", "val_raw_acc"), required=True)
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    candidate_path = Path(args.candidate_checkpoint)
    baseline_path = Path(args.baseline_checkpoint)
    output_path = Path(args.output_checkpoint)
    decision_path = Path(args.decision_json)

    candidate = load_checkpoint(candidate_path)
    baseline = load_checkpoint(baseline_path)

    candidate_metric = metric_value(candidate, args.metric, role="candidate")
    baseline_metric = metric_value(baseline, args.metric, role="baseline")
    accept_candidate = improved(args.metric, baseline_metric, candidate_metric, args.min_delta)

    selected_path = candidate_path if accept_candidate else baseline_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_path, output_path)

    decision = {
        "metric": args.metric,
        "min_delta": args.min_delta,
        "candidate_checkpoint": str(candidate_path),
        "baseline_checkpoint": str(baseline_path),
        "candidate_metric": candidate_metric,
        "baseline_metric": baseline_metric,
        "accepted_candidate": accept_candidate,
        "selected_checkpoint": str(selected_path),
        "output_checkpoint": str(output_path),
    }
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
