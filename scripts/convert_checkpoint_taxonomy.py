#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from metric_learning_pipeline import (
    TRAINING_CLASS_ORDER,
    adapt_checkpoint_state_dict_to_training_taxonomy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a legacy checkpoint into the repo's current 6-class logical taxonomy."
    )
    parser.add_argument("--input-checkpoint", required=True, help="Path to the source checkpoint (.pt).")
    parser.add_argument(
        "--output-checkpoint",
        default="",
        help="Optional output path. Defaults to <input_stem>_6class.pt next to the source checkpoint.",
    )
    return parser.parse_args()


def infer_source_phase(checkpoint: dict[str, Any]) -> dict[str, Any]:
    phase_name = checkpoint.get("phase_name")
    phase_index = checkpoint.get("phase_index")
    if phase_name is not None or phase_index is not None:
        return {
            "phase_name": phase_name,
            "phase_index": phase_index,
            "source": "checkpoint_metadata",
        }

    history = checkpoint.get("history") or []
    if isinstance(history, list) and history:
        last_row = history[-1]
        if isinstance(last_row, dict):
            return {
                "phase_name": last_row.get("phase_name"),
                "phase_index": last_row.get("phase_index"),
                "stage": last_row.get("stage"),
                "epoch_in_phase": last_row.get("epoch_in_phase"),
                "validation_index": last_row.get("validation_index"),
                "source": "history_tail",
            }
    resume = checkpoint.get("resume")
    if isinstance(resume, dict) and resume:
        return {
            "phase_name": resume.get("phase_name"),
            "phase_index": resume.get("phase_index"),
            "stage": resume.get("stage"),
            "source": "resume_state",
        }
    return {"source": "unknown"}


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_6class.pt")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_checkpoint)
    if not input_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {input_path}")

    checkpoint = torch.load(input_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict checkpoint.")

    source_class_names = list(checkpoint.get("class_names") or [])
    if not source_class_names:
        raise ValueError("Checkpoint is missing class_names metadata.")

    source_phase = infer_source_phase(checkpoint)
    print(json.dumps({
        "input_checkpoint": str(input_path),
        "source_phase": source_phase,
        "source_class_names": source_class_names,
        "target_class_names": list(TRAINING_CLASS_ORDER),
    }, indent=2))

    if args.output_checkpoint:
        output_path = Path(args.output_checkpoint)
    else:
        output_path = default_output_path(input_path)

    adapted_state_dict, adaptation_report = adapt_checkpoint_state_dict_to_training_taxonomy(
        checkpoint["model_state_dict"],
        source_class_names,
        list(TRAINING_CLASS_ORDER),
        class_mapping=checkpoint.get("args", {}).get("training_class_mapping") if isinstance(checkpoint.get("args"), dict) else None,
    )

    converted = dict(checkpoint)
    converted["model_state_dict"] = adapted_state_dict
    converted["class_names"] = list(TRAINING_CLASS_ORDER)
    converted["class_to_idx"] = {name: index for index, name in enumerate(TRAINING_CLASS_ORDER)}
    converted["legacy_source_class_names"] = source_class_names
    converted["taxonomy_conversion"] = {
        "source_phase": source_phase,
        "adaptation": adaptation_report,
    }

    args_payload = converted.get("args")
    if isinstance(args_payload, dict):
        args_payload = dict(args_payload)
        args_payload["class_names_resolved"] = list(TRAINING_CLASS_ORDER)
        args_payload["training_class_order"] = list(TRAINING_CLASS_ORDER)
        args_payload["training_excluded_classes"] = ["ewaste"]
        converted["args"] = args_payload

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)
    print(json.dumps({
        "output_checkpoint": str(output_path),
        "adaptation": adaptation_report,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
