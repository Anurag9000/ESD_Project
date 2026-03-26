#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from metric_learning_pipeline import build_parser, run_experiment, save_json


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_run(base_args, classifier_head: str, run_output_dir: Path, run_log_path: Path):
    args = copy.deepcopy(base_args)
    args.classifier_head = classifier_head
    args.skip_supcon = True
    args.supcon_epochs = 0
    args.optimizer = "adamw"
    args.stage_epochs = 0
    args.max_progressive_phases = 0
    args.output_dir = str(run_output_dir)
    args.log_file = str(run_log_path)
    args.resume_checkpoint = str(run_output_dir / "__fresh_start__.pt")
    args.resume_mode = "latest"
    args.resume_phase_index = 0
    args.resume_phase_name = ""
    return args


def compare_confidence_reports(
    ce_confidence: dict[str, Any],
    arcface_confidence: dict[str, Any],
) -> dict[str, Any]:
    ce_overall = ce_confidence.get("overall_average_confidence_on_correct_predictions")
    arcface_overall = arcface_confidence.get("overall_average_confidence_on_correct_predictions")
    if ce_overall is None and arcface_overall is None:
        overall_winner = "tie"
        overall_margin = None
    elif ce_overall is None:
        overall_winner = "arcface"
        overall_margin = None
    elif arcface_overall is None:
        overall_winner = "ce"
        overall_margin = None
    else:
        if ce_overall > arcface_overall:
            overall_winner = "ce"
        elif arcface_overall > ce_overall:
            overall_winner = "arcface"
        else:
            overall_winner = "tie"
        overall_margin = abs(ce_overall - arcface_overall)

    per_class: dict[str, Any] = {}
    for class_name in ce_confidence["per_class"]:
        ce_value = ce_confidence["per_class"][class_name]["average_confidence_on_correct_predictions"]
        arcface_value = arcface_confidence["per_class"][class_name]["average_confidence_on_correct_predictions"]
        if ce_value is None and arcface_value is None:
            winner = "tie"
            margin = None
        elif ce_value is None:
            winner = "arcface"
            margin = None
        elif arcface_value is None:
            winner = "ce"
            margin = None
        else:
            if ce_value > arcface_value:
                winner = "ce"
            elif arcface_value > ce_value:
                winner = "arcface"
            else:
                winner = "tie"
            margin = abs(ce_value - arcface_value)
        per_class[class_name] = {
            "ce_average_confidence_on_correct_predictions": ce_value,
            "arcface_average_confidence_on_correct_predictions": arcface_value,
            "winner": winner,
            "absolute_margin": margin,
        }

    return {
        "overall": {
            "ce_average_confidence_on_correct_predictions": ce_overall,
            "arcface_average_confidence_on_correct_predictions": arcface_overall,
            "winner": overall_winner,
            "absolute_margin": overall_margin,
        },
        "per_class": per_class,
    }


def main() -> int:
    parser = build_parser(use_gabor=False)
    parser.description = "Sequential head-only CE vs ArcFace comparison with final metrics comparison"
    parser.set_defaults(
        output_dir="Results/head_only_ce_vs_arcface_comparison",
        log_file="logs/head_only_ce_vs_arcface_comparison/comparison.log.jsonl",
        optimizer="adamw",
        skip_supcon=True,
        stage_epochs=0,
        max_progressive_phases=0,
    )
    args = parser.parse_args()

    comparison_root = Path(args.output_dir)
    comparison_root.mkdir(parents=True, exist_ok=True)
    comparison_log_dir = Path(args.log_file).parent
    comparison_log_dir.mkdir(parents=True, exist_ok=True)

    ce_output_dir = comparison_root / "ce_head_only"
    arcface_output_dir = comparison_root / "arcface_head_only"
    ce_log_path = comparison_log_dir / "ce_head_only.log.jsonl"
    arcface_log_path = comparison_log_dir / "arcface_head_only.log.jsonl"

    ce_args = configure_run(args, "ce", ce_output_dir, ce_log_path)
    arcface_args = configure_run(args, "arcface", arcface_output_dir, arcface_log_path)

    if run_experiment(ce_args, use_gabor=False) != 0:
        return 1
    if run_experiment(arcface_args, use_gabor=False) != 0:
        return 1

    ce_metrics = load_json(ce_output_dir / "metrics.json")
    arcface_metrics = load_json(arcface_output_dir / "metrics.json")
    ce_confidence = load_json(ce_output_dir / "test_correct_confidence_by_class.json")
    arcface_confidence = load_json(arcface_output_dir / "test_correct_confidence_by_class.json")

    ce_raw_accuracy = ce_metrics["test_metrics"]["raw_accuracy"]
    arcface_raw_accuracy = arcface_metrics["test_metrics"]["raw_accuracy"]
    if ce_raw_accuracy > arcface_raw_accuracy:
        accuracy_winner = "ce"
    elif arcface_raw_accuracy > ce_raw_accuracy:
        accuracy_winner = "arcface"
    else:
        accuracy_winner = "tie"

    comparison = {
        "setup": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "skip_supcon": True,
            "optimizer": "adamw",
            "head_epochs": args.head_epochs,
            "stage_epochs": 0,
            "max_progressive_phases": 0,
            "weighted_sampling": args.weighted_sampling,
        },
        "runs": {
            "ce": {
                "output_dir": str(ce_output_dir),
                "log_file": str(ce_log_path),
                "test_raw_accuracy": ce_raw_accuracy,
                "test_thresholded_accuracy": ce_metrics["test_metrics"]["accuracy"],
            },
            "arcface": {
                "output_dir": str(arcface_output_dir),
                "log_file": str(arcface_log_path),
                "test_raw_accuracy": arcface_raw_accuracy,
                "test_thresholded_accuracy": arcface_metrics["test_metrics"]["accuracy"],
            },
        },
        "comparison": {
            "raw_accuracy": {
                "ce": ce_raw_accuracy,
                "arcface": arcface_raw_accuracy,
                "winner": accuracy_winner,
                "absolute_margin": abs(ce_raw_accuracy - arcface_raw_accuracy),
            },
            "confidence_on_correct_predictions": compare_confidence_reports(ce_confidence, arcface_confidence),
        },
    }

    save_json(comparison_root / "comparison_summary.json", comparison)
    with (comparison_root / "comparison_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Head-Only CE vs ArcFace Comparison\n\n")
        handle.write(f"- Batch size: `{args.batch_size}`\n")
        handle.write(f"- Num workers: `{args.num_workers}`\n")
        handle.write(f"- Raw accuracy winner: `{accuracy_winner}`\n")
        handle.write(
            f"- Raw accuracy margin: `{abs(ce_raw_accuracy - arcface_raw_accuracy):.6f}`"
            f" (`ce={ce_raw_accuracy:.6f}`, `arcface={arcface_raw_accuracy:.6f}`)\n\n"
        )
        overall_confidence = comparison["comparison"]["confidence_on_correct_predictions"]["overall"]
        handle.write("## Confidence On Correct Predictions\n\n")
        handle.write(
            f"- Overall winner: `{overall_confidence['winner']}`"
            f" (`ce={overall_confidence['ce_average_confidence_on_correct_predictions']}`, "
            f"`arcface={overall_confidence['arcface_average_confidence_on_correct_predictions']}`)\n\n"
        )
        handle.write("## Per-Class Confidence Margins\n\n")
        for class_name, row in comparison["comparison"]["confidence_on_correct_predictions"]["per_class"].items():
            handle.write(
                f"- `{class_name}`: winner `{row['winner']}`, margin `{row['absolute_margin']}`, "
                f"`ce={row['ce_average_confidence_on_correct_predictions']}`, "
                f"`arcface={row['arcface_average_confidence_on_correct_predictions']}`\n"
            )

    print(
        {
            "comparison_output_dir": str(comparison_root),
            "batch_size": args.batch_size,
            "raw_accuracy_winner": accuracy_winner,
            "raw_accuracy_margin": abs(ce_raw_accuracy - arcface_raw_accuracy),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
