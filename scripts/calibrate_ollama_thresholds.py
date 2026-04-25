#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ollama_pipeline_state import init_db
from run_ollama_end_to_end_pipeline import (
    IMAGE_EXTS,
    artifact_dirs,
    ensure_ollama_models,
    judge_single_image,
    repo_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Ollama material-dataset thresholds against a pristine local dataset.")
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--output-root", default="review_downloads/ollama_threshold_calibration")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--vision-model", default="qwen2.5vl:3b")
    parser.add_argument("--vision-temperature", type=float, default=0.0)
    parser.add_argument("--target-classes", default="organic,metal,paper")
    parser.add_argument("--eval-classes", default="organic,metal,paper,plastic")
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--pull-missing-models", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


def iter_class_images(class_dir: Path, limit: int) -> list[Path]:
    rows: list[Path] = []
    for path in sorted(class_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rows.append(path)
            if len(rows) >= limit:
                break
    return rows


def main() -> int:
    args = parse_args()
    dataset_root = (repo_root() / args.dataset_root).resolve()
    output_root = (repo_root() / args.output_root).resolve()
    dirs = artifact_dirs(output_root)
    db_path = dirs["manifests"] / "calibration.sqlite"
    init_db(db_path)

    target_classes = [part.strip() for part in args.target_classes.split(",") if part.strip()]
    eval_classes = [part.strip() for part in args.eval_classes.split(",") if part.strip()]
    ensure_ollama_models(args.ollama_host, [args.vision_model], args.pull_missing_models)

    positives: dict[str, dict[str, Any]] = {}
    for class_name in target_classes:
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            continue
        accepted = 0
        total = 0
        class_scores: list[float] = []
        photo_scores: list[float] = []
        for image_path in iter_class_images(class_dir, args.max_per_class):
            result = judge_single_image(
                args.ollama_host,
                args.vision_model,
                image_path,
                class_name,
                class_name,
                target_classes,
                args.vision_temperature,
                db_path,
            )
            total += 1
            if result["final_decision"] == "accepted":
                accepted += 1
            class_scores.append(float(result["class_stage"].get("confidence", 0.0)))
            photo_scores.append(float(result["photo_stage"].get("confidence", 0.0)))
        positives[class_name] = {
            "accepted": accepted,
            "total": total,
            "acceptance_pct": (100.0 * accepted / max(total, 1)),
            "suggested_class_accept_p05": percentile(class_scores, 0.05),
            "suggested_photo_accept_p05": percentile(photo_scores, 0.05),
        }

    negatives: dict[str, Any] = {}
    for eval_class in eval_classes:
        if eval_class in target_classes:
            continue
        class_dir = dataset_root / eval_class
        if not class_dir.is_dir():
            continue
        per_target = defaultdict(lambda: {"accepted": 0, "total": 0})
        any_accept = 0
        total = 0
        for image_path in iter_class_images(class_dir, args.max_per_class):
            total += 1
            accepted_any = False
            for target in target_classes:
                result = judge_single_image(
                    args.ollama_host,
                    args.vision_model,
                    image_path,
                    target,
                    eval_class,
                    target_classes,
                    args.vision_temperature,
                    db_path,
                )
                per_target[target]["total"] += 1
                if result["final_decision"] == "accepted":
                    per_target[target]["accepted"] += 1
                    accepted_any = True
            if accepted_any:
                any_accept += 1
        negatives[eval_class] = {
            "per_target_false_accept": {
                target: {
                    **stats,
                    "false_accept_pct": (100.0 * stats["accepted"] / max(stats["total"], 1)),
                }
                for target, stats in per_target.items()
            },
            "any_target_false_accept_pct": (100.0 * any_accept / max(total, 1)),
            "total": total,
        }

    summary = {
        "vision_model": args.vision_model,
        "target_classes": target_classes,
        "eval_classes": eval_classes,
        "max_per_class": args.max_per_class,
        "positives": positives,
        "negatives": negatives,
        "passes_95pct_positive_gate": {
            category: stats["acceptance_pct"] >= 95.0 for category, stats in positives.items()
        },
    }
    summary_path = dirs["manifests"] / "threshold_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
