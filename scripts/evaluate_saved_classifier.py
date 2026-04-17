#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from metric_learning_pipeline import (
    MetricLearningEfficientNetB0,
    build_datasets,
    class_counts,
    collect_logits_and_labels,
    collapse_logits_and_targets_to_runtime_classes,
    compute_classification_metrics,
    compute_correct_confidence_by_class,
    model_dtype_for_args,
    release_training_memory,
    save_confusion_matrix_plot,
    save_confidence_histogram,
    save_json,
    save_reliability_diagram,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved classifier checkpoint on the current dataset.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--batch-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--selected-class", action="append", default=[])
    parser.add_argument("--other-label", default="other")
    parser.add_argument(
        "--class-mapping", 
        type=str, 
        default="", 
        help="JSON string for custom class mapping, e.g. '{\"Fiber\": [\"paper\", \"cardboard\"]}'"
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    return parser.parse_args()


def make_eval_args(checkpoint_args: dict[str, Any], cli_args: argparse.Namespace) -> argparse.Namespace:
    merged = dict(checkpoint_args)
    merged["dataset_root"] = cli_args.dataset_root or checkpoint_args.get("dataset_root", "Dataset_Final")
    merged["batch_size"] = int(cli_args.batch_size)
    merged["num_workers"] = int(cli_args.num_workers)
    merged["max_eval_batches"] = int(cli_args.max_eval_batches)
    if cli_args.confidence_threshold is not None:
        merged["confidence_threshold"] = float(cli_args.confidence_threshold)
    merged.setdefault("prefetch_factor", None)
    merged.setdefault("augment_repeats", 16)
    merged.setdefault("augment_gaussian_sigmas", 2.0)
    merged.setdefault("image_size", 224)
    merged.setdefault("seed", 42)
    merged.setdefault("auto_split_ratios", "0.7,0.2,0.1")
    merged.setdefault("weights", "default")
    merged.setdefault("backbone", "efficientnet_b0")
    merged.setdefault("embedding_dim", 512)
    merged.setdefault("projection_dim", 256)
    merged.setdefault("precision", "mixed")
    merged.setdefault("log_eval_every_steps", 1000)
    return argparse.Namespace(**merged)


def build_eval_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    return DataLoader(**loader_kwargs)


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        expected_roots = {"evaluation_manifest.json", "evaluation_summary.json", "evaluation.log.jsonl"}
        existing_names = {entry.name for entry in output_dir.iterdir()}
        if not existing_names.issubset(expected_roots | {"train", "val", "test"}):
            raise ValueError(
                f"Refusing to clear non-empty evaluation directory {output_dir} because it contains unrelated files: "
                f"{sorted(existing_names)}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict):
        raise ValueError("Checkpoint is missing args metadata.")

    eval_args = make_eval_args(checkpoint_args, args)
    train_dataset, val_dataset, test_dataset, _, _ = build_datasets(eval_args)
    datasets_by_split = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }

    class_names = list(checkpoint["class_names"])
    if train_dataset.classes != class_names:
        raise ValueError(
            "Checkpoint class names do not match the current dataset classes.\n"
            f"checkpoint={class_names}\n"
            f"dataset={train_dataset.classes}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricLearningEfficientNetB0(
        num_classes=len(class_names),
        weights_mode=eval_args.weights,
        embedding_dim=int(eval_args.embedding_dim),
        projection_dim=int(eval_args.projection_dim),
        args=eval_args,
        backbone_name=str(getattr(eval_args, "backbone", "efficientnet_b0")),
    ).to(device=device, dtype=model_dtype_for_args(eval_args))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    manifest: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(eval_args.dataset_root),
        "selected_classes": list(args.selected_class),
        "other_label": args.other_label,
        "splits": list(args.splits),
        "source_class_names": class_names,
    }
    save_json(output_dir / "evaluation_manifest.json", manifest)

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "splits": {},
    }

    empty_log_path = output_dir / "evaluation.log.jsonl"
    empty_log_path.parent.mkdir(parents=True, exist_ok=True)
    empty_log_path.touch(exist_ok=True)

    import json as json_lib
    class_mapping_dict = json_lib.loads(args.class_mapping) if args.class_mapping else None

    for split in args.splits:
        if split not in datasets_by_split:
            raise ValueError(f"Unsupported split {split!r}; expected one of train/val/test.")
        dataset = datasets_by_split[split]
        loader = build_eval_loader(dataset, eval_args.batch_size, eval_args.num_workers)
        logits, targets = collect_logits_and_labels(
            model,
            loader,
            device,
            eval_args.max_eval_batches,
            empty_log_path,
            int(eval_args.log_eval_every_steps),
            split,
            eval_args,
        )
        collapsed_logits, collapsed_targets, runtime_class_names, collapse_info = collapse_logits_and_targets_to_runtime_classes(
            logits,
            targets,
            class_names,
            selected_classes=list(args.selected_class),
            class_mapping=class_mapping_dict,
            other_label=args.other_label,
        )
        metrics = compute_classification_metrics(
            collapsed_logits,
            collapsed_targets,
            runtime_class_names,
            float(eval_args.confidence_threshold),
        )
        confidence = compute_correct_confidence_by_class(collapsed_logits, collapsed_targets, runtime_class_names)
        probabilities = torch.softmax(torch.from_numpy(collapsed_logits), dim=1).numpy()
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        save_json(split_dir / "metrics.json", metrics)
        save_json(split_dir / "correct_confidence_by_class.json", confidence)
        split_summary = {
            "split": split,
            "num_samples": metrics["num_samples"],
            "raw_accuracy": metrics["raw_accuracy"],
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "expected_calibration_error": metrics["calibration"]["expected_calibration_error"],
            "maximum_calibration_error": metrics["calibration"]["maximum_calibration_error"],
            "brier_score": metrics["calibration"]["brier_score"],
            "negative_log_likelihood": metrics["calibration"]["negative_log_likelihood"],
            "class_names": runtime_class_names,
            "collapse": collapse_info,
            "source_class_counts": class_counts(dataset),
        }
        save_json(split_dir / "summary.json", split_summary)
        save_reliability_diagram(
            split_dir / "reliability_diagram.png",
            metrics["calibration"],
            f"{split.title()} Reliability Diagram",
        )
        save_confidence_histogram(
            split_dir / "confidence_histogram.png",
            probabilities,
            f"{split.title()} Confidence Histogram",
        )
        save_confusion_matrix_plot(
            split_dir / "confusion_matrix.png",
            torch.tensor(metrics["confusion_matrix"]).numpy(),
            runtime_class_names,
            f"{split.title()} Confusion Matrix",
        )
        summary["splits"][split] = split_summary
        release_training_memory(device, loader)

    save_json(output_dir / "evaluation_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
