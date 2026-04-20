#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from metric_learning_pipeline import (
    DEFAULT_BACKBONE_NAME,
    MetricLearningEfficientNetB0,
    adapt_checkpoint_state_dict_to_training_taxonomy,
    build_datasets,
    class_counts,
    collect_logits_and_labels,
    collapse_logits_and_targets_to_runtime_classes,
    compute_classification_metrics,
    compute_correct_confidence_by_class,
    save_classification_report_csv,
    save_confusion_matrix_csv,
    log_json_event,
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
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=2)
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
    parser.add_argument("--evaluation-stage", default="checkpoint_evaluation")
    parser.add_argument("--phase-name", default="")
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
    merged.setdefault("augment_repeats", 1)
    merged.setdefault("augment_gaussian_sigmas", 1.0)
    merged.setdefault("image_size", 224)
    merged.setdefault("seed", 42)
    merged.setdefault("auto_split_ratios", "0.9,0.05,0.05")
    merged.setdefault("weights", "default")
    merged.setdefault("backbone", DEFAULT_BACKBONE_NAME)
    merged.setdefault("embedding_dim", 128)
    merged.setdefault("projection_dim", 128)
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


def reset_generated_evaluation_dir(output_dir: Path) -> None:
    if not output_dir.exists() or not any(output_dir.iterdir()):
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    generated_roots = {
        "evaluation_manifest.json",
        "evaluation_summary.json",
        "evaluation.log.jsonl",
        "train_metrics.csv",
        "val_metrics.csv",
        "test_metrics.csv",
        "train",
        "val",
        "test",
    }
    existing_names = {entry.name for entry in output_dir.iterdir()}
    unrelated = sorted(existing_names - generated_roots)
    if unrelated:
        raise ValueError(
            f"Refusing to clear non-empty evaluation directory {output_dir} because it contains unrelated files: "
            f"{unrelated}"
        )
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    reset_generated_evaluation_dir(output_dir)

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

    source_class_names = list(checkpoint["class_names"])
    class_names = list(train_dataset.classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricLearningEfficientNetB0(
        num_classes=len(class_names),
        weights_mode=eval_args.weights,
        embedding_dim=int(eval_args.embedding_dim),
        projection_dim=int(eval_args.projection_dim),
        args=eval_args,
        backbone_name=str(getattr(eval_args, "backbone", DEFAULT_BACKBONE_NAME)),
    ).to(device=device, dtype=model_dtype_for_args(eval_args))
    model_state_dict = checkpoint["model_state_dict"]
    if source_class_names != class_names:
        model_state_dict, adaptation_report = adapt_checkpoint_state_dict_to_training_taxonomy(
            model_state_dict,
            source_class_names,
            class_names,
            class_mapping=checkpoint_args.get("training_class_mapping"),
        )
    else:
        adaptation_report = {"applied": False, "reason": "already_aligned"}
    model.load_state_dict(model_state_dict)
    model.eval()

    manifest: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(eval_args.dataset_root),
        "selected_classes": list(args.selected_class),
        "other_label": args.other_label,
        "splits": list(args.splits),
        "source_class_names": source_class_names,
        "runtime_class_names": class_names,
        "checkpoint_taxonomy_adaptation": adaptation_report,
        "evaluation_stage": args.evaluation_stage,
        "phase_name": args.phase_name,
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
        log_json_event(
            empty_log_path,
            {
                "event": "validation_started" if split == "val" else "final_evaluation_started",
                "stage": args.evaluation_stage,
                "phase_name": args.phase_name or None,
                "split": split,
                "eval_batches": len(loader),
            },
        )
        logits, targets = collect_logits_and_labels(
            model=model,
            loader=loader,
            device=device,
            max_batches=eval_args.max_eval_batches,
            log_path=empty_log_path,
            log_every_eval_steps=int(eval_args.log_eval_every_steps),
            criterion=None,
            split=split,
            args=eval_args,
            stage=args.evaluation_stage,
            phase_name=args.phase_name or None,
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
        confmat = torch.as_tensor(metrics["confusion_matrix"], dtype=torch.int64).numpy()
        loss_value = float(metrics.get("cross_entropy_loss", metrics.get("loss", 0.0)))
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        save_json(split_dir / "metrics.json", metrics)
        save_json(split_dir / "correct_confidence_by_class.json", confidence)
        save_confusion_matrix_csv(split_dir / f"confmat_counts_{split}.csv", confmat, runtime_class_names, percent=False)
        save_confusion_matrix_csv(split_dir / f"confmat_rate_pct_{split}.csv", confmat, runtime_class_names, percent=True)
        save_classification_report_csv(split_dir / f"classification_report_{split}.csv", metrics, runtime_class_names)
        split_summary = {
            "split": split,
            "num_samples": metrics["num_samples"],
            "loss": loss_value,
            "cross_entropy_loss": loss_value,
            "raw_accuracy": metrics["raw_accuracy"],
            "accuracy": metrics["accuracy"],
            "top1_accuracy": metrics["top1_accuracy"],
            "top3_accuracy": metrics["top3_accuracy"],
            "top5_accuracy": metrics["top5_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_roc_auc_ovr": metrics["macro_roc_auc_ovr"],
            "weighted_roc_auc_ovr": metrics["weighted_roc_auc_ovr"],
            "macro_pr_auc_ovr": metrics["macro_pr_auc_ovr"],
            "weighted_pr_auc_ovr": metrics["weighted_pr_auc_ovr"],
            "cohen_kappa": metrics["cohen_kappa"],
            "mcc": metrics["mcc"],
            "expected_calibration_error": metrics["calibration"]["expected_calibration_error"],
            "maximum_calibration_error": metrics["calibration"]["maximum_calibration_error"],
            "brier_score": metrics["calibration"]["brier_score"],
            "negative_log_likelihood": metrics["calibration"]["negative_log_likelihood"],
            "class_names": runtime_class_names,
            "collapse": collapse_info,
            "source_class_counts": class_counts(dataset),
            "per_class_accuracy": metrics["per_class_accuracy"],
            "per_class_avg_confidence": metrics["per_class_avg_confidence"],
            "per_class": metrics["per_class"],
            "calibration": metrics["calibration"],
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
            confmat,
            runtime_class_names,
            f"{split.title()} Confusion Matrix",
        )
        summary["splits"][split] = split_summary
        log_json_event(
            empty_log_path,
            {
                "event": "validation_finished" if split == "val" else "final_evaluation_finished",
                "stage": args.evaluation_stage,
                "phase_name": args.phase_name or None,
                "split": split,
                "eval_batches": len(loader),
                "loss": loss_value,
                "accuracy": metrics["accuracy"],
                "per_class_accuracy": metrics["per_class_accuracy"],
                "per_class_avg_confidence": metrics["per_class_avg_confidence"],
            },
        )
        release_training_memory(device, loader)

    save_json(output_dir / "evaluation_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
