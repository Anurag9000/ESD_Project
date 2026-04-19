#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from metric_learning_pipeline import (
    DEFAULT_BACKBONE_NAME,
    MetricLearningEfficientNetB0,
    TRAINING_CLASS_ORDER,
    adapt_checkpoint_state_dict_to_training_taxonomy,
    collapse_logits_and_targets_to_runtime_classes,
    compute_classification_metrics,
    compute_correct_confidence_by_class,
    evaluation_tensor_from_image,
    model_dtype_for_args,
    parse_json_class_mapping,
    project_class_name_to_training_taxonomy,
    resolve_runtime_selected_classes,
    release_training_memory,
    save_classification_report_csv,
    save_confusion_matrix_csv,
    save_confidence_histogram,
    save_confusion_matrix_plot,
    save_json,
    save_reliability_diagram,
)

warnings.filterwarnings("ignore", message="Palette images with Transparency", category=UserWarning)


class NoAugDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.samples[index]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img = Image.open(path).convert("RGB")
            tensor = evaluation_tensor_from_image(img, self.image_size)
        except Exception:
            tensor = torch.zeros(3, self.image_size, self.image_size)
        return tensor, label, path


def collate(batch):
    tensors, labels, paths = zip(*batch)
    return torch.stack(tensors), torch.tensor(labels, dtype=torch.long), list(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on an external holdout dataset root.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--selected-class", action="append", default=[])
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Strictly evaluate only the selected runtime classes and ignore all other logits.",
    )
    parser.add_argument("--other-label", default="other")
    parser.add_argument("--class-mapping", default="", help="JSON string for runtime class collapsing.")
    return parser.parse_args()


def make_eval_args(checkpoint_args: dict[str, Any], cli_args: argparse.Namespace) -> argparse.Namespace:
    merged = dict(checkpoint_args)
    merged["dataset_root"] = cli_args.dataset_root
    merged["batch_size"] = int(cli_args.batch_size)
    merged["num_workers"] = int(cli_args.num_workers)
    merged["max_eval_batches"] = int(cli_args.max_eval_batches)
    if cli_args.confidence_threshold is not None:
        merged["confidence_threshold"] = float(cli_args.confidence_threshold)
    merged.setdefault("image_size", 224)
    merged.setdefault("weights", "default")
    merged.setdefault("backbone", DEFAULT_BACKBONE_NAME)
    merged.setdefault("embedding_dim", 128)
    merged.setdefault("projection_dim", 128)
    merged.setdefault("precision", "mixed")
    merged.setdefault("log_eval_every_steps", 1000)
    return argparse.Namespace(**merged)


def build_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate,
    )


def normalize_class_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    return "ewaste" if normalized in {"e_waste", "electronic_waste"} else normalized


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict):
        raise ValueError("Checkpoint is missing args metadata.")

    eval_args = make_eval_args(checkpoint_args, args)
    dataset = datasets.ImageFolder(str(dataset_root))
    source_class_names = list(checkpoint["class_names"])
    runtime_class_names = list(TRAINING_CLASS_ORDER)
    runtime_class_to_idx = {name: idx for idx, name in enumerate(runtime_class_names)}
    remapped_samples: list[tuple[str, int]] = []
    missing_classes: list[str] = []
    holdout_class_remap: dict[str, str] = {}
    dropped_classes: dict[str, int] = {}
    runtime_class_mapping = parse_json_class_mapping(args.class_mapping)
    for path, target in dataset.samples:
        holdout_class_name = dataset.classes[target]
        normalized_holdout_class = normalize_class_name(holdout_class_name)
        class_name = project_class_name_to_training_taxonomy(normalized_holdout_class)
        if class_name is None:
            dropped_classes[holdout_class_name] = dropped_classes.get(holdout_class_name, 0) + 1
            continue
        if class_name not in runtime_class_to_idx:
            missing_classes.append(holdout_class_name)
            continue
        holdout_class_remap[holdout_class_name] = class_name
        remapped_samples.append((path, runtime_class_to_idx[class_name]))
    if missing_classes and not args.class_mapping and not args.selected_class:
        raise ValueError(
            "Holdout dataset contains classes that do not exist in the current runtime taxonomy. "
            f"Missing classes: {sorted(set(missing_classes))}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricLearningEfficientNetB0(
        num_classes=len(runtime_class_names),
        weights_mode=eval_args.weights,
        embedding_dim=int(eval_args.embedding_dim),
        projection_dim=int(eval_args.projection_dim),
        args=eval_args,
        backbone_name=str(getattr(eval_args, "backbone", DEFAULT_BACKBONE_NAME)),
    ).to(device=device, dtype=model_dtype_for_args(eval_args))
    model_state_dict = checkpoint["model_state_dict"]
    if source_class_names != runtime_class_names:
        model_state_dict, adaptation_report = adapt_checkpoint_state_dict_to_training_taxonomy(
            model_state_dict,
            source_class_names,
            runtime_class_names,
            class_mapping=checkpoint_args.get("training_class_mapping"),
        )
    else:
        adaptation_report = {"applied": False, "reason": "already_aligned"}
    model.load_state_dict(model_state_dict)
    model.eval()

    class_mapping_dict = runtime_class_mapping if args.class_mapping else None
    manifest = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "selected_classes": list(args.selected_class),
        "selected_only": bool(args.selected_only),
        "other_label": args.other_label,
        "source_class_names": source_class_names,
        "runtime_class_names": runtime_class_names,
        "checkpoint_taxonomy_adaptation": adaptation_report,
        "holdout_classes": list(dataset.classes),
        "holdout_class_remap": holdout_class_remap,
        "dropped_classes": dropped_classes,
    }
    save_json(output_dir / "holdout_manifest.json", manifest)

    loader = build_loader(NoAugDataset(remapped_samples, int(eval_args.image_size)), eval_args.batch_size, eval_args.num_workers)
    logits, targets, prediction_rows = [], [], []
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True, dtype=model_dtype_for_args(eval_args))
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                emb = model.encode(images)
                batch_logits = model.classify(emb, labels=None)
            batch_logits_np = batch_logits.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            logits.append(batch_logits_np)
            targets.append(labels_np)
            batch_probabilities = torch.softmax(torch.from_numpy(batch_logits_np), dim=1).numpy()
            for path, true_index, prob_row in zip(paths, labels_np, batch_probabilities):
                pred_index = int(np.argmax(prob_row))
                prediction_rows.append(
                    {
                        "path": path,
                        "true_class": runtime_class_names[int(true_index)],
                        "pred_class": runtime_class_names[pred_index],
                        "confidence": float(prob_row[pred_index]),
                        "correct": bool(pred_index == int(true_index)),
                    }
                )

    logits_concat = np.concatenate(logits, axis=0) if logits else np.empty((0, len(runtime_class_names)), dtype=np.float32)
    targets_concat = np.concatenate(targets, axis=0) if targets else np.empty((0,), dtype=np.int64)
    if args.selected_only:
        resolved_selected = resolve_runtime_selected_classes(runtime_class_names, list(args.selected_class))
        selected_indices = [runtime_class_names.index(name) for name in resolved_selected]
        selected_mask = np.asarray([int(target) in selected_indices for target in targets_concat], dtype=bool)
        strict_logits = logits_concat[:, selected_indices]
        strict_targets = np.asarray(
            [selected_indices.index(int(target)) for target in targets_concat[selected_mask]],
            dtype=np.int64,
        )
        collapsed_logits = strict_logits[selected_mask]
        collapsed_targets = strict_targets
        runtime_class_names = resolved_selected
        collapse_info = {
            "mode": "strict_selected_only",
            "selected_classes": list(resolved_selected),
            "collapse_applied": True,
            "dropped_source_classes": [name for name in TRAINING_CLASS_ORDER if name not in resolved_selected],
        }
    else:
        collapsed_logits, collapsed_targets, runtime_class_names, collapse_info = collapse_logits_and_targets_to_runtime_classes(
            logits_concat,
            targets_concat,
            runtime_class_names,
            selected_classes=list(args.selected_class),
            class_mapping=class_mapping_dict,
            other_label=args.other_label,
        )
    metrics = compute_classification_metrics(
        collapsed_logits,
        collapsed_targets,
        runtime_class_names,
        float(eval_args.confidence_threshold if hasattr(eval_args, "confidence_threshold") else 0.0),
    )
    confidence = compute_correct_confidence_by_class(collapsed_logits, collapsed_targets, runtime_class_names)
    probabilities = torch.softmax(torch.from_numpy(collapsed_logits), dim=1).numpy() if collapsed_logits.size else np.empty((0, len(runtime_class_names)), dtype=np.float32)
    confmat = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    loss_value = float(metrics.get("cross_entropy_loss", metrics.get("loss", 0.0)))

    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "correct_confidence_by_class.json", confidence)
    save_confusion_matrix_csv(output_dir / "confmat_counts_test.csv", confmat, runtime_class_names, percent=False)
    save_confusion_matrix_csv(output_dir / "confmat_rate_pct_test.csv", confmat, runtime_class_names, percent=True)
    save_classification_report_csv(output_dir / "classification_report_test.csv", metrics, runtime_class_names)
    save_json(output_dir / "all_predictions.json", prediction_rows)
    wrong_predictions = [row for row in prediction_rows if not row["correct"]]
    save_json(output_dir / "wrong_predictions.json", wrong_predictions)
    wrong_csv_path = output_dir / "wrong_predictions.csv"
    with wrong_csv_path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=["path", "true_class", "pred_class", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(prediction_rows)

    wrong_dir = output_dir / "wrong_predictions"
    wrong_dir.mkdir(parents=True, exist_ok=True)
    for row in wrong_predictions:
        source_path = Path(row["path"])
        class_dir = wrong_dir / row["true_class"] / row["pred_class"]
        class_dir.mkdir(parents=True, exist_ok=True)
        target_path = class_dir / source_path.name
        try:
            shutil.copy2(source_path, target_path)
        except Exception:
            pass
    save_json(
        output_dir / "summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "dataset_root": str(dataset_root),
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
            "per_class_accuracy": metrics["per_class_accuracy"],
            "per_class_avg_confidence": metrics["per_class_avg_confidence"],
            "per_class": metrics["per_class"],
            "calibration": metrics["calibration"],
        },
    )
    save_confusion_matrix_plot(
        output_dir / "confusion_matrix.png",
        confmat,
        runtime_class_names,
        "External Holdout Confusion Matrix",
    )
    save_reliability_diagram(
        output_dir / "reliability_diagram.png",
        metrics["calibration"],
        "External Holdout Reliability Diagram",
    )
    save_confidence_histogram(
        output_dir / "confidence_histogram.png",
        probabilities,
        "External Holdout Confidence Histogram",
    )
    release_training_memory(device, loader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
