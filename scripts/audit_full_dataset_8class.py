#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm

from metric_learning_pipeline import (
    DEFAULT_BACKBONE_NAME,
    MetricLearningEfficientNetB0,
    TRAINING_CLASS_ORDER,
    adapt_checkpoint_state_dict_to_training_taxonomy,
    compute_classification_metrics,
    compute_correct_confidence_by_class,
    evaluation_tensor_from_image,
    model_dtype_for_args,
    project_samples_to_training_taxonomy,
    save_classification_report_csv,
    save_confidence_histogram,
    save_confusion_matrix_csv,
    save_confusion_matrix_plot,
    save_json,
    save_reliability_diagram,
)


TARGET_CLASS_ORDER = list(TRAINING_CLASS_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the entire Dataset_Final root using the current 3-class taxonomy.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint compatible with the 3-class taxonomy.")
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=240)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-split-ratios", default="0.7,0.2,0.1")
    parser.add_argument("--max-batches", type=int, default=0, help="Optional cap per split for debugging.")
    return parser.parse_args()


def parse_ratios(spec: str) -> tuple[float, float, float]:
    parts = [segment.strip() for segment in spec.split(",") if segment.strip()]
    if len(parts) != 3:
        raise ValueError("--auto-split-ratios must contain exactly three comma-separated values.")
    ratios = [float(part) for part in parts]
    if any(value <= 0.0 for value in ratios):
        raise ValueError("--auto-split-ratios values must be positive.")
    total = sum(ratios)
    return (ratios[0] / total, ratios[1] / total, ratios[2] / total)


def allocate_split_counts(sample_count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if sample_count <= 0:
        return (0, 0, 0)
    minimums = [0, 0, 0]
    if sample_count >= 1:
        minimums[0] = 1
    if sample_count >= 2:
        minimums[1] = 1
    if sample_count >= 3:
        minimums[2] = 1
    remaining = sample_count - sum(minimums)
    if remaining <= 0:
        return tuple(minimums)
    raw = [remaining * ratio for ratio in ratios]
    extras = [int(math.floor(value)) for value in raw]
    shortfall = remaining - sum(extras)
    remainders = sorted(((raw[index] - extras[index], index) for index in range(3)), reverse=True)
    for _, index in remainders[:shortfall]:
        extras[index] += 1
    return tuple(minimums[index] + extras[index] for index in range(3))


def source_prefix(path: str) -> str:
    stem = Path(path).stem
    parts = re.split(r"[^a-zA-Z0-9]", stem)
    for part in parts:
        word = re.sub(r"\d+", "", part)
        if word:
            return word.lower()
    return stem[:8].lower()


@dataclass(frozen=True)
class SplitSpec:
    name: str
    samples: list[tuple[str, int]]


class AuditDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], image_size: int) -> None:
        self.samples = list(samples)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.samples[index]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                tensor = evaluation_tensor_from_image(img, self.image_size)
        except Exception:
            tensor = torch.zeros(3, self.image_size, self.image_size)
        return tensor, int(label), path


def collate(batch):
    images, labels, paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(paths)


def build_raw_splits(dataset_root: Path, ratios: tuple[float, float, float], seed: int) -> tuple[list[SplitSpec], dict[str, Any]]:
    dataset = datasets.ImageFolder(str(dataset_root))
    projected_classes, projected_class_to_idx, projected_samples, taxonomy = project_samples_to_training_taxonomy(
        list(dataset.classes),
        list(dataset.samples),
        class_mapping=None,
    )
    dataset.classes = projected_classes
    dataset.class_to_idx = projected_class_to_idx
    dataset.samples = projected_samples

    by_class: dict[int, dict[str, list[tuple[str, int]]]] = {index: {} for index in range(len(dataset.classes))}
    for path, target in dataset.samples:
        by_class[int(target)].setdefault(source_prefix(path), []).append((path, int(target)))

    rng = random.Random(seed)
    split_samples: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    split_counts: dict[str, dict[str, int]] = {name: {} for name in split_samples}

    for class_index, sources in by_class.items():
        class_name = dataset.classes[class_index]
        class_train = class_val = class_test = 0
        for source_name, source_samples in sorted(sources.items()):
            shuffled = list(source_samples)
            rng.shuffle(shuffled)
            train_n, val_n, test_n = allocate_split_counts(len(shuffled), ratios)
            split_samples["train"].extend(shuffled[:train_n])
            split_samples["val"].extend(shuffled[train_n : train_n + val_n])
            split_samples["test"].extend(shuffled[train_n + val_n : train_n + val_n + test_n])
            class_train += train_n
            class_val += val_n
            class_test += test_n
        split_counts["train"][class_name] = class_train
        split_counts["val"][class_name] = class_val
        split_counts["test"][class_name] = class_test

    manifest = {
        "dataset_root": str(dataset_root),
        "taxonomy": list(TARGET_CLASS_ORDER),
        "split_mode": "source_stratified_within_class",
        "seed": int(seed),
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "split_counts": split_counts,
        "source_samples": len(dataset.samples),
        "train_samples": len(split_samples["train"]),
        "val_samples": len(split_samples["val"]),
        "test_samples": len(split_samples["test"]),
    }
    return [SplitSpec(name=name, samples=samples) for name, samples in split_samples.items()], manifest


def make_loader(samples: list[tuple[str, int]], image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        AuditDataset(samples, image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate,
    )


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args") or {}
    if not isinstance(checkpoint_args, dict):
        raise ValueError("Checkpoint is missing args metadata.")

    ratios = parse_ratios(str(checkpoint_args.get("auto_split_ratios", args.auto_split_ratios)))
    splits, manifest = build_raw_splits(dataset_root, ratios=ratios, seed=int(checkpoint_args.get("seed", args.seed)))
    manifest["checkpoint"] = str(checkpoint_path)
    save_json(output_dir / "dataset_manifest.json", manifest)

    class_names = list(checkpoint.get("class_names") or TARGET_CLASS_ORDER)
    if class_names != TARGET_CLASS_ORDER:
        adapted_state, adaptation = adapt_checkpoint_state_dict_to_training_taxonomy(
            checkpoint["model_state_dict"],
            class_names,
            TARGET_CLASS_ORDER,
        )
        checkpoint = dict(checkpoint)
        checkpoint["model_state_dict"] = adapted_state
        class_names = TARGET_CLASS_ORDER
    else:
        adaptation = {"applied": False, "reason": "already_aligned"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricLearningEfficientNetB0(
        num_classes=len(class_names),
        weights_mode=str(checkpoint_args.get("weights", "default")),
        embedding_dim=int(checkpoint_args.get("embedding_dim", 128)),
        projection_dim=int(checkpoint_args.get("projection_dim", 128)),
        args=argparse.Namespace(**checkpoint_args),
        backbone_name=str(checkpoint_args.get("backbone", DEFAULT_BACKBONE_NAME)),
    ).to(device=device, dtype=model_dtype_for_args(argparse.Namespace(**checkpoint_args)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_rows: list[dict[str, Any]] = []
    overall_logits: list[np.ndarray] = []
    overall_targets: list[np.ndarray] = []

    for split in splits:
        loader = make_loader(split.samples, args.image_size, args.batch_size, args.num_workers)
        split_dir = output_dir / split.name
        split_dir.mkdir(parents=True, exist_ok=True)
        logits_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []
        paths_list: list[str] = []

        iterator = loader
        if args.max_batches > 0:
            iterator = iter(loader)

        with torch.no_grad():
            for batch_index, (images, labels, paths) in enumerate(tqdm(loader, desc=f"{split.name}", leave=False), start=1):
                if args.max_batches > 0 and batch_index > args.max_batches:
                    break
                images = images.to(device, non_blocking=True, dtype=model_dtype_for_args(argparse.Namespace(**checkpoint_args)))
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    embeddings = model.encode(images)
                    batch_logits = model.classify(embeddings, labels=None)
                logits_np = batch_logits.detach().cpu().numpy()
                targets_np = labels.detach().cpu().numpy()
                logits_list.append(logits_np)
                targets_list.append(targets_np)
                paths_list.extend(paths)

        split_logits = np.concatenate(logits_list, axis=0) if logits_list else np.empty((0, len(class_names)), dtype=np.float32)
        split_targets = np.concatenate(targets_list, axis=0) if targets_list else np.empty((0,), dtype=np.int64)
        overall_logits.append(split_logits)
        overall_targets.append(split_targets)

        metrics = compute_classification_metrics(split_logits, split_targets, class_names, confidence_threshold=0.0)
        confidence = compute_correct_confidence_by_class(split_logits, split_targets, class_names)
        probabilities = torch.softmax(torch.from_numpy(split_logits), dim=1).numpy() if split_logits.size else np.empty((0, len(class_names)), dtype=np.float32)
        confmat = np.asarray(metrics["confusion_matrix"], dtype=np.int64)

        save_json(split_dir / "metrics.json", metrics)
        save_json(split_dir / "correct_confidence_by_class.json", confidence)
        save_confusion_matrix_csv(split_dir / "confmat_counts.csv", confmat, class_names, percent=False)
        save_confusion_matrix_csv(split_dir / "confmat_rate_pct.csv", confmat, class_names, percent=True)
        save_classification_report_csv(split_dir / "classification_report.csv", metrics, class_names)
        save_confusion_matrix_plot(split_dir / "confusion_matrix.png", confmat, class_names, f"{split.name} Confusion Matrix")
        save_reliability_diagram(split_dir / "reliability_diagram.png", metrics["calibration"], f"{split.name} Reliability Diagram")
        save_confidence_histogram(split_dir / "confidence_histogram.png", probabilities, f"{split.name} Confidence Histogram")

        split_wrong_rows: list[dict[str, Any]] = []
        predictions = split_logits.argmax(axis=1) if split_logits.size else np.empty((0,), dtype=np.int64)
        confidences = probabilities.max(axis=1) if probabilities.size else np.empty((0,), dtype=np.float32)
        for idx, (path, true_index, pred_index, confidence_value) in enumerate(zip(paths_list, split_targets.tolist(), predictions.tolist(), confidences.tolist())):
            row = {
                "split": split.name,
                "path": path,
                "true_class": class_names[int(true_index)],
                "pred_class": class_names[int(pred_index)],
                "confidence": float(confidence_value),
            }
            if int(true_index) != int(pred_index):
                split_wrong_rows.append(row)
            all_rows.append(row)

        save_json(
            split_dir / "summary.json",
            {
                "split": split.name,
                "num_samples": metrics["num_samples"],
                "accuracy": metrics["accuracy"],
                "raw_accuracy": metrics["raw_accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_precision": metrics["weighted_precision"],
                "weighted_recall": metrics["weighted_recall"],
                "weighted_f1": metrics["weighted_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "confusion_matrix": metrics["confusion_matrix"],
                "per_class_accuracy": metrics["per_class_accuracy"],
                "per_class_avg_confidence": metrics["per_class_avg_confidence"],
                "calibration": metrics["calibration"],
                "wrong_count": len(split_wrong_rows),
            },
        )
        save_json(split_dir / "wrong_count.json", {"wrong_count": len(split_wrong_rows), "total": int(metrics["num_samples"])})
        with (split_dir / "wrong_predictions.csv").open("w", encoding="utf-8") as handle:
            handle.write("split,path,true_class,pred_class,confidence\n")
            for row in split_wrong_rows:
                handle.write(
                    f"{row['split']},{row['path']},{row['true_class']},{row['pred_class']},{row['confidence']:.6f}\n"
                )

    all_logits = np.concatenate(overall_logits, axis=0) if overall_logits else np.empty((0, len(class_names)), dtype=np.float32)
    all_targets = np.concatenate(overall_targets, axis=0) if overall_targets else np.empty((0,), dtype=np.int64)
    overall_metrics = compute_classification_metrics(all_logits, all_targets, class_names, confidence_threshold=0.0)
    overall_confidence = compute_correct_confidence_by_class(all_logits, all_targets, class_names)
    overall_probabilities = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy() if all_logits.size else np.empty((0, len(class_names)), dtype=np.float32)
    overall_confmat = np.asarray(overall_metrics["confusion_matrix"], dtype=np.int64)
    save_json(output_dir / "metrics.json", overall_metrics)
    save_json(output_dir / "correct_confidence_by_class.json", overall_confidence)
    save_confusion_matrix_csv(output_dir / "confmat_counts.csv", overall_confmat, class_names, percent=False)
    save_confusion_matrix_csv(output_dir / "confmat_rate_pct.csv", overall_confmat, class_names, percent=True)
    save_classification_report_csv(output_dir / "classification_report.csv", overall_metrics, class_names)
    save_confusion_matrix_plot(output_dir / "confusion_matrix.png", overall_confmat, class_names, "Overall Confusion Matrix")
    save_reliability_diagram(output_dir / "reliability_diagram.png", overall_metrics["calibration"], "Overall Reliability Diagram")
    save_confidence_histogram(output_dir / "confidence_histogram.png", overall_probabilities, "Overall Confidence Histogram")

    with (output_dir / "all_predictions.csv").open("w", encoding="utf-8") as handle:
        handle.write("split,path,true_class,pred_class,confidence\n")
        for row in all_rows:
            handle.write(
                f"{row['split']},{row['path']},{row['true_class']},{row['pred_class']},{row['confidence']:.6f}\n"
            )

    wrong_rows = [row for row in all_rows if row["true_class"] != row["pred_class"]]
    with (output_dir / "wrong_predictions.csv").open("w", encoding="utf-8") as handle:
        handle.write("split,path,true_class,pred_class,confidence\n")
        for row in wrong_rows:
            handle.write(
                f"{row['split']},{row['path']},{row['true_class']},{row['pred_class']},{row['confidence']:.6f}\n"
            )

    save_json(
        output_dir / "summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "dataset_root": str(dataset_root),
            "class_names": class_names,
            "checkpoint_taxonomy_adaptation": adaptation,
            "num_samples": overall_metrics["num_samples"],
            "accuracy": overall_metrics["accuracy"],
            "raw_accuracy": overall_metrics["raw_accuracy"],
            "loss": overall_metrics.get("loss", 0.0),
            "cross_entropy_loss": overall_metrics.get("cross_entropy_loss", 0.0),
            "macro_precision": overall_metrics["macro_precision"],
            "macro_recall": overall_metrics["macro_recall"],
            "macro_f1": overall_metrics["macro_f1"],
            "weighted_precision": overall_metrics["weighted_precision"],
            "weighted_recall": overall_metrics["weighted_recall"],
            "weighted_f1": overall_metrics["weighted_f1"],
            "balanced_accuracy": overall_metrics["balanced_accuracy"],
            "cohen_kappa": overall_metrics["cohen_kappa"],
            "mcc": overall_metrics["mcc"],
            "expected_calibration_error": overall_metrics["calibration"]["expected_calibration_error"],
            "maximum_calibration_error": overall_metrics["calibration"]["maximum_calibration_error"],
            "brier_score": overall_metrics["calibration"]["brier_score"],
            "negative_log_likelihood": overall_metrics["calibration"]["negative_log_likelihood"],
            "per_class_accuracy": overall_metrics["per_class_accuracy"],
            "per_class_avg_confidence": overall_metrics["per_class_avg_confidence"],
            "wrong_count": len(wrong_rows),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
