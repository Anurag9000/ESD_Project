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
    MetricLearningEfficientNetB0,
    collapse_logits_and_targets_to_runtime_classes,
    compute_classification_metrics,
    compute_correct_confidence_by_class,
    evaluation_tensor_from_image,
    model_dtype_for_args,
    release_training_memory,
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img = Image.open(path).convert("RGB")
            tensor = evaluation_tensor_from_image(img, self.image_size)
        except Exception:
            tensor = torch.zeros(3, self.image_size, self.image_size)
        return tensor, label


def collate(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors), torch.tensor(labels, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on an external holdout dataset root.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--selected-class", action="append", default=[])
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
    merged.setdefault("backbone", "convnextv2_nano")
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
    class_names = list(checkpoint["class_names"])
    checkpoint_class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    remapped_samples: list[tuple[str, int]] = []
    missing_classes: list[str] = []
    for path, target in dataset.samples:
        class_name = dataset.classes[target]
        if class_name not in checkpoint_class_to_idx:
            missing_classes.append(class_name)
            continue
        remapped_samples.append((path, checkpoint_class_to_idx[class_name]))
    if missing_classes and not args.class_mapping and not args.selected_class:
        raise ValueError(
            "Holdout dataset contains classes that do not exist in the checkpoint taxonomy. "
            f"Missing classes: {sorted(set(missing_classes))}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricLearningEfficientNetB0(
        num_classes=len(class_names),
        weights_mode=eval_args.weights,
        embedding_dim=int(eval_args.embedding_dim),
        projection_dim=int(eval_args.projection_dim),
        args=eval_args,
        backbone_name=str(getattr(eval_args, "backbone", "convnextv2_nano")),
    ).to(device=device, dtype=model_dtype_for_args(eval_args))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    import json as json_lib

    class_mapping_dict = json_lib.loads(args.class_mapping) if args.class_mapping else None
    manifest = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "selected_classes": list(args.selected_class),
        "other_label": args.other_label,
        "source_class_names": class_names,
        "holdout_classes": list(dataset.classes),
    }
    save_json(output_dir / "holdout_manifest.json", manifest)

    loader = build_loader(NoAugDataset(remapped_samples, int(eval_args.image_size)), eval_args.batch_size, eval_args.num_workers)
    logits, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True, dtype=model_dtype_for_args(eval_args))
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                emb = model.encode(images)
                batch_logits = model.classify(emb, labels=None)
            logits.append(batch_logits.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    logits_concat = np.concatenate(logits, axis=0) if logits else np.empty((0, len(class_names)), dtype=np.float32)
    targets_concat = np.concatenate(targets, axis=0) if targets else np.empty((0,), dtype=np.int64)
    collapsed_logits, collapsed_targets, runtime_class_names, collapse_info = collapse_logits_and_targets_to_runtime_classes(
        logits_concat,
        targets_concat,
        class_names,
        selected_classes=list(args.selected_class),
        class_mapping=class_mapping_dict,
        other_label=args.other_label,
    )
    metrics = compute_classification_metrics(
        collapsed_logits,
        collapsed_targets,
        runtime_class_names,
        float(eval_args.confidence_threshold if hasattr(eval_args, "confidence_threshold") else 0.8),
    )
    confidence = compute_correct_confidence_by_class(collapsed_logits, collapsed_targets, runtime_class_names)
    probabilities = torch.softmax(torch.from_numpy(collapsed_logits), dim=1).numpy() if collapsed_logits.size else np.empty((0, len(runtime_class_names)), dtype=np.float32)

    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "correct_confidence_by_class.json", confidence)
    save_json(
        output_dir / "summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "dataset_root": str(dataset_root),
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
        },
    )
    save_confusion_matrix_plot(
        output_dir / "confusion_matrix.png",
        np.asarray(metrics["confusion_matrix"], dtype=np.int64),
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
