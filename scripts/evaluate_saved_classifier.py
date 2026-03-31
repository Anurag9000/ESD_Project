#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metric_learning_pipeline import (
    DeterministicAugmentedImageFolder,
    MetricLearningEfficientNetB0,
    collect_logits_and_labels,
    compute_classification_metrics,
    compute_correct_confidence_by_class,
    make_loader,
    model_dtype_for_args,
    release_training_memory,
    save_json,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved classifier checkpoint on non-aug and aug test suites.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset_Final"))
    parser.add_argument("--batch-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--log-every-eval-steps", type=int, default=1000)
    parser.add_argument("--use-gabor", action="store_true")
    return parser


def checkpoint_args_namespace(checkpoint: dict[str, Any]) -> argparse.Namespace:
    raw_args = dict(checkpoint.get("args", {}))
    defaults = {
        "dataset_root": "Dataset_Final",
        "image_size": 224,
        "augment_repeats": 16,
        "augment_gaussian_sigmas": 2.0,
        "embedding_dim": 512,
        "projection_dim": 256,
        "precision": "mixed",
        "weights": "default",
        "confidence_threshold": 0.8,
        "seed": 42,
        "gabor_kernel_size": 15,
        "gabor_orientations": 8,
        "gabor_wavelengths": "4,8,16",
        "gabor_sigma": 4.0,
        "gabor_gamma": 0.5,
    }
    for key, value in defaults.items():
        raw_args.setdefault(key, value)
    return argparse.Namespace(**raw_args)


def save_confusion_matrix_with_matplotlib(
    path: Path,
    confusion_matrix: np.ndarray,
    class_names: list[str],
    title: str,
) -> None:
    matrix = np.asarray(confusion_matrix, dtype=np.int64)
    fig_width = max(8.0, 1.25 * len(class_names))
    fig_height = max(6.0, 1.05 * len(class_names))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    max_value = max(int(matrix.max()), 1)
    for row_index in range(matrix.shape[0]):
        row_total = int(matrix[row_index].sum())
        for col_index in range(matrix.shape[1]):
            value = int(matrix[row_index, col_index])
            pct = (100.0 * value / row_total) if row_total > 0 else 0.0
            text_color = "white" if value > 0.55 * max_value else "black"
            ax.text(
                col_index,
                row_index,
                f"{value}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate_split(
    *,
    model: MetricLearningEfficientNetB0,
    device: torch.device,
    eval_args: argparse.Namespace,
    dataset: DeterministicAugmentedImageFolder,
    split_name: str,
    output_dir: Path,
    class_names: list[str],
) -> dict[str, Any]:
    dataset.set_epoch(0)
    loader = make_loader(
        dataset,
        eval_args.batch_size,
        eval_args.num_workers,
        eval_args.prefetch_factor,
        shuffle=False,
    )
    logits, targets = collect_logits_and_labels(
        model=model,
        loader=loader,
        device=device,
        max_batches=eval_args.max_eval_batches,
        log_path=output_dir / "evaluation_log.jsonl",
        log_every_eval_steps=eval_args.log_every_eval_steps,
        split=split_name,
        args=eval_args,
    )
    metrics = compute_classification_metrics(logits, targets, class_names, eval_args.confidence_threshold)
    correct_confidence = compute_correct_confidence_by_class(logits, targets, class_names)
    release_training_memory(device, loader)

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    save_json(split_dir / "metrics.json", metrics)
    save_json(split_dir / "correct_confidence_by_class.json", correct_confidence)
    save_confusion_matrix_with_matplotlib(
        split_dir / "confusion_matrix.png",
        np.asarray(metrics["confusion_matrix"], dtype=np.int64),
        class_names,
        f"{split_name.replace('_', ' ').title()} Confusion Matrix",
    )
    summary = {
        "split": split_name,
        "num_samples": metrics["num_samples"],
        "raw_accuracy": metrics["raw_accuracy"],
        "qualified_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
    }
    save_json(split_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = build_parser()
    cli_args = parser.parse_args()

    checkpoint = torch.load(cli_args.checkpoint, map_location="cpu")
    train_args = checkpoint_args_namespace(checkpoint)
    seed_everything(int(train_args.seed))

    class_names = list(checkpoint["class_names"])
    eval_args = argparse.Namespace(**vars(train_args))
    eval_args.batch_size = cli_args.batch_size
    eval_args.num_workers = cli_args.num_workers
    eval_args.prefetch_factor = cli_args.prefetch_factor
    eval_args.max_eval_batches = cli_args.max_eval_batches
    eval_args.log_every_eval_steps = cli_args.log_every_eval_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricLearningEfficientNetB0(
        num_classes=len(class_names),
        weights_mode=train_args.weights,
        embedding_dim=train_args.embedding_dim,
        projection_dim=train_args.projection_dim,
        use_gabor=cli_args.use_gabor,
        args=train_args,
    ).to(device=device, dtype=model_dtype_for_args(train_args))
    model.load_state_dict(checkpoint.get("best_classifier_state", checkpoint["model_state_dict"]))
    model.eval()

    dataset_root = cli_args.dataset_root
    test_non_aug = DeterministicAugmentedImageFolder(
        dataset_root / "test",
        train_args.image_size,
        1,
        "test",
        train_args.seed,
        train_args.augment_gaussian_sigmas,
        apply_augmentation=False,
    )
    test_aug = DeterministicAugmentedImageFolder(
        dataset_root / "test",
        train_args.image_size,
        train_args.augment_repeats,
        "test",
        train_args.seed,
        train_args.augment_gaussian_sigmas,
        apply_augmentation=True,
    )

    output_dir = cli_args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(cli_args.checkpoint),
        "output_dir": str(output_dir),
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "batch_size": cli_args.batch_size,
        "augment_repeats": train_args.augment_repeats,
        "confidence_threshold": train_args.confidence_threshold,
        "evaluated_splits": ["test_non_aug", "test_aug"],
    }
    save_json(output_dir / "evaluation_manifest.json", manifest)

    summaries = {
        "test_non_aug": evaluate_split(
            model=model,
            device=device,
            eval_args=eval_args,
            dataset=test_non_aug,
            split_name="test_non_aug",
            output_dir=output_dir,
            class_names=class_names,
        ),
        "test_aug": evaluate_split(
            model=model,
            device=device,
            eval_args=eval_args,
            dataset=test_aug,
            split_name="test_aug",
            output_dir=output_dir,
            class_names=class_names,
        ),
    }
    save_json(output_dir / "evaluation_summary.json", summaries)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
