#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from metric_learning_pipeline import (
    compute_classification_metrics,
    compute_correct_confidence_by_class,
    TRAINING_CLASS_ORDER,
    project_class_name_to_training_taxonomy,
    save_classification_report_csv,
    save_confusion_matrix_csv,
    save_confidence_histogram,
    save_confusion_matrix_plot,
    save_json,
    save_reliability_diagram,
)

warnings.filterwarnings("ignore", message="Palette images with Transparency", category=UserWarning)


class HoldoutDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.samples[index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = Image.open(path).convert("RGB")
        return image, label, path


def collate(batch):
    images, labels, paths = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot CLIP on a folder dataset.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--audit-metal-subclasses",
        action="store_true",
        help="Run the legacy metal-subclass audit with the same CLIP model instead of zero-shot holdout evaluation.",
    )
    parser.add_argument("--top-k-examples", type=int, default=50)
    return parser.parse_args()


def make_prompts(class_name: str) -> list[str]:
    name = class_name.replace("_", " ")
    return [
        f"a photo of {name}",
        f"a close-up photo of {name}",
        f"a photo of waste that is {name}",
        f"a product photo of {name}",
    ]


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
    remainders = sorted(
        ((raw[index] - extras[index], index) for index in range(3)),
        reverse=True,
    )
    for _, index in remainders[:shortfall]:
        extras[index] += 1
    return tuple(minimums[index] + extras[index] for index in range(3))


def source_prefix(path: str) -> str:
    stem = Path(path).stem
    parts = re.split(r"[^a-zA-Z0-9]", stem)
    for part in parts:
        cleaned = re.sub(r"\d+", "", part)
        if cleaned:
            return cleaned.lower()
    return stem[:8].lower()


def build_metal_split_membership(dataset_root: Path, seed: int) -> dict[str, list[str]]:
    metal_dir = dataset_root / "metal"
    files = sorted(
        p for p in metal_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    by_source: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_source[source_prefix(str(path))].append(path)

    rng = random.Random(seed)
    split_membership = {"train": [], "val": [], "test": []}
    ratios = (0.9, 0.05, 0.05)
    for _, src_paths in sorted(by_source.items()):
        shuffled = list(src_paths)
        rng.shuffle(shuffled)
        tr_n, va_n, te_n = allocate_split_counts(len(shuffled), ratios)
        split_membership["train"].extend(str(path) for path in shuffled[:tr_n])
        split_membership["val"].extend(str(path) for path in shuffled[tr_n : tr_n + va_n])
        split_membership["test"].extend(str(path) for path in shuffled[tr_n + va_n : tr_n + va_n + te_n])
    return split_membership


def make_metal_prompts(category: str) -> list[str]:
    prompts = {
        "metal_bottle": [
            "a photo of a metal bottle",
            "a photo of a stainless steel bottle",
            "a close-up photo of a metal bottle",
            "a product photo of a reusable metal water bottle",
        ],
        "metal_glass": [
            "a photo of a metal glass",
            "a photo of a metal tumbler",
            "a photo of a stainless steel glass",
            "a close-up photo of a metal tumbler",
        ],
        "metal_utensil": [
            "a photo of a metal utensil",
            "a photo of metal cutlery",
            "a close-up photo of a metal spoon or fork",
            "a photo of stainless steel utensils",
        ],
        "metal_can": [
            "a photo of a metal can",
            "a photo of a tin can",
            "a photo of an aluminum can",
            "a close-up photo of a metal can",
        ],
        "metal_cup": [
            "a photo of a metal cup",
            "a photo of a metal mug",
            "a photo of a stainless steel cup",
            "a close-up photo of a metal mug",
        ],
        "metal_plate": [
            "a photo of a metal plate",
            "a photo of a stainless steel plate",
            "a close-up photo of a metal plate",
            "a photo of metal dinnerware",
        ],
        "metal_bowl": [
            "a photo of a metal bowl",
            "a photo of a stainless steel bowl",
            "a close-up photo of a metal bowl",
            "a photo of metal kitchenware",
        ],
        "metal_container": [
            "a photo of a metal jar",
            "a photo of a metal container",
            "a photo of a tin container",
            "a close-up photo of a metal jar or container",
        ],
        "other_metal": [
            "a photo of an other metal object",
            "a photo of scrap metal",
            "a photo of metal waste",
            "a photo of a generic metal object",
        ],
    }
    return prompts[category]


def run_metal_audit(dataset_root: Path, output_dir: Path, model_name: str, pretrained: str, image_size: int, batch_size: int, num_workers: int, seed: int, top_k_examples: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_membership = build_metal_split_membership(dataset_root, seed)
    rows: list[dict[str, Any]] = []
    for split_name, paths in split_membership.items():
        for path in paths:
            rows.append(
                {
                    "path": path,
                    "split": split_name,
                    "source_prefix": source_prefix(path),
                }
            )

    categories = [
        "metal_bottle",
        "metal_glass",
        "metal_utensil",
        "metal_can",
        "metal_cup",
        "metal_plate",
        "metal_bowl",
        "metal_container",
        "other_metal",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()

    text_features = []
    for category in categories:
        tokens = tokenizer(make_metal_prompts(category)).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        text_features.append(features.mean(dim=0, keepdim=True))
    text_features = torch.cat(text_features, dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    class ImagePathDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]], image_size: int) -> None:
            self.rows = rows
            self.image_size = image_size

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str, str]:
            row = self.rows[index]
            with Image.open(row["path"]) as img:
                img = img.convert("RGB")
                return img, row["split"], row["path"], row["source_prefix"]

    def collate(batch):
        images, splits, paths, prefixes = zip(*batch)
        return list(images), list(splits), list(paths), list(prefixes)

    dataset = ImagePathDataset(rows, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate,
    )

    predictions: list[dict[str, Any]] = []
    category_counter = Counter()
    split_counter = {"train": Counter(), "val": Counter(), "test": Counter()}
    sample_probs: dict[str, list[tuple[float, str]]] = {category: [] for category in categories}

    with torch.no_grad():
        for images, splits, paths, prefixes in loader:
            batch = torch.stack([preprocess(image) for image in images]).to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                image_features = model.encode_image(batch)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = 100.0 * image_features @ text_features.t()

            probs_np = torch.softmax(logits, dim=1).detach().cpu().numpy()
            for path, split_name, prefix, prob_row in zip(paths, splits, prefixes, probs_np):
                pred_index = int(np.argmax(prob_row))
                pred_name = categories[pred_index]
                confidence = float(prob_row[pred_index])
                row = {
                    "path": path,
                    "split": split_name,
                    "source_prefix": prefix,
                    "pred_category": pred_name,
                    "confidence": confidence,
                }
                predictions.append(row)
                category_counter[pred_name] += 1
                split_counter[split_name][pred_name] += 1
                sample_probs[pred_name].append((confidence, path))

    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "split", "source_prefix", "pred_category", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)

    for category in categories:
        top_dir = output_dir / "top_examples" / category
        top_dir.mkdir(parents=True, exist_ok=True)
        for rank, (_, path) in enumerate(sorted(sample_probs[category], reverse=True)[: max(0, int(top_k_examples))], start=1):
            source = Path(path)
            target = top_dir / f"{rank:03d}_{source.name}"
            try:
                shutil.copy2(source, target)
            except Exception:
                pass

    summary = {
        "dataset_root": str(dataset_root),
        "metal_image_count": len(predictions),
        "split_counts": {split: len(split_membership[split]) for split in ("train", "val", "test")},
        "predicted_counts": dict(category_counter),
        "predicted_counts_by_split": {
            split: dict(counter) for split, counter in split_counter.items()
        },
        "categories": categories,
        "model_name": model_name,
        "pretrained": pretrained,
        "top_k_examples": int(top_k_examples),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


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
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.audit_metal_subclasses:
        return run_metal_audit(
            dataset_root=dataset_root,
            output_dir=output_dir,
            model_name=args.model_name,
            pretrained=args.pretrained,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=42,
            top_k_examples=args.top_k_examples,
        )

    dataset = datasets.ImageFolder(str(dataset_root))
    class_names = list(TRAINING_CLASS_ORDER)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device)
    model.eval()

    text_features = []
    for class_name in class_names:
        prompts = make_prompts(class_name)
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        text_features.append(features.mean(dim=0, keepdim=True))
    text_features = torch.cat(text_features, dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    remapped_samples: list[tuple[str, int]] = []
    holdout_class_remap: dict[str, str] = {}
    for path, target in dataset.samples:
        holdout_class_name = dataset.classes[target]
        normalized = holdout_class_name.strip().lower().replace(" ", "_")
        projected = project_class_name_to_training_taxonomy(normalized)
        if projected is None or projected not in class_names:
            continue
        holdout_class_remap[holdout_class_name] = projected
        remapped_samples.append((path, class_names.index(projected)))

    loader = build_loader(HoldoutDataset(remapped_samples, args.image_size), args.batch_size, args.num_workers)

    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for images, labels, paths in loader:
            batch_tensors = torch.stack([preprocess(image) for image in images]).to(device)
            labels = labels.to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                image_features = model.encode_image(batch_tensors)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = 100.0 * image_features @ text_features.t()
            logits_np = logits.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            all_logits.append(logits_np)
            all_targets.append(labels_np)
            probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
            for path, true_index, prob_row in zip(paths, labels_np, probs):
                pred_index = int(np.argmax(prob_row))
                all_rows.append(
                    {
                        "path": path,
                        "true_class": class_names[int(true_index)],
                        "pred_class": class_names[pred_index],
                        "confidence": float(prob_row[pred_index]),
                        "correct": bool(pred_index == int(true_index)),
                    }
                )

    logits_concat = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, len(class_names)), dtype=np.float32)
    targets_concat = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0,), dtype=np.int64)
    metrics = compute_classification_metrics(logits_concat, targets_concat, class_names, confidence_threshold=0.0)
    confidence = compute_correct_confidence_by_class(logits_concat, targets_concat, class_names)
    confmat = np.asarray(metrics["confusion_matrix"], dtype=np.int64)

    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "correct_confidence_by_class.json", confidence)
    save_confusion_matrix_csv(output_dir / "confmat_counts_test.csv", confmat, class_names, percent=False)
    save_confusion_matrix_csv(output_dir / "confmat_rate_pct_test.csv", confmat, class_names, percent=True)
    save_classification_report_csv(output_dir / "classification_report_test.csv", metrics, class_names)
    probabilities = torch.softmax(torch.from_numpy(logits_concat), dim=1).numpy() if logits_concat.size else np.empty((0, len(class_names)), dtype=np.float32)
    save_confidence_histogram(output_dir / "confidence_histogram.png", probabilities, "CLIP Confidence Histogram")
    save_reliability_diagram(output_dir / "reliability_diagram.png", metrics["calibration"], "CLIP Reliability Diagram")
    save_confusion_matrix_plot(output_dir / "confusion_matrix.png", confmat, class_names, "CLIP Confusion Matrix")
    save_json(output_dir / "all_predictions.json", all_rows)
    wrong_rows = [row for row in all_rows if not row["correct"]]
    save_json(output_dir / "wrong_predictions.json", wrong_rows)
    with (output_dir / "wrong_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "true_class", "pred_class", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(all_rows)

    wrong_dir = output_dir / "wrong_predictions"
    wrong_dir.mkdir(parents=True, exist_ok=True)
    for row in wrong_rows:
        source_path = Path(row["path"])
        class_dir = wrong_dir / row["true_class"] / row["pred_class"]
        class_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source_path, class_dir / source_path.name)
        except Exception:
            pass

    save_json(
        output_dir / "summary.json",
        {
            "dataset_root": str(dataset_root),
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
            "class_names": class_names,
            "per_class_accuracy": metrics["per_class_accuracy"],
            "per_class_avg_confidence": metrics["per_class_avg_confidence"],
            "wrong_count": len(wrong_rows),
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "holdout_class_remap": holdout_class_remap,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
