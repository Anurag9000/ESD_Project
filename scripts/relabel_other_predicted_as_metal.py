#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from metric_learning_pipeline import (
    MetricLearningEfficientNetB0,
    evaluation_tensor_from_image,
    model_dtype_for_args,
    seed_everything,
)


SPLITS = ("train", "val", "test")
CLASSES = ("organic", "paper", "other", "metal")


class OtherImagesDataset(Dataset[tuple[torch.Tensor, str, str]]):
    def __init__(self, items: list[tuple[str, Path]], image_size: int) -> None:
        self.items = items
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        split, path = self.items[index]
        image = Image.open(path).convert("RGB")
        return evaluation_tensor_from_image(image, self.image_size), split, str(path)


def checkpoint_args_namespace(checkpoint: dict[str, Any]) -> argparse.Namespace:
    raw_args = dict(checkpoint.get("args", {}))
    defaults = {
        "image_size": 224,
        "embedding_dim": 512,
        "projection_dim": 256,
        "precision": "mixed",
        "weights": "default",
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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def gather_other_images(dataset_root: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for split in SPLITS:
        other_dir = dataset_root / split / "other"
        for path in sorted(other_dir.iterdir()):
            if path.is_file():
                items.append((split, path))
    return items


def predict_metal_paths(
    *,
    checkpoint_path: Path,
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint_args_namespace(checkpoint)
    seed_everything(int(train_args.seed))

    class_names = list(checkpoint["class_names"])
    metal_index = class_names.index("metal")
    use_gabor = bool(checkpoint.get("args", {}).get("weights") == "gabor") or bool(
        checkpoint.get("args", {}).get("use_gabor", False)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricLearningEfficientNetB0(
        num_classes=len(class_names),
        weights_mode=train_args.weights,
        embedding_dim=train_args.embedding_dim,
        projection_dim=train_args.projection_dim,
        use_gabor=use_gabor,
        args=train_args,
    ).to(device=device, dtype=model_dtype_for_args(train_args))
    model.load_state_dict(checkpoint.get("best_classifier_state", checkpoint["model_state_dict"]))
    model.eval()

    items = gather_other_images(dataset_root)
    dataset = OtherImagesDataset(items, int(train_args.image_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    moved: list[dict[str, Any]] = []
    with torch.inference_mode():
        for tensors, splits, paths in loader:
            tensors = tensors.to(device=device, dtype=model_dtype_for_args(train_args), non_blocking=True)
            embeddings = model.encode(tensors)
            logits = model.classify(embeddings)
            probabilities = torch.softmax(logits.float(), dim=1)
            predicted_indices = probabilities.argmax(dim=1).cpu().tolist()
            confidences = probabilities.max(dim=1).values.cpu().tolist()
            metal_scores = probabilities[:, metal_index].cpu().tolist()

            for split, path_str, predicted_index, confidence, metal_score in zip(splits, paths, predicted_indices, confidences, metal_scores):
                if int(predicted_index) == metal_index:
                    moved.append(
                        {
                            "split": split,
                            "source_path": path_str,
                            "predicted_label": class_names[int(predicted_index)],
                            "prediction_confidence": float(confidence),
                            "metal_probability": float(metal_score),
                        }
                    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "class_names": class_names,
        "num_other_images_scored": len(items),
        "num_relabelled_other_to_metal": len(moved),
        "relabelled_counts_by_split": dict(Counter(item["split"] for item in moved)),
    }
    return moved, summary


def move_paths_and_update_metadata(dataset_root: Path, moved: list[dict[str, Any]]) -> dict[str, Any]:
    moved_by_source = {entry["source_path"]: entry for entry in moved}

    for entry in moved:
        source = Path(entry["source_path"])
        target = dataset_root / entry["split"] / "metal" / source.name
        if target.exists() and target != source:
            raise FileExistsError(f"Refusing to overwrite existing file: {target}")
        source.rename(target)
        entry["target_path"] = str(target)

    metadata_paths = {
        "dataset": dataset_root / "dataset_metadata.json",
        **{split: dataset_root / f"{split}_metadata.json" for split in SPLITS},
    }
    loaded_metadata = {name: load_json(path) for name, path in metadata_paths.items()}

    for name, entries in loaded_metadata.items():
        for record in entries:
            source_path = record["file_path"]
            if source_path in moved_by_source:
                target_path = moved_by_source[source_path]["target_path"]
                record["file_path"] = target_path
                record["label"] = "metal"
        save_json(metadata_paths[name], entries)

    split_class_counts: dict[str, dict[str, int]] = {}
    split_totals: dict[str, int] = {}
    class_totals: Counter[str] = Counter()
    for split in SPLITS:
        entries = loaded_metadata[split]
        counts = Counter(entry["label"] for entry in entries)
        split_class_counts[split] = {label: counts.get(label, 0) for label in CLASSES}
        split_totals[split] = len(entries)
        class_totals.update(counts)

    dataset_split_report = load_json(dataset_root / "dataset_split_report.json")
    dataset_split_report["class_totals"] = {label: class_totals.get(label, 0) for label in CLASSES}
    dataset_split_report["split_class_counts"] = split_class_counts
    dataset_split_report["split_totals"] = split_totals
    save_json(dataset_root / "dataset_split_report.json", dataset_split_report)

    reorg_report = load_json(dataset_root / "dataset_reorganization_report.json")
    previous_final_counts = reorg_report.get("final_class_counts", {})
    relabel_count = len(moved)
    reorg_report["final_class_counts"] = {label: class_totals.get(label, 0) for label in CLASSES}
    class_moves = dict(reorg_report.get("class_moves", {}))
    class_moves["other->metal_model"] = relabel_count
    reorg_report["class_moves"] = class_moves
    reorg_report["model_assisted_relabel"] = {
        "source_folder": "other",
        "target_folder": "metal",
        "moved_image_count": relabel_count,
        "previous_final_class_counts": previous_final_counts,
        "updated_final_class_counts": reorg_report["final_class_counts"],
    }
    save_json(dataset_root / "dataset_reorganization_report.json", reorg_report)

    split_summary_lines = [
        "# Split Summary",
        "",
        f"Seed: {dataset_split_report['seed']}",
        "",
        "Ratios:",
        f"- train: {int(round(dataset_split_report['ratios']['train'] * 100))}%",
        f"- val: {int(round(dataset_split_report['ratios']['val'] * 100))}%",
        f"- test: {int(round(dataset_split_report['ratios']['test'] * 100))}%",
        "",
        "Counts by split and class:",
    ]
    for split in SPLITS:
        split_summary_lines.append(f"- {split}: {split_totals[split]}")
        for label in CLASSES:
            split_summary_lines.append(f"  - {label}: {split_class_counts[split][label]}")
    save_text(dataset_root / "split_summary.md", "\n".join(split_summary_lines) + "\n")

    total_images = sum(split_totals.values())
    readiness_lines = [
        "# Training Readiness Summary",
        "",
        f"Total images: {total_images}",
        "",
        "Final classes:",
    ]
    for label in CLASSES:
        count = class_totals.get(label, 0)
        readiness_lines.append(f"- {label}: {count} ({(100.0 * count / total_images):.2f}%)")
    readiness_lines.extend(
        [
            "",
            "Applied changes:",
            "- moved all images from `plastic` into `other`",
            "- moved metallic items from `other` into `metal` using filename prefixes `metal*` and `alum*`",
            "- moved additional model-predicted metallic items from `other` into `metal`",
            "- regenerated every file with normalized filenames",
            "- corrected extensions based on actual image content",
            "- stripped EXIF and GPS metadata",
        ]
    )
    save_text(dataset_root / "training_readiness_summary.md", "\n".join(readiness_lines) + "\n")

    return {
        "class_totals": {label: class_totals.get(label, 0) for label in CLASSES},
        "split_class_counts": split_class_counts,
        "split_totals": split_totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Move `other` images predicted as metal into the metal class and rewrite metadata.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset_Final"))
    parser.add_argument("--batch-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    moved, summary = predict_metal_paths(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    updated_counts = move_paths_and_update_metadata(args.dataset_root, moved)

    report = {
        **summary,
        **updated_counts,
        "moved_items": moved,
    }
    save_json(args.dataset_root / "model_assisted_other_to_metal_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "moved_items"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
