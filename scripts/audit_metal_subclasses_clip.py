#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the Dataset_Final/metal subset with zero-shot CLIP.")
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-examples", type=int, default=50)
    return parser.parse_args()


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


def make_prompts(category: str) -> list[str]:
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


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_membership = build_metal_split_membership(dataset_root, args.seed)
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
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device)
    model.eval()

    text_features = []
    for category in categories:
        tokens = tokenizer(make_prompts(category)).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        text_features.append(features.mean(dim=0, keepdim=True))
    text_features = torch.cat(text_features, dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    dataset = ImagePathDataset(rows, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
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
                probabilities = torch.softmax(logits, dim=1)

            probs_np = probabilities.detach().cpu().numpy()
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

    top_k = max(0, int(args.top_k_examples))
    for category in categories:
        top_dir = output_dir / "top_examples" / category
        top_dir.mkdir(parents=True, exist_ok=True)
        for rank, (_, path) in enumerate(sorted(sample_probs[category], reverse=True)[:top_k], start=1):
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
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "top_k_examples": top_k,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
