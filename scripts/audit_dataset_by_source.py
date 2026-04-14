#!/usr/bin/env python3
"""
Dataset Source Audit Tool — Flat Output
========================================
Samples 10 images per source-batch per class into a FLAT per-class folder.
Every sampled file is renamed as:  <source-batch-name>_sample_NNN.jpg

Output layout:
    dataset_audit/
        metal/
            bottles-and-cans_sample_001.jpg
            bottles-and-cans_sample_002.jpg
            count-coins-image-dataset_sample_001.jpg
            ...
        plastic/
            ...

Usage:
    python scripts/audit_dataset_by_source.py
    python scripts/audit_dataset_by_source.py --n 15 --classes metal plastic
"""

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


def extract_source_prefix(filename: str) -> str:
    """
    Ultra-aggressive 'word-only' grouping:
    Splits by non-alphanumeric, strips all digits, and takes the first 
    resulting non-empty word. This collapses versions and numbered files 
    (e.g., 'plastic123', 'plastic234', 'trash-v2' -> 'plastic', 'trash').
    """
    stem = Path(filename).stem
    # Split by any character that isn't a letter or number
    parts = re.split(r'[^a-zA-Z0-9]', stem)
    for p in parts:
        # Remove all digits from the part
        word = re.sub(r'\d+', '', p)
        if word:
            return word.lower()
            
    # Fallback to first 8 chars if no pure words found
    return stem[:8].lower()


def safe_name(s: str, max_len: int = 80) -> str:
    """Replace characters that are bad in filenames, and truncate to max_len."""
    cleaned = re.sub(r'[^\w\-]', '_', s)
    return cleaned[:max_len]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Dataset_Final")
    parser.add_argument("--out",     default="dataset_audit")
    parser.add_argument("--n",       type=int, default=10)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--classes", nargs="*")
    args = parser.parse_args()

    random.seed(args.seed)
    dataset_root = Path(args.dataset)
    out_root     = Path(args.out)

    if not dataset_root.is_dir():
        print(f"ERROR: dataset dir not found: {dataset_root}")
        return

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_classes = sorted([
        d.name for d in dataset_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if args.classes:
        all_classes = [c for c in all_classes if c in args.classes]

    total_sources = 0
    total_sampled = 0

    for class_name in all_classes:
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            continue

        # Group images by source-batch prefix
        groups: dict[str, list[Path]] = defaultdict(list)
        for f in class_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith("."):
                groups[extract_source_prefix(f.name)].append(f)

        # One flat output folder per class — NO subfolders
        out_dir = out_root / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*55}")
        print(f"  {class_name.upper()}  ({len(groups)} batches, "
              f"{sum(len(v) for v in groups.values())} images)")
        print(f"{'='*55}")

        for source, files in sorted(groups.items()):
            sample = random.sample(files, min(args.n, len(files)))
            prefix = safe_name(source)
            for i, src in enumerate(sorted(sample), 1):
                dst = out_dir / f"{prefix}_sample_{i:03d}{src.suffix.lower()}"
                shutil.copy2(src, dst)
            print(f"  [{len(files):>6}]  {source}  →  {len(sample)} samples")
            total_sources += 1
            total_sampled += len(sample)

    print(f"\n{'='*55}")
    print(f"  DONE — {total_sources} batches, {total_sampled} images")
    print(f"  Location: {out_root.resolve()}")
    print(f"{'='*55}")
    print("\nEach file inside a class folder is named:")
    print("  <source-batch-name>_sample_NNN.jpg")
    print("\nTell me any source-batch name that looks contaminated and I will purge it.")


if __name__ == "__main__":
    main()
