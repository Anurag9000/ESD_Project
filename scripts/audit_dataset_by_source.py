#!/usr/bin/env python3
"""
Dataset Source Audit Tool
=========================
For every class in Dataset_Final, groups images by their source-batch prefix
(everything before the last underscore+hex/number suffix), then randomly
samples 10 images from each batch into a structured audit folder.

Output layout:
    dataset_audit/
        metal/
            bottles-and-cans/        ← source batch name
                sample_001.jpg
                sample_002.jpg
                ...
            count-coins-image-dataset/
                sample_001.jpg
                ...
        plastic/
            ...

Usage:
    python scripts/audit_dataset_by_source.py
    python scripts/audit_dataset_by_source.py --dataset Dataset_Final --out dataset_audit --n 10
"""

import argparse
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Prefix extraction logic
# ──────────────────────────────────────────────────────────────────────────────

def extract_source_prefix(filename: str) -> str:
    """
    Given a filename like:
        bottles-and-cans_008edc25.jpg          → 'bottles-and-cans'
        biodegradable1234_0.jpg                → 'biodegradable1234'
        crop_0_abc123.jpg                      → 'crop_0'
        202512-paper100.jpg                    → '202512-paper100.jpg'  (no separable prefix)
        20240626_162343.jpg                    → '20240626'

    Rule: strip the file extension, then split on underscore from the RIGHT.
    If the rightmost segment looks like a hash (hex) or pure number, it is
    considered a suffix and dropped. Otherwise the whole stem is the prefix.
    """
    stem = Path(filename).stem  # remove extension

    # Split on the LAST underscore
    parts = stem.rsplit('_', 1)
    if len(parts) == 2:
        suffix = parts[1]
        # treat as a suffix if it is purely hex OR purely numeric
        if re.fullmatch(r'[0-9a-fA-F]{4,}', suffix) or re.fullmatch(r'\d+', suffix):
            return parts[0]

    # No strippable suffix → whole stem is the source key
    return stem


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audit dataset by source batch")
    parser.add_argument("--dataset", default="Dataset_Final",
                        help="Root dataset directory (default: Dataset_Final)")
    parser.add_argument("--out", default="dataset_audit",
                        help="Output audit directory (default: dataset_audit)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of sample images per source batch (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", nargs="*",
                        help="Limit to specific classes (default: all)")
    args = parser.parse_args()

    random.seed(args.seed)
    dataset_root = Path(args.dataset)
    out_root = Path(args.out)

    if not dataset_root.is_dir():
        print(f"ERROR: dataset dir not found: {dataset_root}")
        return

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    all_classes = sorted([
        d.name for d in dataset_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and d.name not in {"auto_split_manifest.json"}
    ])

    if args.classes:
        all_classes = [c for c in all_classes if c in args.classes]

    total_sources = 0
    total_sampled = 0

    for class_name in all_classes:
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            continue

        # Group files by source prefix
        source_groups: dict[str, list[Path]] = defaultdict(list)
        for f in class_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith("."):
                prefix = extract_source_prefix(f.name)
                source_groups[prefix].append(f)

        print(f"\n{'='*60}")
        print(f"  Class: {class_name}  ({len(source_groups)} source batches, "
              f"{sum(len(v) for v in source_groups.values())} images total)")
        print(f"{'='*60}")

        for source_name, files in sorted(source_groups.items()):
            # Sample up to N images randomly
            sample = random.sample(files, min(args.n, len(files)))

            out_dir = out_root / class_name / source_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for i, src_file in enumerate(sorted(sample), 1):
                dst = out_dir / f"sample_{i:03d}{src_file.suffix.lower()}"
                shutil.copy2(src_file, dst)

            print(f"  [{len(files):>6} imgs]  {source_name:<60} → sampled {len(sample)}")
            total_sources += 1
            total_sampled += len(sample)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Total source batches : {total_sources}")
    print(f"  Total images sampled : {total_sampled}")
    print(f"  Audit folder         : {out_root.resolve()}")
    print(f"{'='*60}")
    print(f"\nOpen '{out_root}/' in your file manager to visually inspect each batch.")
    print("For each batch that looks contaminated, note the source name and tell me.")


if __name__ == "__main__":
    main()
