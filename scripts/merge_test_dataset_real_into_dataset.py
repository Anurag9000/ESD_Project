#!/usr/bin/env python3
"""Merge the curated Test_Dataset_Real holdout into Dataset_Final.

This helper exists for the explicit user request to fold the external real
holdout into the main dataset tree before training. It only imports the
supported training classes (organic, metal, paper) and skips plastic.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


CLASS_MAP = {
    "organic": "organic",
    "metal": "metal",
    "paper": "paper",
}


@dataclass(frozen=True)
class MergeStats:
    copied: int = 0
    skipped_existing: int = 0
    skipped_missing: int = 0


def unique_destination(dst_dir: Path, src_name: str) -> Path:
    candidate = dst_dir / src_name
    if not candidate.exists():
        return candidate
    stem = Path(src_name).stem
    suffix = Path(src_name).suffix
    index = 1
    while True:
        alt = dst_dir / f"{stem}_{index}{suffix}"
        if not alt.exists():
            return alt
        index += 1


def merge_class(source_dir: Path, dest_dir: Path, label: str, metadata: list[dict], dry_run: bool) -> MergeStats:
    if not source_dir.exists():
        return MergeStats(skipped_missing=1)
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped_existing = 0
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        target = unique_destination(dest_dir, path.name)
        if target.exists():
            skipped_existing += 1
            continue
        rel_path = f"Dataset_Final/{label}/{target.name}"
        metadata.append({"file_path": rel_path, "label": label})
        if not dry_run:
            shutil.copy2(path, target)
        copied += 1
    return MergeStats(copied=copied, skipped_existing=skipped_existing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="Test_Dataset_Real")
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--metadata-file", default="dataset_metadata.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-source", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    dataset_root = Path(args.dataset_root)
    metadata_path = dataset_root / args.metadata_file

    if not source_root.is_dir():
        raise SystemExit(f"source root not found: {source_root}")
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root not found: {dataset_root}")
    if not metadata_path.is_file():
        raise SystemExit(f"metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    if not isinstance(metadata, list):
        raise SystemExit("dataset_metadata.json must contain a JSON list")

    total_copied = 0
    total_skipped_existing = 0
    total_skipped_missing = 0

    for source_class, target_class in CLASS_MAP.items():
        stats = merge_class(
            source_root / source_class.capitalize(),
            dataset_root / target_class,
            target_class,
            metadata,
            args.dry_run,
        )
        total_copied += stats.copied
        total_skipped_existing += stats.skipped_existing
        total_skipped_missing += stats.skipped_missing

    if not args.dry_run:
        backup_path = metadata_path.with_suffix(metadata_path.suffix + ".bak")
        shutil.copy2(metadata_path, backup_path)
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
            fh.write("\n")

    if args.delete_source and not args.dry_run:
        shutil.rmtree(source_root)

    print(
        json.dumps(
            {
                "source_root": str(source_root),
                "dataset_root": str(dataset_root),
                "copied": total_copied,
                "skipped_existing": total_skipped_existing,
                "skipped_missing": total_skipped_missing,
                "dry_run": args.dry_run,
                "delete_source": args.delete_source,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
