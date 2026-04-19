#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete mispredicted dataset images and update dataset metadata JSONs.")
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--wrong-predictions-csv", required=True)
    parser.add_argument("--metadata-json", default="Dataset_Final/dataset_metadata.json")
    parser.add_argument("--metadata-backup-json", default="Dataset_Final/dataset_metadata.backup_20260417_193153.json")
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--delete-empty-parents", action="store_true")
    return parser.parse_args()


def load_wrong_paths(csv_path: Path) -> list[Path]:
    wrong_paths: list[Path] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path = row.get("path", "").strip()
            if path:
                wrong_paths.append(Path(path))
    return wrong_paths


def rewrite_metadata(metadata_path: Path, deleted_paths: set[str]) -> dict[str, Any]:
    if not metadata_path.exists():
        return {"path": str(metadata_path), "updated": False, "reason": "missing"}
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {"path": str(metadata_path), "updated": False, "reason": "not_a_list"}
    before = len(data)
    filtered = [item for item in data if item.get("file_path") not in deleted_paths]
    metadata_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
    return {
        "path": str(metadata_path),
        "updated": True,
        "before": before,
        "after": len(filtered),
        "removed": before - len(filtered),
    }


def remove_empty_parent_dirs(paths: list[Path], dataset_root: Path) -> list[str]:
    removed: list[str] = []
    class_dirs = set()
    for path in paths:
        try:
            rel = path.relative_to(dataset_root)
        except ValueError:
            continue
        if len(rel.parts) >= 2:
            class_dirs.add(dataset_root / rel.parts[0])
    for class_dir in sorted(class_dirs):
        if not class_dir.is_dir():
            continue
        try:
            next(class_dir.iterdir())
        except StopIteration:
            class_dir.rmdir()
            removed.append(str(class_dir))
    return removed


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    csv_path = Path(args.wrong_predictions_csv)
    metadata_json = Path(args.metadata_json)
    metadata_backup_json = Path(args.metadata_backup_json)
    wrong_paths = load_wrong_paths(csv_path)
    unique_paths = sorted({str(path) for path in wrong_paths})

    deleted_files: list[str] = []
    missing_files: list[str] = []
    for path_str in unique_paths:
        path = Path(path_str)
        if path.exists():
            path.unlink()
            deleted_files.append(path_str)
        else:
            missing_files.append(path_str)

    empty_dirs_removed: list[str] = []
    if args.delete_empty_parents:
        empty_dirs_removed = remove_empty_parent_dirs([Path(p) for p in deleted_files], dataset_root)

    metadata_updates = [
        rewrite_metadata(metadata_json, set(deleted_files)),
        rewrite_metadata(metadata_backup_json, set(deleted_files)),
    ]

    summary = {
        "dataset_root": str(dataset_root),
        "wrong_predictions_csv": str(csv_path),
        "requested_deletions": len(unique_paths),
        "deleted_files": len(deleted_files),
        "missing_files": len(missing_files),
        "missing_paths": missing_files[:20],
        "empty_dirs_removed": empty_dirs_removed,
        "metadata_updates": metadata_updates,
    }
    print(json.dumps(summary, indent=2))
    if args.output_summary:
        Path(args.output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
