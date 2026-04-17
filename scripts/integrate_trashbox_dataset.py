#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TARGET_CLASS_MAP = {
    "cardboard": "paper",
    "e-waste": "ewaste",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
}

PLASTIC_SUBCLASS_MAP = {
    "plastic bags": "soft_plastic",
    "cigarette butt": "soft_plastic",
    "plastic bottles": "hard_plastic",
    "plastic containers": "hard_plastic",
    "plastic cups": "hard_plastic",
}

FLAT_PLASTIC_DISTRIBUTION = {
    "hard_plastic": 1325,
    "soft_plastic": 808,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class IntegrationStats:
    copied: int = 0
    skipped_duplicate: int = 0
    skipped_conflict: int = 0
    skipped_medical: int = 0
    skipped_unsupported: int = 0
    flat_plastic_hard: int = 0
    flat_plastic_soft: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate TrashBox into Dataset_Final with metadata sync.")
    parser.add_argument("--source-root", default="dataset_audit/incoming/TrashBox")
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--metadata-file", default="dataset_metadata.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep the TrashBox source tree after integration instead of deleting it from the audit folder.",
    )
    return parser.parse_args()


def sanitize_token(text: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    token = token.strip("_")
    return token or "unknown"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_existing_hash_index(dataset_root: Path) -> dict[str, tuple[str, str]]:
    existing_hashes: dict[str, tuple[str, str]] = {}
    for class_dir in sorted([path for path in dataset_root.iterdir() if path.is_dir() and not path.name.startswith(".")]):
        class_name = class_dir.name
        for image_path in sorted([path for path in class_dir.rglob("*") if path.is_file() and not path.name.startswith(".")]):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            digest = hash_file(image_path)
            existing_hashes.setdefault(digest, (class_name, image_path.relative_to(dataset_root).as_posix()))
    return existing_hashes


def load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    if not metadata_path.exists():
        return []
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Metadata file {metadata_path} must contain a JSON list.")
    return payload


def save_metadata(metadata_path: Path, records: list[dict[str, Any]]) -> None:
    backup_path = metadata_path.with_name(
        f"{metadata_path.stem}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{metadata_path.suffix}"
    )
    if metadata_path.exists():
        shutil.copy2(metadata_path, backup_path)
    tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(metadata_path)


def copy_image(
    source_path: Path,
    dataset_root: Path,
    target_label: str,
    source_tree: str,
    source_class: str,
    source_subclass: str | None,
    stats: IntegrationStats,
    seen_hashes: dict[str, str],
    existing_hashes: dict[str, tuple[str, str]],
    metadata_paths: set[str],
    dry_run: bool,
) -> dict[str, Any] | None:
    digest = hash_file(source_path)
    existing_match = existing_hashes.get(digest)
    if existing_match is not None:
        existing_label, _ = existing_match
        if existing_label != target_label:
            stats.skipped_conflict += 1
        else:
            stats.skipped_duplicate += 1
        return None

    existing_target = seen_hashes.get(digest)
    if existing_target is not None:
        if existing_target != target_label:
            stats.skipped_conflict += 1
        else:
            stats.skipped_duplicate += 1
        return None

    existing_target_path = dataset_root / target_label / f"trashbox_{sanitize_token(source_tree)}_{sanitize_token(source_class)}"
    if source_subclass:
        existing_target_path = existing_target_path.with_name(
            f"{existing_target_path.name}_{sanitize_token(source_subclass)}"
        )
    filename = f"{existing_target_path.name}_{digest[:12]}{source_path.suffix.lower()}"
    destination = dataset_root / target_label / filename
    suffix = 1
    while str(destination.relative_to(dataset_root)).replace("\\", "/") in metadata_paths or destination.exists():
        destination = dataset_root / target_label / f"{existing_target_path.name}_{digest[:12]}_{suffix}{source_path.suffix.lower()}"
        suffix += 1

    relative_destination = destination.relative_to(dataset_root).as_posix()
    if relative_destination in metadata_paths:
        stats.skipped_duplicate += 1
        seen_hashes[digest] = target_label
        return None

    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    seen_hashes[digest] = target_label
    metadata_paths.add(relative_destination)
    stats.copied += 1
    return {
        "file_path": f"{dataset_root.name}/{relative_destination}",
        "label": target_label,
        "source_dataset": "TrashBox",
        "source_tree": source_tree,
        "source_class": source_class,
        "source_subclass": source_subclass,
        "source_path": source_path.as_posix(),
        "source_sha256": digest,
        "integration_rule": f"{source_class}->{target_label}" if source_subclass is None else f"{source_class}/{source_subclass}->{target_label}",
    }


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    dataset_root = Path(args.dataset_root)
    metadata_path = dataset_root / args.metadata_file

    if not source_root.is_dir():
        raise FileNotFoundError(f"TrashBox source root not found: {source_root}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    metadata = load_metadata(metadata_path)
    metadata_paths = {str(entry.get("file_path", "")).replace("\\", "/") for entry in metadata}
    existing_hashes = build_existing_hash_index(dataset_root)
    seen_hashes: dict[str, str] = {}
    stats = IntegrationStats()
    new_records: list[dict[str, Any]] = []

    source_trees = [
        ("TrashBox_train_dataset_subfolders", source_root / "TrashBox_train_dataset_subfolders"),
        ("TrashBox_train_set", source_root / "TrashBox_train_set"),
    ]

    flat_plastic_files: list[Path] = []
    labelled_plastic_files: list[tuple[Path, str, str]] = []

    for tree_name, tree_root in source_trees:
        if not tree_root.is_dir():
            continue
        for class_dir in sorted([path for path in tree_root.iterdir() if path.is_dir() and not path.name.startswith(".")]):
            class_name = class_dir.name
            if class_name == "medical":
                stats.skipped_medical += sum(1 for _ in class_dir.rglob("*") if _.is_file() and not _.name.startswith("."))
                continue
            if class_name == "plastic":
                if tree_name == "TrashBox_train_dataset_subfolders":
                    for subclass_dir in sorted([path for path in class_dir.iterdir() if path.is_dir() and not path.name.startswith(".")]):
                        for image_path in sorted([p for p in subclass_dir.iterdir() if p.is_file() and not p.name.startswith(".")]):
                            labelled_plastic_files.append((image_path, tree_name, f"plastic/{subclass_dir.name}"))
                else:
                    for image_path in sorted([p for p in class_dir.iterdir() if p.is_file() and not p.name.startswith(".")]):
                        flat_plastic_files.append(image_path)
                continue

            target_label = TARGET_CLASS_MAP.get(class_name)
            if target_label is None:
                stats.skipped_unsupported += sum(1 for _ in class_dir.rglob("*") if _.is_file() and not _.name.startswith("."))
                continue
            for image_path in sorted([p for p in class_dir.rglob("*") if p.is_file() and not p.name.startswith(".")]):
                record = copy_image(
                    image_path,
                    dataset_root,
                    target_label,
                    tree_name,
                    class_name,
                    None,
                    stats,
                    seen_hashes,
                    existing_hashes,
                    metadata_paths,
                    args.dry_run,
                )
                if record is not None:
                    new_records.append(record)

    for image_path, tree_name, source_subclass in labelled_plastic_files:
        subclass_name = source_subclass.split("/", 1)[1]
        target_label = PLASTIC_SUBCLASS_MAP.get(subclass_name)
        if target_label is None:
            stats.skipped_unsupported += 1
            continue
        record = copy_image(
            image_path,
            dataset_root,
            target_label,
            tree_name,
            "plastic",
            subclass_name,
            stats,
            seen_hashes,
            existing_hashes,
            metadata_paths,
            args.dry_run,
        )
        if record is not None:
            if target_label == "hard_plastic":
                stats.flat_plastic_hard += 1
            else:
                stats.flat_plastic_soft += 1
            new_records.append(record)

    if flat_plastic_files:
        hard_weight = FLAT_PLASTIC_DISTRIBUTION["hard_plastic"]
        soft_weight = FLAT_PLASTIC_DISTRIBUTION["soft_plastic"]
        total_weight = hard_weight + soft_weight
        for image_path in flat_plastic_files:
            digest = hash_file(image_path)
            if digest in seen_hashes:
                if seen_hashes[digest] == "hard_plastic":
                    stats.skipped_duplicate += 1
                else:
                    stats.skipped_duplicate += 1
                continue
            bucket = int(digest[:16], 16) % total_weight
            target_label = "soft_plastic" if bucket < soft_weight else "hard_plastic"
            record = copy_image(
                image_path,
                dataset_root,
                target_label,
                "TrashBox_train_set",
                "plastic",
                "flat_plastic",
                stats,
                seen_hashes,
                existing_hashes,
                metadata_paths,
                args.dry_run,
            )
            if record is not None:
                if target_label == "hard_plastic":
                    stats.flat_plastic_hard += 1
                else:
                    stats.flat_plastic_soft += 1
                new_records.append(record)

    if not args.dry_run and new_records:
        metadata.extend(new_records)
        metadata.sort(key=lambda row: str(row.get("file_path", "")))
        save_metadata(metadata_path, metadata)

    if not args.keep_source and not args.dry_run:
        shutil.rmtree(source_root)

    label_counts = Counter(str(record.get("label", "")) for record in new_records)
    report = {
        "source_root": source_root.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "copied": stats.copied,
        "skipped_duplicate": stats.skipped_duplicate,
        "skipped_conflict": stats.skipped_conflict,
        "skipped_medical": stats.skipped_medical,
        "skipped_unsupported": stats.skipped_unsupported,
        "flat_plastic_hard": stats.flat_plastic_hard,
        "flat_plastic_soft": stats.flat_plastic_soft,
        "label_counts": dict(sorted(label_counts.items())),
        "dry_run": args.dry_run,
        "keep_source": args.keep_source,
    }
    print(json.dumps(report, indent=2))
    if not args.dry_run:
        (dataset_root / "trashbox_integration_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
