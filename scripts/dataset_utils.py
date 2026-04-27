#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


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
