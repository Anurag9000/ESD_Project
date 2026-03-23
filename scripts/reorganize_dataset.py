#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

from PIL import Image, ImageOps


FINAL_CLASSES = ("organic", "paper", "other", "metal")
METAL_PREFIXES = ("metal", "alum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reorganize the dataset for image classification by remapping classes, "
            "normalizing filenames, correcting extensions, and stripping metadata."
        )
    )
    parser.add_argument(
        "--source-root",
        default="Dataset_Final",
        help="Existing dataset root containing class folders and dataset_metadata.json",
    )
    parser.add_argument(
        "--rebuilt-root",
        default="Dataset_Final_rebuilt",
        help="Temporary output directory for the rebuilt dataset",
    )
    parser.add_argument(
        "--backup-zip",
        default="Dataset_Final_original_backup.zip",
        help="Backup name for the original zip before writing the rebuilt archive",
    )
    return parser.parse_args()


def looks_metallic(filename: str) -> bool:
    stem = Path(filename).stem.lower()
    return any(stem.startswith(prefix) for prefix in METAL_PREFIXES)


def target_label(source_label: str, filename: str) -> str:
    if source_label == "plastic":
        return "other"
    if source_label == "other" and looks_metallic(filename):
        return "metal"
    return source_label


def choose_output_format(image: Image.Image, actual_format: str | None) -> tuple[str, str]:
    actual = (actual_format or "").upper()
    has_alpha = "A" in image.getbands() or "transparency" in image.info

    if actual == "PNG":
        return "PNG", ".png"
    if actual == "WEBP":
        return ("PNG", ".png") if has_alpha else ("JPEG", ".jpg")
    if actual in {"JPEG", "MPO"}:
        return "JPEG", ".jpg"
    return ("PNG", ".png") if has_alpha else ("JPEG", ".jpg")


def normalize_for_output(image: Image.Image, output_format: str) -> Image.Image:
    if output_format == "JPEG":
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", image.size, (255, 255, 255))
            alpha = image.getchannel("A")
            background.paste(image.convert("RGBA"), mask=alpha)
            return background
        if image.mode == "P":
            if "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.getchannel("A"))
                return background
            return image.convert("RGB")
        if image.mode in {"L", "RGB"}:
            return image.convert("RGB") if image.mode != "RGB" else image.copy()
        return image.convert("RGB")

    if output_format == "PNG":
        if image.mode == "CMYK":
            return image.convert("RGB")
        return image.copy()

    return image.copy()


def metadata_path_from_entry(project_root: Path, file_path: str) -> Path:
    rel = PureWindowsPath(file_path).as_posix()
    return project_root / rel


def save_image(image: Image.Image, destination: Path, output_format: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {}
    if output_format == "JPEG":
        save_kwargs.update({"quality": 95, "optimize": True})
    elif output_format == "PNG":
        save_kwargs.update({"optimize": True})
    image.save(destination, format=output_format, **save_kwargs)


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()
    source_root = project_root / args.source_root
    rebuilt_root = project_root / args.rebuilt_root
    metadata_file = source_root / "dataset_metadata.json"

    if not source_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {source_root}")
    if rebuilt_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing rebuilt root: {rebuilt_root}")

    with metadata_file.open("r", encoding="utf-8") as handle:
        source_metadata = json.load(handle)

    for label in FINAL_CLASSES:
        (rebuilt_root / label).mkdir(parents=True, exist_ok=True)

    counters = Counter()
    class_counts = Counter()
    source_counts = Counter()
    moved_counts = Counter()
    extension_counts = Counter()
    actual_format_counts = Counter()
    output_format_counts = Counter()
    mismatched_extensions = Counter()
    exif_removed = 0
    gps_removed = 0

    new_metadata: list[dict[str, str]] = []

    for index, entry in enumerate(source_metadata, start=1):
        source_label = entry["label"]
        source_path = metadata_path_from_entry(project_root, entry["file_path"])
        original_name = source_path.name
        final_label = target_label(source_label, original_name)

        if not source_path.exists():
            raise FileNotFoundError(f"Metadata entry missing on disk: {source_path}")

        with Image.open(source_path) as image:
            actual_format = image.format
            exif = image.getexif()
            if exif and len(exif):
                exif_removed += 1
                if 34853 in exif or 0x8825 in exif:
                    gps_removed += 1

            transposed = ImageOps.exif_transpose(image)
            transposed.load()

            output_format, extension = choose_output_format(transposed, actual_format)
            normalized = normalize_for_output(transposed, output_format)

        counters[final_label] += 1
        new_name = f"{final_label}_{counters[final_label]:06d}{extension}"
        destination = rebuilt_root / final_label / new_name
        save_image(normalized, destination, output_format)

        source_counts[source_label] += 1
        class_counts[final_label] += 1
        actual_format_counts[(source_label, actual_format or "UNKNOWN")] += 1
        output_format_counts[output_format] += 1
        extension_counts[extension] += 1

        source_extension = source_path.suffix.lower()
        expected_extension = {
            "JPEG": ".jpg",
            "PNG": ".png",
        }.get((actual_format or "").upper(), source_extension)
        if source_extension != expected_extension:
            mismatched_extensions[source_label] += 1

        if source_label != final_label:
            moved_counts[f"{source_label}->{final_label}"] += 1

        new_metadata.append(
            {
                "file_path": f"{source_root.name}/{final_label}/{new_name}",
                "label": final_label,
            }
        )

        if index % 1000 == 0:
            print(f"Processed {index}/{len(source_metadata)} images")

    report = {
        "source_root": source_root.name,
        "final_classes": list(FINAL_CLASSES),
        "source_image_count": len(source_metadata),
        "final_image_count": len(new_metadata),
        "source_class_counts": dict(sorted(source_counts.items())),
        "final_class_counts": dict(sorted(class_counts.items())),
        "class_moves": dict(sorted(moved_counts.items())),
        "source_format_counts": {
            f"{label}:{fmt}": count
            for (label, fmt), count in sorted(actual_format_counts.items())
        },
        "output_format_counts": dict(sorted(output_format_counts.items())),
        "output_extension_counts": dict(sorted(extension_counts.items())),
        "mismatched_source_extensions_by_class": dict(sorted(mismatched_extensions.items())),
        "exif_removed": exif_removed,
        "gps_removed": gps_removed,
        "metal_rule": {
            "source_folder": "other",
            "filename_prefixes": list(METAL_PREFIXES),
        },
        "filename_scheme": "<class>_<6-digit-sequence>.<normalized-extension>",
        "path_separator_in_metadata": "forward_slash",
    }

    with (rebuilt_root / "dataset_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(new_metadata, handle, indent=2)
        handle.write("\n")

    with (rebuilt_root / "dataset_reorganization_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    final_total = sum(class_counts.values())
    lines = [
        "# Training Readiness Summary",
        "",
        f"Total images: {final_total}",
        "",
        "Final classes:",
    ]
    for label in FINAL_CLASSES:
        count = class_counts[label]
        pct = (count / final_total) * 100 if final_total else 0
        lines.append(f"- {label}: {count} ({pct:.2f}%)")
    lines.extend(
        [
            "",
            "Applied changes:",
            "- moved all images from `plastic` into `other`",
            "- moved metallic items from `other` into `metal` using filename prefixes `metal*` and `alum*`",
            "- regenerated every file with normalized filenames",
            "- corrected extensions based on actual image content",
            "- stripped EXIF and GPS metadata",
        ]
    )
    (rebuilt_root / "training_readiness_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("Rebuild complete")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
