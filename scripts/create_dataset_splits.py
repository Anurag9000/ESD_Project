#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path


CLASSES = ("organic", "paper", "other", "metal")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val/test splits for the dataset."
    )
    parser.add_argument(
        "--dataset-root",
        default="Dataset_Final",
        help="Dataset root containing top-level class folders",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: total * ratio for split, ratio in ratios.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remainder = total - sum(counts.values())

    ordered = sorted(
        SPLITS,
        key=lambda split: (raw[split] - counts[split], ratios[split]),
        reverse=True,
    )
    for split in ordered[:remainder]:
        counts[split] += 1
    return counts


def write_json(path: Path, payload: list[dict[str, str]] | dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }

    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    for cls in CLASSES:
        class_dir = dataset_root / cls
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class directory not found: {class_dir}")

    split_dirs = [dataset_root / split for split in SPLITS]
    if any(path.exists() for path in split_dirs):
        raise FileExistsError("Refusing to run because split directories already exist")

    rng = random.Random(args.seed)
    metadata_all: list[dict[str, str]] = []
    metadata_by_split = {split: [] for split in SPLITS}
    report_counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    class_totals = Counter()

    for split in SPLITS:
        for cls in CLASSES:
            (dataset_root / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        class_dir = dataset_root / cls
        files = sorted(path for path in class_dir.iterdir() if path.is_file())
        class_totals[cls] = len(files)
        counts = split_counts(len(files), ratios)
        report_counts["train"][cls] = counts["train"]
        report_counts["val"][cls] = counts["val"]
        report_counts["test"][cls] = counts["test"]

        rng.shuffle(files)
        boundaries = {
            "train": counts["train"],
            "val": counts["train"] + counts["val"],
        }
        chunks = {
            "train": files[: boundaries["train"]],
            "val": files[boundaries["train"] : boundaries["val"]],
            "test": files[boundaries["val"] :],
        }

        for split in SPLITS:
            for source_path in chunks[split]:
                destination = dataset_root / split / cls / source_path.name
                shutil.move(str(source_path), str(destination))
                entry = {
                    "file_path": f"{dataset_root.name}/{split}/{cls}/{source_path.name}",
                    "label": cls,
                    "split": split,
                }
                metadata_all.append(entry)
                metadata_by_split[split].append(entry)

        class_dir.rmdir()
        print(
            f"{cls}: train={counts['train']} val={counts['val']} test={counts['test']}"
        )

    write_json(dataset_root / "dataset_metadata.json", metadata_all)
    for split in SPLITS:
        write_json(dataset_root / f"{split}_metadata.json", metadata_by_split[split])

    summary = {
        "dataset_root": dataset_root.name,
        "seed": args.seed,
        "ratios": ratios,
        "class_totals": dict(class_totals),
        "split_class_counts": report_counts,
        "split_totals": {
            split: sum(report_counts[split].values()) for split in SPLITS
        },
    }
    write_json(dataset_root / "dataset_split_report.json", summary)

    lines = [
        "# Split Summary",
        "",
        f"Seed: {args.seed}",
        "",
        "Ratios:",
        f"- train: {args.train_ratio:.0%}",
        f"- val: {args.val_ratio:.0%}",
        f"- test: {args.test_ratio:.0%}",
        "",
        "Counts by split and class:",
    ]
    for split in SPLITS:
        lines.append(f"- {split}: {sum(report_counts[split].values())}")
        for cls in CLASSES:
            lines.append(f"  - {cls}: {report_counts[split][cls]}")
    (dataset_root / "split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
