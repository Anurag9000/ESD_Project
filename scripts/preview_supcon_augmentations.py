#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from metric_learning_pipeline import build_datasets, seed_everything, training_tensor_from_image
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from metric_learning_pipeline import build_datasets, seed_everything, training_tensor_from_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the original image beside several SupCon train-time augmentation views."
    )
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--image", default="", help="Optional explicit image path. Defaults to the first image in the split.")
    parser.add_argument("--output-dir", default="Results/supcon_augmentation_previews")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-mapping", type=str, default="")
    parser.add_argument("--auto-split-ratios", default="0.9,0.05,0.05")
    parser.add_argument("--runtime-bad-sample-cleanup", action="store_true")
    parser.add_argument("--augment-repeats", type=int, default=1)
    parser.add_argument("--augment-gaussian-sigmas", type=float, default=1.0)
    parser.add_argument("--camera-color-cast-probability", type=float, default=1.0)
    parser.add_argument("--camera-color-cast-strength", type=float, default=0.50)
    parser.add_argument("--camera-color-cast-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-views", type=int, default=6)
    return parser


def _load_source_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _make_panel(images: list[np.ndarray], titles: list[str], output_path: Path) -> None:
    count = len(images)
    fig, axes = plt.subplots(1, count, figsize=(4 * count, 4), constrained_layout=True)
    if count == 1:
        axes = [axes]
    for ax, image, title in zip(axes, images, titles, strict=True):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, _, _ = build_datasets(args)
    split_dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}[args.split]
    if args.image:
        image_path = Path(args.image)
    else:
        if len(split_dataset) <= 0:
            raise RuntimeError("No samples found in the selected split.")
        image_path = Path(split_dataset.samples[0][0])

    source_image = _load_source_image(image_path)
    source_array = np.asarray(source_image)

    view_images: list[np.ndarray] = [source_array]
    view_titles: list[str] = ["original"]
    for index in range(max(1, int(args.num_views))):
        rng = random.Random(args.seed + 10_003 * (index + 1))
        tensor = training_tensor_from_image(
            source_image,
            args.image_size,
            rng,
            args.augment_gaussian_sigmas,
            camera_color_cast_probability=args.camera_color_cast_probability,
            camera_color_cast_strength=args.camera_color_cast_strength,
        )
        view_images.append(tensor.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy())
        view_titles.append(f"view {index + 1}")

    output_path = output_dir / f"{args.split}_supcon_augmentations.png"
    _make_panel(view_images, view_titles, output_path)
    print(output_path)
    print(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
