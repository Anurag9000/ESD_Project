#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    from metric_learning_pipeline import (
        CAMERA_COLOR_CAST_STRENGTH,
        apply_camera_color_cast,
        build_datasets,
        seed_everything,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from metric_learning_pipeline import (
        CAMERA_COLOR_CAST_STRENGTH,
        apply_camera_color_cast,
        build_datasets,
        seed_everything,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a pink-tint sweep and a model-input preview using the repo's fixed preprocessing."
    )
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--image", default="", help="Optional explicit image path. Defaults to the first image in the split.")
    parser.add_argument("--output-dir", default="Results/pink_tint_previews")
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
    parser.add_argument(
        "--tint-strengths",
        default="0.0,0.10,0.20,0.30,0.40,0.50,0.60",
        help="Comma-separated tint strengths for the sweep panel.",
    )
    return parser
def _load_source_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _aspect_preserving_base_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = TF.resize(image, image_size, interpolation=InterpolationMode.BILINEAR)
    cropped = TF.center_crop(resized, [image_size, image_size])
    return TF.to_tensor(cropped).clamp(0.0, 1.0)


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

    _, _, test_dataset, _, _ = build_datasets(args)
    if args.image:
        image_path = Path(args.image)
    else:
        if len(test_dataset) <= 0:
            raise RuntimeError("No samples found in the selected split.")
        image_path = Path(test_dataset.samples[0][0])

    strengths = [float(segment.strip()) for segment in args.tint_strengths.split(",") if segment.strip()]
    if not strengths:
        strengths = [0.0, CAMERA_COLOR_CAST_STRENGTH]

    source_image = _load_source_image(image_path)
    base_tensor = _aspect_preserving_base_tensor(source_image, args.image_size)

    sweep_images: list[np.ndarray] = []
    sweep_titles: list[str] = []
    for strength in strengths:
        tinted = apply_camera_color_cast(
            base_tensor.clone(),
            rng=random.Random(0),
            gaussian_sigmas=0.0,
            probability=1.0,
            strength=strength,
        )
        sweep_images.append(tinted.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy())
        sweep_titles.append(f"strength={strength:.2f}")

    sweep_path = output_dir / f"{args.split}_pink_tint_sweep.png"
    _make_panel(sweep_images, sweep_titles, sweep_path)

    model_input = apply_camera_color_cast(
        base_tensor.clone(),
        rng=random.Random(0),
        gaussian_sigmas=0.0,
        probability=1.0,
        strength=CAMERA_COLOR_CAST_STRENGTH,
    )
    model_view = model_input.clamp(0.0, 1.0)
    model_path = output_dir / f"{args.split}_model_input_view.png"
    _make_panel(
        [
            np.asarray(source_image),
            model_view.permute(1, 2, 0).cpu().numpy(),
        ],
        ["source image", "model input after resize/crop + tint"],
        model_path,
    )

    print(sweep_path)
    print(model_path)
    print(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
