#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from train_phase0_mim import (
        RepoSafeConvNeXtMIM,
        SpatialMaskGenerator,
        save_phase0_reconstruction_preview,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from train_phase0_mim import (
        RepoSafeConvNeXtMIM,
        SpatialMaskGenerator,
        save_phase0_reconstruction_preview,
    )

try:
    from metric_learning_pipeline import build_datasets, seed_everything
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from metric_learning_pipeline import build_datasets, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Phase 0 MIM reconstructions from a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a Phase 0 checkpoint such as best.pt or last.pt.")
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--backbone", default="convnextv2_nano")
    parser.add_argument("--weights", choices=("default", "none"), default="default")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--decoder-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-mapping", type=str, default="")
    parser.add_argument("--auto-split-ratios", default="0.9,0.05,0.05")
    parser.add_argument("--runtime-bad-sample-cleanup", action="store_true")
    parser.add_argument("--augment-repeats", type=int, default=1)
    parser.add_argument("--augment-gaussian-sigmas", type=float, default=1.0)
    parser.add_argument("--camera-color-cast-probability", type=float, default=1.0)
    parser.add_argument("--camera-color-cast-strength", type=float, default=0.50)
    parser.add_argument("--camera-color-cast-eval", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    seed_everything(args.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_dataset, val_dataset, test_dataset, _, _ = build_datasets(args)
    split_map = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    dataset = split_map[args.split]
    loader = DataLoader(
        dataset,
        batch_size=max(1, min(args.batch_size, args.sample_count)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RepoSafeConvNeXtMIM(args.backbone, args.weights, input_res=args.image_size, decoder_dim=args.decoder_dim).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()

    batch = next(iter(loader))
    if len(batch) == 3:
        images, _, _ = batch
    else:
        images, _ = batch
    images = images.to(device)
    mask_generator = SpatialMaskGenerator(args.image_size, args.patch_size, args.mask_ratio)
    pixel_mask, _ = mask_generator(images.shape[0], device)

    with torch.no_grad():
        reconstructed = model(images, pixel_mask)

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "reconstruction_previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_checkpoint_preview.png"
    save_phase0_reconstruction_preview(
        output_path,
        originals=images,
        pixel_mask=pixel_mask,
        reconstructed=reconstructed,
        epoch=0,
        global_step=0,
        sample_count=args.sample_count,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
