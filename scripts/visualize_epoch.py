#!/usr/bin/env python3
"""
visualize_epoch.py

Per-epoch visualization engine for the ESD waste classification pipeline.
Generates canonical artefacts from the COMPLETE test set (no augmentations):

  1. umap_thumbnail_map.png    — UMAP layout of test embeddings with thumbnails

Usage (standalone):
    python scripts/visualize_epoch.py \\
        --checkpoint <path/to/last.pt> \\
        --dataset-root Dataset_Final \\
        --output-dir Results/<run>/visualizations \\
        --epoch 3

Called automatically by the pipeline after every validation pass.
"""

from __future__ import annotations

import argparse
import sys
import math
import csv
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
import umap
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Import from the same scripts package ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from metric_learning_pipeline import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_BACKBONE_NAME,
    MetricLearningEfficientNetB0,
    adapt_checkpoint_state_dict_to_training_taxonomy,
    build_datasets,
    model_dtype_for_args,
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _make_clean_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _denorm(tensor_hwc: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalisation → uint8 HWC."""
    img = tensor_hwc * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 1. UMAP thumbnail embedding map
# ─────────────────────────────────────────────────────────────────────────────

def generate_umap_thumbnail_map(
    model: MetricLearningEfficientNetB0,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    epoch_label: str,
    max_samples: int = 0,
    thumbnail_limit: int = 0,
    thumb_size: int = 48,
) -> None:
    print("  [UMAP] Extracting embeddings, predictions, and thumbnails …")
    model.eval()
    model_dtype = next(model.parameters()).dtype
    all_emb, all_pred, all_thumb = [], [], []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="  UMAP embeddings", leave=False):
            images_gpu = images.to(device=device, dtype=model_dtype)
            emb = model.encode(images_gpu)
            logits = model.classify(emb)
            probs = F.softmax(logits, dim=1)
            _, preds = probs.max(dim=1)

            all_emb.append(emb.cpu().float().numpy())
            all_pred.extend(preds.cpu().tolist())
            for image in images:
                image_np = _denorm(image.permute(1, 2, 0).numpy())
                thumb = Image.fromarray(image_np).resize((thumb_size, thumb_size), Image.BILINEAR)
                all_thumb.append(np.array(thumb))

            if max_samples > 0 and sum(chunk.shape[0] for chunk in all_emb) >= max_samples:
                break

    emb_np = np.concatenate(all_emb, axis=0).astype(np.float32, copy=False)
    pred_np = np.array(all_pred[: len(emb_np)])
    thumb_np = all_thumb[: len(emb_np)]
    if max_samples > 0 and len(emb_np) > max_samples:
        emb_np = emb_np[:max_samples]
        pred_np = pred_np[:max_samples]
        thumb_np = thumb_np[:max_samples]

    pca_dim = min(50, emb_np.shape[1], max(2, emb_np.shape[0] - 1))
    if pca_dim < emb_np.shape[1]:
        emb_np = PCA(n_components=pca_dim, random_state=42).fit_transform(emb_np)

    print(f"  [UMAP] Running on {len(emb_np)} samples …")
    coords = umap.UMAP(
        n_components=2,
        random_state=42,
        init="spectral",
        n_neighbors=min(30, max(2, len(emb_np) - 1)),
        min_dist=0.05,
        metric="euclidean",
    ).fit_transform(emb_np)

    output_dir.mkdir(parents=True, exist_ok=True)
    palette = matplotlib.colormaps.get_cmap("tab10").resampled(len(class_names))
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_span = max(float(x_max - x_min), 1e-6)
    y_span = max(float(y_max - y_min), 1e-6)
    norm_x = (coords[:, 0] - x_min) / x_span
    norm_y = (coords[:, 1] - y_min) / y_span

    render_count = len(coords) if thumbnail_limit <= 0 else min(len(coords), thumbnail_limit)
    candidate_order = np.lexsort((norm_y, norm_x))
    if render_count < len(candidate_order):
        candidate_order = candidate_order[:render_count]

    # Pack every sample into a distinct cell, using the UMAP location as the
    # preferred cell and a greedy search for the nearest free neighbour when
    # that slot is already occupied.
    cols = max(1, int(math.ceil(math.sqrt(render_count * (x_span / y_span)))))
    rows = max(1, int(math.ceil(render_count / cols)))
    cell = max(thumb_size + 4, 8)
    pad = 2
    top_margin = 92
    left_margin = 24
    canvas_w = left_margin * 2 + cols * cell
    canvas_h = top_margin + rows * cell + 24
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    def _find_free_slot(target_r: int, target_c: int, occupied: np.ndarray) -> tuple[int, int]:
        if not occupied[target_r, target_c]:
            return target_r, target_c
        max_radius = max(rows, cols)
        for radius in range(1, max_radius):
            r0 = max(0, target_r - radius)
            r1 = min(rows - 1, target_r + radius)
            c0 = max(0, target_c - radius)
            c1 = min(cols - 1, target_c + radius)
            for c in range(c0, c1 + 1):
                if not occupied[r0, c]:
                    return r0, c
                if not occupied[r1, c]:
                    return r1, c
            for r in range(r0 + 1, r1):
                if not occupied[r, c0]:
                    return r, c0
                if not occupied[r, c1]:
                    return r, c1
        for r in range(rows):
            for c in range(cols):
                if not occupied[r, c]:
                    return r, c
        return target_r, target_c

    occupied = np.zeros((rows, cols), dtype=bool)
    sample_count = 0
    for idx in candidate_order:
        target_r = int(round(norm_y[idx] * max(rows - 1, 0)))
        target_c = int(round(norm_x[idx] * max(cols - 1, 0)))
        slot_r, slot_c = _find_free_slot(target_r, target_c, occupied)
        occupied[slot_r, slot_c] = True
        sample_count += 1

        x = left_margin + slot_c * cell + pad
        y = top_margin + slot_r * cell + pad
        thumb = Image.fromarray(thumb_np[idx])
        canvas.paste(thumb, (x, y))
        draw.rectangle(
            [x - 1, y - 1, x + thumb_size + 1, y + thumb_size + 1],
            outline=tuple(int(v * 255) for v in palette(int(pred_np[idx]))[:3]),
            width=2,
        )

    draw.text(
        (left_margin, 18),
        f"UMAP Thumbnail Map of Test Embeddings — {epoch_label}",
        fill=(245, 245, 245),
        font=title_font,
    )
    draw.text(
        (left_margin, 42),
        f"All {len(coords):,} test samples are placed on the canvas; thumbnails shift to the nearest free cell when a slot is occupied.",
        fill=(210, 210, 210),
        font=small_font,
    )
    draw.text(
        (left_margin, 60),
        "Border color = predicted class.",
        fill=(210, 210, 210),
        font=small_font,
    )
    legend_y = 76
    legend_x = left_margin
    for cls_id, cls_name in enumerate(class_names):
        color = tuple(int(v * 255) for v in palette(cls_id)[:3])
        draw.rectangle([legend_x, legend_y, legend_x + 10, legend_y + 10], fill=color, outline=(255, 255, 255))
        draw.text((legend_x + 14, legend_y - 1), cls_name, fill=(230, 230, 230), font=small_font)
        legend_x += max(110, 10 + 14 + len(cls_name) * 6)
        if legend_x > canvas_w - 160:
            legend_x = left_margin
            legend_y += 14

    output_path = output_dir / "umap_thumbnail_map.png"
    canvas.save(output_path)
    print(f"  [UMAP] Saved → {output_path}")

    np.savez_compressed(
        output_dir / "umap_embeddings_test_set.npz",
        coords=coords.astype(np.float32, copy=False),
        predicted_class_indices=pred_np.astype(np.int64, copy=False),
        class_names=np.asarray(class_names, dtype=object),
        epoch_label=np.asarray([epoch_label], dtype=object),
        thumbnail_size=np.asarray([thumb_size], dtype=np.int64),
        sample_count=np.asarray([len(coords)], dtype=np.int64),
    )
    with (output_dir / "umap_embeddings_test_set.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "umap_x", "umap_y", "predicted_class_index", "predicted_class_name"])
        for index, (coord, pred_index) in enumerate(zip(coords, pred_np)):
            writer.writerow([index, f"{coord[0]:.8f}", f"{coord[1]:.8f}", int(pred_index), class_names[int(pred_index)]])
    print(f"  [UMAP] Saved → {output_dir / 'umap_embeddings_test_set.npz'}")
    print(f"  [UMAP] Saved → {output_dir / 'umap_embeddings_test_set.csv'}")

# Main entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    checkpoint_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    epoch: int | str,
    batch_size: int = 128,
    num_workers: int = 4,
    umap_max_samples: int = 0,
    umap_thumbnail_limit: int = 0,
    umap_thumb_size: int = 18,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_label = f"Epoch {epoch}" if str(epoch).isdigit() else str(epoch)

    print(f"\n[VIZ] Starting epoch visualizations → {output_dir}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = argparse.Namespace(**ckpt["args"])
    ckpt_args.dataset_root = str(dataset_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = model_dtype_for_args(ckpt_args)

    # Load ONLY the test split (clean, no augmentation)
    _, _, test_dataset, _, _ = build_datasets(ckpt_args)
    loader = _make_clean_loader(test_dataset, batch_size, num_workers)
    class_names = test_dataset.classes
    backbone_name = str(getattr(ckpt_args, "backbone", DEFAULT_BACKBONE_NAME))

    # Build model
    model = MetricLearningEfficientNetB0(
        num_classes=len(class_names),
        weights_mode="none",
        embedding_dim=ckpt_args.embedding_dim,
        projection_dim=ckpt_args.projection_dim,
        args=ckpt_args,
        backbone_name=backbone_name,
    ).to(device=device, dtype=dtype)
    model_state_dict = ckpt["model_state_dict"]
    if list(ckpt["class_names"]) != class_names:
        model_state_dict, adaptation_report = adapt_checkpoint_state_dict_to_training_taxonomy(
            model_state_dict,
            list(ckpt["class_names"]),
            class_names,
            class_mapping=getattr(ckpt_args, "training_class_mapping", None),
        )
    else:
        adaptation_report = {"applied": False, "reason": "already_aligned"}
    model.load_state_dict(model_state_dict)
    model.eval()

    # ── 1. UMAP thumbnail map ────────────────────────────────────────────────
    generate_umap_thumbnail_map(
        model, loader, class_names, device,
        output_dir, epoch_label,
        max_samples=umap_max_samples,
        thumbnail_limit=umap_thumbnail_limit,
        thumb_size=umap_thumb_size,
    )

    print(f"[VIZ] All visualizations saved to {output_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a UMAP thumbnail embedding map and all-layer activation "
            "maps from a training checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a .pt checkpoint (last.pt or step_last.pt).",
    )
    parser.add_argument(
        "--dataset-root", default="Dataset_Final",
        help="Dataset root matching the one used during training.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory where the three PNG artefacts will be saved.",
    )
    parser.add_argument(
        "--epoch", default="initial",
        help="Epoch label to embed in the plot titles (default: 'initial').",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--umap-thumb-size", type=int, default=48,
        help="Thumbnail size for the UMAP thumbnail map (default 48).",
    )
    parser.add_argument(
        "--umap-max-samples", type=int, default=0,
        help="Optional cap for UMAP samples. Use 0 for the full test set.",
    )
    parser.add_argument(
        "--umap-thumbnail-limit", type=int, default=0,
        help="Maximum number of thumbnails rendered on the UMAP map. Use 0 to render all thumbnails.",
    )
    args = parser.parse_args()
    run(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        epoch=args.epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        umap_max_samples=args.umap_max_samples,
        umap_thumbnail_limit=args.umap_thumbnail_limit,
        umap_thumb_size=args.umap_thumb_size,
    )


if __name__ == "__main__":
    main()
