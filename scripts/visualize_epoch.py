#!/usr/bin/env python3
"""
visualize_epoch.py

Per-epoch visualization engine for the ESD waste classification pipeline.
Generates canonical artefacts from the COMPLETE test set (no augmentations):

  1. umap_thumbnail_map.png    — UMAP layout of test embeddings with thumbnails
  2. all_layer_activations.png — mean spatial activation for the backbone's major
                                  feature stages (per-layer = mean over channels)
                                  Why per-layer, not per-neuron?
                                  Per-layer = one summary map per stage = usable signal.

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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No display needed — pure file output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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


class _LayerHook:
    """Forward hook that captures the output of a single module."""
    def __init__(self, module: torch.nn.Module) -> None:
        self._handle = module.register_forward_hook(self._fn)
        self.out: torch.Tensor | None = None

    def _fn(self, _m, _i, output: torch.Tensor) -> None:
        self.out = output.detach()

    def remove(self) -> None:
        self._handle.remove()


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
    thumbnail_limit: int = 1024,
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
    fig, ax = plt.subplots(figsize=(16, 14), dpi=150)

    point_colors = np.array([palette(int(pred)) for pred in pred_np])
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=8,
        alpha=0.10,
        c=point_colors,
        linewidths=0,
    )

    if thumbnail_limit > 0 and len(coords) > thumbnail_limit:
        rng = np.random.default_rng(42)
        selected = np.sort(rng.choice(len(coords), size=thumbnail_limit, replace=False))
    else:
        selected = np.arange(len(coords))

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 1e-6)
    zoom = max(0.18, min(0.58, 0.32 * (48 / max(thumb_size, 1))))

    for idx in selected:
        color = palette(int(pred_np[idx]))
        imagebox = OffsetImage(thumb_np[idx], zoom=zoom)
        ab = AnnotationBbox(
            imagebox,
            (coords[idx, 0], coords[idx, 1]),
            frameon=True,
            pad=0.12,
            bboxprops=dict(edgecolor=color, linewidth=1.1, boxstyle="round,pad=0.05"),
        )
        ax.add_artist(ab)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=cls_name,
                   markerfacecolor=palette(cls_id), markersize=8)
        for cls_id, cls_name in enumerate(class_names)
    ]
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=True, ncol=2)
    ax.set_title(
        f"UMAP Thumbnail Map of Test Embeddings — {epoch_label}\n"
        f"Thumbnails are colored by predicted class; background points show the full test embedding cloud.",
        fontsize=14,
    )
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.12)
    ax.set_xlim(x_min - 0.05 * span, x_max + 0.05 * span)
    ax.set_ylim(y_min - 0.05 * span, y_max + 0.05 * span)
    fig.tight_layout()
    output_path = output_dir / "umap_thumbnail_map.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [UMAP] Saved → {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. All-layer activations
# ─────────────────────────────────────────────────────────────────────────────

def backbone_activation_specs(model: MetricLearningEfficientNetB0) -> list[tuple[str, torch.nn.Module]]:
    backbone = model.backbone
    if hasattr(backbone, "features"):
        return [
            ("Stage 0 · Stem", backbone.features[0]),
            ("Stage 1 · Features", backbone.features[1]),
            ("Stage 2 · Features", backbone.features[2]),
            ("Stage 3 · Features", backbone.features[3]),
            ("Stage 4 · Features", backbone.features[4]),
            ("Stage 5 · Features", backbone.features[5]),
            ("Stage 6 · Features", backbone.features[6]),
            ("Stage 7 · Features", backbone.features[7]),
            ("Stage 8 · Head", backbone.features[8]),
        ]
    if hasattr(backbone, "stem") and hasattr(backbone, "stages"):
        specs: list[tuple[str, torch.nn.Module]] = [("Stage 0 · Stem", backbone.stem)]
        for idx, stage in enumerate(backbone.stages):
            specs.append((f"Stage {idx + 1} · ConvNeXt Stage {idx}", stage))
        if hasattr(backbone, "norm_pre"):
            specs.append(("Final Norm", backbone.norm_pre))
        return specs
    children = [
        (name, module)
        for name, module in backbone.named_children()
        if any(parameter.requires_grad or parameter.numel() > 0 for parameter in module.parameters(recurse=True))
    ]
    if not children:
        raise ValueError("Backbone exposes no hookable child modules for visualization.")
    return [(f"Stage {idx} · {name}", module) for idx, (name, module) in enumerate(children)]

def generate_layer_activations(
    model: MetricLearningEfficientNetB0,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    epoch_label: str,
    sample_limit: int = 0,
) -> None:
    """
    Aggregate mean spatial activations across the entire test set by default.
    The hook set follows the backbone's native stage layout. We average over:
      - batch dimension  (all images)
      - channel dimension (all learned filters)
    resulting in one (H×W) heatmap per stage — a true "What does this stage attend to?"
    """
    stage_specs = backbone_activation_specs(model)
    print(f"  [ActMaps] Hooking all {len(stage_specs)} backbone stages …")
    model.eval()
    model_dtype = next(model.parameters()).dtype

    hooks = {name: _LayerHook(module) for name, module in stage_specs}

    accumulators: dict[str, np.ndarray | None] = {name: None for name in hooks}
    counts = {name: 0 for name in hooks}

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="  LayerActivations", leave=False):
            model.encode(images.to(device=device, dtype=model_dtype))   # forward — all hooks fire

            for name, hook in hooks.items():
                if hook.out is None:
                    continue
                # shape: (B, C, H, W) — average over B and C → (H, W)
                spatial = hook.out.mean(dim=(0, 1)).cpu().float().numpy()
                if accumulators[name] is None:
                    accumulators[name] = spatial.copy()
                else:
                    accumulators[name] += spatial
                counts[name] += 1

            if sample_limit > 0 and sum(counts.values()) // len(counts) * loader.batch_size >= sample_limit:
                break

    for hook in hooks.values():
        hook.remove()

    # ── Plot: 3×3 grid ────────────────────────────────────────────────────────
    n = len(stage_specs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(ncols * 5, nrows * 4 + 1), dpi=130, constrained_layout=True)
    fig.suptitle(
        f"All Backbone Layer Activations — {epoch_label}\n"
        f"(Mean over channels and {'all test images' if sample_limit <= 0 else f'{sample_limit:,} test images'})",
        fontsize=14, y=0.99,
    )
    gs = gridspec.GridSpec(nrows, ncols, hspace=0.45, wspace=0.25)

    for plot_i, (name, acc) in enumerate(accumulators.items()):
        ax = fig.add_subplot(gs[plot_i // ncols, plot_i % ncols])
        if acc is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(name, fontsize=8)
            continue
        amap = acc / max(counts[name], 1)
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        im = ax.imshow(amap, cmap="inferno", interpolation="nearest")
        ax.set_title(name, fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ActMaps] Saved → {output_path}")
# ─────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    checkpoint_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    epoch: int | str,
    batch_size: int = 128,
    num_workers: int = 4,
    sample_limit_activations: int = 0,
    umap_max_samples: int = 0,
    umap_thumbnail_limit: int = 1024,
    umap_thumb_size: int = 48,
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── 1. UMAP thumbnail map ────────────────────────────────────────────────
    generate_umap_thumbnail_map(
        model, loader, class_names, device,
        output_dir, epoch_label,
        max_samples=umap_max_samples,
        thumbnail_limit=umap_thumbnail_limit,
        thumb_size=umap_thumb_size,
    )

    # ── 2. All-layer activations ─────────────────────────────────────────────
    generate_layer_activations(
        model, loader, device,
        output_dir / "all_layer_activations.png", epoch_label,
        sample_limit=sample_limit_activations,
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
        "--sample-limit-activations", type=int, default=2048,
        help="Max images used to build the activation average. Use 0 for all test images.",
    )
    parser.add_argument(
        "--umap-thumb-size", type=int, default=48,
        help="Thumbnail size for the UMAP thumbnail map (default 48).",
    )
    parser.add_argument(
        "--umap-max-samples", type=int, default=0,
        help="Optional cap for UMAP samples. Use 0 for the full test set.",
    )
    parser.add_argument(
        "--umap-thumbnail-limit", type=int, default=1024,
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
        sample_limit_activations=args.sample_limit_activations,
        umap_max_samples=args.umap_max_samples,
        umap_thumbnail_limit=args.umap_thumbnail_limit,
        umap_thumb_size=args.umap_thumb_size,
    )


if __name__ == "__main__":
    main()
