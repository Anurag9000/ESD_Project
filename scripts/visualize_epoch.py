#!/usr/bin/env python3
"""
visualize_epoch.py

Per-epoch visualization engine for the ESD waste classification pipeline.
Generates canonical artefacts from the COMPLETE test set (no augmentations):

  1. tsne_all_classes.png      — one global t-SNE of every test embedding
  2. per_class_tsne/*.png      — one t-SNE highlight plot per class
  3. all_layer_activations.png — mean spatial activation for ALL 9 EfficientNet-B0
                                  backbone feature stages (per-layer = mean over channels)
                                  Why per-layer, not per-neuron?
                                  A single EfficientNet stage has up to 192 channels.
                                  Per-neuron = thousands of tiny maps = uninterpretable noise.
                                  Per-layer = one summary map per stage = usable signal.
  4. test_full_atlas.png       — full test-set atlas mosaic (every test image)

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
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Import from the same scripts package ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from metric_learning_pipeline import (
    IMAGENET_MEAN,
    IMAGENET_STD,
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
# 1. Per-class t-SNE
# ─────────────────────────────────────────────────────────────────────────────

def generate_tsne_suite(
    model: MetricLearningEfficientNetB0,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    epoch_label: str,
    max_samples: int = 0,
) -> None:
    print("  [t-SNE] Extracting embeddings …")
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  t-SNE embeddings", leave=False):
            emb = model.encode(images.to(device))  # (B, 128)
            all_emb.append(emb.cpu().float().numpy())
            all_lbl.extend(labels.tolist())
            if max_samples > 0 and sum(chunk.shape[0] for chunk in all_emb) >= max_samples:
                break

    emb_np = np.concatenate(all_emb, axis=0).astype(np.float32, copy=False)
    lbl_np = np.array(all_lbl[: len(emb_np)])
    if max_samples > 0 and len(emb_np) > max_samples:
        emb_np = emb_np[:max_samples]
        lbl_np = lbl_np[:max_samples]

    pca_dim = min(50, emb_np.shape[1], max(2, emb_np.shape[0] - 1))
    if pca_dim < emb_np.shape[1]:
        emb_np = PCA(n_components=pca_dim, random_state=42).fit_transform(emb_np)

    print(f"  [t-SNE] Running on {len(emb_np)} samples …")
    coords = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        angle=0.5,
        perplexity=min(40, len(emb_np) - 1),
    ).fit_transform(emb_np)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_class_dir = output_dir / "per_class_tsne"
    per_class_dir.mkdir(parents=True, exist_ok=True)

    palette = matplotlib.colormaps.get_cmap("tab10").resampled(len(class_names))
    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    for cls_id, cls_name in enumerate(class_names):
        mask = lbl_np == cls_id
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[palette(cls_id)],
            label=cls_name,
            s=6,
            alpha=0.5,
            linewidths=0,
        )
    ax.legend(markerscale=4, loc="upper right", framealpha=0.8)
    ax.set_title(f"Per-Class Embedding Geometry — {epoch_label}\n"
                 f"({len(emb_np):,} test images, no augmentation)", fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    output_path = output_dir / "tsne_all_classes.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [t-SNE] Saved → {output_path}")

    for cls_id, cls_name in enumerate(class_names):
        mask = lbl_np == cls_id
        fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
        ax.scatter(coords[:, 0], coords[:, 1], c="#999999", s=5, alpha=0.12, linewidths=0)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[palette(cls_id)],
            s=7,
            alpha=0.8,
            linewidths=0,
        )
        ax.set_title(
            f"t-SNE Highlight — {cls_name} — {epoch_label}\n"
            f"({int(mask.sum()):,} samples highlighted)",
            fontsize=13,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        fig.tight_layout()
        per_class_path = per_class_dir / f"{cls_name}.png"
        fig.savefig(per_class_path, bbox_inches="tight")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. All-layer activations
# ─────────────────────────────────────────────────────────────────────────────

_EFFICIENTNET_B0_STAGES = {
    "Stage 0 · Stem (Conv2d)":        0,
    "Stage 1 · MBConv1 ×1":           1,
    "Stage 2 · MBConv6 ×2":           2,
    "Stage 3 · MBConv6 ×2":           3,
    "Stage 4 · MBConv6 ×3":           4,
    "Stage 5 · MBConv6 ×3":           5,
    "Stage 6 · MBConv6 ×4":           6,
    "Stage 7 · MBConv6 ×1":           7,
    "Stage 8 · Head Conv (1280ch)":   8,
}

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
    ALL 9 EfficientNet-B0 backbone stages.  We average over:
      - batch dimension  (all images)
      - channel dimension (all learned filters)
    resulting in one (H×W) heatmap per stage — a true "What does this stage attend to?"
    """
    print(f"  [ActMaps] Hooking all {len(_EFFICIENTNET_B0_STAGES)} backbone stages …")
    model.eval()

    hooks = {
        name: _LayerHook(model.backbone.features[idx])
        for name, idx in _EFFICIENTNET_B0_STAGES.items()
    }

    accumulators: dict[str, np.ndarray | None] = {name: None for name in hooks}
    counts = {name: 0 for name in hooks}

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="  LayerActivations", leave=False):
            model.encode(images.to(device))   # forward — all hooks fire

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
    n = len(_EFFICIENTNET_B0_STAGES)
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


def generate_per_class_sample_activations(
    model: MetricLearningEfficientNetB0,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    epoch_label: str,
    samples_per_class: int = 10,
) -> None:
    """
    Pick 10 samples from each class and generate a 4x3 grid for EACH one:
    - [0,0]: Original Image
    - [0,1...]: Stages of the backbone
    """
    print(f"  [SampleActs] Picking {samples_per_class} samples per class for audit …")
    model.eval()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Selection buckets
    n_cls = len(class_names)
    buckets: dict[int, list[tuple[torch.Tensor, int]]] = {i: [] for i in range(n_cls)}
    
    with torch.no_grad():
        for images, targets in loader:
            for i in range(len(images)):
                t = int(targets[i])
                if len(buckets[t]) < samples_per_class:
                    buckets[t].append((images[i], t))
            if all(len(v) >= samples_per_class for v in buckets.values()):
                break
                
    # Flatten selected samples
    selected_samples = []
    for cls_id in range(n_cls):
        selected_samples.extend(buckets[cls_id])
        
    # Setup hooks
    hooks = {
        name: _LayerHook(model.backbone.features[idx])
        for name, idx in _EFFICIENTNET_B0_STAGES.items()
    }
    stage_names = list(_EFFICIENTNET_B0_STAGES.keys())
    
    print(f"  [SampleActs] Generating {len(selected_samples)} individual maps …")
    for idx, (img_tensor, target_id) in enumerate(tqdm(selected_samples, desc="  Sample activations", leave=False)):
        model.eval()
        with torch.no_grad():
            img_batch = img_tensor.unsqueeze(0).to(device)
            emb = model.encode(img_batch)
            logits = model.classify(emb)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            pred = int(pred[0])
            conf = float(conf[0])
            
            # Capture activations
            activations = {name: hook.out[0].mean(dim=0).cpu().float().numpy() for name, hook in hooks.items()}
            
        # Draw 4x3 Grid
        fig = plt.figure(figsize=(18, 22), dpi=100)
        gs = gridspec.GridSpec(4, 3, hspace=0.3, wspace=0.2)
        
        # Original Image
        ax0 = fig.add_subplot(gs[0, 0])
        img_np = _denorm(img_tensor.permute(1, 2, 0).numpy())
        ax0.imshow(img_np)
        ax0.set_title(f"Original: {class_names[target_id]}", fontsize=12, fontweight="bold")
        ax0.axis("off")
        
        # 9 Stages
        for s_idx, s_name in enumerate(stage_names):
            row = (s_idx + 1) // 3
            col = (s_idx + 1) % 3
            ax = fig.add_subplot(gs[row, col])
            amap = activations[s_name]
            amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
            im = ax.imshow(amap, cmap="magma")
            ax.set_title(f"Stage {s_idx}\n{s_name.split('·')[-1].strip()}", fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        # Stats summary
        ax_stat = fig.add_subplot(gs[3, 1:])
        ax_stat.axis("off")
        color = "green" if pred == target_id else "red"
        status = "CORRECT" if pred == target_id else "MISCLASSIFIED"
        stats_text = (
            f"Ground Truth: {class_names[target_id]}\n"
            f"Prediction: {class_names[pred]} ({status})\n"
            f"Confidence: {conf:.2%}\n"
            f"Epoch: {epoch_label}"
        )
        ax_stat.text(0.1, 0.5, stats_text, fontsize=16, color=color, va="center", fontweight="bold")
        
        plt.savefig(output_dir / f"{class_names[target_id]}_sample_{idx%samples_per_class}.png", bbox_inches="tight")
        plt.close(fig)
        
    for hook in hooks.values():
        hook.remove()
    print(f"  [SampleActs] Saved {len(selected_samples)} samples to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Full test-set classification Atlas
# ─────────────────────────────────────────────────────────────────────────────

def generate_test_atlas(
    model: MetricLearningEfficientNetB0,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    output_path: Path,
    epoch_label: str,
    thumb_size: int = 18,
    max_columns: int = 192,
    max_images: int = 0,
) -> None:
    """
    Render the COMPLETE test set into one atlas image.
    Images are sorted by predicted class, then confidence, then actual class.
    Border colour: green = correct, red = misclassified.
    """
    print("  [Atlas] Collecting the full test set …")
    model.eval()

    entries: list[tuple[int, int, float, np.ndarray]] = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="  Atlas collect", leave=False):
            images_gpu = images.to(device)
            emb = model.encode(images_gpu)
            logits = model.classify(emb)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            for i in range(len(images)):
                pred = int(preds[i])
                img = images[i].permute(1, 2, 0).numpy()
                img = _denorm(img)
                img_pil = Image.fromarray(img).resize(
                    (thumb_size, thumb_size), Image.BILINEAR
                )
                entries.append((pred, int(targets[i]), float(confs[i]), np.array(img_pil)))
                if max_images > 0 and len(entries) >= max_images:
                    break
            if max_images > 0 and len(entries) >= max_images:
                break

    entries.sort(key=lambda item: (item[0], -item[2], item[1]))

    border = 2
    cell = thumb_size + 2 * border
    count = len(entries)
    columns = min(max_columns, max(1, math.ceil(math.sqrt(count))))
    rows = max(1, math.ceil(count / columns))
    title_h = 44
    legend_h = 28
    canvas_h = title_h + legend_h + (rows * cell)
    canvas_w = columns * cell
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 24

    for index, (pred, actual, _conf, thumb) in enumerate(entries):
        row = index // columns
        col = index % columns
        y_top = title_h + legend_h + row * cell
        x_left = col * cell
        colour = (50, 200, 80) if actual == pred else (220, 50, 50)
        canvas[y_top : y_top + cell, x_left : x_left + cell] = colour
        canvas[
            y_top + border : y_top + border + thumb_size,
            x_left + border : x_left + border + thumb_size,
        ] = thumb

    final_img = Image.fromarray(canvas)
    from PIL import ImageDraw, ImageFont
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15
        )
    except OSError:
        title_font = ImageFont.load_default()
    draw = ImageDraw.Draw(final_img)
    draw.text(
        (8, 10),
        f"Full Test Set Atlas — {epoch_label} ({count:,} images, no augmentation)",
        fill=(255, 255, 255),
        font=title_font,
    )
    draw.text(
        (8, title_h + 4 - legend_h),
        "Sorted by predicted class then confidence. Green border = correct, red border = misclassified.",
        fill=(210, 210, 210),
        font=title_font if thumb_size >= 24 else ImageFont.load_default(),
    )
    final_img.save(output_path)
    print(f"  [Atlas] Saved → {output_path}")


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
    atlas_thumb_size: int = 48,
    tsne_max_samples: int = 0,
    atlas_max_images: int = 0,
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

    # Load ONLY the test split (clean, no augmentation)
    _, _, test_dataset, _, _ = build_datasets(ckpt_args)
    loader = _make_clean_loader(test_dataset, batch_size, num_workers)
    class_names = test_dataset.classes

    # Build model
    model = MetricLearningEfficientNetB0(
        num_classes=len(class_names),
        weights_mode="none",
        embedding_dim=ckpt_args.embedding_dim,
        projection_dim=ckpt_args.projection_dim,
        args=ckpt_args,
    ).to(device=device, dtype=model_dtype_for_args(ckpt_args))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── 1. Per-class t-SNE ───────────────────────────────────────────────────
    generate_tsne_suite(
        model, loader, class_names, device,
        output_dir, epoch_label,
        max_samples=tsne_max_samples,
    )

    # ── 2. All-layer activations ─────────────────────────────────────────────
    generate_layer_activations(
        model, loader, device,
        output_dir / "all_layer_activations.png", epoch_label,
        sample_limit=sample_limit_activations,
    )

    generate_per_class_sample_activations(
        model, loader, class_names, device,
        output_dir / "per_class_samples", epoch_label,
        samples_per_class=10,
    )

    # ── 3. Full atlas ────────────────────────────────────────────────────────
    generate_test_atlas(
        model, loader, class_names, device,
        output_dir / "test_full_atlas.png", epoch_label,
        thumb_size=atlas_thumb_size,
        max_images=atlas_max_images,
    )
    print(f"[VIZ] All visualizations saved to {output_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-class t-SNE, all-layer activation maps, and a "
            "full test-set classification atlas from a training checkpoint."
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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--sample-limit-activations", type=int, default=2048,
        help="Max images used to build the activation average. Use 0 for all test images.",
    )
    parser.add_argument(
        "--atlas-thumb-size", type=int, default=48,
        help="Thumbnail size for the full test atlas (default 48).",
    )
    parser.add_argument(
        "--tsne-max-samples", type=int, default=0,
        help="Optional cap for t-SNE samples. Use 0 for the full test set.",
    )
    parser.add_argument(
        "--atlas-max-images", type=int, default=0,
        help="Optional cap for full-atlas images. Use 0 for the full test set.",
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
        atlas_thumb_size=args.atlas_thumb_size,
        tsne_max_samples=args.tsne_max_samples,
        atlas_max_images=args.atlas_max_images,
    )


if __name__ == "__main__":
    main()
