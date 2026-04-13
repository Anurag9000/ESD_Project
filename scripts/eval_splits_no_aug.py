#!/usr/bin/env python3
"""
eval_splits_no_aug.py — Evaluate the model on train / val / test splits
with NO augmentation (centre-crop + normalize only).

Uses the EXACT same deterministic split assignment as the training pipeline:
    seed=42, per-class stratified 70/20/10 shuffle via random.Random,
    with the plastic → [soft_plastic, hard_plastic, plastic] class mapping.

Outputs per split (written to --output-dir):
    confmat_counts_{split}.csv          raw confusion matrix
    confmat_rate_pct_{split}.csv        confusion matrix as % of true class
    confusion_matrix_{split}.png        colour-coded confusion matrix heatmap
    summary_{split}.json               overall + per-class accuracy

Usage (from repo root, venv activated):
    python scripts/eval_splits_no_aug.py \\
        --checkpoint Results/efficientnet_b0_master_run/loss_cleanup/best.pt \\
        --dataset-root Dataset_Final \\
        --batch-size 224
"""
from __future__ import annotations

import argparse
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency",
    category=UserWarning,
)

try:
    from metric_learning_pipeline import (
        MetricLearningEfficientNetB0,
        evaluation_tensor_from_image,
        model_dtype_for_args,
        allocate_split_counts,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from metric_learning_pipeline import (
        MetricLearningEfficientNetB0,
        evaluation_tensor_from_image,
        model_dtype_for_args,
        allocate_split_counts,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class NoAugDataset(Dataset):
    """Single-pass no-augmentation dataset for a pre-built sample list."""

    def __init__(self, samples: list[tuple[str, int]], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img = Image.open(path).convert("RGB")
            tensor = evaluation_tensor_from_image(img, self.image_size)
        except Exception:
            tensor = torch.zeros(3, self.image_size, self.image_size)
        return tensor, label


def _collate(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors), torch.tensor(labels, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Split logic — mirrors build_auto_split_datasets() in metric_learning_pipeline
# ─────────────────────────────────────────────────────────────────────────────

_CLASS_MAPPING: dict[str, list[str]] = {
    "plastic": ["soft_plastic", "hard_plastic", "plastic"]
}


def build_splits(
    dataset_root: Path,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
) -> tuple[
    list[tuple[str, int]],
    list[tuple[str, int]],
    list[tuple[str, int]],
    list[str],
]:
    """
    Mirrors build_auto_split_datasets() exactly — scans disk, applies class
    mapping, shuffles per-class with the same RNG seed, then splits 70/20/10.

    Returns (train_samples, val_samples, test_samples, class_names).
    """
    base = datasets.ImageFolder(str(dataset_root))

    # Build reverse map: source_class → target_class
    reverse_map: dict[str, str] = {}
    for target, sources in _CLASS_MAPPING.items():
        for s in sources:
            reverse_map[s] = target

    # Derive new class list after merging
    new_classes_set = set(base.classes)
    for target, sources in _CLASS_MAPPING.items():
        for s in sources:
            new_classes_set.discard(s)
        new_classes_set.add(target)
    new_classes = sorted(new_classes_set)
    new_class_to_idx = {cls: i for i, cls in enumerate(new_classes)}

    # Re-map all samples
    remapped: list[tuple[str, int]] = []
    for path, old_target in base.samples:
        old_class = base.classes[old_target]
        new_class = reverse_map.get(old_class, old_class)
        remapped.append((path, new_class_to_idx[new_class]))

    # Group by class index
    by_class: dict[int, list[tuple[str, int]]] = {i: [] for i in range(len(new_classes))}
    for path, target in remapped:
        by_class[target].append((path, target))

    # Shuffle + split (identical RNG state to training)
    rng = random.Random(seed)
    train_s: list[tuple[str, int]] = []
    val_s:   list[tuple[str, int]] = []
    test_s:  list[tuple[str, int]] = []

    for class_idx in range(len(new_classes)):
        shuffled = list(by_class[class_idx])
        rng.shuffle(shuffled)
        n_train, n_val, n_test = allocate_split_counts(len(shuffled), ratios)
        train_s.extend(shuffled[:n_train])
        val_s.extend(shuffled[n_train : n_train + n_val])
        test_s.extend(shuffled[n_train + n_val : n_train + n_val + n_test])

    return train_s, val_s, test_s, new_classes


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(
    path: Path,
    confmat: np.ndarray,
    class_names: list[str],
    percent: bool = False,
) -> None:
    header = ["true_class\\pred_class"] + class_names
    rows = []
    for i, cls in enumerate(class_names):
        rt = int(confmat[i].sum())
        if percent:
            row = [cls] + (
                [f"{confmat[i, j] / rt * 100:.3f}" for j in range(len(class_names))]
                if rt > 0
                else ["0.000"] * len(class_names)
            )
        else:
            row = [cls] + [str(int(confmat[i, j])) for j in range(len(class_names))]
        rows.append(row)
    path.write_text(
        "\n".join(",".join(r) for r in [header] + rows),
        encoding="utf-8",
    )


def _plot_confmat(
    confmat: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    n = len(class_names)
    row_sums = confmat.sum(axis=1, keepdims=True)
    norm = np.where(row_sums > 0, confmat / row_sums, 0.0) * 100.0

    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), max(8, n * 0.8)))
    im = ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("% of True Class", fontsize=9)

    thresh = 50.0
    for i in range(n):
        for j in range(n):
            val = norm[i, j]
            ax.text(
                j, i, f"{val:.1f}",
                ha="center", va="center", fontsize=7,
                color="white" if val > thresh else "black",
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-split evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_split(
    split_name: str,
    samples: list[tuple[str, int]],
    model: torch.nn.Module,
    device: torch.device,
    class_names: list[str],
    image_size: int,
    batch_size: int,
    use_autocast: bool,
    output_dir: Path,
) -> dict[str, Any]:
    n = len(class_names)
    ds = NoAugDataset(samples, image_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=_collate,
        pin_memory=(device.type == "cuda"),
    )

    confmat = np.zeros((n, n), dtype=np.int64)
    total_batches = len(loader)

    print(f"\n  ── {split_name.upper()} ─────────────────────────────────────────")
    print(f"     {len(samples):,} images · {total_batches} batches")

    with torch.no_grad():
        for b_idx, (images, labels) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_autocast):
                emb = model.encode(images)
                logits = model.classify(emb, labels=None)
            preds = logits.argmax(dim=1).cpu().numpy()
            for tl, pl in zip(labels.numpy(), preds):
                confmat[tl][pl] += 1
            if b_idx % 100 == 0 or b_idx == total_batches:
                pct = b_idx / total_batches * 100
                print(
                    f"     [{b_idx:>5}/{total_batches}]  ({pct:5.1f}%)", end="\r"
                )

    correct = int(np.trace(confmat))
    total   = int(confmat.sum())
    raw_acc = correct / max(1, total) * 100
    print(
        f"     Done ─ Raw Accuracy: \033[1m{raw_acc:.4f}%\033[0m"
        f"  ({correct:,} / {total:,})                    "
    )

    # --- Save outputs ---
    _write_csv(output_dir / f"confmat_counts_{split_name}.csv",   confmat, class_names)
    _write_csv(output_dir / f"confmat_rate_pct_{split_name}.csv", confmat, class_names, percent=True)
    _plot_confmat(
        confmat, class_names,
        title=(
            f"{split_name.upper()} Split — No-Aug Confusion Matrix\n"
            f"Raw Accuracy: {raw_acc:.4f}%  ({correct:,} / {total:,})"
        ),
        output_path=output_dir / f"confusion_matrix_{split_name}.png",
    )

    per_class: dict[str, dict[str, Any]] = {}
    for i, cls in enumerate(class_names):
        rt = int(confmat[i].sum())
        per_class[cls] = {
            "total":   rt,
            "correct": int(confmat[i][i]),
            "acc_pct": round(confmat[i][i] / max(1, rt) * 100, 4),
        }

    summary: dict[str, Any] = {
        "split":          split_name,
        "num_samples":    total,
        "raw_accuracy_pct": round(raw_acc, 6),
        "correct":        correct,
        "wrong":          total - correct,
        "per_class":      per_class,
    }
    (output_dir / f"summary_{split_name}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True,
                   help="Path to model checkpoint (best.pt).")
    p.add_argument("--dataset-root", default="Dataset_Final")
    p.add_argument("--output-dir",   default="Results/eval_cleaned_splits")
    p.add_argument("--batch-size",   type=int, default=224)
    p.add_argument("--splits",       nargs="+", default=["train", "val", "test"])
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'═'*65}")
    print(f"  eval_splits_no_aug   [{ts}]")
    print(f"{'═'*65}")
    print(f"\n  Checkpoint   : {args.checkpoint}")
    print(f"  Dataset      : {dataset_root}")
    print(f"  Splits       : {args.splits}")
    print(f"  Output dir   : {output_dir}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt          = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args     = ckpt.get("args", {})
    class_names   = list(ckpt["class_names"])
    num_classes   = len(class_names)
    image_size    = int(ckpt_args.get("image_size",    224))
    embedding_dim = int(ckpt_args.get("embedding_dim", 128))
    projection_dim= int(ckpt_args.get("projection_dim",128))
    weights_mode  = str(ckpt_args.get("weights",   "default"))
    precision     = str(ckpt_args.get("precision", "mixed"))

    model_args   = argparse.Namespace(precision=precision)
    model_dtype  = model_dtype_for_args(model_args)
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = device.type == "cuda"

    print(f"  Device       : {device}")
    print(f"  Classes ({num_classes:>3}) : {class_names}\n")

    model = MetricLearningEfficientNetB0(
        num_classes=num_classes,
        weights_mode=weights_mode,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        args=model_args,
    ).to(device=device, dtype=model_dtype)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Build splits (mirrors pipeline logic exactly) ─────────────────────────
    print(f"  Building splits from disk (seed={args.seed}) …")
    train_s, val_s, test_s, disk_classes = build_splits(
        dataset_root, seed=args.seed
    )
    print(f"    train={len(train_s):,}  val={len(val_s):,}  test={len(test_s):,}")

    split_map = {"train": train_s, "val": val_s, "test": test_s}

    # ── Evaluate each split ───────────────────────────────────────────────────
    all_summaries: list[dict[str, Any]] = []
    for split_name in args.splits:
        samples = split_map.get(split_name, [])
        if not samples:
            print(f"\n  WARNING: no samples for '{split_name}' — skipping.")
            continue
        summary = evaluate_split(
            split_name=split_name,
            samples=samples,
            model=model,
            device=device,
            class_names=class_names,
            image_size=image_size,
            batch_size=args.batch_size,
            use_autocast=use_autocast,
            output_dir=output_dir,
        )
        all_summaries.append(summary)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  FINAL SUMMARY")
    print(f"{'─'*65}")
    print(f"  {'Split':<8} {'Samples':>10} {'Correct':>9} {'Wrong':>7} {'Raw Acc':>10}")
    print(f"  {'─'*8}   {'─'*10}   {'─'*9}   {'─'*7}   {'─'*10}")
    for s in all_summaries:
        print(
            f"  {s['split']:<8} {s['num_samples']:>10,}"
            f" {s['correct']:>9,} {s['wrong']:>7,}"
            f" {s['raw_accuracy_pct']:>9.4f}%"
        )
    print(f"{'═'*65}")
    print(f"\n  Outputs → {output_dir}/\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
