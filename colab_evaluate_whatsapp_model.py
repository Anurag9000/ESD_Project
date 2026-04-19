#!/usr/bin/env python3
"""Evaluate this repo's best.pt / best.pt.zip checkpoint in Google Colab.

Expected Colab setup:
  1. Upload the WhatsApp file to /content/best.pt.zip.
  2. Run:
       !python /content/colab_evaluate_whatsapp_model.py

The script downloads the metal/organic/paper tar datasets, uses the last 20%
of each class as the test split, reconstructs the repo model from the
checkpoint metadata, and writes metrics/predictions to /content/colab_eval_outputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace


DATASETS = {
    "organic": "1YSAVPhoIYBw29dmkLc7nY13IefCOdm-4",
    "metal": "1d5d8xc0hHmIXxWl44CH_ihuDhSJMMR21",
    "paper": "1o_mX_AcbZIz3E_JixrlTP4x0wZ5ITzcC",
}

REPO_URL = "https://github.com/Anurag9000/ESD_Project.git"
REPO_DIR = Path("/content/ESD_Project")
DEFAULT_MODEL_PATH = Path("/content/best.pt.zip")
DEFAULT_DATASET_DIR = Path("/content/dataset")
DEFAULT_OUTPUT_DIR = Path("/content/colab_eval_outputs")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def install_if_missing(module_name: str, pip_name: str | None = None) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    package = pip_name or module_name
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def ensure_dependencies() -> None:
    install_if_missing("gdown")
    install_if_missing("timm")
    install_if_missing("sklearn", "scikit-learn")
    install_if_missing("seaborn")
    install_if_missing("matplotlib")


def ensure_repo(repo_dir: Path) -> None:
    metric_file = repo_dir / "scripts" / "metric_learning_pipeline.py"
    if metric_file.exists():
        return
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    subprocess.check_call(["git", "clone", "-q", REPO_URL, str(repo_dir)])


def safe_extract_tar(tar: tarfile.TarFile, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    for member in tar.getmembers():
        member_path = (target_dir / member.name).resolve()
        if not str(member_path).startswith(str(target_dir) + os.sep):
            raise RuntimeError(f"Unsafe tar member path blocked: {member.name}")
    tar.extractall(path=target_dir)


def download_and_extract_datasets(base_dir: Path) -> None:
    import gdown

    base_dir.mkdir(parents=True, exist_ok=True)
    for class_name, file_id in DATASETS.items():
        tar_path = Path(f"/content/{class_name}.tar")
        extract_path = base_dir / class_name
        if extract_path.exists() and any(extract_path.rglob("*")):
            print(f"{class_name}: already extracted at {extract_path}")
            continue

        print(f"Downloading {class_name} dataset...")
        gdown.download(id=file_id, output=str(tar_path), quiet=False)
        if not tar_path.exists() or tar_path.stat().st_size == 0:
            raise FileNotFoundError(f"Download failed or empty file: {tar_path}")

        print(f"Extracting {class_name} dataset...")
        extract_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            safe_extract_tar(tar, extract_path)
        tar_path.unlink(missing_ok=True)


def clean_images(base_dir: Path) -> int:
    from PIL import Image

    illegal_formats = {"WEBP", "MPO", "GIF"}
    deleted = 0
    for path in list(base_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as img:
                if img.format in illegal_formats:
                    path.unlink(missing_ok=True)
                    deleted += 1
                else:
                    img.verify()
        except Exception:
            path.unlink(missing_ok=True)
            deleted += 1
    return deleted


def resolve_checkpoint_path(model_path: Path, output_dir: Path) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Upload best.pt.zip to /content first, "
            "or pass --model-path /path/to/file."
        )

    if model_path.suffix.lower() == ".zip":
        extract_dir = output_dir / "extracted_model"
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(model_path, "r") as zf:
                zf.extractall(extract_dir)
            pt_files = sorted(extract_dir.rglob("*.pt"))
            if pt_files:
                print(f"Extracted checkpoint from zip: {pt_files[0]}")
                return pt_files[0]
        except zipfile.BadZipFile:
            # torch.save checkpoints are often zip containers internally. If WhatsApp
            # only renamed best.pt to best.pt.zip, torch.load can still read it.
            print("File is not a normal zip archive; trying torch.load directly.")

    return model_path


def torch_load_checkpoint(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model(checkpoint: dict, repo_dir: Path, device):
    sys.path.insert(0, str(repo_dir / "scripts"))
    from metric_learning_pipeline import MetricLearningEfficientNetB0, model_dtype_for_args

    checkpoint_args = dict(checkpoint.get("args") or {})
    checkpoint_args.setdefault("backbone", "convnextv2_nano")
    checkpoint_args.setdefault("weights", "none")
    checkpoint_args.setdefault("embedding_dim", 128)
    checkpoint_args.setdefault("projection_dim", 128)
    checkpoint_args.setdefault("image_size", 224)
    checkpoint_args.setdefault("precision", "mixed")

    model_args = SimpleNamespace(**checkpoint_args)
    class_names = list(checkpoint["class_names"])

    # Use weights_mode="none" deliberately: the checkpoint immediately overwrites
    # every parameter, and this avoids pretrained-weight downloads/rate limits.
    model = MetricLearningEfficientNetB0(
        num_classes=len(class_names),
        weights_mode="none",
        embedding_dim=int(model_args.embedding_dim),
        projection_dim=int(model_args.projection_dim),
        args=model_args,
        backbone_name=str(model_args.backbone),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device=device, dtype=model_dtype_for_args(model_args))
    model.eval()
    return model, model_args, class_names


class WasteTestDataset:
    def __init__(self, base_dir: Path, image_size: int, repo_dir: Path) -> None:
        self.base_dir = base_dir
        self.image_size = image_size
        self.class_names = ["metal", "organic", "paper"]
        self.class_to_idx = {"metal": 0, "organic": 1, "paper": 2}
        self.samples: list[tuple[Path, int]] = []
        sys.path.insert(0, str(repo_dir / "scripts"))
        from metric_learning_pipeline import evaluation_tensor_from_image

        self.evaluation_tensor_from_image = evaluation_tensor_from_image

        for class_name in self.class_names:
            class_dir = base_dir / class_name
            files = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            split_index = int(len(files) * 0.8)
            test_files = files[split_index:]
            for path in test_files:
                self.samples.append((path, self.class_to_idx[class_name]))

        print(f"Total test images: {len(self.samples)}")
        for class_name in self.class_names:
            label = self.class_to_idx[class_name]
            count = sum(1 for _, y in self.samples if y == label)
            print(f"  {class_name}: {count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
            tensor = self.evaluation_tensor_from_image(image, self.image_size)
        return tensor, label, str(path)


def collate_batch(batch):
    import torch

    images, labels, paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(paths)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    sys.path.insert(0, str(args.repo_dir / "scripts"))
    from metric_learning_pipeline import model_dtype_for_args

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_checkpoint_path(args.model_path, args.output_dir)
    checkpoint = torch_load_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected checkpoint dict with key 'model_state_dict'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, model_args, checkpoint_class_names = build_model(checkpoint, args.repo_dir, device)

    print("Checkpoint class order:")
    for index, name in enumerate(checkpoint_class_names):
        print(f"  {index}: {name}")

    dataset = WasteTestDataset(args.dataset_dir, int(model_args.image_size), args.repo_dir)
    if len(dataset) == 0:
        raise RuntimeError("No test images found. Check dataset extraction paths.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    runtime_class_names = ["Metal", "Organic", "Paper"]
    y_true: list[int] = []
    y_pred: list[int] = []
    rows: list[dict] = []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True, dtype=model_dtype_for_args(model_args))
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                embeddings = model.encode(images)
                logits8 = model.classify(embeddings, labels=None)
            probs8 = torch.softmax(logits8.float(), dim=1)

            class_to_idx = {name: index for index, name in enumerate(checkpoint_class_names)}
            required = ["metal", "organic", "paper"]
            missing = [name for name in required if name not in class_to_idx]
            if missing:
                raise ValueError(f"Checkpoint is missing required classes for this 3-class eval: {missing}")
            if not args.force_three_class_only and "ewaste" in class_to_idx:
                metal_prob = probs8[:, class_to_idx["metal"]] + probs8[:, class_to_idx["ewaste"]]
                organic_prob = probs8[:, class_to_idx["organic"]]
                paper_prob = probs8[:, class_to_idx["paper"]]
                probs3 = torch.stack([metal_prob, organic_prob, paper_prob], dim=1)
                mode = "legacy checkpoint: metal=ewaste+metal merged"
            else:
                probs3 = probs8[:, [class_to_idx["metal"], class_to_idx["organic"], class_to_idx["paper"]]]
                mode = "metal/organic/paper logits only"

            conf3, preds3 = probs3.max(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds3.cpu().numpy().tolist())

            for path, true_idx, pred_idx, confidence, probs in zip(
                paths,
                labels.cpu().numpy().tolist(),
                preds3.cpu().numpy().tolist(),
                conf3.cpu().numpy().tolist(),
                probs3.cpu().numpy().tolist(),
            ):
                rows.append(
                    {
                        "path": path,
                        "file_name": Path(path).name,
                        "true_class": runtime_class_names[int(true_idx)],
                        "predicted_class": runtime_class_names[int(pred_idx)],
                        "top1_confidence_pct": float(confidence) * 100.0,
                        "true_class_confidence_pct": float(probs[int(true_idx)]) * 100.0,
                        "prob_metal_pct": float(probs[0]) * 100.0,
                        "prob_organic_pct": float(probs[1]) * 100.0,
                        "prob_paper_pct": float(probs[2]) * 100.0,
                    }
                )

    accuracy = float(accuracy_score(y_true, y_pred))
    report_text = classification_report(y_true, y_pred, target_names=runtime_class_names, digits=4)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=runtime_class_names,
        digits=4,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
    wrong_rows = [row for row in rows if row["true_class"] != row["predicted_class"]]

    print(f"\nEvaluation mode: {mode}")
    print(f"Final test accuracy: {accuracy * 100.0:.2f}%")
    print("\nClassification report:")
    print(report_text)
    print(f"Wrong predictions: {len(wrong_rows)} / {len(rows)}")
    for row in wrong_rows:
        print(
            f"{row['file_name']} | true={row['true_class']} | "
            f"pred={row['predicted_class']} | conf={row['top1_confidence_pct']:.2f}% | "
            f"true_conf={row['true_class_confidence_pct']:.2f}%"
        )

    write_csv(args.output_dir / "predictions.csv", rows)
    write_csv(args.output_dir / "wrong_predictions.csv", wrong_rows)
    np_cm = cm.tolist()
    np_cm_pct = cm_pct.tolist()
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_dir": str(args.dataset_dir),
        "evaluation_mode": mode,
        "num_images": len(rows),
        "accuracy": accuracy,
        "accuracy_pct": accuracy * 100.0,
        "wrong_predictions": len(wrong_rows),
        "class_names": runtime_class_names,
        "confusion_matrix_counts": np_cm,
        "confusion_matrix_row_pct": np_cm_pct,
        "classification_report": report_dict,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=runtime_class_names, yticklabels=runtime_class_names)
    plt.title("Confusion Matrix Counts")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix_counts.png", dpi=180)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues", xticklabels=runtime_class_names, yticklabels=runtime_class_names)
    plt.title("Confusion Matrix Row %")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix_percent.png", dpi=180)
    plt.show()

    print(f"\nSaved outputs to: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab evaluator for ESD best.pt.zip checkpoints.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--repo-dir", type=Path, default=REPO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--force-three-class-only",
        action="store_true",
        help="Use only original logits [metal, organic, paper]. For legacy 8-class checkpoints, default also merges ewaste into Metal.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dependencies()
    ensure_repo(args.repo_dir)
    download_and_extract_datasets(args.dataset_dir)
    deleted = clean_images(args.dataset_dir)
    print(f"Cleaned dataset. Deleted {deleted} corrupt/unsupported files.")
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
