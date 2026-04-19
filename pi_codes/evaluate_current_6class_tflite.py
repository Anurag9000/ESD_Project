#!/usr/bin/env python3
"""
Pi-side TFLite evaluator for the current 3-class waste model.

Current head order:
    0 -> organic
    1 -> metal
    2 -> paper

This script evaluates only the 3 logits for:
    - Organic
    - Metal
    - Paper
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

try:
    from ai_edge_litert.interpreter import Interpreter
except Exception:  # pragma: no cover
    # If you are using standard tflite_runtime on the Pi, swap the import above
    # for:
    # from tflite_runtime.interpreter import Interpreter
    raise


# ==============================
# 1. CONFIGURATION
# ==============================
MODEL_PATH = "model/current_best_3class_quantized.tflite"
DATASET_DIR = "REDACTED_DATA_ROOT"
OUTPUT_DIR = "pi_eval_outputs"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.80

# Full 3-class head order from the current model.
MODEL_CLASSES = ["organic", "metal", "paper"]

# Only these folders are expected in your current Pi test set.
EVAL_CLASSES = ["organic", "metal", "paper"]

# Current 3-class head indices:
#   0 organic
#   1 metal
#   2 paper
STRICT_EVAL_HEAD_INDICES = [0, 1, 2]


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def build_interpreter(model_path: str) -> tuple[Interpreter, dict, dict]:
    print(f"Loading TFLite model from {model_path} ...")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(
        f"Model ready. Input dtype={input_details[0]['dtype']}, "
        f"output dtype={output_details[0]['dtype']}"
    )
    return interpreter, input_details[0], output_details[0]


def infer_input_layout(input_shape: list[int] | tuple[int, ...]) -> str:
    # Most TFLite exports are NHWC: [1, H, W, 3].
    # Some exports can be NCHW: [1, 3, H, W].
    if len(input_shape) != 4:
        raise ValueError(f"Unexpected input shape: {input_shape}")
    if input_shape[-1] == 3:
        return "NHWC"
    if input_shape[1] == 3:
        return "NCHW"
    raise ValueError(f"Cannot infer channel layout from shape: {input_shape}")


def preprocess_image(image_path: str, input_details: dict) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Match the training/eval normalization used by the repo.
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - imagenet_mean) / imagenet_std

    layout = infer_input_layout(input_details["shape"])
    if layout == "NHWC":
        arr = arr[np.newaxis, ...]
    else:
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

    dtype = input_details["dtype"]
    if dtype in (np.uint8, np.int8):
        scale, zero_point = input_details.get("quantization", (0.0, 0))
        if scale == 0:
            raise ValueError("Quantized input model reports zero scale.")
        arr = np.round(arr / scale + zero_point)
        info = np.iinfo(dtype)
        arr = np.clip(arr, info.min, info.max).astype(dtype)
    else:
        arr = arr.astype(dtype)
    return arr


def dequantize_output(raw_output: np.ndarray, output_details: dict) -> np.ndarray:
    dtype = output_details["dtype"]
    if dtype in (np.uint8, np.int8):
        scale, zero_point = output_details.get("quantization", (0.0, 0))
        if scale == 0:
            raise ValueError("Quantized output model reports zero scale.")
        return (raw_output.astype(np.float32) - zero_point) * scale
    return raw_output.astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def predict(interpreter: Interpreter, input_details: dict, output_details: dict, image_path: str):
    tensor = preprocess_image(image_path, input_details)
    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()

    raw = interpreter.get_tensor(output_details["index"])[0]
    logits_or_probs = dequantize_output(raw, output_details)

    # Strict 3-logit evaluation: only metal / organic / paper participate.
    # Head index mapping:
    #   2 -> metal
    #   3 -> organic
    #   4 -> paper
    logits_3 = np.asarray([logits_or_probs[i] for i in STRICT_EVAL_HEAD_INDICES], dtype=np.float32)

    # If the export already returns probabilities, keep them. Otherwise softmax.
    if (
        np.all(logits_3 >= 0.0)
        and np.all(logits_3 <= 1.0 + 1e-3)
        and abs(float(logits_3.sum()) - 1.0) < 0.25
    ):
        probs = logits_3
    else:
        probs = softmax(logits_3)

    pred_idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    return pred_idx, conf, probs


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate current 3-class TFLite model on Organic/Metal/Paper")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the current 3-class TFLite model")
    parser.add_argument("--dataset", default=DATASET_DIR, help="Dataset root containing Organic/Metal/Paper folders")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Where to write CSVs and summary files")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Low-confidence cutoff")
    args = parser.parse_args()

    interpreter, input_details, output_details = build_interpreter(args.model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Display class names for the current 3-class head.
    print("\nCurrent 3-class head:")
    for i, name in enumerate(MODEL_CLASSES):
        print(f"  {i}: {name}")
    print("\nStrict eval slice:")
    for i, name in zip(STRICT_EVAL_HEAD_INDICES, EVAL_CLASSES):
        print(f"  head[{i}] -> {name}")

    # Only these classes are expected in the evaluation dataset.
    eval_class_to_idx = {name: i for i, name in enumerate(EVAL_CLASSES)}
    labels = list(EVAL_CLASSES)
    label_to_idx = {name: i for i, name in enumerate(labels)}

    rows = []
    y_true = []
    y_pred = []
    class_stats = {name: {"correct": 0, "total": 0} for name in EVAL_CLASSES}
    overall_correct = 0
    overall_total = 0
    source_counts = Counter()

    dataset_root = Path(args.dataset)
    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue
        expected = normalize_name(folder.name)
        if expected not in eval_class_to_idx:
            continue

        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                continue
            source_counts[expected] += 1
            try:
                pred_idx, conf, probs = predict(interpreter, input_details, output_details, str(img_path))
                pred_name = EVAL_CLASSES[pred_idx]

                # Strict 3-way evaluation: the model decision is always taken
                # from the three selected logits only.
                true_name = expected
                true_idx = label_to_idx[true_name]
                pred_eval_idx = label_to_idx[pred_name]

                y_true.append(true_idx)
                y_pred.append(pred_eval_idx)
                overall_total += 1
                class_stats[true_name]["total"] += 1
                if pred_name == true_name:
                    overall_correct += 1
                    class_stats[true_name]["correct"] += 1

                rows.append(
                    {
                        "image_path": str(img_path),
                        "true_class": true_name,
                        "predicted_class": pred_name,
                        "predicted_eval_class": pred_name,
                        "confidence": float(conf),
                        "threshold": float(args.threshold),
                        "top1_pct": float(conf) * 100.0,
                        "raw_probabilities": json.dumps(
                            {name: float(prob) for name, prob in zip(MODEL_CLASSES, probs.tolist())}
                        ),
                    }
                )
            except Exception as exc:
                print(f"Warning: could not process {img_path}: {exc}")

    if overall_total == 0:
        print("No images found in Metal/Organic/Paper folders.")
        return 1

    cm = confusion_matrix(y_true, y_pred, labels=[label_to_idx[k] for k in labels])
    report = classification_report(
        y_true,
        y_pred,
        labels=[label_to_idx[k] for k in labels],
        target_names=labels,
        zero_division=0,
        digits=4,
    )

    # Save outputs.
    pred_csv = out_dir / "predictions.csv"
    with pred_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_path": args.model,
        "dataset": str(dataset_root),
        "model_classes": MODEL_CLASSES,
        "evaluation_classes": labels,
        "source_counts": dict(source_counts),
        "overall_total": overall_total,
        "overall_correct": overall_correct,
        "overall_accuracy": overall_correct / overall_total,
        "per_class_accuracy": {
            cls: (class_stats[cls]["correct"] / class_stats[cls]["total"] if class_stats[cls]["total"] else None)
            for cls in EVAL_CLASSES
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "classification_report.txt").write_text(report)

    print("\n" + "=" * 40)
    print("CURRENT MODEL EVALUATION")
    print("=" * 40)
    print(f"Model path: {args.model}")
    print(f"Dataset: {dataset_root}")
    print(f"Target classes: {EVAL_CLASSES}")
    print(f"Overall accuracy: {summary['overall_accuracy'] * 100:.2f}% ({overall_correct}/{overall_total})")
    print("\nPer-class accuracy:")
    for cls in EVAL_CLASSES:
        acc = summary["per_class_accuracy"][cls]
        print(f"  {cls}: {acc * 100:.2f}%" if acc is not None else f"  {cls}: n/a")
    print("\nConfusion matrix labels:")
    print(labels)
    print(cm)
    print("\nClassification report:")
    print(report)
    print(f"Saved predictions to: {pred_csv}")
    print(f"Saved summary to: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
