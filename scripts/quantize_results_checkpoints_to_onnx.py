#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from PIL import Image

try:
    from metric_learning_pipeline import TRAINING_CLASS_ORDER, evaluation_tensor_from_image
except ModuleNotFoundError:
    from scripts.metric_learning_pipeline import TRAINING_CLASS_ORDER, evaluation_tensor_from_image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize grouped Results ONNX checkpoints to INT8.")
    parser.add_argument(
        "--results-dir",
        default="Results/convnextv2_nano.fcmae_ft_in22k_in1k",
        help="Directory containing the grouped onnx_models/ tree and output subfolders.",
    )
    parser.add_argument("--calibration-root", default="Dataset_Final", help="Dataset root used to calibrate INT8.")
    parser.add_argument("--verification-root", default="Test_Dataset_Real", help="Dataset root used to verify INT8.")
    parser.add_argument("--calibration-samples", type=int, default=256, help="Number of calibration images to use.")
    parser.add_argument("--verification-samples", type=int, default=0, help="Optional cap on verification images. 0 = all usable.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing INT8 exports.")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Skip existing INT8 exports.")
    parser.add_argument("--opset", type=int, default=17, help="Unused compatibility flag; kept for reporting.")
    return parser.parse_args()


def collect_image_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            paths.append(path)
    return paths


def project_label_from_path(path: Path) -> str | None:
    parent = path.parent.name.strip().lower()
    if parent not in TRAINING_CLASS_ORDER:
        return None
    return parent


def sample_paths(paths: list[Path], limit: int, seed: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    rng = random.Random(seed)
    chosen = list(paths)
    rng.shuffle(chosen)
    return chosen[:limit]


class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], image_size: int) -> None:
        self.image_paths = image_paths
        self.image_size = image_size
        self.index = 0
        self.input_name = "image"

    def get_next(self) -> dict[str, np.ndarray] | None:
        while self.index < len(self.image_paths):
            image_path = self.image_paths[self.index]
            self.index += 1
            try:
                image = Image.open(image_path).convert("RGB")
                tensor = evaluation_tensor_from_image(image, self.image_size).unsqueeze(0).numpy().astype(np.float32)
                return {self.input_name: tensor}
            except Exception:
                continue
        return None

    def rewind(self) -> None:
        self.index = 0


def load_session(model_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 4
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])


def run_session(session: ort.InferenceSession, image_paths: list[Path], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    input_name = session.get_inputs()[0].name
    logits_list: list[np.ndarray] = []
    labels: list[int] = []
    class_to_idx = {name: idx for idx, name in enumerate(TRAINING_CLASS_ORDER)}
    for image_path in image_paths:
        label_name = project_label_from_path(image_path)
        if label_name is None:
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = evaluation_tensor_from_image(image, image_size).unsqueeze(0).numpy().astype(np.float32)
            logits = session.run(None, {input_name: tensor})[0]
        except Exception:
            continue
        logits_list.append(np.asarray(logits[0], dtype=np.float32))
        labels.append(class_to_idx[label_name])
    if not logits_list:
        return np.zeros((0, len(TRAINING_CLASS_ORDER)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(logits_list, axis=0), np.asarray(labels, dtype=np.int64)


def accuracy_and_confmat(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    if logits.size == 0 or labels.size == 0:
        return 0.0, np.zeros((len(TRAINING_CLASS_ORDER), len(TRAINING_CLASS_ORDER)), dtype=np.int64)
    preds = logits.argmax(axis=1)
    acc = float((preds == labels).mean())
    confmat = np.zeros((len(TRAINING_CLASS_ORDER), len(TRAINING_CLASS_ORDER)), dtype=np.int64)
    for target, pred in zip(labels.tolist(), preds.tolist()):
        confmat[int(target), int(pred)] += 1
    return acc, confmat


def verify_quantized_model(fp32_session: ort.InferenceSession, int8_session: ort.InferenceSession, verification_paths: list[Path], image_size: int) -> dict[str, Any]:
    fp32_logits, labels = run_session(fp32_session, verification_paths, image_size)
    int8_logits, _ = run_session(int8_session, verification_paths, image_size)
    if fp32_logits.shape != int8_logits.shape:
        raise RuntimeError(f"Logit shape mismatch: fp32={fp32_logits.shape}, int8={int8_logits.shape}")
    fp32_acc, fp32_confmat = accuracy_and_confmat(fp32_logits, labels)
    int8_acc, int8_confmat = accuracy_and_confmat(int8_logits, labels)
    agreement = float((fp32_logits.argmax(axis=1) == int8_logits.argmax(axis=1)).mean()) if fp32_logits.size else 0.0
    return {
        "fp32_accuracy": fp32_acc,
        "int8_accuracy": int8_acc,
        "top1_agreement": agreement,
        "fp32_confusion_matrix": fp32_confmat.tolist(),
        "int8_confusion_matrix": int8_confmat.tolist(),
        "max_abs_diff": float(np.max(np.abs(fp32_logits - int8_logits))) if fp32_logits.size else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(fp32_logits - int8_logits))) if fp32_logits.size else 0.0,
        "num_samples": int(labels.shape[0]),
    }


def export_quantized_model(
    fp32_path: Path,
    out_path: Path,
    calibration_paths: list[Path],
    verification_paths: list[Path],
    overwrite: bool,
) -> dict[str, Any]:
    if out_path.exists() and not overwrite:
        return {"checkpoint": str(fp32_path), "quantized": str(out_path), "status": "skipped_exists"}

    session = load_session(fp32_path)
    image_size = int(session.get_inputs()[0].shape[2] or 224)
    calibrator = ImageCalibrationDataReader(calibration_paths, image_size)

    if out_path.exists():
        out_path.unlink()
    out_data_path = out_path.with_suffix(out_path.suffix + ".data")
    if out_data_path.exists():
        out_data_path.unlink()

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(out_path),
        calibration_data_reader=calibrator,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=True,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "EnableSubgraph": True,
        },
    )

    int8_session = load_session(out_path)
    verification = verify_quantized_model(session, int8_session, verification_paths, image_size)
    verification.update(
        {
            "checkpoint": str(fp32_path),
            "quantized": str(out_path),
            "status": "quantized",
            "image_size": image_size,
        }
    )
    return verification


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    calibration_root = Path(args.calibration_root)
    verification_root = Path(args.verification_root)

    fp32_checkpoints = sorted((results_dir / "onnx_models").glob("*/*.onnx"))
    if not fp32_checkpoints:
        print(f"No fp32 ONNX checkpoints found in {results_dir / 'onnx_models'}")
        return 1

    calibration_paths = sample_paths(collect_image_paths(calibration_root), args.calibration_samples, args.seed)
    verification_paths = collect_image_paths(verification_root)
    verification_paths = [path for path in verification_paths if project_label_from_path(path) is not None]
    if args.verification_samples > 0 and len(verification_paths) > args.verification_samples:
        verification_paths = sample_paths(verification_paths, args.verification_samples, args.seed + 1)

    if not calibration_paths:
        print(f"No calibration images found under {calibration_root}")
        return 1
    if not verification_paths:
        print(f"No verification images found under {verification_root}")
        return 1

    reports: list[dict[str, Any]] = []
    for fp32_path in fp32_checkpoints:
        out_path = results_dir / "onnx_quantised_models" / fp32_path.parent.name / f"{fp32_path.stem}_quantised_int8.onnx"
        report = export_quantized_model(fp32_path, out_path, calibration_paths, verification_paths, args.overwrite)
        reports.append(report)
        print(json.dumps(report, sort_keys=True))

    report_path = results_dir / "onnx_quantised_models" / "onnx_int8_quantization_report.json"
    report_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
