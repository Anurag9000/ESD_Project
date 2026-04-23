#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

try:
    from metric_learning_pipeline import MetricLearningEfficientNetB0
except ModuleNotFoundError:
    from scripts.metric_learning_pipeline import MetricLearningEfficientNetB0


class ExportableClassifier(nn.Module):
    def __init__(self, model: MetricLearningEfficientNetB0) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.encode(image)
        return self.model.classify(embeddings, labels=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export grouped Results checkpoints to ONNX.")
    parser.add_argument(
        "--results-dir",
        default="Results/convnextv2_nano.fcmae_ft_in22k_in1k",
        help="Directory containing the grouped pt_models/ tree and output subfolders.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify ONNX against PyTorch after export.")
    parser.add_argument("--no-verify", dest="verify", action="store_false", help="Skip numerical verification.")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing ONNX files.")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Keep existing ONNX files.")
    parser.add_argument(
        "--pattern",
        default="pt_models/*/*.pt",
        help="Glob pattern relative to the selected results root for checkpoints to export.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format in {path}: {type(checkpoint)!r}")


def build_model(checkpoint: dict[str, Any]) -> ExportableClassifier:
    args_dict = dict(checkpoint.get("args", {}))
    class_names = list(checkpoint.get("class_names", []))
    num_classes = len(class_names) if class_names else 3

    backbone = str(args_dict.get("backbone", "convnextv2_nano"))
    image_size = int(args_dict.get("image_size", 224))
    embedding_dim = int(args_dict.get("embedding_dim", 512))
    projection_dim = int(args_dict.get("projection_dim", 256))

    model_args = SimpleNamespace(
        backbone=backbone,
        image_size=image_size,
    )
    model = MetricLearningEfficientNetB0(
        num_classes=num_classes,
        weights_mode="none",
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        args=model_args,
        backbone_name=backbone,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    wrapper = ExportableClassifier(model)
    wrapper.eval()
    return wrapper


def export_one(pt_path: Path, onnx_path: Path, opset: int, verify: bool, overwrite: bool) -> dict[str, Any]:
    checkpoint = load_checkpoint(pt_path)
    wrapper = build_model(checkpoint)
    image_size = int(dict(checkpoint.get("args", {})).get("image_size", 224))

    if onnx_path.exists() and not overwrite:
        return {
            "checkpoint": str(pt_path),
            "onnx": str(onnx_path),
            "status": "skipped_exists",
        }

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if onnx_path.exists():
        onnx_path.unlink()

    dummy = torch.randn(2, 3, image_size, image_size, dtype=torch.float32)
    with torch.no_grad():
        torch_logits = wrapper(dummy).cpu().numpy()

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    report: dict[str, Any] = {
        "checkpoint": str(pt_path),
        "onnx": str(onnx_path),
        "status": "exported",
        "image_size": image_size,
        "opset": opset,
        "verify": verify,
    }

    if verify:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        onnx_logits = session.run(None, {input_name: dummy.numpy()})[0]
        max_abs = float(np.max(np.abs(torch_logits - onnx_logits)))
        mean_abs = float(np.mean(np.abs(torch_logits - onnx_logits)))
        report["max_abs_diff"] = max_abs
        report["mean_abs_diff"] = mean_abs
        report["verified"] = bool(np.allclose(torch_logits, onnx_logits, rtol=1e-3, atol=1e-4))
        if not report["verified"]:
            raise RuntimeError(
                f"ONNX verification failed for {pt_path.name}: max_abs_diff={max_abs:.6g}, mean_abs_diff={mean_abs:.6g}"
            )

    return report


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    checkpoints = sorted(results_dir.glob(args.pattern))
    if not checkpoints:
        print(f"No checkpoints found under {results_dir} matching {args.pattern}")
        return 1

    reports: list[dict[str, Any]] = []
    for pt_path in checkpoints:
        onnx_path = results_dir / "onnx_models" / pt_path.parent.name / f"{pt_path.stem}.onnx"
        report = export_one(pt_path, onnx_path, args.opset, args.verify, args.overwrite)
        reports.append(report)
        print(json.dumps(report, sort_keys=True))

    report_path = results_dir / "onnx_models" / "onnx_export_report.json"
    report_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
