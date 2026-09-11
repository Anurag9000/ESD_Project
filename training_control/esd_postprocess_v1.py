#!/usr/bin/env python3
"""Post-training transactions for one ESD checkpoint.

These wrappers call the repository's native evaluation/export implementations with
explicit checkpoint/output paths so OPF can schedule them independently instead of
hiding them inside shell pipelines.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def evaluate(checkpoint: Path, output_dir: Path, dataset_root: Path) -> int:
    return subprocess.call(
        [
            sys.executable,
            "scripts/evaluate_saved_classifier.py",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--dataset-root",
            str(dataset_root),
            "--splits",
            "val",
            "test",
        ],
        cwd=ROOT,
    )


def torchscript(checkpoint: Path) -> int:
    env = os.environ.copy()
    env["CHECKPOINT_PATH"] = str(checkpoint)
    return subprocess.call([sys.executable, "scripts/export_to_torchscript.py"], cwd=ROOT, env=env)


def onnx(checkpoint: Path, output_path: Path) -> int:
    from export_results_checkpoints_to_onnx import export_one

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = export_one(checkpoint, output_path, 17, True, True)
    _atomic_json(output_path.with_suffix(".report.json"), report)
    return 0


def quantize(onnx_path: Path, output_path: Path, calibration_root: Path, verification_root: Path) -> int:
    from quantize_results_checkpoints_to_onnx import (
        collect_image_paths,
        export_quantized_model,
        project_label_from_path,
        sample_paths,
    )

    calibration = sample_paths(collect_image_paths(calibration_root), 256, 42)
    verification = [path for path in collect_image_paths(verification_root) if project_label_from_path(path) is not None]
    if not calibration:
        raise RuntimeError(f"no calibration images under {calibration_root}")
    if not verification:
        raise RuntimeError(f"no verification images under {verification_root}")
    report = export_quantized_model(onnx_path, output_path, calibration, verification, True)
    _atomic_json(output_path.with_suffix(".report.json"), report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--output-dir", required=True)
    p_eval.add_argument("--dataset-root", default="Dataset_Final")

    p_ts = sub.add_parser("torchscript")
    p_ts.add_argument("--checkpoint", required=True)

    p_onnx = sub.add_parser("onnx")
    p_onnx.add_argument("--checkpoint", required=True)
    p_onnx.add_argument("--output", required=True)

    p_quant = sub.add_parser("quantize")
    p_quant.add_argument("--onnx", required=True)
    p_quant.add_argument("--output", required=True)
    p_quant.add_argument("--calibration-root", default="Dataset_Final")
    p_quant.add_argument("--verification-root", default="Test_Dataset_Real")

    args = parser.parse_args()
    if args.action == "evaluate":
        return evaluate(Path(args.checkpoint), Path(args.output_dir), Path(args.dataset_root))
    if args.action == "torchscript":
        return torchscript(Path(args.checkpoint))
    if args.action == "onnx":
        return onnx(Path(args.checkpoint), Path(args.output))
    if args.action == "quantize":
        return quantize(Path(args.onnx), Path(args.output), Path(args.calibration_root), Path(args.verification_root))
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
