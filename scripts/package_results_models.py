#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


MODEL_DIRS = ("pt_models", "onnx_models", "onnx_quantised_models")
SKIP_DIR_NAMES = {
    "pt_models",
    "onnx_models",
    "onnx_quantised_models",
    "__pycache__",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exhaustive model-only Results package trees.")
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root containing the training artifacts to scan for best.pt equivalents.",
    )
    parser.add_argument(
        "--target-root",
        required=True,
        help="Root where pt_models/, onnx_models/, and onnx_quantised_models/ will be materialized.",
    )
    parser.add_argument(
        "--prune-non-model-dirs",
        action="store_true",
        help="Remove non-model directories from the target root after packaging.",
    )
    parser.add_argument(
        "--clean-model-dirs",
        action="store_true",
        help="Remove the target model directories before copying sources. Use only when you want a full rebuild.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset forwarded to the exporter.")
    parser.add_argument(
        "--export-pattern",
        default=None,
        help="Optional checkpoint glob relative to pt_models/ for the ONNX export phase.",
    )
    parser.add_argument(
        "--quantize-pattern",
        default=None,
        help="Optional fp32 ONNX glob relative to onnx_models/ for the INT8 quantization phase.",
    )
    parser.add_argument("--calibration-samples", type=int, default=256, help="INT8 calibration sample count.")
    parser.add_argument("--verification-samples", type=int, default=0, help="INT8 verification sample cap.")
    parser.add_argument("--no-verify", dest="verify", action="store_false", help="Skip ONNX numerical verification.")
    parser.add_argument("--verify", dest="verify", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Keep existing exports.")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    parser.add_argument(
        "--keep-report-files",
        action="store_true",
        help="Keep exporter/packager JSON reports. Default is to remove them so the result tree stays model-only.",
    )
    return parser.parse_args()


def is_model_artifact(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def source_alias_for_checkpoint(source_root: Path, checkpoint_path: Path) -> str | None:
    rel = checkpoint_path.relative_to(source_root)
    parts = rel.parts
    if not parts:
        return None

    # Already-packaged entries are kept by name.
    if len(parts) == 3 and parts[0] == "pt_models" and parts[2].endswith(".pt"):
        return parts[1]

    if parts[0] == "progressive":
        if len(parts) == 2 and parts[1] == "best.pt":
            return "progressive_best"
        if len(parts) == 2 and parts[1] == "supcon_best.pt":
            return "progressive_supcon_best"
        if len(parts) == 4 and parts[1] == "phases" and parts[3] == "best.pt":
            return f"{parts[2]}_best"

    if parts[0] == "loss_cleanup":
        if len(parts) == 2 and parts[1] == "best.pt":
            return "loss_cleanup_best"
        if len(parts) == 2 and parts[1] == "accepted_best.pt":
            return "loss_cleanup_accepted_best"
        if len(parts) == 4 and parts[1] == "phases" and parts[3] == "best.pt":
            return f"loss_cleanup_{parts[2]}_best"
        if len(parts) == 3 and parts[1].startswith("iteration_") and parts[2] == "best.pt":
            return f"loss_cleanup_{parts[1]}_best"

    if parts[0] == "rawacc_refine":
        if len(parts) == 2 and parts[1] == "best.pt":
            return "rawacc_refine_best"
        if len(parts) == 2 and parts[1] == "accepted_best.pt":
            return "rawacc_refine_accepted_best"
        if len(parts) == 4 and parts[1] == "phases" and parts[3] == "best.pt":
            return f"rawacc_refine_{parts[2]}_best"
        if len(parts) == 3 and parts[1].startswith("iteration_") and parts[2] == "best.pt":
            return f"rawacc_refine_{parts[1]}_best"

    return None


def collect_source_checkpoints(source_root: Path) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    for path in (
        sorted(source_root.rglob("best.pt"))
        + sorted(source_root.rglob("accepted_best.pt"))
        + sorted(source_root.rglob("supcon_best.pt"))
    ):
        if not path.is_file():
            continue
        if is_model_artifact(path):
            continue
        alias = source_alias_for_checkpoint(source_root, path)
        if alias is None:
            continue
        existing = candidates.get(alias)
        if existing is None:
            candidates[alias] = path
            continue
        # Prefer the more direct source path over already-packaged copies.
        existing_is_packaged = "pt_models" in existing.parts
        candidate_is_packaged = "pt_models" in path.parts
        if existing_is_packaged and not candidate_is_packaged:
            candidates[alias] = path
        elif existing_is_packaged == candidate_is_packaged and len(path.parts) < len(existing.parts):
            candidates[alias] = path
    return candidates


def collect_packaged_checkpoints(pt_models_root: Path) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    if not pt_models_root.exists():
        return candidates
    for path in sorted(pt_models_root.glob("*/*.pt")):
        if not path.is_file():
            continue
        alias = path.parent.name
        existing = candidates.get(alias)
        if existing is None or len(path.parts) < len(existing.parts):
            candidates[alias] = path
    return candidates


def checkpoint_score(path: Path) -> tuple[int, float, float, str]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return (0, float("-inf"), float("-inf"), str(path))

    def first_float(*keys: str) -> float | None:
        for key in keys:
            value = checkpoint.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    raw_acc = first_float("best_val_raw_acc", "val_raw_acc", "best_val_acc", "val_acc")
    val_loss = first_float("best_val_loss", "val_loss")
    if raw_acc is not None:
        loss_score = -val_loss if val_loss is not None else float("-inf")
        return (2, raw_acc, loss_score, str(path))
    if val_loss is not None:
        return (1, -val_loss, float("-inf"), str(path))
    return (0, float("-inf"), float("-inf"), str(path))


def select_overall_best_checkpoint(sources: dict[str, Path]) -> tuple[str, Path] | None:
    best_alias: str | None = None
    best_path: Path | None = None
    best_score: tuple[int, float, float, str] | None = None
    for alias, path in sources.items():
        score = checkpoint_score(path)
        if best_score is None or score > best_score:
            best_alias = alias
            best_path = path
            best_score = score
    if best_alias is None or best_path is None:
        return None
    return best_alias, best_path


def clean_model_dirs(target_root: Path) -> None:
    for name in MODEL_DIRS:
        path = target_root / name
        if path.exists():
            shutil.rmtree(path)


def prune_non_model_dirs(target_root: Path) -> None:
    for child in target_root.iterdir():
        if child.name in MODEL_DIRS:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_sources_to_pt_models(target_root: Path, sources: dict[str, Path]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    pt_models_root = target_root / "pt_models"
    pt_models_root.mkdir(parents=True, exist_ok=True)
    for alias, source in sorted(sources.items()):
        out_dir = pt_models_root / alias
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{alias}.pt"
        if source.resolve() != out_path.resolve():
            shutil.copy2(source, out_path)
        manifest.append(
            {
                "alias": alias,
                "source": str(source),
                "target": str(out_path),
            }
        )
    return manifest


def run_exporters(
    target_root: Path,
    opset: int,
    verify: bool,
    overwrite: bool,
    export_pattern: str | None,
    quantize_pattern: str | None,
    calibration_samples: int,
    verification_samples: int,
) -> None:
    export_cmd = [
        sys.executable,
        "scripts/export_results_checkpoints_to_onnx.py",
        "--results-dir",
        str(target_root),
        "--opset",
        str(opset),
    ]
    if export_pattern is not None:
        export_cmd.extend(["--pattern", export_pattern])
    export_cmd.append("--verify" if verify else "--no-verify")
    export_cmd.append("--overwrite" if overwrite else "--no-overwrite")
    subprocess.run(export_cmd, check=True)

    quant_cmd = [
        sys.executable,
        "scripts/quantize_results_checkpoints_to_onnx.py",
        "--results-dir",
        str(target_root),
        "--calibration-samples",
        str(calibration_samples),
        "--verification-samples",
        str(verification_samples),
    ]
    if quantize_pattern is not None:
        quant_cmd.extend(["--pattern", quantize_pattern])
    quant_cmd.append("--overwrite" if overwrite else "--no-overwrite")
    subprocess.run(quant_cmd, check=True)


def cleanup_report_files(target_root: Path) -> None:
    for rel_path in (
        Path("model_package_manifest.json"),
        Path("onnx_models") / "onnx_export_report.json",
        Path("onnx_quantised_models") / "onnx_int8_quantization_report.json",
    ):
        path = target_root / rel_path
        if path.exists():
            path.unlink()


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    target_root = Path(args.target_root)

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    target_root.mkdir(parents=True, exist_ok=True)

    sources = collect_source_checkpoints(source_root)
    if not sources:
        sources = collect_packaged_checkpoints(target_root / "pt_models")
        if not sources:
            print(f"No model checkpoints found under {source_root} or {target_root / 'pt_models'}")
            return 1

    overall_best = select_overall_best_checkpoint(sources)
    if overall_best is not None:
        sources = dict(sources)
        sources["overall_best"] = overall_best[1]

    source_is_target = source_root.resolve() == target_root.resolve()
    if args.clean_model_dirs and not source_is_target:
        clean_model_dirs(target_root)

    manifest = copy_sources_to_pt_models(target_root, sources)
    run_exporters(
        target_root,
        opset=args.opset,
        verify=args.verify,
        overwrite=args.overwrite,
        export_pattern=args.export_pattern,
        quantize_pattern=args.quantize_pattern,
        calibration_samples=args.calibration_samples,
        verification_samples=args.verification_samples,
    )

    if args.prune_non_model_dirs:
        prune_non_model_dirs(target_root)

    if args.keep_report_files:
        manifest_path = target_root / "model_package_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"target_root": str(target_root), "count": len(manifest), "manifest": str(manifest_path)}, indent=2))
    else:
        cleanup_report_files(target_root)
        print(json.dumps({"target_root": str(target_root), "count": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
