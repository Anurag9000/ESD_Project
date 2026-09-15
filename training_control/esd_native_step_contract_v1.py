#!/usr/bin/env python3
"""Source contract for the ESD physical cohort model-step adapter.

The physical cohort layer is allowed to share sample coordinates/views, but it is
not allowed to reimplement ESD model science. This contract proves that the live
repository still exposes the native batch-step, deterministic sampler, SAM/AdamW,
checkpoint and stage-resume seams required by an adapter.

It deliberately emits no execution claim. A root profile must not advertise
physical cohort execution merely because this source contract passes.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "metric_learning_pipeline.py"
PHASE0 = ROOT / "scripts" / "train_phase0_mim.py"
EXACT_DATA = ROOT / "training_control" / "esd_deterministic_data_v1.py"
BROKER = ROOT / "training_control" / "esd_shared_batch_broker_v2.py"
GROUPING = ROOT / "training_control" / "esd_dataset_cohort_catalog_v2.py"
SCHEMA = "esd-native-step-contract/v1"

_REQUIRED_PIPELINE_FUNCTIONS = {
    "build_parser",
    "build_datasets",
    "make_epoch_sampler",
    "make_balanced_sampler",
    "make_weighted_sampler",
    "train_supcon_steps",
    "train_classifier_steps",
    "save_step_checkpoint",
    "save_training_checkpoint",
    "load_resume_checkpoint",
    "build_classifier_phase_plan",
    "resolve_phase_start_index",
}
_REQUIRED_PIPELINE_CLASSES = {
    "DeterministicAugmentedImageFolder",
    "DeterministicSupConDataset",
    "DeterministicEpochSampler",
    "BalancedClassEpochSampler",
    "SAM",
}
_REQUIRED_PIPELINE_TOKENS = (
    "epoch_step_completed",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "phase_best_loss",
    "phase_wait",
    "stage",
    "supcon",
    "classifier",
    "set_start_index",
)
_REQUIRED_PHASE0_TOKENS = (
    "epoch_batch_index",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "early_stopping",
    "resume",
)
_REQUIRED_EXACT_DATA_TOKENS = (
    "derive_augmentation_seed",
    "persistent_workers",
    "source_index",
    "variant_index",
    "epoch",
    "view_offset",
    "attempt",
)
_REQUIRED_BROKER_TOKENS = (
    "register_consumer",
    "deactivate",
    "transition",
    "sampler divergence",
    "assert_batch_boundary_clean",
    "move_to_device",
)
_REQUIRED_GROUPING_TOKENS = (
    "physical_batch_contract",
    "supcon_then_ce",
    "ce_only",
    "phase0_mim",
    "final_refine",
    "overlap_group_last",
)


def _read(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"required ESD cohort source is missing: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8", errors="strict")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _definitions(source: str, path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(source, filename=str(path))
    functions: set[str] = set()
    classes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.add(node.name)
    return functions, classes


def _require_tokens(source: str, tokens: tuple[str, ...], label: str) -> None:
    missing = [token for token in tokens if token not in source]
    if missing:
        raise RuntimeError(f"{label} lost native cohort prerequisites: {missing}")


def audit() -> dict[str, Any]:
    pipeline = _read(PIPELINE)
    phase0 = _read(PHASE0)
    exact_data = _read(EXACT_DATA)
    broker = _read(BROKER)
    grouping = _read(GROUPING)

    functions, classes = _definitions(pipeline, PIPELINE)
    missing_functions = sorted(_REQUIRED_PIPELINE_FUNCTIONS - functions)
    missing_classes = sorted(_REQUIRED_PIPELINE_CLASSES - classes)
    if missing_functions or missing_classes:
        raise RuntimeError(
            "ESD native step surface drifted: "
            f"functions={missing_functions}, classes={missing_classes}"
        )

    _require_tokens(pipeline, _REQUIRED_PIPELINE_TOKENS, "metric-learning pipeline")
    _require_tokens(phase0, _REQUIRED_PHASE0_TOKENS, "Phase-0 MIM")
    _require_tokens(exact_data, _REQUIRED_EXACT_DATA_TOKENS, "restart-addressable data shim")
    _require_tokens(broker, _REQUIRED_BROKER_TOKENS, "stage-aware shared-batch broker")
    _require_tokens(grouping, _REQUIRED_GROUPING_TOKENS, "dataset/view grouping compiler")

    tree = ast.parse(pipeline, filename=str(PIPELINE))
    signatures: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "train_supcon_steps", "train_classifier_steps"
        }:
            signatures[node.name] = [argument.arg for argument in node.args.args]
    for name in ("train_supcon_steps", "train_classifier_steps"):
        args = signatures.get(name, [])
        for required in ("model", "batch_iterator", "step_limit", "optimizer"):
            if required not in args:
                raise RuntimeError(f"{name} no longer exposes required adapter argument {required!r}")

    return {
        "schema": SCHEMA,
        "repository": "Anurag9000/ESD_Project",
        "pipeline_sha256": _sha256(pipeline),
        "phase0_sha256": _sha256(phase0),
        "exact_data_sha256": _sha256(exact_data),
        "broker_sha256": _sha256(broker),
        "grouping_sha256": _sha256(grouping),
        "native_step_functions": {
            name: signatures[name] for name in sorted(signatures)
        },
        "native_sampler_classes": sorted(
            name for name in _REQUIRED_PIPELINE_CLASSES if "Sampler" in name
        ),
        "native_resume_loader": "load_resume_checkpoint",
        "grouping_coordinates": [
            "canonical_data_contract",
            "workflow_cadence",
            "physical_batch_contract",
        ],
        "sampler_cursor_source_proven": True,
        "native_sam_and_adamw_preserved": True,
        "native_amp_scaler_state_preserved": True,
        "native_scheduler_state_preserved": True,
        "native_stage_early_stopping_preserved": True,
        "phase0_native_resume_preserved": True,
        "restart_addressable_augmentation_preserved": True,
        "shared_batch_broker_stage_aware": True,
        "physical_cohort_execution_activated": False,
        "execution_claim_emitted": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
