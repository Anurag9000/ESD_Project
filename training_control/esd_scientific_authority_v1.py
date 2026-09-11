#!/usr/bin/env python3
"""Exhaustive source-derived ESD scientific/training authority.

The old account-wide launcher only guessed generic trainer filenames.  This authority
represents the actual finite scientific selector space exposed by the live ESD
trainers while keeping continuous hyperparameters configurable rather than inventing
an infinite Cartesian product.

Covered finite surfaces:
* every registered backbone (Atto/Femto/Pico/Nano/Tiny/EfficientNetV2-S);
* pretrained vs scratch initialization;
* AdamW vs SAM;
* mixed/FP32/FP64 precision;
* progressive vs full-model classifier tuning;
* val-loss vs raw-accuracy classifier stopping;
* balanced/weighted/shuffle sampling;
* SupCon+CE vs CE-only methodology;
* Phase-0 MIM raw-MSE vs patch-normalized-MSE and constant vs cosine schedule;
* MIM-seeded supervised integration;
* confidence-gap, per-class weighting, and every ordered targeted-confusion loss;
* final train-only refinement and evaluation/TorchScript/ONNX/INT8 deployment paths.

All training transactions use the repository's native step-level checkpoint state
and semantic early stopping.  ``esd_pressure_runner_v1.py`` bridges those native
checkpoints to the OPF checkpoint-request acknowledgement protocol; it does not
reimplement scheduling/resource admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import os
from pathlib import Path
import sys
from typing import Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = os.environ.get("ESD_DATASET_ROOT", "Dataset_Final")
VERIFICATION_ROOT = os.environ.get("ESD_VERIFICATION_ROOT", "Test_Dataset_Real")
ARTIFACT_ROOT = Path(os.environ.get("ESD_CENTRAL_ARTIFACT_ROOT", "Results/central_training"))
LOG_ROOT = Path(os.environ.get("ESD_CENTRAL_LOG_ROOT", "logs/central_training"))

BACKBONES = (
    "atto",
    "femto",
    "convnextv2_pico",
    "convnextv2_nano",
    "convnextv2_tiny",
    "efficientnetv2_s",
)
WEIGHTS = ("default", "none")
OPTIMIZERS = ("adamw", "sam")
PRECISIONS = ("mixed", "32", "64")
CLASSIFIER_MODES = ("progressive", "full_model")
CLASSIFIER_METRICS = ("val_loss", "val_raw_acc")
SAMPLING_STRATEGIES = ("balanced", "weighted", "shuffle")
SUPCON_MODES = (True, False)
MIM_LOSSES = ("raw_mse", "patch_normalized_mse")
MIM_SCHEDULERS = ("warmup_constant", "warmup_cosine")
CLASSES = ("organic", "metal", "paper")

EXPECTED_MAIN_SOURCE_TOKENS = (
    'BACKBONE_REGISTRY: dict[str, BackboneSpec]',
    '--optimizer", choices=["sam", "adamw"]',
    '--precision", choices=("mixed", "32", "64")',
    '--classifier-train-mode", choices=("progressive", "full_model")',
    '--classifier-early-stopping-metric", choices=("val_loss", "val_raw_acc")',
    'choices=("balanced", "weighted", "shuffle")',
    '--skip-supcon',
    '--resume-checkpoint',
    'save_step_checkpoint(',
    '"optimizer_state_dict"',
    '"scheduler_state_dict"',
    '"scaler_state_dict"',
    '"epoch_step_completed"',
    'supcon_early_stopping_patience',
    'head_early_stopping_patience',
    'stage_early_stopping_patience',
    'confidence_gap_penalty_weight',
    'class_loss_weight_map_resolved',
    'targeted_confusion_penalties_resolved',
)
EXPECTED_MIM_SOURCE_TOKENS = (
    'PHASE0_LOSS_MODE_RAW_MSE = "raw_mse"',
    'PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE = "patch_normalized_mse"',
    'PHASE0_SCHEDULER_MODE_WARMUP_CONSTANT = "warmup_constant"',
    'PHASE0_SCHEDULER_MODE_WARMUP_COSINE = "warmup_cosine"',
    '--resume-checkpoint',
    '"optimizer_state_dict"',
    '"scaler_state_dict"',
    '"scheduler_state_dict"',
    '"epoch_batch_index"',
    '--early-stopping-patience',
    'train_loss_window_best_loss',
)
EXPECTED_PRESSURE_TOKENS = (
    'TRAINING_CHECKPOINT_REQUEST_FILE',
    'TRAINING_CHECKPOINT_ACK_FILE',
    'checkpoint_ready',
    'scheduler_owns_process_control',
)


@dataclass(frozen=True, slots=True)
class SourceAudit:
    main_selector_matrix_size: int
    phase0_matrix_size: int
    phase0_integration_size: int
    auxiliary_objective_job_count: int
    canonical_deployment_count: int

    @property
    def expected_training_jobs(self) -> int:
        return (
            self.main_selector_matrix_size
            + self.phase0_matrix_size
            + self.phase0_integration_size
            + self.auxiliary_objective_job_count
            + 2 * len(BACKBONES) * len(WEIGHTS)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "backbones": list(BACKBONES),
            "weights": list(WEIGHTS),
            "optimizers": list(OPTIMIZERS),
            "precisions": list(PRECISIONS),
            "classifier_modes": list(CLASSIFIER_MODES),
            "classifier_metrics": list(CLASSIFIER_METRICS),
            "sampling_strategies": list(SAMPLING_STRATEGIES),
            "supcon_modes": ["supcon_ce", "ce_only"],
            "mim_losses": list(MIM_LOSSES),
            "mim_schedulers": list(MIM_SCHEDULERS),
            "classes": list(CLASSES),
            "main_selector_matrix_size": self.main_selector_matrix_size,
            "phase0_matrix_size": self.phase0_matrix_size,
            "phase0_integration_size": self.phase0_integration_size,
            "auxiliary_objective_job_count": self.auxiliary_objective_job_count,
            "expected_training_jobs": self.expected_training_jobs,
            "canonical_deployment_count": self.canonical_deployment_count,
            "continuous_hyperparameters_remain_tunable": True,
            "blind_infinite_cartesian_product_emitted": False,
            "source_configuration_only": True,
            "execution_claim_emitted": False,
            "training_claim_emitted": False,
        }


def _source(path: str) -> str:
    target = ROOT / path
    if not target.is_file():
        raise FileNotFoundError(target)
    return target.read_text(encoding="utf-8", errors="replace")


def audit_source_contract() -> SourceAudit:
    main = _source("scripts/metric_learning_pipeline.py")
    mim = _source("scripts/train_phase0_mim.py")
    pressure = _source("training_control/esd_pressure_runner_v1.py")
    for backbone in BACKBONES:
        if f'"{backbone}"' not in main:
            raise RuntimeError(f"ESD backbone {backbone!r} disappeared from the live registry")
    missing_main = [token for token in EXPECTED_MAIN_SOURCE_TOKENS if token not in main]
    missing_mim = [token for token in EXPECTED_MIM_SOURCE_TOKENS if token not in mim]
    missing_pressure = [token for token in EXPECTED_PRESSURE_TOKENS if token not in pressure]
    if missing_main or missing_mim or missing_pressure:
        raise RuntimeError(
            "ESD scientific source contract drifted: "
            f"main={missing_main} mim={missing_mim} pressure={missing_pressure}"
        )
    ordered_pairs = tuple((a, b) for a in CLASSES for b in CLASSES if a != b)
    main_matrix = len(tuple(itertools.product(
        BACKBONES, WEIGHTS, OPTIMIZERS, PRECISIONS, CLASSIFIER_MODES,
        CLASSIFIER_METRICS, SAMPLING_STRATEGIES, SUPCON_MODES,
    )))
    phase0_matrix = len(BACKBONES) * len(WEIGHTS) * len(MIM_LOSSES) * len(MIM_SCHEDULERS)
    # One MIM-seeded canonical supervised job per Phase-0 scientific cell.
    phase0_integration = phase0_matrix
    # confidence gap + one class-weight job per class + every ordered confusion pair,
    # repeated for every architecture so each model family exercises extended losses.
    objective_variants_per_backbone = 1 + len(CLASSES) + len(ordered_pairs)
    objective_jobs = len(BACKBONES) * objective_variants_per_backbone
    deployment = len(BACKBONES) * len(WEIGHTS) * 4
    return SourceAudit(main_matrix, phase0_matrix, phase0_integration, objective_jobs, deployment)


def _slug(*parts: object) -> str:
    return "__".join(str(part).replace("/", "-").replace(" ", "_") for part in parts)


def _training_contract(checkpoint_root: Path) -> dict[str, object]:
    return {
        "is_training_job": True,
        "device_capable": True,
        "resume_strategy": "exact_checkpoint",
        "checkpoint_contract": {"exact_resume": True},
        "checkpoint_artifacts": [
            str(checkpoint_root / "step_last.pt"),
            str(checkpoint_root / "last.pt"),
        ],
        "early_stopping": True,
        "early_stopping_applicable": True,
        "exact_resume_source": "scripts/metric_learning_pipeline.py:save_step_checkpoint",
        "early_stopping_source": "scripts/metric_learning_pipeline.py:semantic phase patience",
        "pressure_checkpoint_source": "training_control/esd_pressure_runner_v1.py",
        "cooperative_checkpoint_ack": True,
    }


def _pressure_command(output: Path, completion: Path, child: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "training_control/esd_pressure_runner_v1.py",
        "--checkpoint-root",
        str(output),
        "--completion-marker",
        str(completion),
        "--",
        *[str(value) for value in child],
    ]


def _main_job(
    *,
    backbone: str,
    weights: str,
    optimizer: str,
    precision: str,
    classifier_mode: str,
    classifier_metric: str,
    sampling: str,
    supcon: bool,
    family: str = "supervised-selector-matrix",
    suffix: str = "",
    extra_args: Sequence[str] = (),
    depends_on: Sequence[str] = ("audit-esd-scientific-authority",),
    phase0_checkpoint: str = "",
) -> dict[str, object]:
    mode = "supcon_ce" if supcon else "ce_only"
    parts = [backbone, weights, optimizer, precision, classifier_mode, classifier_metric, sampling, mode]
    if suffix:
        parts.append(suffix)
    slug = _slug(*parts)
    output = ARTIFACT_ROOT / family / slug
    log = LOG_ROOT / family / f"{slug}.jsonl"
    completion = output / "COMPLETE.json"
    child = [
        sys.executable,
        "scripts/metric_learning_pipeline.py",
        "--dataset-root",
        DATASET_ROOT,
        "--output-dir",
        str(output),
        "--log-file",
        str(log),
        "--backbone",
        backbone,
        "--weights",
        weights,
        "--optimizer",
        optimizer,
        "--precision",
        precision,
        "--classifier-train-mode",
        classifier_mode,
        "--classifier-early-stopping-metric",
        classifier_metric,
        "--sampling-strategy",
        sampling,
        "--grad-accum-steps",
        "1",
        "--resume-checkpoint",
        str(output / "step_last.pt"),
        "--seed",
        os.environ.get("ESD_CENTRAL_SEED", "42"),
    ]
    if not supcon:
        child.append("--skip-supcon")
    if phase0_checkpoint:
        child.extend(["--phase0-encoder-checkpoint", phase0_checkpoint])
    child.extend(str(value) for value in extra_args)
    job = {
        "id": f"train:{family}:{slug}",
        "command": _pressure_command(output, completion, child),
        "phase": "training",
        "family": family,
        "depends_on": list(depends_on),
        "completion_artifacts": [str(completion), str(output / "best.pt")],
        "backbone": backbone,
        "weights": weights,
        "optimizer": optimizer,
        "precision": precision,
        "classifier_train_mode": classifier_mode,
        "classifier_early_stopping_metric": classifier_metric,
        "sampling_strategy": sampling,
        "supcon_enabled": supcon,
        "task": "three_class_waste_classification",
        "class_order": list(CLASSES),
        "source_compatible": True,
        "execution_claim_emitted": False,
    }
    job.update(_training_contract(output))
    return job


def _phase0_job(backbone: str, weights: str, loss: str, scheduler: str) -> dict[str, object]:
    slug = _slug(backbone, weights, loss, scheduler)
    output = ARTIFACT_ROOT / "phase0_mim" / slug
    log = LOG_ROOT / "phase0_mim" / f"{slug}.jsonl"
    completion = output / "COMPLETE.json"
    child = [
        sys.executable,
        "scripts/train_phase0_mim.py",
        "--dataset-root",
        DATASET_ROOT,
        "--output-dir",
        str(output),
        "--log-file",
        str(log),
        "--backbone",
        backbone,
        "--weights",
        weights,
        "--loss-mode",
        loss,
        "--scheduler-mode",
        scheduler,
        "--resume-checkpoint",
        str(output / "step_last.pt"),
        "--seed",
        os.environ.get("ESD_CENTRAL_SEED", "42"),
    ]
    if scheduler == "warmup_cosine":
        child.extend(["--warmup-steps", "1000", "--total-steps", "50000", "--max-steps", "50000"])
    job = {
        "id": f"train:phase0_mim:{slug}",
        "command": _pressure_command(output, completion, child),
        "phase": "training",
        "family": "phase0_mim",
        "depends_on": ["audit-esd-scientific-authority"],
        "completion_artifacts": [str(completion), str(output / "phase0_encoder_final.pth")],
        "backbone": backbone,
        "weights": weights,
        "mim_loss": loss,
        "mim_scheduler": scheduler,
        "task": "masked_image_modeling",
        "source_compatible": True,
        "execution_claim_emitted": False,
        "exact_resume_source": "scripts/train_phase0_mim.py:save_phase0_checkpoint",
        "early_stopping_source": "scripts/train_phase0_mim.py:effective-batch plateau early stopping",
    }
    job.update(_training_contract(output))
    job["exact_resume_source"] = "scripts/train_phase0_mim.py:save_phase0_checkpoint"
    job["early_stopping_source"] = "scripts/train_phase0_mim.py:effective-batch plateau early stopping"
    return job


def _canonical_main_id(backbone: str, weights: str) -> str:
    slug = _slug(backbone, weights, "adamw", "mixed", "progressive", "val_loss", "balanced", "supcon_ce")
    return f"train:supervised-selector-matrix:{slug}"


def iter_jobs() -> Iterator[dict[str, object]]:
    audit_source_contract()

    phase0_ids: dict[tuple[str, str, str, str], str] = {}
    for backbone, weights, loss, scheduler in itertools.product(BACKBONES, WEIGHTS, MIM_LOSSES, MIM_SCHEDULERS):
        job = _phase0_job(backbone, weights, loss, scheduler)
        phase0_ids[(backbone, weights, loss, scheduler)] = str(job["id"])
        yield job

    for values in itertools.product(
        BACKBONES,
        WEIGHTS,
        OPTIMIZERS,
        PRECISIONS,
        CLASSIFIER_MODES,
        CLASSIFIER_METRICS,
        SAMPLING_STRATEGIES,
        SUPCON_MODES,
    ):
        yield _main_job(
            backbone=values[0],
            weights=values[1],
            optimizer=values[2],
            precision=values[3],
            classifier_mode=values[4],
            classifier_metric=values[5],
            sampling=values[6],
            supcon=values[7],
        )

    # Exercise every Phase-0 scientific recipe as an initialization route into the
    # canonical supervised pipeline without multiplying it across unrelated axes.
    for (backbone, weights, loss, scheduler), phase0_id in sorted(phase0_ids.items()):
        phase0_output = ARTIFACT_ROOT / "phase0_mim" / _slug(backbone, weights, loss, scheduler)
        yield _main_job(
            backbone=backbone,
            weights=weights,
            optimizer="adamw",
            precision="mixed",
            classifier_mode="progressive",
            classifier_metric="val_loss",
            sampling="balanced",
            supcon=True,
            family="phase0_seeded_supervised",
            suffix=_slug(loss, scheduler),
            depends_on=(phase0_id,),
            phase0_checkpoint=str(phase0_output / "phase0_encoder_final.pth"),
        )

    # Additional differentiable loss surfaces. Continuous weights are intentionally
    # tunable; these jobs prove each distinct mechanism is executable.
    ordered_pairs = [(a, b) for a in CLASSES for b in CLASSES if a != b]
    for backbone in BACKBONES:
        yield _main_job(
            backbone=backbone, weights="default", optimizer="adamw", precision="mixed",
            classifier_mode="progressive", classifier_metric="val_loss", sampling="balanced", supcon=True,
            family="objective-surface", suffix="confidence_gap",
            extra_args=("--confidence-gap-penalty-weight", "0.1"),
        )
        for class_name in CLASSES:
            yield _main_job(
                backbone=backbone, weights="default", optimizer="adamw", precision="mixed",
                classifier_mode="progressive", classifier_metric="val_loss", sampling="balanced", supcon=True,
                family="objective-surface", suffix=f"class_weight_{class_name}",
                extra_args=("--class-loss-weight", f"{class_name}=1.25"),
            )
        for true_name, predicted_name in ordered_pairs:
            yield _main_job(
                backbone=backbone, weights="default", optimizer="adamw", precision="mixed",
                classifier_mode="progressive", classifier_metric="val_loss", sampling="balanced", supcon=True,
                family="objective-surface", suffix=f"targeted_{true_name}_to_{predicted_name}",
                extra_args=("--targeted-confusion-penalty", f"{true_name}:{predicted_name}:0.1"),
            )

    # Train-only cleanup/refinement is a distinct repository methodology. One job
    # per architecture+initialization is enough to cover the mechanism because the
    # main selector matrix above already covers optimizer/precision/sampling modes.
    for backbone, weights in itertools.product(BACKBONES, WEIGHTS):
        parent = _canonical_main_id(backbone, weights)
        main_slug = _slug(backbone, weights, "adamw", "mixed", "progressive", "val_loss", "balanced", "supcon_ce")
        source_output = ARTIFACT_ROOT / "supervised-selector-matrix" / main_slug
        refine_output = ARTIFACT_ROOT / "final_refine" / _slug(backbone, weights)
        completion = refine_output / "COMPLETE.json"
        child = [
            sys.executable,
            "scripts/metric_learning_pipeline.py",
            "--train-only",
            "--dataset-root",
            DATASET_ROOT,
            "--output-dir",
            str(refine_output),
            "--log-file",
            str(LOG_ROOT / "final_refine" / f"{_slug(backbone, weights)}.jsonl"),
            "--backbone",
            backbone,
            "--weights",
            weights,
            "--resume-checkpoint",
            str(source_output / "best.pt"),
            "--sampling-strategy",
            "balanced",
            "--grad-accum-steps",
            "1",
        ]
        job = {
            "id": f"train:final_refine:{_slug(backbone, weights)}",
            "command": _pressure_command(refine_output, completion, child),
            "phase": "training",
            "family": "final_refine",
            "depends_on": [parent],
            "completion_artifacts": [str(completion)],
            "backbone": backbone,
            "weights": weights,
            "task": "train_loss_cleanup_refinement",
            "source_compatible": True,
            "execution_claim_emitted": False,
        }
        job.update(_training_contract(refine_output))
        yield job

    # One verified deployment chain per architecture+initialization.
    for backbone, weights in itertools.product(BACKBONES, WEIGHTS):
        parent = _canonical_main_id(backbone, weights)
        main_slug = _slug(backbone, weights, "adamw", "mixed", "progressive", "val_loss", "balanced", "supcon_ce")
        checkpoint = ARTIFACT_ROOT / "supervised-selector-matrix" / main_slug / "best.pt"
        deploy_root = ARTIFACT_ROOT / "deployment" / _slug(backbone, weights)
        eval_id = f"evaluate:{_slug(backbone, weights)}"
        yield {
            "id": eval_id,
            "command": [
                sys.executable, "training_control/esd_postprocess_v1.py", "evaluate",
                "--checkpoint", str(checkpoint), "--output-dir", str(deploy_root / "evaluation"),
                "--dataset-root", DATASET_ROOT,
            ],
            "phase": "evaluation",
            "family": "checkpoint-evaluation",
            "device_capable": True,
            "is_training_job": False,
            "depends_on": [parent],
            "resume_strategy": "restart_exact",
            "deterministic": True,
            "idempotent": True,
            "atomic_outputs": True,
            "checkpoint_contract": {"exact_resume": True, "deterministic": True, "idempotent": True, "atomic_outputs": True},
            "early_stopping_applicable": False,
            "early_stopping_exception_reason": "evaluation-only transaction",
            "completion_artifacts": [str(deploy_root / "evaluation" / "evaluation_summary.json")],
        }
        ts_id = f"export:torchscript:{_slug(backbone, weights)}"
        yield {
            "id": ts_id,
            "command": [sys.executable, "training_control/esd_postprocess_v1.py", "torchscript", "--checkpoint", str(checkpoint)],
            "phase": "export",
            "family": "torchscript-export",
            "device_capable": False,
            "is_training_job": False,
            "depends_on": [parent],
            "resume_strategy": "restart_exact",
            "deterministic": True,
            "idempotent": True,
            "atomic_outputs": True,
            "checkpoint_contract": {"exact_resume": True, "deterministic": True, "idempotent": True, "atomic_outputs": True},
            "early_stopping_applicable": False,
            "early_stopping_exception_reason": "export-only transaction",
            "completion_artifacts": [str(checkpoint.parent / "best_scripted.pt")],
        }
        onnx_path = deploy_root / "model.onnx"
        onnx_id = f"export:onnx:{_slug(backbone, weights)}"
        yield {
            "id": onnx_id,
            "command": [sys.executable, "training_control/esd_postprocess_v1.py", "onnx", "--checkpoint", str(checkpoint), "--output", str(onnx_path)],
            "phase": "export",
            "family": "onnx-export",
            "device_capable": False,
            "is_training_job": False,
            "depends_on": [parent],
            "resume_strategy": "restart_exact",
            "deterministic": True,
            "idempotent": True,
            "atomic_outputs": True,
            "checkpoint_contract": {"exact_resume": True, "deterministic": True, "idempotent": True, "atomic_outputs": True},
            "early_stopping_applicable": False,
            "early_stopping_exception_reason": "export-only transaction",
            "completion_artifacts": [str(onnx_path), str(onnx_path.with_suffix(".report.json"))],
        }
        int8_path = deploy_root / "model_int8.onnx"
        yield {
            "id": f"export:int8:{_slug(backbone, weights)}",
            "command": [
                sys.executable, "training_control/esd_postprocess_v1.py", "quantize",
                "--onnx", str(onnx_path), "--output", str(int8_path),
                "--calibration-root", DATASET_ROOT, "--verification-root", VERIFICATION_ROOT,
            ],
            "phase": "export",
            "family": "int8-quantization",
            "device_capable": False,
            "is_training_job": False,
            "depends_on": [onnx_id],
            "resume_strategy": "restart_exact",
            "deterministic": True,
            "idempotent": True,
            "atomic_outputs": True,
            "checkpoint_contract": {"exact_resume": True, "deterministic": True, "idempotent": True, "atomic_outputs": True},
            "early_stopping_applicable": False,
            "early_stopping_exception_reason": "quantization/export transaction",
            "completion_artifacts": [str(int8_path), str(int8_path.with_suffix(".report.json"))],
        }
