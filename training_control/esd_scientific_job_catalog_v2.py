#!/usr/bin/env python3
"""ESD scientific job catalog v2: v1 science + restart-addressable data entrypoint.

All finite scientific choices and job identities remain owned by
``esd_scientific_authority_v1``.  v2 changes only the executable entrypoint for
jobs that invoke ``scripts/metric_learning_pipeline.py`` so individual execution
and future dataset-cohort execution share the same exact-replay augmentation
contract.  Phase-0 MIM and non-training deployment transactions are untouched.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "training_control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

import esd_scientific_authority_v1 as base  # noqa: E402

SCHEMA = "esd-scientific-job-catalog/v2"
LEGACY_ENTRY = "scripts/metric_learning_pipeline.py"
EXACT_ENTRY = "training_control/run_metric_learning_exact_v1.py"


def _rewrite(job: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(job)
    command = [str(value) for value in (row.get("command") or [])]
    replacements = 0
    rewritten: list[str] = []
    for value in command:
        if value.replace("\\", "/") == LEGACY_ENTRY:
            rewritten.append(EXACT_ENTRY)
            replacements += 1
        else:
            rewritten.append(value)
    row["command"] = rewritten
    if replacements:
        if replacements != 1:
            raise RuntimeError(f"{row.get('id')}: expected exactly one metric-learning entrypoint, found {replacements}")
        row["dataset_replay_contract"] = "esd-restart-addressable-data/v1"
        row["dataset_replay_source"] = "training_control/esd_deterministic_data_v1.py"
        row["cpu_capable"] = True
        row["gpu_capable"] = True
        row["backend_selection"] = "torch cuda when scheduler exposes GPU, otherwise cpu"
    return row


def iter_jobs() -> Iterator[dict[str, Any]]:
    for job in base.iter_jobs():
        yield _rewrite(dict(job))


def audit_source_contract():
    return base.audit_source_contract()


def catalog_metadata() -> dict[str, Any]:
    jobs = list(iter_jobs())
    training = [job for job in jobs if bool(job.get("is_training_job"))]
    exact_data = [job for job in training if job.get("dataset_replay_contract") == "esd-restart-addressable-data/v1"]
    return {
        "schema": SCHEMA,
        "repository": "Anurag9000/ESD_Project",
        "source_authority": "training_control/esd_scientific_authority_v1.py",
        "training_jobs": len(training),
        "restart_addressable_supervised_jobs": len(exact_data),
        "phase0_and_other_training_jobs": len(training) - len(exact_data),
        "legacy_science_changed": False,
        "execution_claim_emitted": False,
    }


# Re-export finite source constants for account-wide scientific introspection.
for _name in (
    "BACKBONES",
    "WEIGHTS",
    "OPTIMIZERS",
    "PRECISIONS",
    "CLASSIFIER_MODES",
    "CLASSIFIER_METRICS",
    "SAMPLING_STRATEGIES",
    "SUPCON_MODES",
    "MIM_LOSSES",
    "MIM_SCHEDULERS",
    "CLASSES",
):
    globals()[_name] = getattr(base, _name)


__all__ = ["SCHEMA", "audit_source_contract", "catalog_metadata", "iter_jobs"]
