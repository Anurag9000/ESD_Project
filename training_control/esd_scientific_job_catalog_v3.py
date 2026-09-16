#!/usr/bin/env python3
"""ESD v3 physical dataset-cohort training authority.

v2 remains the complete logical scientific inventory. This layer replaces only
source-proven supervised/final-refinement metric-learning optimizer jobs with one
physical lockstep parent per execution-safe dataset lane. Phase-0 MIM jobs stay
native because they have exact resume but no public one-batch adapter seam yet.

Every replaced logical job maps to exactly one parent. Dependencies are rewritten
through that map, parent prerequisites are the union of child prerequisites, and
self-dependencies created by coalescing are removed. Dataset parents/lanes retain
the ordering produced by ``esd_dataset_cohort_catalog_v2``: descending family/model
count with overlap last and explicit stream/workflow/batch compatibility lanes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "training_control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

import esd_scientific_job_catalog_v2 as logical  # noqa: E402
import esd_dataset_cohort_catalog_v2 as cohorts  # noqa: E402

SCHEMA = "esd-scientific-job-catalog/v3"
WORKER = "training_control/esd_supervised_cohort_pressure_v1.py"
STATE_ROOT = ".training_control/cohorts/esd-supervised-v1"
SUPPORTED = frozenset({"supcon_then_ce", "ce_only", "final_refine"})


def _hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()[:12]


def _rows() -> list[dict[str, Any]]:
    rows = [dict(row) for row in logical.iter_jobs()]
    ids = [str(row.get("id") or "") for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("ESD v2 logical authority emitted duplicate/empty IDs")
    return rows


def _deps(row: Mapping[str, Any]) -> list[str]:
    value = row.get("depends_on") or []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"job {row.get('id')} has invalid depends_on")
    return [str(item) for item in value]


def _replace_deps(row: dict[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    row["depends_on"] = list(dict.fromkeys(mapping.get(dep, dep) for dep in _deps(row)))
    return row


def _compile() -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    compiled = cohorts.compile_catalog()
    logical_rows = _rows()
    training_by_id = {
        str(row["id"]): row for row in logical_rows if bool(row.get("is_training_job"))
    }
    mapping: dict[str, str] = {}
    parents: list[dict[str, Any]] = []

    for group in compiled["dataset_groups"]:
        lane_index = 0
        for lane in group["lanes"]:
            if str(lane["workflow_cadence"]) not in SUPPORTED:
                continue
            lane_index += 1
            member_ids = [str(value) for value in lane["logical_job_ids"]]
            missing = sorted(set(member_ids) - set(training_by_id))
            if missing:
                raise RuntimeError(f"ESD physical lane references missing logical jobs: {missing[:20]}")
            physical_id = (
                f"esd-cohort:g{int(group['group_index']):03d}:"
                f"l{lane_index:03d}:{_hash(member_ids)}"
            )
            for job_id in member_ids:
                if job_id in mapping:
                    raise RuntimeError(f"ESD logical job mapped to two physical parents: {job_id}")
                mapping[job_id] = physical_id
            parent = dict(lane)
            parent.update(
                {
                    "physical_id": physical_id,
                    "dataset_key": group["dataset_key"],
                    "dataset_roots": list(group.get("dataset_roots") or []),
                    "dataset_group_index": int(group["group_index"]),
                    "dataset_group_overlap": bool(group.get("overlap")),
                }
            )
            parents.append(parent)

    eligible = {
        job_id: dict(compiled["logical_jobs"][job_id])
        for job_id in mapping
    }
    expected_supported = {
        str(job_id)
        for job_id, row in compiled["logical_jobs"].items()
        if str(row.get("workflow_cadence")) in SUPPORTED
    }
    if set(mapping) != expected_supported:
        raise RuntimeError(
            "ESD v3 does not map every supported logical optimizer exactly once: "
            f"missing={sorted(expected_supported - set(mapping))[:20]}, "
            f"extra={sorted(set(mapping) - expected_supported)[:20]}"
        )
    return compiled, mapping, eligible, parents


def iter_jobs() -> Iterator[dict[str, Any]]:
    rows = _rows()
    row_by_id = {str(row["id"]): row for row in rows}
    compiled, mapping, eligible, parents = _compile()

    for raw in rows:
        job_id = str(raw["id"])
        if job_id in eligible:
            continue
        row = _replace_deps(dict(raw), mapping)
        row["catalog_schema"] = SCHEMA
        row["logical_authority"] = "training_control/esd_scientific_job_catalog_v2.py"
        yield row

    for lane in parents:
        member_ids = [str(value) for value in lane["logical_job_ids"]]
        physical_id = str(lane["physical_id"])
        source_rows = [row_by_id[job_id] for job_id in member_ids]
        parent_deps = {
            mapping.get(dep, dep)
            for row in source_rows
            for dep in _deps(row)
        }
        parent_deps.discard(physical_id)
        completion = sorted(
            {
                str(path)
                for row in source_rows
                for path in (row.get("completion_artifacts") or [])
            }
        )
        checkpoints = sorted(
            {
                str(path)
                for row in source_rows
                for path in (row.get("checkpoint_artifacts") or [])
            }
        )
        certificate = (
            "artifacts/training_control/esd_cohort_"
            f"g{int(lane['dataset_group_index']):03d}_{lane['lane_key'][:12]}.json"
        )
        families = sorted({str(row.get("family") or "unclassified") for row in source_rows})
        yield {
            "id": physical_id,
            "command": [
                sys.executable,
                WORKER,
                "--lane-key", str(lane["lane_key"]),
                "--backend", "auto",
                "--transaction-root", STATE_ROOT,
                "--certificate", certificate,
            ],
            "phase": "training",
            "family": "esd/dataset-cohort",
            "is_training_job": True,
            "device_capable": True,
            "cpu_capable": True,
            "gpu_capable": True,
            "depends_on": sorted(parent_deps),
            "dataset": str(lane["dataset_key"]),
            "datasets": list(lane["dataset_roots"]),
            "dataset_group_index": int(lane["dataset_group_index"]),
            "dataset_group_overlap": bool(lane["dataset_group_overlap"]),
            "dataset_group_member_count": len(member_ids),
            "dataset_group_distinct_model_families": len(families),
            "dataset_group_model_families": families,
            "logical_job_ids": member_ids,
            "lane_key": str(lane["lane_key"]),
            "workflow_cadence": str(lane["workflow_cadence"]),
            "classifier_phase_plan": str(lane["classifier_phase_plan"]),
            "physical_batch_contract": str(lane["physical_batch_contract"]),
            "shared_raw_batch": True,
            "shared_compatible_views": True,
            "uniform_batch_size": True,
            "gpu_first_cpu_fallback": True,
            "resume_strategy": "exact_native_step_checkpoint_plus_cohort_round_rollback",
            "checkpoint_contract": {
                "exact_resume": True,
                "native_model_optimizer_scheduler_scaler_rng": True,
                "native_step_checkpoint_per_batch": True,
                "shared_raw_batch_cursor": True,
                "cohort_round_transaction_rollback": True,
                "semantic_early_stopping_native": True,
                "phase_divergence_dynamic_grouping": True,
                "residency_window_safe": False,
            },
            "checkpoint_artifacts": [STATE_ROOT, *checkpoints],
            "completion_artifacts": [*completion, certificate],
            "early_stopping": True,
            "early_stopping_applicable": True,
            "exact_resume_source": "training_control/run_esd_supervised_cohort_v1.py:CohortCheckpointJournal",
            "early_stopping_source": "scripts/metric_learning_pipeline.py:native run_experiment",
            "dataset_cohort": True,
            "group_order_rule": "descending distinct model-family count then models, overlap last; compatibility lanes preserve workflow/phase/batch stream equivalence",
            "logical_authority": "training_control/esd_scientific_job_catalog_v2.py",
            "physical_authority": "training_control/esd_scientific_job_catalog_v3.py",
            "cohort_compiler_authority": "training_control/esd_dataset_cohort_catalog_v2.py",
            "pressure_bridge": WORKER,
            "catalog_schema": SCHEMA,
            "execution_claim_emitted": False,
        }


def catalog_metadata() -> dict[str, Any]:
    compiled, mapping, _eligible, parents = _compile()
    phase0_ids = sorted(
        job_id
        for job_id, row in compiled["logical_jobs"].items()
        if str(row.get("workflow_cadence")) == "phase0_mim"
    )
    return {
        "schema": SCHEMA,
        "repository": "Anurag9000/ESD_Project",
        "logical_authority": "training_control/esd_scientific_job_catalog_v2.py",
        "physical_authority": "training_control/esd_scientific_job_catalog_v3.py",
        "cohort_compiler_authority": "training_control/esd_dataset_cohort_catalog_v2.py",
        "logical_training_jobs": int(compiled["logical_training_jobs"]),
        "cohorted_logical_jobs": len(mapping),
        "physical_supervised_cohorts": len(parents),
        "native_phase0_mim_jobs": phase0_ids,
        "all_supported_logical_jobs_mapped_once": True,
        "phase0_mim_remains_native_until_batch_step_adapter": True,
        "group_order_rule": compiled["group_order"],
        "cpu_variant": True,
        "gpu_first_variant": True,
        "shared_raw_batch": True,
        "shared_compatible_views": True,
        "uniform_batch_size": True,
        "exact_resume": True,
        "residency_window_safe": False,
        "root_activated": True,
        "physical_supervised_refinement_execution_activated": True,
        "physical_phase0_mim_execution_activated": False,
        "execution_claim_emitted": False,
    }


if __name__ == "__main__":
    print(json.dumps(catalog_metadata(), indent=2, sort_keys=True))
