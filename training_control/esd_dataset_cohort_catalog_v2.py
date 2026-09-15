#!/usr/bin/env python3
"""Execution-safe refinement of the ESD dataset cohort catalog.

v1 proves lossless dataset-parent grouping.  v2 keeps that one-to-one scientific
inventory unchanged but refines the *physical lockstep lane* with two additional
coordinates that materially affect whether models can consume the same next
batch at the same time:

* workflow cadence (SupCon+CE, CE-only, Phase-0 MIM, final refinement); and
* native/requested physical batch cardinality.

This prevents a scientifically invalid optimization where two jobs read the same
folder tree but are in different training phases, use a different number of
views, or request different physical batch sizes.  Model architecture,
initialization, optimizer, precision and objective weights remain outside stream
identity when they do not change example selection.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "training_control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

import esd_dataset_cohort_catalog_v1 as v1  # noqa: E402

SCHEMA = "esd-dataset-cohort-catalog/v2"
OVERLAP_KEY = v1.OVERLAP_KEY


def _child_command(command: Sequence[str]) -> list[str]:
    values = [str(value) for value in command]
    try:
        index = values.index("--")
    except ValueError:
        return values
    return values[index + 1 :]


def _option(command: Sequence[str], name: str) -> str | None:
    values = list(command)
    for index, token in enumerate(values):
        if token == name:
            if index + 1 >= len(values) or str(values[index + 1]).startswith("--"):
                return "true"
            return str(values[index + 1])
    return None


def _workflow(row: Mapping[str, Any]) -> str:
    surface = str(row.get("surface") or "")
    if surface == "phase0_mim":
        return "phase0_mim"
    if surface == "final_refine":
        return "final_refine"
    if surface == "supervised_metric":
        return "supcon_then_ce" if row.get("supcon_enabled") is True else "ce_only"
    raise ValueError(f"{row.get('job_id')}: unsupported ESD workflow surface {surface!r}")


def _native_batch_contract(row: Mapping[str, Any]) -> str:
    child = _child_command(row.get("command") or [])
    explicit = _option(child, "--batch-size")
    if explicit is not None:
        try:
            value = int(explicit)
        except ValueError as exc:
            raise ValueError(f"{row.get('job_id')}: invalid --batch-size {explicit!r}") from exc
        if value <= 0:
            raise ValueError(f"{row.get('job_id')}: --batch-size must be positive")
        return f"explicit:{value}"
    # Omitted defaults are kept symbolic rather than guessed here.  All members
    # of one workflow call the same native parser, so this is an exact equality
    # contract while leaving the parser itself authoritative for the numeric
    # default.  The runtime adapter resolves and records that numeric value before
    # any model is admitted to a physical lane.
    return f"native-default:{_workflow(row)}"


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def compile_catalog() -> dict[str, Any]:
    base = v1.compile_catalog()
    base_groups = {str(group["dataset_key"]): group for group in base["dataset_groups"]}
    base_lane_contract: dict[tuple[str, str], Mapping[str, Any]] = {}
    for group in base["dataset_groups"]:
        for lane in group["lanes"]:
            base_lane_contract[(str(group["dataset_key"]), str(lane["lane_key"]))] = dict(lane["data_contract"])

    groups: dict[str, dict[str, Any]] = {}
    logical: dict[str, dict[str, Any]] = {}
    for job_id, raw in base["logical_jobs"].items():
        row = dict(raw)
        parent_key = str(row["dataset_parent"])
        base_lane_key = str(row["lane_key"])
        data_contract = base_lane_contract[(parent_key, base_lane_key)]
        workflow = _workflow(row)
        batch_contract = _native_batch_contract(row)
        lane_contract = {
            "data_contract": data_contract,
            "workflow_cadence": workflow,
            "physical_batch_contract": batch_contract,
        }
        lane_key = _digest(lane_contract)

        source_group = base_groups[parent_key]
        group = groups.setdefault(
            parent_key,
            {
                "dataset_key": parent_key,
                "dataset_roots": list(source_group.get("dataset_roots") or []),
                "overlap": bool(source_group.get("overlap")),
                "lanes": {},
                "logical_job_ids": [],
                "families": set(),
            },
        )
        lane = group["lanes"].setdefault(
            lane_key,
            {
                "lane_key": lane_key,
                "base_lane_key": base_lane_key,
                "lane_contract": lane_contract,
                "workflow_cadence": workflow,
                "physical_batch_contract": batch_contract,
                "logical_job_ids": [],
                "families": set(),
            },
        )
        family = str(row.get("family") or "unclassified")
        lane["logical_job_ids"].append(str(job_id))
        lane["families"].add(family)
        group["logical_job_ids"].append(str(job_id))
        group["families"].add(family)
        row["lane_key_v1"] = base_lane_key
        row["lane_key"] = lane_key
        row["workflow_cadence"] = workflow
        row["physical_batch_contract"] = batch_contract
        logical[str(job_id)] = row

    parent_rows: list[dict[str, Any]] = []
    for group in groups.values():
        lanes: list[dict[str, Any]] = []
        for lane in group["lanes"].values():
            lane["logical_job_ids"] = sorted(lane["logical_job_ids"])
            lane["families"] = sorted(lane["families"])
            lane["model_family_count"] = len(lane["families"])
            lane["model_count"] = len(lane["logical_job_ids"])
            lanes.append(lane)
        lanes.sort(
            key=lambda lane: (
                -int(lane["model_family_count"]),
                -int(lane["model_count"]),
                str(lane["workflow_cadence"]),
                str(lane["physical_batch_contract"]),
                str(lane["lane_key"]),
            )
        )
        group["lanes"] = lanes
        group["logical_job_ids"] = sorted(group["logical_job_ids"])
        group["families"] = sorted(group["families"])
        group["model_family_count"] = len(group["families"])
        group["model_count"] = len(group["logical_job_ids"])
        group["lane_count"] = len(lanes)
        parent_rows.append(group)

    parent_rows.sort(
        key=lambda group: (
            bool(group["overlap"]),
            -int(group["model_family_count"]),
            -int(group["model_count"]),
            str(group["dataset_key"]),
        )
    )
    for index, group in enumerate(parent_rows, start=1):
        group["group_index"] = index

    expected = set(base["logical_jobs"])
    observed = [
        job_id
        for group in parent_rows
        for lane in group["lanes"]
        for job_id in lane["logical_job_ids"]
    ]
    if set(observed) != expected or len(observed) != len(expected):
        raise RuntimeError(
            "ESD v2 physical catalog is not a one-to-one refinement of v1: "
            f"missing={sorted(expected - set(observed))[:20]}, "
            f"extra={sorted(set(observed) - expected)[:20]}, "
            f"duplicates={len(observed) - len(set(observed))}"
        )

    return {
        "schema": SCHEMA,
        "base_schema": base["schema"],
        "repository": base["repository"],
        "logical_training_jobs": len(logical),
        "dataset_groups": parent_rows,
        "logical_jobs": logical,
        "overlap_group_last": all(not group["overlap"] for group in parent_rows[:-1]) if parent_rows else True,
        "group_order": "descending_distinct_model_families_then_models_overlap_last",
        "lane_rule": "canonical_data_contract_plus_workflow_cadence_plus_physical_batch_contract",
        "uniform_batch_required_inside_lane": True,
        "effective_batch_preserved_or_lane_split": True,
        "cpu_variant_required": True,
        "gpu_first_variant_required": True,
        "auto_backend": "gpu_if_available_else_cpu",
        "execution_claim_emitted": False,
    }


def metadata() -> dict[str, Any]:
    compiled = compile_catalog()
    groups = compiled["dataset_groups"]
    workflows: dict[str, int] = defaultdict(int)
    for group in groups:
        for lane in group["lanes"]:
            workflows[str(lane["workflow_cadence"])] += int(lane["model_count"])
    return {
        "schema": SCHEMA,
        "base_schema": compiled["base_schema"],
        "logical_training_jobs": compiled["logical_training_jobs"],
        "dataset_group_count": len(groups),
        "compatibility_lane_count": sum(int(group["lane_count"]) for group in groups),
        "workflow_model_counts": dict(sorted(workflows.items())),
        "largest_dataset_group_models": max((int(group["model_count"]) for group in groups), default=0),
        "largest_lane_models": max(
            (int(lane["model_count"]) for group in groups for lane in group["lanes"]),
            default=0,
        ),
        "overlap_groups": sum(bool(group["overlap"]) for group in groups),
        "lossless_refinement": True,
        "execution_claim_emitted": False,
    }


def iter_dataset_groups() -> Iterator[dict[str, Any]]:
    yield from compile_catalog()["dataset_groups"]


if __name__ == "__main__":
    print(json.dumps(metadata(), indent=2, sort_keys=True))


__all__ = ["SCHEMA", "OVERLAP_KEY", "compile_catalog", "iter_dataset_groups", "metadata"]
