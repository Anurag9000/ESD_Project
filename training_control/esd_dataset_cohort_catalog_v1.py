#!/usr/bin/env python3
"""Lossless dataset-centric physical training catalog for ESD.

This module does *not* change the scientific job inventory.  The immutable
``esd_scientific_authority_v1`` / executable v2 catalog remain the source of
truth.  It classifies every training transaction into one dataset parent and a
source-proven compatibility lane so a physical cohort runner can share batches
without pretending incompatible sampling/preprocessing streams are identical.

The user-level grouping rule is represented literally:

* one parent for each single training dataset;
* multi-dataset/overlap parents are sorted last;
* parents are ordered by descending distinct model-family count;
* within a dataset parent, lanes separate incompatible sample streams (sampling
  strategy, supervised vs MIM vs final-refinement, initialization dependency,
  and any command-line data-shaping options);
* logical jobs are mapped exactly once and are never silently dropped.

A lane is the smallest unit allowed to share one synchronized raw-batch cursor.
Different model/loss/optimizer/precision choices can share a lane because those
choices do not select different examples.  Sampling strategy or data-shaping
choices *do* split a lane because they change which example is next.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "training_control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

import esd_scientific_job_catalog_v2 as catalog  # noqa: E402

SCHEMA = "esd-dataset-cohort-catalog/v1"
OVERLAP_KEY = "__overlap__"
DEFAULT_DATASET_ROOT = os.environ.get("ESD_DATASET_ROOT", "Dataset_Final")

# Options that alter source identity, sample ordering, image construction, or the
# exact train stream.  Model/optimizer/loss switches deliberately do not appear.
_DATA_OPTIONS = frozenset(
    {
        "--dataset-root",
        "--split-mode",
        "--split-manifest",
        "--train-fraction",
        "--image-size",
        "--augment-repeats",
        "--gaussian-sigmas",
        "--sampling-strategy",
        "--seed",
        "--class-mapping",
        "--train-only",
        "--max-train-samples",
        "--exclude-sources",
        "--include-sources",
        "--phase0-encoder-checkpoint",
    }
)
_FLAG_OPTIONS = frozenset({"--train-only"})


def _normalized_command(job: Mapping[str, Any]) -> list[str]:
    command = [str(value) for value in (job.get("command") or [])]
    if not command:
        raise ValueError(f"{job.get('id')}: empty command")
    # Scientific jobs are wrapped by esd_pressure_runner_v1.py.  The child begins
    # after the explicit '--' separator.  Non-wrapped commands remain untouched.
    try:
        cut = command.index("--")
    except ValueError:
        return command
    child = command[cut + 1 :]
    if not child:
        raise ValueError(f"{job.get('id')}: pressure wrapper has no child command")
    return child


def _option_map(command: Sequence[str]) -> dict[str, list[str | bool]]:
    out: dict[str, list[str | bool]] = defaultdict(list)
    i = 0
    while i < len(command):
        token = str(command[i])
        if token in _FLAG_OPTIONS:
            out[token].append(True)
            i += 1
            continue
        if token.startswith("--"):
            if i + 1 < len(command) and not str(command[i + 1]).startswith("--"):
                out[token].append(str(command[i + 1]))
                i += 2
            else:
                out[token].append(True)
                i += 1
            continue
        i += 1
    return dict(out)


def _first(options: Mapping[str, Sequence[str | bool]], name: str, default: str = "") -> str:
    values = options.get(name) or ()
    if not values:
        return default
    return str(values[-1])


def _training_surface(job: Mapping[str, Any], child: Sequence[str]) -> str:
    normalized = [value.replace("\\", "/") for value in child]
    if "scripts/train_phase0_mim.py" in normalized:
        return "phase0_mim"
    if "training_control/run_metric_learning_exact_v1.py" in normalized or "scripts/metric_learning_pipeline.py" in normalized:
        options = _option_map(child)
        return "final_refine" if "--train-only" in options else "supervised_metric"
    raise ValueError(
        f"{job.get('id')}: unclassified retained training entrypoint: {shlex.join(list(child))}"
    )


def _dataset_roots(job: Mapping[str, Any], child: Sequence[str]) -> tuple[str, ...]:
    options = _option_map(child)
    roots: list[str] = []
    root = _first(options, "--dataset-root", DEFAULT_DATASET_ROOT).strip()
    if root:
        roots.append(root.replace("\\", "/"))
    # Keep this generic for future catalog growth.  A training command that later
    # grows multiple explicit roots will automatically enter the final overlap
    # parent rather than being falsely merged into a single-dataset stream.
    for option in ("--secondary-dataset-root", "--aux-dataset-root", "--replay-dataset-root"):
        value = _first(options, option, "").strip()
        if value:
            roots.append(value.replace("\\", "/"))
    roots = sorted(dict.fromkeys(roots))
    if not roots:
        raise ValueError(f"{job.get('id')}: training job has no dataset root")
    return tuple(roots)


def _canonical_data_contract(job: Mapping[str, Any], child: Sequence[str]) -> dict[str, Any]:
    options = _option_map(child)
    surface = _training_surface(job, child)
    selected: dict[str, Any] = {}
    for name in sorted(_DATA_OPTIONS):
        if name in _FLAG_OPTIONS:
            if name in options:
                selected[name] = True
            continue
        values = options.get(name)
        if values:
            selected[name] = [str(value) for value in values]
    # Defaults that are scientifically relevant but may be omitted by the catalog
    # are represented explicitly.  This prevents an omitted default and an
    # explicit equivalent value from being classified as unrelated streams.
    selected.setdefault("--dataset-root", [DEFAULT_DATASET_ROOT])
    selected.setdefault("--seed", [os.environ.get("ESD_CENTRAL_SEED", "42")])
    if surface in {"supervised_metric", "final_refine"}:
        selected.setdefault("--sampling-strategy", [str(job.get("sampling_strategy") or "balanced")])
    return {
        "surface": surface,
        "dataset_roots": list(_dataset_roots(job, child)),
        "data_options": selected,
        "class_order": list(job.get("class_order") or catalog.CLASSES),
        "dataset_replay_contract": job.get("dataset_replay_contract"),
    }


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def logical_training_jobs() -> list[dict[str, Any]]:
    rows = [dict(job) for job in catalog.iter_jobs() if bool(job.get("is_training_job"))]
    ids = [str(row.get("id") or "") for row in rows]
    if any(not value for value in ids):
        raise ValueError("training catalog contains an empty job id")
    if len(ids) != len(set(ids)):
        raise ValueError("training catalog contains duplicate job ids")
    return rows


def compile_catalog() -> dict[str, Any]:
    jobs = logical_training_jobs()
    logical: dict[str, dict[str, Any]] = {}
    parents: dict[str, dict[str, Any]] = {}

    for job in jobs:
        job_id = str(job["id"])
        child = _normalized_command(job)
        roots = _dataset_roots(job, child)
        contract = _canonical_data_contract(job, child)
        lane_key = _digest(contract)
        parent_key = roots[0] if len(roots) == 1 else OVERLAP_KEY
        parent = parents.setdefault(
            parent_key,
            {
                "dataset_key": parent_key,
                "dataset_roots": list(roots) if parent_key == OVERLAP_KEY else [parent_key],
                "overlap": parent_key == OVERLAP_KEY,
                "lanes": {},
                "logical_job_ids": [],
                "families": set(),
            },
        )
        if parent_key == OVERLAP_KEY:
            parent["dataset_roots"] = sorted(set(parent["dataset_roots"]) | set(roots))
        lane = parent["lanes"].setdefault(
            lane_key,
            {
                "lane_key": lane_key,
                "data_contract": contract,
                "surface": contract["surface"],
                "logical_job_ids": [],
                "families": set(),
            },
        )
        family = str(job.get("family") or "unclassified")
        lane["logical_job_ids"].append(job_id)
        lane["families"].add(family)
        parent["logical_job_ids"].append(job_id)
        parent["families"].add(family)
        logical[job_id] = {
            "job_id": job_id,
            "family": family,
            "dataset_parent": parent_key,
            "lane_key": lane_key,
            "surface": contract["surface"],
            "dataset_roots": list(roots),
            "backbone": job.get("backbone"),
            "weights": job.get("weights"),
            "optimizer": job.get("optimizer"),
            "precision": job.get("precision"),
            "sampling_strategy": job.get("sampling_strategy"),
            "supcon_enabled": job.get("supcon_enabled"),
            "depends_on": list(job.get("depends_on") or []),
            "completion_artifacts": list(job.get("completion_artifacts") or []),
            "command": list(job.get("command") or []),
        }

    parent_rows: list[dict[str, Any]] = []
    for parent in parents.values():
        lanes = []
        for lane in parent["lanes"].values():
            lane["logical_job_ids"] = sorted(lane["logical_job_ids"])
            lane["families"] = sorted(lane["families"])
            lane["model_family_count"] = len(lane["families"])
            lane["model_count"] = len(lane["logical_job_ids"])
            lanes.append(lane)
        lanes.sort(key=lambda row: (-int(row["model_family_count"]), -int(row["model_count"]), str(row["lane_key"])))
        parent["lanes"] = lanes
        parent["logical_job_ids"] = sorted(parent["logical_job_ids"])
        parent["families"] = sorted(parent["families"])
        parent["model_family_count"] = len(parent["families"])
        parent["model_count"] = len(parent["logical_job_ids"])
        parent["lane_count"] = len(lanes)
        parent_rows.append(parent)

    parent_rows.sort(
        key=lambda row: (
            bool(row["overlap"]),
            -int(row["model_family_count"]),
            -int(row["model_count"]),
            str(row["dataset_key"]),
        )
    )
    for index, parent in enumerate(parent_rows, start=1):
        parent["group_index"] = index

    expected = set(logical)
    observed = {
        job_id
        for parent in parent_rows
        for lane in parent["lanes"]
        for job_id in lane["logical_job_ids"]
    }
    if observed != expected:
        raise RuntimeError(
            "ESD cohort catalog is not lossless: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    appearances: dict[str, int] = defaultdict(int)
    for parent in parent_rows:
        for lane in parent["lanes"]:
            for job_id in lane["logical_job_ids"]:
                appearances[job_id] += 1
    duplicates = sorted(job_id for job_id, count in appearances.items() if count != 1)
    if duplicates:
        raise RuntimeError(f"ESD logical jobs do not map exactly once: {duplicates[:20]}")

    return {
        "schema": SCHEMA,
        "repository": "Anurag9000/ESD_Project",
        "logical_training_jobs": len(logical),
        "dataset_groups": parent_rows,
        "logical_jobs": logical,
        "overlap_group_last": all(
            not row["overlap"] for row in parent_rows[:-1]
        ) if parent_rows else True,
        "group_order": "descending_distinct_model_families_then_models_overlap_last",
        "lane_rule": "share_batch_only_when_canonical_data_contract_matches",
        "cpu_variant_required": True,
        "gpu_first_variant_required": True,
        "auto_backend": "gpu_if_available_else_cpu",
        "execution_claim_emitted": False,
    }


def iter_dataset_groups() -> Iterator[dict[str, Any]]:
    yield from compile_catalog()["dataset_groups"]


def metadata() -> dict[str, Any]:
    compiled = compile_catalog()
    groups = compiled["dataset_groups"]
    return {
        "schema": SCHEMA,
        "logical_training_jobs": compiled["logical_training_jobs"],
        "dataset_group_count": len(groups),
        "compatibility_lane_count": sum(int(group["lane_count"]) for group in groups),
        "largest_dataset_group_models": max((int(group["model_count"]) for group in groups), default=0),
        "largest_lane_models": max(
            (int(lane["model_count"]) for group in groups for lane in group["lanes"]),
            default=0,
        ),
        "overlap_groups": sum(bool(group["overlap"]) for group in groups),
        "lossless": True,
        "execution_claim_emitted": False,
    }


if __name__ == "__main__":
    print(json.dumps(metadata(), indent=2, sort_keys=True))


__all__ = [
    "SCHEMA",
    "OVERLAP_KEY",
    "compile_catalog",
    "iter_dataset_groups",
    "logical_training_jobs",
    "metadata",
]
