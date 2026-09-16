#!/usr/bin/env python3
"""Pressure-aware process bridge for one ESD supervised physical cohort.

The native cohort worker owns shared-batch/model science and exact round rollback.
This bridge owns OPF process semantics: it launches the worker, watches the
scheduler checkpoint request, and acknowledges only while the cohort transaction
journal is clean.  The acknowledgement points at a manifest of every logical
member's restart checkpoint/completion frontier, never at an invented monolithic
model checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "training_control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

import esd_dataset_cohort_catalog_v2 as catalog  # noqa: E402
import run_esd_supervised_cohort_v1 as worker  # noqa: E402

SCHEMA = "esd-supervised-cohort-pressure/v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-key", required=True)
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--transaction-root",
        type=Path,
        default=Path(".training_control/cohorts/esd-supervised-v1"),
    )
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--max-resident-models", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=0.20)
    return parser.parse_args()


def _lane_state(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path, dict[str, Path]]:
    compiled = catalog.compile_catalog()
    lane, logical = worker._lane(compiled, args.lane_key)
    digest = hashlib.sha256(str(args.lane_key).encode("utf-8")).hexdigest()[:16]
    transaction_dir = (ROOT / args.transaction_root / digest).resolve()
    checkpoints = {
        consumer_id: worker._step_checkpoint_for(row)
        for consumer_id, row in logical.items()
    }
    return lane, logical, transaction_dir, checkpoints


def _completion_ready(row: dict[str, Any]) -> bool:
    artifacts = [ROOT / str(path) for path in (row.get("completion_artifacts") or [])]
    return bool(artifacts) and all(path.exists() for path in artifacts)


def _frontier_manifest(
    *,
    lane: dict[str, Any],
    logical: dict[str, dict[str, Any]],
    checkpoints: dict[str, Path],
    transaction_dir: Path,
) -> dict[str, Any] | None:
    inflight = transaction_dir / "inflight.json"
    if inflight.exists():
        return None
    members: dict[str, Any] = {}
    for consumer_id, row in logical.items():
        checkpoint = checkpoints[consumer_id]
        completed = _completion_ready(row)
        if not checkpoint.is_file() and not completed:
            return None
        members[consumer_id] = {
            "step_checkpoint": str(checkpoint) if checkpoint.is_file() else None,
            "step_checkpoint_size": checkpoint.stat().st_size if checkpoint.is_file() else None,
            "completed": completed,
            "completion_artifacts": list(row.get("completion_artifacts") or []),
        }
    return {
        "schema": SCHEMA,
        "status": "clean_shared_batch_frontier",
        "lane_key": lane["lane_key"],
        "dataset_key": lane["dataset_key"],
        "workflow_cadence": lane["workflow_cadence"],
        "classifier_phase_plan": lane["classifier_phase_plan"],
        "physical_batch_contract": lane["physical_batch_contract"],
        "members": members,
        "restart_exact": True,
        "shared_cursor_committed": True,
        "trainer_owned_member_checkpoints": True,
    }


def main() -> int:
    args = _parse()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be > 0")
    lane, logical, transaction_dir, checkpoints = _lane_state(args)

    certificate = args.certificate if args.certificate.is_absolute() else ROOT / args.certificate
    manifest_path = transaction_dir / "committed_frontier.json"
    request_raw = (os.environ.get("TRAINING_CHECKPOINT_REQUEST_FILE") or "").strip()
    ack_raw = (os.environ.get("TRAINING_CHECKPOINT_ACK_FILE") or "").strip()
    request_path = Path(request_raw) if request_raw else None
    ack_path = Path(ack_raw) if ack_raw else None
    acknowledged_mtime: int | None = None

    command = [
        sys.executable,
        str(CONTROL / "run_esd_supervised_cohort_v1.py"),
        "--lane-key", args.lane_key,
        "--backend", args.backend,
        "--transaction-root", str(args.transaction_root),
    ]
    if args.max_resident_models > 0:
        command.extend(["--max-resident-models", str(args.max_resident_models)])

    child = subprocess.Popen(command, cwd=ROOT)
    try:
        while child.poll() is None:
            if request_path is not None and ack_path is not None and request_path.is_file():
                request_mtime = request_path.stat().st_mtime_ns
                if request_mtime != acknowledged_mtime:
                    frontier = _frontier_manifest(
                        lane=lane,
                        logical=logical,
                        checkpoints=checkpoints,
                        transaction_dir=transaction_dir,
                    )
                    if frontier is not None:
                        frontier["request_mtime_ns"] = request_mtime
                        _atomic_json(manifest_path, frontier)
                        _atomic_json(
                            ack_path,
                            {
                                "schema_version": 1,
                                "status": "checkpoint_ready",
                                "checkpoint": str(manifest_path),
                                "request_mtime_ns": request_mtime,
                                "restart_exact": True,
                                "shared_raw_batch_cursor": True,
                                "cohort_transaction_clean": True,
                                "scheduler_owns_process_control": True,
                            },
                        )
                        acknowledged_mtime = request_mtime
            time.sleep(args.poll_seconds)
        code = int(child.wait())
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    if code != 0:
        return code

    frontier = _frontier_manifest(
        lane=lane,
        logical=logical,
        checkpoints=checkpoints,
        transaction_dir=transaction_dir,
    )
    if frontier is None:
        # Successful native completion should have either member completion
        # artifacts or an exact step checkpoint for every logical child.
        raise RuntimeError("ESD cohort exited successfully without a complete restart frontier")
    _atomic_json(manifest_path, frontier)
    _atomic_json(
        certificate,
        {
            **frontier,
            "status": "complete",
            "physical_worker": "training_control/run_esd_supervised_cohort_v1.py",
            "pressure_bridge": "training_control/esd_supervised_cohort_pressure_v1.py",
            "requested_backend": args.backend,
            "model_count": len(logical),
            "certificate": str(certificate),
            "execution_claim_emitted": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
