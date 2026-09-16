#!/usr/bin/env python3
"""Execute one ESD supervised/refinement lockstep cohort lane.

This is the physical worker for the source-proven metric-learning surfaces.  It
reuses the repository's native ``run_experiment`` lifecycle for every logical
model and only intercepts the already-public one-batch SupCon/classifier seams.
No loss, optimizer, SAM, AMP, scheduler, validation, phase or early-stopping
science is reimplemented here.

Crash consistency is cohort-transactional: each native step writes its exact
``step_last.pt`` before returning.  Before a shared-batch round this worker copies
every resident model's previous step checkpoint into a transaction snapshot and
writes an inflight journal.  The journal is committed only after every live model
has completed the round.  A later process that finds an inflight journal restores
all members to that common pre-round frontier before admitting another batch.

Phase-0 MIM is intentionally rejected here.  It has exact native resume but does
not yet expose the one-batch step function required for honest lockstep sharing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import threading
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "training_control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

SCHEMA = "esd-supervised-cohort-worker/v1"
SUPPORTED_WORKFLOWS = frozenset({"supcon_then_ce", "ce_only", "final_refine"})


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _configure_backend(requested: str) -> str:
    requested = str(requested).strip().lower()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"unsupported ESD cohort backend {requested!r}")
    if requested == "cpu":
        resolved = "cpu"
    else:
        available = _gpu_available()
        if requested == "gpu" and not available:
            raise RuntimeError("ESD cohort GPU backend required but CUDA is unavailable")
        resolved = "gpu" if available else "cpu"
    if resolved == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HIP_VISIBLE_DEVICES"] = ""
        os.environ["ROCR_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["CUPY_ACCELERATORS"] = ""
    os.environ["ESD_TRAINING_BACKEND"] = resolved
    os.environ["ESD_TRAINING_BACKEND_RESOLVED"] = resolved
    return resolved


def _child_argv(command: list[str]) -> list[str]:
    values = [str(value) for value in command]
    if "--" in values:
        values = values[values.index("--") + 1 :]
    if values and Path(values[0]).name.lower().startswith("python"):
        values = values[1:]
    if not values:
        raise ValueError("ESD logical command has no native child entrypoint")
    entry = values[0].replace("\\", "/")
    allowed = {
        "training_control/run_metric_learning_exact_v1.py",
        "scripts/metric_learning_pipeline.py",
    }
    if entry not in allowed and not any(entry.endswith("/" + item) for item in allowed):
        raise ValueError(f"unsupported ESD cohort child entrypoint: {entry!r}")
    return values[1:]


def _option(argv: list[str], name: str, default: str = "") -> str:
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            return str(argv[index + 1])
    return default


def _step_checkpoint_for(row: Mapping[str, Any]) -> Path:
    argv = _child_argv(list(row.get("command") or []))
    output = _option(argv, "--output-dir", "Results/metric_learning_experiment")
    return (ROOT / output).resolve() / "step_last.pt"


def _lane(compiled: Mapping[str, Any], lane_key: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    matches: list[dict[str, Any]] = []
    for group in compiled["dataset_groups"]:
        for lane in group["lanes"]:
            if str(lane["lane_key"]) == lane_key:
                row = dict(lane)
                row["dataset_key"] = group["dataset_key"]
                row["group_index"] = group["group_index"]
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"ESD lane key {lane_key!r} resolved to {len(matches)} lanes")
    lane = matches[0]
    if str(lane["workflow_cadence"]) not in SUPPORTED_WORKFLOWS:
        raise ValueError(
            f"ESD lane {lane_key} uses {lane['workflow_cadence']!r}; Phase-0 MIM "
            "must remain on its native exact-resume path until it exposes a batch-step seam"
        )
    logical = {
        job_id: dict(compiled["logical_jobs"][job_id])
        for job_id in lane["logical_job_ids"]
    }
    if not logical:
        raise ValueError(f"ESD lane {lane_key} contains no models")
    return lane, logical


def _capture_rng() -> dict[str, Any]:
    import numpy as np
    import torch
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class CohortCheckpointJournal:
    def __init__(self, *, lane_key: str, checkpoints: Mapping[str, Path], root: Path) -> None:
        self.lane_key = str(lane_key)
        self.checkpoints = {str(key): Path(value) for key, value in checkpoints.items()}
        digest = hashlib.sha256(self.lane_key.encode("utf-8")).hexdigest()[:16]
        self.root = root / digest
        self.journal = self.root / "inflight.json"
        self.snapshots = self.root / "snapshots"

    def recover_if_needed(self) -> None:
        if not self.journal.is_file():
            return
        payload = json.loads(self.journal.read_text(encoding="utf-8"))
        if payload.get("status") != "inflight":
            self._clear()
            return
        members = payload.get("members") or {}
        for consumer_id, record in members.items():
            target = self.checkpoints.get(str(consumer_id))
            if target is None:
                raise RuntimeError(f"journal references unknown ESD consumer {consumer_id!r}")
            existed = bool(record.get("existed"))
            snapshot = self.root / str(record.get("snapshot"))
            if existed:
                if not snapshot.is_file():
                    raise RuntimeError(f"missing ESD cohort recovery snapshot: {snapshot}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snapshot, target)
            elif target.exists():
                target.unlink()
        self._clear()

    def before_round(self, generation: int, arrivals: Mapping[str, Any]) -> None:
        if self.journal.exists():
            raise RuntimeError("previous ESD cohort transaction journal was not cleared")
        self.snapshots.mkdir(parents=True, exist_ok=True)
        members: dict[str, Any] = {}
        for consumer_id in sorted(arrivals):
            target = self.checkpoints[consumer_id]
            snapshot = self.snapshots / f"{consumer_id}.pt"
            existed = target.is_file()
            if existed:
                shutil.copy2(target, snapshot)
            elif snapshot.exists():
                snapshot.unlink()
            members[consumer_id] = {
                "checkpoint": str(target),
                "existed": existed,
                "snapshot": str(snapshot.relative_to(self.root)),
            }
        _atomic_json(
            self.journal,
            {
                "schema": SCHEMA,
                "status": "inflight",
                "lane_key": self.lane_key,
                "generation": int(generation),
                "members": members,
            },
        )

    def after_round(self, generation: int, arrivals: Mapping[str, Any]) -> None:
        if not self.journal.is_file():
            raise RuntimeError("ESD cohort round completed without an inflight journal")
        payload = json.loads(self.journal.read_text(encoding="utf-8"))
        if int(payload.get("generation", -1)) != int(generation):
            raise RuntimeError("ESD cohort transaction generation drift")
        payload["status"] = "committed"
        _atomic_json(self.journal, payload)
        self._clear()

    def _clear(self) -> None:
        if self.journal.exists():
            self.journal.unlink()
        if self.snapshots.exists():
            shutil.rmtree(self.snapshots)


def _load_pipeline(consumer_id: str):
    from esd_deterministic_data_v1 import install
    path = ROOT / "scripts" / "metric_learning_pipeline.py"
    safe = hashlib.sha256(consumer_id.encode("utf-8")).hexdigest()[:20]
    name = f"_esd_cohort_pipeline_{safe}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import ESD metric-learning pipeline for {consumer_id}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    install(module)
    return module


def _run_member(*, consumer_id: str, row: Mapping[str, Any], coordinator: Any, errors: list[BaseException]) -> None:
    from esd_lockstep_orchestrator_v1 import (
        inspectable_limited_batches,
        make_native_step_proxy,
    )
    try:
        pipeline = _load_pipeline(consumer_id)
        native_supcon = pipeline.train_supcon_steps
        native_classifier = pipeline.train_classifier_steps
        pipeline.limited_batches = inspectable_limited_batches
        pipeline.train_supcon_steps = make_native_step_proxy(
            coordinator=coordinator,
            consumer_id=consumer_id,
            stage="supcon",
            native_step=native_supcon,
        )
        pipeline.train_classifier_steps = make_native_step_proxy(
            coordinator=coordinator,
            consumer_id=consumer_id,
            stage="classifier",
            native_step=native_classifier,
        )
        argv = _child_argv(list(row.get("command") or []))
        parser = pipeline.build_parser()
        args = parser.parse_args(argv)
        # Exact restart of a physical cohort always prefers the native batch-step
        # checkpoint in the declared output directory when one already exists.
        step_checkpoint = Path(args.output_dir) / "step_last.pt"
        if not getattr(args, "resume_checkpoint", "") and step_checkpoint.is_file():
            args.resume_checkpoint = str(step_checkpoint)
        code = int(pipeline.run_experiment(args))
        if code != 0:
            raise RuntimeError(f"native ESD trainer {consumer_id} returned {code}")
    except BaseException as exc:
        errors.append(exc)
        coordinator.abort(exc)
    finally:
        coordinator.finish(consumer_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-key", required=True)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "gpu"),
        default=(os.environ.get("ESD_TRAINING_BACKEND") or "auto").strip().lower(),
    )
    parser.add_argument(
        "--transaction-root",
        type=Path,
        default=Path(".training_control/cohorts/esd-supervised-v1"),
    )
    parser.add_argument(
        "--max-resident-models",
        type=int,
        default=int(os.environ.get("ESD_COHORT_MAX_RESIDENT_MODELS", "0") or 0),
        help="Fail closed above this resident count; 0 means no artificial cap.",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    resolved_backend = _configure_backend(args.backend)

    # Import cohort machinery only after CPU/GPU environment resolution.
    import esd_dataset_cohort_catalog_v2 as catalog
    from esd_lockstep_orchestrator_v1 import LockstepCoordinator
    from esd_shared_batch_broker_v2 import SharedBatchBroker

    compiled = catalog.compile_catalog()
    lane, logical = _lane(compiled, args.lane_key)
    consumer_ids = sorted(logical)
    if args.max_resident_models > 0 and len(consumer_ids) > args.max_resident_models:
        raise RuntimeError(
            f"ESD lane has {len(consumer_ids)} models but resident cap is "
            f"{args.max_resident_models}; residency-window execution has not been "
            "declared transaction-safe for this worker, so it refuses to split the batch"
        )

    checkpoints = {consumer_id: _step_checkpoint_for(row) for consumer_id, row in logical.items()}
    journal = CohortCheckpointJournal(
        lane_key=args.lane_key,
        checkpoints=checkpoints,
        root=(ROOT / args.transaction_root).resolve(),
    )
    journal.recover_if_needed()

    summary = {
        "schema": SCHEMA,
        "lane_key": args.lane_key,
        "dataset_key": lane["dataset_key"],
        "group_index": lane["group_index"],
        "workflow_cadence": lane["workflow_cadence"],
        "classifier_phase_plan": lane["classifier_phase_plan"],
        "physical_batch_contract": lane["physical_batch_contract"],
        "model_count": len(consumer_ids),
        "requested_backend": args.backend,
        "resolved_backend": resolved_backend,
        "phase0_supported": False,
        "transactional_round_rollback": True,
        "residency_window_safe": False,
        "execution_claim_emitted": not args.audit_only,
    }
    if args.audit_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    broker = SharedBatchBroker(device="cuda" if resolved_backend == "gpu" else "cpu")
    coordinator = LockstepCoordinator(
        consumer_ids=consumer_ids,
        lane_key=args.lane_key,
        broker=broker,
        capture_rng=_capture_rng,
        restore_rng=_restore_rng,
        before_round=journal.before_round,
        after_round=journal.after_round,
    )
    errors: list[BaseException] = []
    threads: list[threading.Thread] = []
    try:
        # Start sequentially through first arrival.  Native trainers seed/model-build
        # using process-global RNG; serial admission prevents initialization races.
        for consumer_id in consumer_ids:
            thread = threading.Thread(
                target=_run_member,
                kwargs={
                    "consumer_id": consumer_id,
                    "row": logical[consumer_id],
                    "coordinator": coordinator,
                    "errors": errors,
                },
                name=f"esd-cohort-{consumer_id}",
                daemon=False,
            )
            thread.start()
            threads.append(thread)
            event = coordinator.first_arrival_event(consumer_id)
            # A job can legitimately complete before a training step (for example
            # a fully completed exact resume); then finish() removes it from active
            # membership and no first-arrival wait is required.
            while thread.is_alive() and not event.wait(timeout=0.1):
                if errors:
                    break
            if errors:
                break
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError("ESD physical cohort failed") from errors[0]
        broker.assert_batch_boundary_clean()
    finally:
        for thread in threads:
            if thread.is_alive():
                coordinator.abort(RuntimeError("ESD cohort worker shutting down"))
        for thread in threads:
            thread.join(timeout=5)
        broker.close()

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
