#!/usr/bin/env python3
"""Run one ESD trainer while bridging native step checkpoints to OPF pressure control.

The ESD trainers already persist restart-exact step checkpoints (model, optimizer,
scheduler/scaler and in-epoch progress) and auto-resume them.  This wrapper adds the
missing process-level contract: when the OPF runtime publishes
``TRAINING_CHECKPOINT_REQUEST_FILE``, acknowledge only after a native resumable
checkpoint exists.  The scheduler remains responsible for pause/termination and
relaunch; this file never reimplements admission or resource scheduling.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--completion-marker", default="")
    parser.add_argument("--poll-seconds", type=float, default=0.20)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_candidates(root: Path) -> list[Path]:
    preferred = [root / "step_last.pt", root / "last.pt"]
    output = [path for path in preferred if path.is_file()]
    for path in sorted(root.glob("step-*.pt")):
        if path.is_file() and path not in output:
            output.append(path)
    return output


def _latest_checkpoint(root: Path) -> Path | None:
    candidates = _checkpoint_candidates(root)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.stat().st_size, path.name))


def main() -> int:
    args = _parser().parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a child trainer command is required after --")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be > 0")

    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    request_raw = (os.environ.get("TRAINING_CHECKPOINT_REQUEST_FILE") or "").strip()
    ack_raw = (os.environ.get("TRAINING_CHECKPOINT_ACK_FILE") or "").strip()
    request_path = Path(request_raw) if request_raw else None
    ack_path = Path(ack_raw) if ack_raw else None
    acknowledged_request_mtime: int | None = None

    child = subprocess.Popen(command)
    try:
        while child.poll() is None:
            if request_path is not None and request_path.is_file() and ack_path is not None:
                request_mtime = request_path.stat().st_mtime_ns
                if acknowledged_request_mtime != request_mtime:
                    checkpoint = _latest_checkpoint(checkpoint_root)
                    if checkpoint is not None:
                        _atomic_json(
                            ack_path,
                            {
                                "schema_version": 1,
                                "status": "checkpoint_ready",
                                "checkpoint": str(checkpoint.resolve()),
                                "checkpoint_size": checkpoint.stat().st_size,
                                "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
                                "request_mtime_ns": request_mtime,
                                "restart_exact": True,
                                "trainer_owns_checkpoint_state": True,
                                "scheduler_owns_process_control": True,
                            },
                        )
                        acknowledged_request_mtime = request_mtime
            time.sleep(args.poll_seconds)
        return_code = int(child.wait())
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    if return_code == 0 and args.completion_marker:
        marker = Path(args.completion_marker)
        _atomic_json(
            marker,
            {
                "schema_version": 1,
                "status": "complete",
                "checkpoint_root": str(checkpoint_root),
                "latest_checkpoint": (
                    None if _latest_checkpoint(checkpoint_root) is None else str(_latest_checkpoint(checkpoint_root))
                ),
                "restart_exact": True,
            },
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
