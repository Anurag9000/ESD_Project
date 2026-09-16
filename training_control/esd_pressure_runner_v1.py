#!/usr/bin/env python3
"""Run one ESD trainer while bridging native checkpoints to OPF pressure control.

The wrapper is also the repository-wide backend boundary. ``auto`` is GPU-first
and falls back to CPU, ``gpu`` requires a usable accelerator, and ``cpu`` hides
CUDA/HIP devices before the child imports torch/CuPy/JAX/TensorFlow.  Scientific
trainers continue to own model/loss/optimizer/scheduler/scaler/checkpoint state;
this wrapper owns only process/backend/pressure semantics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--completion-marker", default="")
    parser.add_argument("--poll-seconds", type=float, default=0.20)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "gpu"),
        default=(os.environ.get("ESD_TRAINING_BACKEND") or "auto").strip().lower(),
    )
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


def _gpu_available() -> bool:
    """Probe cheaply without making a GPU allocation in the training process."""
    try:
        import torch  # type: ignore

        if bool(torch.cuda.is_available()):
            return True
    except Exception:
        pass
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            probe = subprocess.run(
                [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
                check=False,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                return True
        except Exception:
            pass
    return False


def _resolve_backend(requested: str) -> str:
    requested = str(requested).strip().lower()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"unsupported ESD backend {requested!r}")
    available = _gpu_available()
    if requested == "gpu" and not available:
        raise RuntimeError("ESD GPU backend was required but no usable GPU was detected")
    return "gpu" if requested == "gpu" or (requested == "auto" and available) else "cpu"


def _child_environment(resolved_backend: str) -> dict[str, str]:
    env = dict(os.environ)
    env["ESD_TRAINING_BACKEND_RESOLVED"] = resolved_backend
    env["ESD_TRAINING_BACKEND"] = resolved_backend
    if resolved_backend == "cpu":
        # Hide accelerator runtimes before child imports.  CuPy, torch, JAX and
        # TensorFlow all respect CUDA visibility; JAX is pinned explicitly too.
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
        env["ROCR_VISIBLE_DEVICES"] = ""
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["CUPY_ACCELERATORS"] = ""
    return env


def main() -> int:
    args = _parser().parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a child trainer command is required after --")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be > 0")

    resolved_backend = _resolve_backend(args.backend)
    child_env = _child_environment(resolved_backend)

    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    request_raw = (os.environ.get("TRAINING_CHECKPOINT_REQUEST_FILE") or "").strip()
    ack_raw = (os.environ.get("TRAINING_CHECKPOINT_ACK_FILE") or "").strip()
    request_path = Path(request_raw) if request_raw else None
    ack_path = Path(ack_raw) if ack_raw else None
    acknowledged_request_mtime: int | None = None

    child = subprocess.Popen(command, env=child_env)
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
                                "schema_version": 2,
                                "status": "checkpoint_ready",
                                "checkpoint": str(checkpoint.resolve()),
                                "checkpoint_size": checkpoint.stat().st_size,
                                "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
                                "request_mtime_ns": request_mtime,
                                "restart_exact": True,
                                "trainer_owns_checkpoint_state": True,
                                "scheduler_owns_process_control": True,
                                "requested_backend": args.backend,
                                "resolved_backend": resolved_backend,
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
        latest = _latest_checkpoint(checkpoint_root)
        _atomic_json(
            marker,
            {
                "schema_version": 2,
                "status": "complete",
                "checkpoint_root": str(checkpoint_root),
                "latest_checkpoint": None if latest is None else str(latest),
                "restart_exact": True,
                "requested_backend": args.backend,
                "resolved_backend": resolved_backend,
            },
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
