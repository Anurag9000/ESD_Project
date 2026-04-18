#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from finalize_refinement_acceptance import load_checkpoint, metric_value
except ModuleNotFoundError:
    from scripts.finalize_refinement_acceptance import load_checkpoint, metric_value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iteration_config_path(iteration_dir: Path) -> Path:
    return iteration_dir / "iteration_config.json"


def decision_path(iteration_dir: Path) -> Path:
    return iteration_dir / "acceptance_decision.json"


def best_eval_checkpoint(iteration_dir: Path) -> Path:
    return iteration_dir / "best.pt"


def latest_checkpoint(iteration_dir: Path) -> Path:
    step_last = iteration_dir / "step_last.pt"
    if step_last.exists():
        return step_last
    return iteration_dir / "last.pt"


def evaluation_checkpoint(iteration_dir: Path) -> Path:
    candidate = best_eval_checkpoint(iteration_dir)
    if candidate.exists():
        return candidate
    candidate = latest_checkpoint(iteration_dir)
    return candidate


def iteration_finished(iteration_dir: Path) -> bool:
    return best_eval_checkpoint(iteration_dir).exists() and latest_checkpoint(iteration_dir).exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively run late-stage full-model refinement until threshold miss.")
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--base-log-file", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--backbone", default="convnextv2_nano")
    parser.add_argument("--weights", default="default")
    parser.add_argument("--metric", choices=("val_loss", "val_raw_acc"), required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--initial-head-lr", type=float, required=True)
    parser.add_argument("--initial-backbone-lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--sampling-strategy", choices=("balanced", "weighted", "shuffle"), default="balanced")
    parser.add_argument("--weighted-sampling", action="store_true", help="Legacy alias for --sampling-strategy weighted.")
    parser.add_argument("--skip-supcon", action="store_true")
    parser.add_argument("--resume-phase-index", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def state_path(base_output_dir: Path) -> Path:
    return base_output_dir / "recursive_state.json"


def accepted_checkpoint_path(base_output_dir: Path) -> Path:
    return base_output_dir / "accepted_best.pt"


def accepted_evaluation_dir(base_output_dir: Path) -> Path:
    return base_output_dir / "accepted_evaluation_suite"


def base_log_path_for_iteration(base_log_file: Path, iteration_index: int, use_root_dir: bool) -> Path:
    if iteration_index == 1 and use_root_dir:
        return base_log_file
    suffix = "".join(base_log_file.suffixes)
    stem = base_log_file.name[: -len(suffix)] if suffix else base_log_file.name
    return base_log_file.with_name(f"{stem}_iter{iteration_index:03d}{suffix}")


def init_state(args: argparse.Namespace, base_output_dir: Path, base_log_file: Path, initial_checkpoint: Path) -> dict[str, Any]:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    initial_metric = metric_value(load_checkpoint(initial_checkpoint), args.metric, role="baseline")

    root_has_active = (base_output_dir / "last.pt").exists() or (base_output_dir / "step_last.pt").exists()
    use_root_first = root_has_active or not any(base_output_dir.glob("iteration_*"))

    state = {
        "metric": args.metric,
        "threshold": args.threshold,
        "initial_checkpoint": str(initial_checkpoint),
        "initial_head_lr": args.initial_head_lr,
        "initial_backbone_lr": args.initial_backbone_lr,
        "accepted_checkpoint": str(initial_checkpoint),
        "accepted_metric": initial_metric,
        "next_head_lr": args.initial_head_lr,
        "next_backbone_lr": args.initial_backbone_lr,
        "iterations": [],
        "active_iteration_dir": str(base_output_dir) if use_root_first else None,
        "active_iteration_index": 1 if use_root_first else None,
        "active_log_file": str(base_log_file) if use_root_first else None,
        "stopped": False,
        "stop_reason": None,
    }

    accepted_path = accepted_checkpoint_path(base_output_dir)
    if not accepted_path.exists():
        shutil.copy2(initial_checkpoint, accepted_path)
    return state


def load_or_init_state(args: argparse.Namespace, base_output_dir: Path, base_log_file: Path, initial_checkpoint: Path) -> dict[str, Any]:
    path = state_path(base_output_dir)
    if path.exists():
        return load_json(path)
    state = init_state(args, base_output_dir, base_log_file, initial_checkpoint)
    save_json(path, state)
    return state


def write_state(base_output_dir: Path, state: dict[str, Any]) -> None:
    save_json(state_path(base_output_dir), state)


def ensure_iteration_config(
    *,
    iteration_dir: Path,
    iteration_index: int,
    baseline_checkpoint: str,
    baseline_metric: float,
    head_lr: float,
    backbone_lr: float,
    metric: str,
    threshold: float,
) -> dict[str, Any]:
    config_path = iteration_config_path(iteration_dir)
    if config_path.exists():
        return load_json(config_path)
    config = {
        "iteration_index": iteration_index,
        "baseline_checkpoint": baseline_checkpoint,
        "baseline_metric": baseline_metric,
        "head_lr": head_lr,
        "backbone_lr": backbone_lr,
        "metric": metric,
        "threshold": threshold,
    }
    save_json(config_path, config)
    return config


def next_iteration_index(base_output_dir: Path, state: dict[str, Any]) -> int:
    seen = [entry["iteration_index"] for entry in state.get("iterations", [])]
    if state.get("active_iteration_index") is not None:
        seen.append(int(state["active_iteration_index"]))
    if not seen:
        return 1
    return max(seen) + 1


def next_iteration_dir(base_output_dir: Path, index: int) -> Path:
    if index == 1:
        return base_output_dir
    return base_output_dir / f"iteration_{index:03d}"


def determine_resume(iteration_dir: Path, baseline_checkpoint: Path) -> tuple[Path, str]:
    step_last = iteration_dir / "step_last.pt"
    if step_last.exists():
        return step_last, "latest"
    last = iteration_dir / "last.pt"
    if last.exists():
        return last, "latest"
    return baseline_checkpoint, "global_best"


def run_training(
    *,
    args: argparse.Namespace,
    iteration_dir: Path,
    log_file: Path,
    baseline_checkpoint: Path,
    head_lr: float,
    backbone_lr: float,
) -> None:
    iteration_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    resume_checkpoint, resume_mode = determine_resume(iteration_dir, baseline_checkpoint)

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    command = [
        sys.executable,
        "scripts/train_efficientnet_b0_progressive.py",
        "--dataset-root",
        args.dataset_root,
        "--backbone",
        args.backbone,
        "--weights",
        args.weights,
        "--optimizer",
        args.optimizer,
        "--batch-size",
        str(args.batch_size),
        "--stage-early-stopping-patience",
        str(args.patience),
        "--classifier-train-mode",
        "progressive",
        "--classifier-early-stopping-metric",
        args.metric,
        "--head-lr",
        f"{head_lr:.12g}",
        "--backbone-lr",
        f"{backbone_lr:.12g}",
        "--ce-max-unfreeze-modules",
        "40",
        "--output-dir",
        str(iteration_dir),
        "--log-file",
        str(log_file),
        "--resume-checkpoint",
        str(resume_checkpoint),
        "--resume-mode",
        resume_mode,
        "--resume-phase-index",
        str(args.resume_phase_index),
    ]
    sampling_strategy = "weighted" if args.weighted_sampling else args.sampling_strategy
    command.extend(["--sampling-strategy", sampling_strategy])
    if args.skip_supcon:
        command.append("--skip-supcon")
    command.extend(extra_args)
    subprocess.run(command, check=True)


def finalize_iteration(
    *,
    iteration_dir: Path,
    config: dict[str, Any],
    base_output_dir: Path,
    metric: str,
    threshold: float,
) -> dict[str, Any]:
    candidate_checkpoint = evaluation_checkpoint(iteration_dir)
    baseline_checkpoint = Path(config["baseline_checkpoint"])
    selected_output = iteration_dir / "accepted_best.pt"
    decision_file = decision_path(iteration_dir)
    subprocess.run(
        [
            sys.executable,
            "scripts/finalize_refinement_acceptance.py",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--baseline-checkpoint",
            str(baseline_checkpoint),
            "--output-checkpoint",
            str(selected_output),
            "--decision-json",
            str(decision_file),
            "--metric",
            metric,
            "--min-delta",
            str(threshold),
        ],
        check=True,
    )
    decision = load_json(decision_file)
    accepted_output = accepted_checkpoint_path(base_output_dir)
    if selected_output.resolve() != accepted_output.resolve():
        shutil.copy2(selected_output, accepted_output)
    return decision


def evaluate_iteration(iteration_dir: Path, dataset_root: str, batch_size: int) -> None:
    checkpoint = evaluation_checkpoint(iteration_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"No evaluation checkpoint found in {iteration_dir}")
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_saved_classifier.py",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(iteration_dir / "evaluation_suite"),
            "--dataset-root",
            dataset_root,
            "--batch-size",
            str(batch_size),
        ],
        check=True,
    )


def sync_accepted_evaluation(base_output_dir: Path, iteration_dir: Path) -> None:
    accepted_eval = accepted_evaluation_dir(base_output_dir)
    if accepted_eval.exists():
        shutil.rmtree(accepted_eval)
    source = iteration_dir / "evaluation_suite"
    if source.exists():
        shutil.copytree(source, accepted_eval)


def append_iteration_record(state: dict[str, Any], config: dict[str, Any], decision: dict[str, Any], iteration_dir: Path, log_file: Path) -> None:
    for existing in state.get("iterations", []):
        if existing["iteration_dir"] == str(iteration_dir):
            existing.update(
                {
                    "head_lr": config["head_lr"],
                    "backbone_lr": config["backbone_lr"],
                    "baseline_checkpoint": config["baseline_checkpoint"],
                    "baseline_metric": config["baseline_metric"],
                    "candidate_metric": decision["candidate_metric"],
                    "accepted_candidate": decision["accepted_candidate"],
                    "selected_checkpoint": decision["selected_checkpoint"],
                    "log_file": str(log_file),
                }
            )
            return
    state.setdefault("iterations", []).append(
        {
            "iteration_index": config["iteration_index"],
            "iteration_dir": str(iteration_dir),
            "log_file": str(log_file),
            "head_lr": config["head_lr"],
            "backbone_lr": config["backbone_lr"],
            "baseline_checkpoint": config["baseline_checkpoint"],
            "baseline_metric": config["baseline_metric"],
            "candidate_metric": decision["candidate_metric"],
            "accepted_candidate": decision["accepted_candidate"],
            "selected_checkpoint": decision["selected_checkpoint"],
        }
    )


def main() -> int:
    args = parse_args()
    base_output_dir = Path(args.base_output_dir)
    base_log_file = Path(args.base_log_file)
    initial_checkpoint = Path(args.initial_checkpoint)

    state = load_or_init_state(args, base_output_dir, base_log_file, initial_checkpoint)
    if state.get("stopped") and state.get("active_iteration_dir") is None:
        print(
            json.dumps(
                {
                    "event": "recursive_refinement_stopped",
                    "reason": state.get("stop_reason"),
                    "accepted_checkpoint": state.get("accepted_checkpoint"),
                    "accepted_metric": state.get("accepted_metric"),
                },
                indent=2,
            )
        )
        return 0

    while True:
        if state.get("active_iteration_dir") is None:
            iteration_index = next_iteration_index(base_output_dir, state)
            iteration_dir = next_iteration_dir(base_output_dir, iteration_index)
            use_root_dir = iteration_index == 1 and iteration_dir == base_output_dir
            log_file = base_log_path_for_iteration(base_log_file, iteration_index, use_root_dir)
            state["active_iteration_dir"] = str(iteration_dir)
            state["active_iteration_index"] = iteration_index
            state["active_log_file"] = str(log_file)
            write_state(base_output_dir, state)
        else:
            iteration_dir = Path(state["active_iteration_dir"])
            iteration_index = int(state["active_iteration_index"])
            log_file = Path(state["active_log_file"])

        config = ensure_iteration_config(
            iteration_dir=iteration_dir,
            iteration_index=iteration_index,
            baseline_checkpoint=state["accepted_checkpoint"],
            baseline_metric=float(state["accepted_metric"]),
            head_lr=float(state["next_head_lr"]),
            backbone_lr=float(state["next_backbone_lr"]),
            metric=args.metric,
            threshold=args.threshold,
        )

        if not iteration_finished(iteration_dir):
            run_training(
                args=args,
                iteration_dir=iteration_dir,
                log_file=log_file,
                baseline_checkpoint=Path(config["baseline_checkpoint"]),
                head_lr=float(config["head_lr"]),
                backbone_lr=float(config["backbone_lr"]),
            )

        evaluate_iteration(iteration_dir, args.dataset_root, args.batch_size)
        decision = finalize_iteration(
            iteration_dir=iteration_dir,
            config=config,
            base_output_dir=base_output_dir,
            metric=args.metric,
            threshold=args.threshold,
        )
        append_iteration_record(state, config, decision, iteration_dir, log_file)
        state["active_iteration_dir"] = None
        state["active_iteration_index"] = None
        state["active_log_file"] = None

        if bool(decision["accepted_candidate"]):
            state["accepted_checkpoint"] = str(accepted_checkpoint_path(base_output_dir))
            state["accepted_metric"] = float(decision["candidate_metric"])
            state["next_head_lr"] = float(config["head_lr"]) / 2.0
            state["next_backbone_lr"] = float(config["backbone_lr"]) / 2.0
            state["stopped"] = False
            state["stop_reason"] = None
            sync_accepted_evaluation(base_output_dir, iteration_dir)
            write_state(base_output_dir, state)
            continue

        state["stopped"] = True
        state["stop_reason"] = "threshold_not_met"
        write_state(base_output_dir, state)
        break

    print(
        json.dumps(
            {
                "event": "recursive_refinement_complete",
                "metric": args.metric,
                "accepted_checkpoint": state["accepted_checkpoint"],
                "accepted_metric": state["accepted_metric"],
                "next_head_lr": state["next_head_lr"],
                "next_backbone_lr": state["next_backbone_lr"],
                "stop_reason": state["stop_reason"],
                "iterations_completed": len(state.get("iterations", [])),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
