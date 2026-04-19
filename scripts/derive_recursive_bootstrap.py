#!/usr/bin/env python3
"""derive_recursive_bootstrap.py

Reads the recursive_state.json produced by run_recursive_refinement.py
after Stage 5 (val_loss refinement) and outputs shell variable assignments
for BOOTSTRAP_CHECKPOINT, BOOTSTRAP_HEAD_LR, BOOTSTRAP_BACKBONE_LR
that run_full_training_pipeline.sh `eval`s to initialise Stage 6 (val_acc).

Usage (called from within run_full_training_pipeline.sh):
    eval "$(python scripts/derive_recursive_bootstrap.py \
        --state-json path/to/recursive_state.json \
        --fallback-checkpoint path/to/fallback.pt \
        --fallback-head-lr 1e-4 \
        --fallback-backbone-lr 5e-5 \
        --halve-lrs \
        --use-half-backbone-for-both)"

All output is shell-safe `key=value` lines consumed by `eval`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-json", required=True, help="Path to recursive_state.json")
    p.add_argument("--fallback-checkpoint", required=True, help="Checkpoint to use if state has no accepted_checkpoint")
    p.add_argument("--fallback-head-lr", type=float, default=1e-4)
    p.add_argument("--fallback-backbone-lr", type=float, default=5e-5)
    p.add_argument(
        "--halve-lrs",
        action="store_true",
        help="Halve the bootstrapped LRs (conservative starting point for next stage)",
    )
    p.add_argument(
        "--use-half-backbone-for-both",
        action="store_true",
        help="Use half the backbone LR as the head LR too (ultra-conservative head warm-start)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    state_path = Path(args.state_json)
    if not state_path.exists():
        # No state file — emit fallbacks silently
        _emit(args.fallback_checkpoint, args.fallback_head_lr, args.fallback_backbone_lr)
        return 0

    try:
        state: dict = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"# WARNING: could not read state JSON: {exc}", file=sys.stderr)
        _emit(args.fallback_checkpoint, args.fallback_head_lr, args.fallback_backbone_lr)
        return 0

    # Best accepted checkpoint from Stage 5
    checkpoint = state.get("accepted_checkpoint") or args.fallback_checkpoint
    if not Path(checkpoint).exists():
        checkpoint = args.fallback_checkpoint

    # Final LRs used by Stage 5's last accepted iteration
    head_lr: float = float(state.get("next_head_lr", args.fallback_head_lr))
    backbone_lr: float = float(state.get("next_backbone_lr", args.fallback_backbone_lr))

    # Apply halving if requested (conservative warm-start for Stage 6)
    if args.halve_lrs:
        head_lr /= 2.0
        backbone_lr /= 2.0

    # Optionally set head LR = halved backbone LR (ultra-conservative)
    if args.use_half_backbone_for_both:
        head_lr = backbone_lr

    _emit(checkpoint, head_lr, backbone_lr)
    return 0


def _emit(checkpoint: str, head_lr: float, backbone_lr: float) -> None:
    """Print shell variable assignments consumed by eval in the shell script."""
    print(f"BOOTSTRAP_CHECKPOINT={_shell_quote(str(checkpoint))}")
    print(f"BOOTSTRAP_HEAD_LR={head_lr:g}")
    print(f"BOOTSTRAP_BACKBONE_LR={backbone_lr:g}")


def _shell_quote(s: str) -> str:
    """Minimal single-quote shell escaping."""
    return "'" + s.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
