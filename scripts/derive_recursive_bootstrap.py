#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive next-stage bootstrap checkpoint/LRs from a recursive refinement state file.")
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--fallback-checkpoint", type=Path, required=True)
    parser.add_argument("--fallback-head-lr", type=float, required=True)
    parser.add_argument("--fallback-backbone-lr", type=float, required=True)
    parser.add_argument("--halve-lrs", action="store_true")
    parser.add_argument("--use-half-backbone-for-both", action="store_true")
    args = parser.parse_args()

    state = load_json(args.state_json)
    accepted_checkpoint = Path(state.get("accepted_checkpoint") or args.fallback_checkpoint)
    if not accepted_checkpoint.exists():
        accepted_checkpoint = args.fallback_checkpoint

    accepted_head_lr = state.get("initial_head_lr", args.fallback_head_lr)
    accepted_backbone_lr = state.get("initial_backbone_lr", args.fallback_backbone_lr)
    for record in state.get("iterations", []):
        if record.get("accepted_candidate"):
            accepted_head_lr = float(record["head_lr"])
            accepted_backbone_lr = float(record["backbone_lr"])

    if args.use_half_backbone_for_both:
        next_lr = accepted_backbone_lr / 2.0 if args.halve_lrs else accepted_backbone_lr
        next_head_lr = next_lr
        next_backbone_lr = next_lr
    else:
        next_head_lr = accepted_head_lr / 2.0 if args.halve_lrs else accepted_head_lr
        next_backbone_lr = accepted_backbone_lr / 2.0 if args.halve_lrs else accepted_backbone_lr

    print(f"BOOTSTRAP_CHECKPOINT={accepted_checkpoint}")
    print(f"BOOTSTRAP_HEAD_LR={next_head_lr:.12g}")
    print(f"BOOTSTRAP_BACKBONE_LR={next_backbone_lr:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
