#!/usr/bin/env python3

from __future__ import annotations

import os

try:
    from metric_learning_pipeline import DEFAULT_BACKBONE_NAME, build_parser, run_experiment
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from metric_learning_pipeline import DEFAULT_BACKBONE_NAME, build_parser, run_experiment


def main() -> int:
    parser = build_parser()
    default_output_dir = parser.get_default("output_dir")
    default_log_file = parser.get_default("log_file")
    args = parser.parse_args()
    backbone_name = getattr(args, "backbone", os.getenv("BACKBONE_NAME", DEFAULT_BACKBONE_NAME))
    if args.output_dir == default_output_dir:
        args.output_dir = f"Results/{backbone_name}_progressive_six_classes"
    if args.log_file == default_log_file:
        args.log_file = f"logs/{backbone_name}_progressive_six_classes.log.jsonl"
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
