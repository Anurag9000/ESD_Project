#!/usr/bin/env python3

from __future__ import annotations

from metric_learning_pipeline import build_parser, run_experiment


def main() -> int:
    parser = build_parser()
    parser.set_defaults(
        output_dir="Results/convnextv2_nano_progressive_six_classes",
        log_file="logs/convnextv2_nano_progressive_six_classes.log.jsonl",
    )
    args = parser.parse_args()
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
