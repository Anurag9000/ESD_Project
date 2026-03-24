#!/usr/bin/env python3

from __future__ import annotations

from metric_learning_pipeline import build_parser, run_experiment


def main() -> int:
    parser = build_parser(use_gabor=True)
    parser.set_defaults(
        output_dir="Results/efficientnet_v2_l_gabor_metric_learning",
        log_file="logs/efficientnet_v2_l_gabor_metric_learning.log.jsonl",
    )
    args = parser.parse_args()
    return run_experiment(args, use_gabor=True)


if __name__ == "__main__":
    raise SystemExit(main())
