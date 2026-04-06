def build_parser() -> argparse.ArgumentParser:
    description = (
        "Metric-learning EfficientNet B0 training with deterministic 16x split-safe augmentation, "
        "SupCon warmup, cross-entropy classification, progressive unfreezing, and paper-style metrics"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--supcon-epochs", type=int, default=100)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--stage-epochs", type=int, default=20)
    parser.add_argument("--unfreeze-chunk-size", type=int, default=50)
    parser.add_argument("--max-progressive-phases", type=int, default=0)
    parser.add_argument("--skip-supcon", action="store_true")
    parser.add_argument("--classifier-train-mode", choices=("progressive", "full_model"), default="progressive")
    parser.add_argument("--classifier-early-stopping-metric", choices=("val_loss", "val_raw_acc"), default="val_loss")
    parser.add_argument(
        "--reject-current-phase-on-global-miss",
        action="store_true",
        default=False,
        help=(
            "Opt-in current-phase rejection only. When enabled, a completed progressive phase that "
            "fails to beat the global best checkpoint on the selected classifier early-stopping metric "
            "is not used to initialize the next phase. Future phases still continue normally. Disabled "
            "by default."
        ),
    )
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--supcon-head-lr", type=float, default=3e-4)
    parser.add_argument("--supcon-backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["sam", "adamw"], default="sam")
    parser.add_argument("--precision", choices=("mixed", "32", "64"), default="mixed")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--confidence-gap-penalty-weight", type=float, default=0.0)
    parser.add_argument("--class-loss-weight", action="append", default=[])
    parser.add_argument("--targeted-confusion-penalty", action="append", default=[])
    parser.add_argument("--weighted-sampling", action="store_true")
    parser.add_argument("--weights", choices=("default", "none"), default="default")
    parser.add_argument("--augment-repeats", type=int, default=16)
    parser.add_argument("--augment-gaussian-sigmas", type=float, default=0.5)
    parser.add_argument("--supcon-unfreeze-backbone-modules", type=int, default=0)
    parser.add_argument("--output-dir", default="Results/metric_learning_experiment")
    parser.add_argument("--log-file", default="logs/metric_learning_experiment.log.jsonl")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-mode", choices=("latest", "global_best", "phase_best"), default="latest")
    parser.add_argument("--resume-phase-index", type=int, default=0)
    parser.add_argument("--resume-phase-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--log-eval-every-steps", type=int, default=1000)
    parser.add_argument("--eval-every-epochs", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--supcon-early-stopping-patience", type=int, default=5)
    parser.add_argument("--head-early-stopping-patience", type=int, default=5)
    parser.add_argument("--stage-early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=1024)
    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    return parser
