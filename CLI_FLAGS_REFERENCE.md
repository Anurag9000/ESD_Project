# CLI Flags Reference

This file documents the live training, evaluation, and utility flags in the current repo state.
Defaults are the values used when the flag is omitted. Boolean defaults are shown as `true` / `false`.
Live logs now print pure accuracy plus `per_class_accuracy` and `per_class_avg_confidence`; thresholded accuracy fields are no longer emitted.

## Main Trainer: `scripts/metric_learning_pipeline.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--dataset-root` | `Dataset_Final` | Root of the physical dataset tree. |
| `--image-size` | `224` | Training crop / resize size. |
| `--batch-size` | `224` | Physical batch size per optimizer step. |
| `--num-workers` | `2` | `DataLoader` worker count. |
| `--prefetch-factor` | `1` | Prefetch depth per worker. |
| `--embedding-dim` | `128` | Width of the learned embedding layer. |
| `--projection-dim` | `128` | Width of the SupCon projection head. |
| `--unfreeze-chunk-size` | `20` | Number of backbone leaf modules unfrozen per progressive step. |
| `--skip-supcon` | `false` | Skip the SupCon stage entirely when enabled. |
| `--classifier-train-mode` | `progressive` | Choose `progressive` unfreezing or `full_model` training; `full_model` is still capped by `--ce-max-unfreeze-modules`. |
| `--classifier-early-stopping-metric` | `val_loss` | Classifier phase selection metric: `val_loss` or `val_raw_acc`. |
| `--reject-current-phase-on-global-miss` / `--no-reject-current-phase-on-global-miss` | `true` | Gate whether a phase that fails to beat the global best may seed the next phase. |
| `--supcon-temperature` | `0.07` | Temperature for the SupCon loss. |
| `--supcon-head-lr` | `3e-3` | Learning rate for the SupCon head warmup. |
| `--supcon-backbone-lr` | `5e-5` | Learning rate for the SupCon backbone phases. |
| `--head-lr` | `1e-3` | Learning rate for classifier head warmup. |
| `--backbone-lr` | `1e-5` | Learning rate for classifier backbone phases. |
| `--weight-decay` | `1e-4` | Optimizer weight decay. |
| `--backbone` | `convnextv2_nano` | Backbone registry choice. Supported values are the keys in `BACKBONE_REGISTRY`. |
| `--optimizer` | `adamw` | Optimizer family: `adamw` or `sam`. |
| `--precision` | `mixed` | Training precision: `mixed`, `32`, or `64`. |
| `--adam-beta1` | `0.9` | Adam beta1. |
| `--adam-beta2` | `0.999` | Adam beta2. |
| `--label-smoothing` | `0.1` | Cross-entropy label smoothing. |
| `--confidence-gap-penalty-weight` | `0.0` | Optional confidence-gap regularizer weight. |
| `--class-loss-weight` | `[]` | Repeatable `NAME=WEIGHT` overrides for per-class loss weighting. |
| `--targeted-confusion-penalty` | `[]` | Repeatable `TRUE_CLASS:PREDICTED_CLASS:WEIGHT` penalties. |
| `--train-loss-validation-patience` | `500` | Trigger validation after this many consecutive train steps without a new best train loss in the current phase. |
| `--validation-patience` | `1000` | Force validation after this many consecutive train steps since the last validation checkpoint in the current phase. |
| `--sampling-strategy` | `balanced` | Train-loader sampling mode: `balanced`, `weighted`, or `shuffle`. |
| `--weighted-sampling` | legacy alias | Sets `--sampling-strategy weighted`. |
| `--no-weighted-sampling` | legacy alias | Sets `--sampling-strategy shuffle`. |
| `--weights` | `default` | Use pretrained weights (`default`) or scratch initialization (`none`). |
| `--augment-repeats` | `16` | Deterministic augmentation variants per source image. |
| `--augment-gaussian-sigmas` | `0.5` | Stochastic augmentation sigma scale. |
| `--supcon-unfreeze-backbone-modules` | `40` | Upper bound on how deep SupCon may unfreeze. |
| `--ce-max-unfreeze-modules` | `40` | Upper bound on how deep CE may unfreeze, including the `full_model` path. |
| `--output-dir` | `Results/metric_learning_experiment` | Main training output root. |
| `--log-file` | `logs/metric_learning_experiment.log.jsonl` | Structured JSONL log path. |
| `--resume-checkpoint` | `""` | Explicit checkpoint to resume from. |
| `--class-mapping` | `""` | JSON string for training-time class merging. |
| `--auto-split-ratios` | `0.7,0.2,0.1` | Train / val / test split ratios. |
| `--resume-mode` | `latest` | Resume source: `latest`, `global_best`, or `phase_best`. |
| `--resume-phase-index` | `0` | Explicit resume phase index override. |
| `--resume-phase-name` | `""` | Explicit resume phase name override. |
| `--seed` | `42` | RNG seed. |
| `--max-train-batches` | `0` | Debug cap on training batches; `0` means no cap. |
| `--max-eval-batches` | `0` | Debug cap on eval batches; `0` means no cap. |
| `--log-every-steps` | `1` | Training-step logging cadence. |
| `--log-eval-every-steps` | `1` | Eval-step logging cadence. |
| `--epoch-visualizations` / `--no-epoch-visualizations` | `true` | Enable or disable startup and phase-end visual audits. |
| `--epoch-visualization-batch-size` | `128` | Batch size used by visualization loaders. |
| `--epoch-visualization-num-workers` | `2` | Worker count used by visualization loaders. |
| `--epoch-visualization-umap-thumb-size` | `48` | Thumbnail size for the UMAP thumbnail map. |
| `--epoch-visualization-umap-max-samples` | `0` | Cap on UMAP samples; `0` means the full test set. |
| `--epoch-visualization-umap-thumbnail-limit` | `0` | Maximum thumbnails rendered on the UMAP map; `0` means render all thumbnails. |
| `--runtime-bad-sample-cleanup` | `false` | Delete unreadable samples during training and remove metadata entries. |
| `--confidence-threshold` | `0.80` | Confidence threshold used in threshold-aware accuracy metrics. |
| `--supcon-early-stopping-patience` | `1` | Patience for SupCon phases. |
| `--head-early-stopping-patience` | `1` | Patience for classifier head warmup. |
| `--stage-early-stopping-patience` | `1` | Patience for later classifier phases. |
| `--early-stopping-min-delta` | `0.0` | Minimum improvement required to reset patience. |
| `--warmup-epochs` | `0` | Scheduler warmup measured in epochs. |
| `--warmup-steps` | `1024` | Scheduler warmup measured in steps. |
| `--sam-rho` | `0.05` | SAM neighborhood radius. |
| `--grad-accum-steps` | `1` | Gradient accumulation factor. |
| `--run-final-test` | `false` | Run the final protected test pass at the end of training. |

## Trainer Wrapper: `scripts/train_efficientnet_b0_progressive.py`

This wrapper does not add new flags. It reuses `scripts/metric_learning_pipeline.py` and overrides only:

| Setting | Default used by wrapper | What it does |
| --- | --- | --- |
| `output_dir` | `Results/convnextv2_nano_progressive_all_classes` | Output root for the progressive pretraining run. |
| `log_file` | `logs/convnextv2_nano_progressive_all_classes.log.jsonl` | Log file for the progressive pretraining run. |

## Recursive Refinement: `scripts/run_recursive_refinement.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--base-output-dir` | required | Root output directory for recursive refinement. |
| `--base-log-file` | required | JSONL log file for recursive refinement. |
| `--initial-checkpoint` | required | Seed checkpoint for the first refinement pass. |
| `--backbone` | `convnextv2_nano` | Backbone name forwarded to the recursive trainer. |
| `--weights` | `default` | Pretrained-weight mode forwarded to the recursive trainer. |
| `--metric` | required | Accept/reject metric: `val_loss` or `val_raw_acc`. |
| `--threshold` | required | Stop threshold for recursive refinement. |
| `--initial-head-lr` | required | Initial head learning rate for refinement. |
| `--initial-backbone-lr` | required | Initial backbone learning rate for refinement. |
| `--batch-size` | `224` | Batch size forwarded to the trainer. |
| `--num-workers` | `2` | DataLoader worker count forwarded to the trainer. |
| `--prefetch-factor` | `1` | Prefetch depth per worker forwarded to the trainer. |
| `--patience` | `1` | Early-stopping patience forwarded to the trainer. |
| `--dataset-root` | `Dataset_Final` | Dataset root forwarded to the trainer. |
| `--optimizer` | `adamw` | Optimizer forwarded to the trainer. |
| `--sampling-strategy` | `balanced` | Sampling strategy forwarded to the trainer. |
| `--weighted-sampling` | legacy alias | Sets weighted sampling for refinement. |
| `--skip-supcon` | `false` | Skip SupCon in recursive refinement. |
| `--resume-phase-index` | `1` | Resume phase index forwarded to the trainer. |
| `--min-delta` | `0.0` | Minimum metric delta required for accept/reject. |

## Evaluation: `scripts/evaluate_saved_classifier.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to evaluate. |
| `--output-dir` | required | Output directory for metrics and plots. |
| `--dataset-root` | `""` | Dataset root; falls back to checkpoint/runtime context when empty. |
| `--batch-size` | `224` | Evaluation batch size. |
| `--num-workers` | `2` | Evaluation loader worker count. |
| `--max-eval-batches` | `0` | Debug cap on evaluation batches. |
| `--confidence-threshold` | `None` | Optional confidence threshold override; otherwise inherit runtime/checkpoint defaults. |
| `--selected-class` | `[]` | Repeatable class filter. |
| `--other-label` | `other` | Label name used for collapsed non-selected classes. |
| `--class-mapping` | `""` | JSON string for runtime class collapsing. |
| `--splits` | `["val", "test"]` | Splits to evaluate. |

The saved-classifier evaluator now writes `metrics.json`, `summary.json`, `correct_confidence_by_class.json`, `confmat_counts_<split>.csv`, `confmat_rate_pct_<split>.csv`, `classification_report_<split>.csv`, `confusion_matrix.png`, `reliability_diagram.png`, and `confidence_histogram.png` for each requested split.

## External Holdout: `scripts/evaluate_external_holdout.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to evaluate. |
| `--dataset-root` | required | External holdout dataset root. |
| `--output-dir` | required | Output directory for holdout reports. |
| `--batch-size` | `224` | Evaluation batch size. |
| `--num-workers` | `2` | Evaluation loader worker count. |
| `--max-eval-batches` | `0` | Debug cap on evaluation batches. |
| `--confidence-threshold` | `None` | Optional confidence threshold override. |
| `--selected-class` | `[]` | Repeatable class filter. |
| `--other-label` | `other` | Label name used for collapsed non-selected classes. |
| `--class-mapping` | `""` | JSON string for runtime class collapsing. |

## Clean Split Evaluation: `scripts/eval_splits_no_aug.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to evaluate. |
| `--dataset-root` | `Dataset_Final` | Dataset root. |
| `--output-dir` | `Results/eval_cleaned_splits` | Output directory for split metrics and plots. |
| `--batch-size` | `224` | Evaluation batch size. |
| `--splits` | `["train", "val", "test"]` | Splits to evaluate. |
| `--seed` | `42` | Deterministic split seed. |
| `--class-mapping` | `""` | Optional JSON class merge map. |

## Grad-CAM: `scripts/gradcam_classifier.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to inspect. |
| `--dataset-root` | required | Dataset root used for sampling images. |
| `--output-dir` | required | Output directory for Grad-CAM overlays. |
| `--class-name` | `[]` | Repeatable class filter for overlays. |
| `--samples-per-class` | `4` | Number of overlays generated per class. |
| `--seed` | `42` | Sampling seed. |

## Dataset Audit: `scripts/audit_dataset_by_source.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--dataset` | `Dataset_Final` | Dataset root to audit. |
| `--out` | `dataset_audit` | Audit output root. |
| `--n` | `10` | Number of images sampled per source batch. |
| `--seed` | `42` | Sampling seed. |
| `--classes` | unset | Optional subset of classes to audit. |

## TrashBox Integration: `scripts/integrate_trashbox_dataset.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--source-root` | `dataset_audit/incoming/TrashBox` | Source tree for TrashBox input. |
| `--dataset-root` | `Dataset_Final` | Destination dataset root. |
| `--metadata-file` | `dataset_metadata.json` | Metadata JSON file to update. |
| `--dry-run` | `false` | Simulate integration without writing changes. |
| `--keep-source` | `false` | Preserve the source tree instead of deleting it after import. |

## Refinement Acceptance: `scripts/finalize_refinement_acceptance.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--candidate-checkpoint` | required | Candidate checkpoint to judge. |
| `--baseline-checkpoint` | required | Baseline checkpoint to compare against. |
| `--output-checkpoint` | required | Destination for the accepted checkpoint. |
| `--decision-json` | required | JSON file recording the accept/reject decision. |
| `--metric` | required | Decision metric: `val_loss` or `val_raw_acc`. |
| `--min-delta` | `0.0` | Minimum improvement required to accept a candidate. |

## Recursive Bootstrap Derivation: `scripts/derive_recursive_bootstrap.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--state-json` | required | Path to `recursive_state.json`. |
| `--fallback-checkpoint` | required | Checkpoint to use if the state has no accepted checkpoint. |
| `--fallback-head-lr` | `1e-4` | Fallback head learning rate. |
| `--fallback-backbone-lr` | `5e-5` | Fallback backbone learning rate. |
| `--halve-lrs` | `false` | Halve the bootstrapped learning rates. |
| `--use-half-backbone-for-both` | `false` | Use the halved backbone LR for both head and backbone. |

## Inference: `scripts/smartbin_infer.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--model` | `best.onnx` | ONNX model to load. |
| `--image` | unset | Classify a single image. |
| `--folder` | unset | Classify every image in a folder. |
| `--camera` | `None` | Run live camera inference on the selected camera index. |

## Wrapper-Managed Environment Knobs

These are not `argparse` flags, but they control the shell wrappers and the active training run.

| Knob | Default | What it does |
| --- | --- | --- |
| `RUN_STAMP` | auto-generated current timestamp | Run identity used by the wrappers. Reuse the same value to resume the same run tree. |
| `RUN_ROOT` | `Results/convnextv2_nano_all_classes_<RUN_STAMP>` in `run_training.sh`, `Results/convnextv2_nano_master_run` in `run_full_training_pipeline.sh` | Output root for checkpoints and artifacts. |
| `LOG_ROOT` | `logs/convnextv2_nano_all_classes_<RUN_STAMP>` in `run_training.sh`, `logs/convnextv2_nano_master_run` in `run_full_training_pipeline.sh` | Log root for JSONL/CSV artifacts. |
| `DATASET_ROOT` | `Dataset_Final` | Corpus root used by both wrappers. |
| `INITIAL_CHECKPOINT` | defaults to `$RUN_ROOT/progressive/best.pt` in `run_full_training_pipeline.sh` | Seed checkpoint for the recursive refinement stages. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` when unset | CUDA allocator setting injected by the wrappers for safer memory behavior. |

## Wrapper Behavior That Overrides CLI Inputs

- `run_training.sh` ignores user-supplied `--dataset-root`, `--batch-size`, `--output-dir`, and `--log-file` because those are wrapper-managed.
- `run_training.sh` always injects `--dataset-root "$DATASET_ROOT"` and `--sampling-strategy balanced` into the progressive trainer.
- `run_training.sh` auto-resumes from `step_last.pt` first, then `last.pt`.
- `run_full_training_pipeline.sh` ignores user-supplied `--dataset-root`, `--batch-size`, `--output-dir`, `--log-file`, `--resume-checkpoint`, `--resume-mode`, `--resume-phase-index`, `--classifier-train-mode`, `--classifier-early-stopping-metric`, `--head-lr`, `--backbone-lr`, `--stage-early-stopping-patience`, and `--optimizer`.
- `scripts/run_recursive_refinement.py` validates pass-through trainer flags against the main trainer parser and raises on unsupported options instead of silently ignoring them.
- `run_full_training_pipeline.sh` hardcodes the recursive refinement recipe to `batch-size 224`, `patience 1`, `resume-phase-index 1`, `optimizer adamw`, `sampling-strategy balanced`, `skip-supcon`, `classifier-train-mode progressive`, and `ce-max-unfreeze-modules 40` for both recursive passes.
