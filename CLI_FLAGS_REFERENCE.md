# CLI Flags Reference

This file documents the live training, evaluation, and utility flags in the current repo state.
Defaults are the values used when the flag is omitted. Boolean defaults are shown as `true` / `false`.
The next-run logical training taxonomy is strictly 3 classes: `organic`, `metal`, `paper`.
Live logs now print pure accuracy plus `per_class_accuracy` and `per_class_avg_confidence`; thresholded accuracy fields are no longer emitted.

## Main Trainer: `scripts/metric_learning_pipeline.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--dataset-root` | `Dataset_Final` | Root of the physical dataset tree. |
| `--image-size` | `224` | Training crop / resize size. |
| `--batch-size` | `320` | Physical batch size per optimizer step. |
| `--num-workers` | `2` | `DataLoader` worker count. |
| `--prefetch-factor` | `1` | Prefetch depth per worker. |
| `--embedding-dim` | `128` | Width of the learned embedding layer. |
| `--projection-dim` | `128` | Width of the SupCon projection head. |
| `--unfreeze-chunk-size` | `20` | Number of backbone leaf modules unfrozen per progressive step. |
| `--skip-supcon` | `false` | Skip the SupCon stage entirely when enabled. |
| `--classifier-train-mode` | `progressive` | Choose `progressive` unfreezing or `full_model` training; both modes preserve `--frozen-core-backbone-modules`. |
| `--classifier-early-stopping-metric` | `val_loss` | Classifier phase selection metric: `val_loss` or `val_raw_acc`. |
| `--reject-current-phase-on-global-miss` / `--no-reject-current-phase-on-global-miss` | `true` | Gate whether a phase that fails to beat the global best may seed the next phase. |
| `--supcon-temperature` | `0.07` | Temperature for the SupCon loss. |
| `--supcon-head-lr` | `3e-3` | Learning rate for the SupCon head warmup. |
| `--supcon-backbone-lr` | `5e-5` | Learning rate for the SupCon backbone phases. |
| `--head-lr` | `1e-3` | Learning rate for classifier head warmup. |
| `--backbone-lr` | `1e-5` | Learning rate for classifier backbone phases. |
| `--weight-decay` | `1e-4` | Optimizer weight decay. |
| `--backbone` | `convnextv2_nano` | Backbone selection. The trainer accepts any timm backbone name; registered aliases resolve pretrained/scratch defaults cleanly. |
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
| `--phase0-encoder-checkpoint` | `""` | Optional Phase 0 masked-image-modeling encoder checkpoint to seed the backbone before SupCon/CE starts. |
| `--augment-repeats` | `16` | Deterministic augmentation variants per source image. |
| `--augment-gaussian-sigmas` | `0.5` | Stochastic augmentation sigma scale. |
| `--camera-color-cast-probability` | `0.65` | Probability of applying Pi-camera style magenta/pink white-balance cast to train augmentations. Validation/test stay clean. |
| `--camera-color-cast-strength` | `0.24` | Maximum strength of the magenta/pink cast augmentation. |
| `--frozen-core-backbone-modules` | `40` | Number of earliest backbone leaf modules kept frozen in every SupCon, CE, and recursive phase. Default freezes the stem/core 40 modules. |
| `--supcon-unfreeze-backbone-modules` | `None` | Optional extra cap on how many tail modules SupCon may unfreeze. When unset, SupCon may train only the tail left after the frozen 40-module core. |
| `--ce-max-unfreeze-modules` | `None` | Optional extra cap on how many tail modules CE may unfreeze, including the `full_model` path. When unset, CE may train only the tail left after the frozen 40-module core. |
| `--output-dir` | `Results/metric_learning_experiment` | Main training output root. |
| `--log-file` | `logs/metric_learning_experiment.log.jsonl` | Structured JSONL log path. |
| `--resume-checkpoint` | `""` | Explicit checkpoint to resume from. |
| `--class-mapping` | `""` | JSON string for training-time class merging; the default 3-class pipeline ignores extra physical folders. |
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
| `--supcon-early-stopping-patience` | `3` | Patience for SupCon phases. |
| `--head-early-stopping-patience` | `3` | Patience for classifier head warmup. |
| `--stage-early-stopping-patience` | `3` | Patience for later classifier phases. |
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
| `output_dir` | `Results/<backbone>_progressive_three_classes` | Output root for the progressive pretraining run when not set explicitly. |
| `log_file` | `logs/<backbone>_progressive_three_classes.log.jsonl` | Log file for the progressive pretraining run when not set explicitly. |

## Phase 0 MIM Launcher: `scripts/train_phase0_mim.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--dataset-root` | `Dataset_Final` | Dataset root used to build the clean train split. |
| `--output-dir` | `Results/phase0_mim` | Phase 0 checkpoint output root. |
| `--log-file` | `logs/phase0_mim.log.jsonl` | Structured Phase 0 JSONL log. |
| `--backbone` | `convnextv2_nano` | Backbone selection used for the encoder. Any timm backbone string is accepted. |
| `--weights` | `default` | For Phase 0, `default` means use the pure `.fcmae` backbone weights; `none` means scratch init. |
| `--image-size` | `224` | Input resolution for masking and reconstruction. |
| `--augment-repeats` | `16` | Passed through to the repo dataset builder for split construction. |
| `--augment-gaussian-sigmas` | `0.5` | Passed through to the repo dataset builder for split construction. |
| `--camera-color-cast-probability` | `0.65` | Passed through to the repo dataset builder so Phase 0 MIM sees the same Pi-camera color-cast augmentation on train images. |
| `--camera-color-cast-strength` | `0.24` | Maximum Phase 0 train-time magenta/pink cast strength. |
| `--class-mapping` | `""` | Optional JSON merge map passed to the repo dataset builder. |
| `--auto-split-ratios` | `0.7,0.2,0.1` | Auto-split ratios when the dataset root has no explicit train/val/test layout. |
| `--runtime-bad-sample-cleanup` | `false` | Mirror the main trainer's runtime bad-sample cleanup behavior. |
| `--batch-size` | `8` | Micro-batch size for Phase 0 masking reconstruction. Uses the same class-balanced sampler as the supervised stages. Combined with `--grad-accum-steps 40` this gives an effective batch size of 320. |
| `--num-workers` | `2` | DataLoader worker count. |
| `--prefetch-factor` | `1` | Prefetch depth per worker. |
| `--epochs` | `0` | Phase 0 epoch cap. `0` means run until early stopping or max-steps termination. |
| `--grad-accum-steps` | `40` | Gradient accumulation factor for Phase 0. The default effective batch size is 320. |
| `--mask-ratio` | `0.6` | Fraction of patches masked before reconstruction. |
| `--patch-size` | `32` | Patch size used by the spatial mask generator. |
| `--decoder-dim` | `512` | Hidden width of the reconstruction decoder. |
| `--learning-rate` | `1.5e-4` | AdamW learning rate for Phase 0. |
| `--weight-decay` | `0.05` | AdamW weight decay for Phase 0. |
| `--seed` | `42` | RNG seed. |
| `--train-loss-window` | `5000` | Number of effective optimizer batches in one Phase 0 plateau window. With defaults, this is `5000 x 320` train images processed. |
| `--early-stopping-patience` | `3` | Stop Phase 0 after this many full effective-batch windows without a new best effective-batch reconstruction loss. |
| `--early-stopping-min-delta` | `1e-4` | Minimum effective-batch loss decrease required to reset Phase 0 patience. |
| `--resume-checkpoint` | `""` | Optional Phase 0 checkpoint to resume from. |

## Recursive Refinement: `scripts/run_recursive_refinement.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--base-output-dir` | required | Root output directory for recursive refinement. |
| `--base-log-file` | required | JSONL log file for recursive refinement. |
| `--initial-checkpoint` | required | Seed checkpoint for the first refinement pass. |
| `--backbone` | `convnextv2_nano` | Backbone name forwarded to the recursive trainer. Any timm backbone string is accepted. |
| `--weights` | `default` | Pretrained-weight mode forwarded to the recursive trainer. |
| `--metric` | required | Accept/reject metric: `val_loss` or `val_raw_acc`. |
| `--threshold` | required | Stop threshold for recursive refinement. |
| `--initial-head-lr` | required | Initial head learning rate for refinement. |
| `--initial-backbone-lr` | required | Initial backbone learning rate for refinement. |
| `--batch-size` | `320` | Batch size forwarded to the trainer. |
| `--num-workers` | `2` | DataLoader worker count forwarded to the trainer. |
| `--prefetch-factor` | `1` | Prefetch depth per worker forwarded to the trainer. |
| `--patience` | `3` | Early-stopping patience forwarded to the trainer. |
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
| `--batch-size` | `320` | Evaluation batch size. |
| `--num-workers` | `2` | Evaluation loader worker count. |
| `--max-eval-batches` | `0` | Debug cap on evaluation batches. |
| `--confidence-threshold` | `None` | Optional confidence threshold override; otherwise inherit runtime/checkpoint defaults. |
| `--selected-class` | `[]` | Repeatable class filter. |
| `--other-label` | `other` | Label name used for collapsed non-selected classes. |
| `--class-mapping` | `""` | JSON string for runtime class collapsing. |
| `--splits` | `["val", "test"]` | Splits to evaluate. |
| `--evaluation-stage` | `checkpoint_evaluation` | Stage label written to terminal, JSONL, and CSV eval logs. Recursive refinement sets this to `recursive_refinement`; final test sets this to `final_test_evaluation`. |
| `--phase-name` | `""` | Optional phase label written to eval logs, such as `val_loss_iteration_001`, `val_raw_acc_iteration_001`, or `final_test`. |

The saved-classifier evaluator now writes `metrics.json`, `summary.json`, `correct_confidence_by_class.json`, `confmat_counts_<split>.csv`, `confmat_rate_pct_<split>.csv`, `classification_report_<split>.csv`, `confusion_matrix.png`, `reliability_diagram.png`, and `confidence_histogram.png` for each requested split.

## External Holdout: `scripts/evaluate_external_holdout.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to evaluate. |
| `--dataset-root` | required | External holdout dataset root. |
| `--output-dir` | required | Output directory for holdout reports. |
| `--batch-size` | `320` | Evaluation batch size. |
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
| `--batch-size` | `320` | Evaluation batch size. |
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
| `RUN_ROOT` | `Results/<backbone>_three_classes_<RUN_STAMP>` in `run_training.sh`, `Results/<backbone>_master_run` in `run_full_training_pipeline.sh` | Output root for checkpoints and artifacts. |
| `LOG_ROOT` | `logs/<backbone>_three_classes_<RUN_STAMP>` in `run_training.sh`, `logs/<backbone>_master_run` in `run_full_training_pipeline.sh` | Log root for JSONL/CSV artifacts. |
| `DATASET_ROOT` | `Dataset_Final` | Corpus root used by both wrappers. |
| `INITIAL_CHECKPOINT` | defaults to `$RUN_ROOT/progressive/best.pt` in `run_full_training_pipeline.sh` | Seed checkpoint for the recursive refinement stages. |
| `RECURSIVE_ACCEPTANCE_MIN_DELTA` | `0.0` | Minimum improvement required to accept a recursive candidate in `run_full_training_pipeline.sh`. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` when unset | CUDA allocator setting injected by the wrappers for safer memory behavior. |

## Wrapper Behavior That Overrides CLI Inputs

- `run_training.sh` ignores user-supplied `--dataset-root`, `--batch-size`, `--output-dir`, and `--log-file` because those are wrapper-managed.
- `run_training.sh` always injects `--dataset-root "$DATASET_ROOT"` and `--sampling-strategy balanced` into the progressive trainer.
- `run_training.sh` supports optional Phase 0 MIM pretraining via `--phase0-mim` plus `--phase0-mim-*` controls; when enabled it exports `phase0_mim/phase0_encoder_final.pth` and passes it into the progressive trainer via `--phase0-encoder-checkpoint`.
- `--phase0-mim-train-loss-window` controls the Phase 0 early-stopping window in effective optimizer batches. The default is `5000`.
- Phase 0 uses the same balanced class sampler as SupCon and CE. It defaults to `batch-size 8` with `grad-accum-steps 40`, giving the same effective batch size of 320 without exceeding 6GB VRAM.
- Re-running the exact same `run_training.sh` command is the resume path. The wrapper skips completed Phase 0/progressive stages and resumes the incomplete stage from that stage's own `step_last.pt` first, then `last.pt`.
- `run_full_training_pipeline.sh` ignores user-supplied `--dataset-root`, `--batch-size`, `--output-dir`, `--log-file`, `--resume-checkpoint`, `--resume-mode`, `--resume-phase-index`, `--classifier-train-mode`, `--classifier-early-stopping-metric`, `--head-lr`, `--backbone-lr`, `--stage-early-stopping-patience`, and `--optimizer`.
- `scripts/run_recursive_refinement.py` validates pass-through trainer flags against the main trainer parser and raises on unsupported options instead of silently ignoring them.
- `run_full_training_pipeline.sh` hardcodes the recursive refinement recipe to `batch-size 320`, `patience 3`, `resume-phase-index 1`, `optimizer adamw`, `sampling-strategy balanced`, `skip-supcon`, `classifier-train-mode full_model`, and `frozen-core-backbone-modules 40` for both recursive passes. This means recursive refinement starts directly in `ce_full_model`: classifier head plus the backbone tail after the frozen 40-module stem/core train from the first step. Recursive candidate scoring evaluates only the validation split; the test split is reserved for the final test bundle.
