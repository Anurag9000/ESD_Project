# ESD Repo Blueprint

## Purpose

This repository is a focused waste-material image-classification project built around a realistic local-training constraint:

- target hardware is an NVIDIA RTX 3050 laptop GPU with 6 GB VRAM
- the dataset is moderately large but class-imbalanced
- the objective is not just raw classification, but robust classification under real-world image variation

The repo is designed to answer a practical question:

> How far can a compact EfficientNet-B0-based metric-learning pipeline be pushed for waste classification on limited consumer hardware, and does a Gabor front-end help?

In concrete terms, the project is trying to produce a strong 4-class waste/material classifier with:

- high validation and test accuracy
- strong robustness to real-world visual variation
- deterministic and reproducible augmentation behavior
- resumable long-running training
- compatibility with constrained local hardware

Current classification-loss decision:

- the repo is now CE-only for supervised classification
- a direct head-only comparison showed CE outperforming the removed margin-loss branch on both raw accuracy and confidence-on-correct-predictions

## Problem Setting

The current classification problem is a 4-class image classification task:

- `metal`
- `organic`
- `other`
- `paper`

The dataset has already gone through class cleanup and restructuring:

- the old `plastic` class was merged into `other`
- metallic content previously mixed into `other` was separated into `metal`

So the repo is not just training on a raw dump of images. It reflects a curated taxonomy decision intended to make the downstream model cleaner and more practical.

## Dataset Strategy

The local dataset is expected at `Dataset_Final/` with split-specific folders:

- `Dataset_Final/train`
- `Dataset_Final/val`
- `Dataset_Final/test`

Current split counts on disk:

| Split | Total | metal | organic | other | paper |
|---|---:|---:|---:|---:|---:|
| train | 39,317 | 2,494 | 6,305 | 21,107 | 9,411 |
| val | 11,234 | 713 | 1,801 | 6,031 | 2,689 |
| test | 5,616 | 356 | 901 | 3,015 | 1,344 |

Total source images: `56,167`

The repo’s augmentation logic turns the train split into an online deterministic augmentation bank. By default:

- `augment_repeats = 16`
- effective train bank size becomes `39,317 * 16 = 629,072`

Important design choice:

- `val` is clean
- `test` is augmented by default

That means the repo is intentionally separating:

- model selection on cleaner validation data
- robustness measurement on a harsher augmented test setting

## Core Training Philosophy

The repo is built around a metric-learning-style classification pipeline rather than a plain classifier-only fine-tune.

The sequence is:

1. start from pretrained `efficientnet_b0`
2. replace the stock classifier with a learned embedding stack
3. optionally warm up with supervised contrastive learning (`SupCon`)
4. switch to cross-entropy supervised classification
5. progressively unfreeze the backbone from the tail
6. early-stop each stage on validation performance
7. restore the best validation checkpoint before moving on
8. run final validation and test evaluation

This is trying to combine:

- compact pretrained CNN transfer learning
- embedding-space structure from contrastive learning
- stronger class separation from cross-entropy
- gradual feature adaptation rather than fully-unfrozen-from-step-1 fine-tuning

## Model Variants

There are two matched experimental arms.

### Plain Variant

Entry points:

- [run_non_gabor.sh](/home/anurag-basistha/Projects/ESD/run_non_gabor.sh)
- [scripts/train_efficientnet_b0_progressive.py](/home/anurag-basistha/Projects/ESD/scripts/train_efficientnet_b0_progressive.py)

This is the baseline arm:

- pretrained `efficientnet_b0`
- no handcrafted front-end
- metric-learning head stack on top

### Gabor Variant

Entry points:

- [run_gabor.sh](/home/anurag-basistha/Projects/ESD/run_gabor.sh)
- [scripts/train_efficientnet_b0_gabor_progressive.py](/home/anurag-basistha/Projects/ESD/scripts/train_efficientnet_b0_gabor_progressive.py)

This arm adds:

- fixed multi-orientation Gabor filters on grayscale input
- fusion of Gabor responses with RGB channels
- learnable `1x1` adapter to project fused channels back into a 3-channel input for EfficientNet-B0

This is an ablation, not a completely different architecture. The point is to test whether an explicitly texture-sensitive front-end helps this task.

## Shared Training Pipeline

The real center of the repo is:

- [scripts/metric_learning_pipeline.py](/home/anurag-basistha/Projects/ESD/scripts/metric_learning_pipeline.py)

That file owns:

- dataset loading
- deterministic augmentation
- weighted sampling
- SupCon logic
- cross-entropy logic
- optimizer and scheduler setup
- progressive unfreezing plan
- checkpointing and resume
- logging
- final metrics

The wrapper scripts are intentionally thin; nearly all behavior is centralized in the shared pipeline.

## Representation and Heads

The repo is not a “512-class” model. The important dimensions are:

- EfficientNet-B0 pooled feature width: `1280`
- embedding dimension: `512`
- SupCon projection dimension: `256`
- final classification classes: `4`

So the learning structure is:

- backbone feature extraction
- `1280 -> 512` embedding projection
- optional SupCon projection head
- cross-entropy classification head from the `512`-dimensional embedding

This is why `512` appears throughout the repo. It is the embedding width, not the number of classes.

## Training Stages

### Stage 1: SupCon Warmup

Optional and controlled by `--skip-supcon`.

Intent:

- encourage better geometry in embedding space
- learn invariances across augmented views of the same source image

Current default:

- `supcon_epochs = 200`
- `supcon_temperature = 0.07`
- `supcon_unfreeze_backbone_modules = 0`

By default, this means:

- the backbone remains frozen during SupCon
- only the metric-learning heads are adapted during this stage

### Stage 2: cross-entropy Head-Only

This is the first supervised classification stage after SupCon or immediately if SupCon is skipped.

Intent:

- adapt the new embedding/classification heads first
- avoid immediately disturbing pretrained backbone features

### Stage 3+: Progressive Unfreezing

The backbone is unfrozen from the tail in chunks.

Current defaults:

- `unfreeze_chunk_size = 20`
- `max_progressive_phases = 0`

Meaning:

- unfreeze the last `20` parameter-bearing backbone modules, then `40`, then `60`, etc.
- `0` means no artificial cap; progression continues until the whole planned backbone span is covered

This is designed to make fine-tuning more stable on limited hardware and on a dataset that can still overfit despite augmentation.

## Optimization Stack

Current default parser values:

- `optimizer = sam`
- `precision = mixed`
- `batch_size = 512`
- `num_workers = 4`
- `prefetch_factor = None` meaning PyTorch default if omitted
- `supcon_head_lr = 3e-4`
- `supcon_backbone_lr = 1e-4`
- `head_lr = 1e-3`
- `backbone_lr = 1e-4`
- `weight_decay = 1e-4`
- `sam_rho = 0.05`

### Why the LRs differ

The repo intentionally uses a larger head LR than backbone LR.

Reason:

- the head stack is newly learned and needs faster adaptation
- the pretrained backbone should move more conservatively

So the repo is not using a single LR everywhere by design.

### Mixed Precision

Mixed precision is the default.

That means:

- not everything is forced to `float16`
- autocast is used where appropriate
- master optimization state remains in safer precision

Goal:

- save VRAM
- speed up training on CUDA hardware
- preserve most FP32 behavior

### SAM

`SAM` is the default optimizer wrapper in the repo, but it is explicitly optional:

- `--optimizer sam`
- `--optimizer adamw`

The repo is set up so the user can run the full pipeline with or without SAM and compare outcomes.

## Confidence-Threshold Overlay

The repo currently includes a custom confidence-aware classification overlay:

- `confidence_threshold = 0.80`

There are two distinct behaviors here.

### Metric-Level Thresholding

A prediction only counts as “correct” for the thresholded metric if:

- the predicted class matches the label
- predicted confidence is at least `0.80`

That affects:

- training `running_acc`
- validation `val_acc`
- final saved thresholded metrics

The raw unthresholded accuracy is also tracked separately.

### Loss-Level Confidence Penalty

For non-mixed samples only:

- if the model predicts the correct class
- but true-class confidence is below `0.80`
- an additional squared penalty is added

This is intended to push correct predictions to become more decisive, not merely correct.

This is compatible with cross-entropy in implementation terms, but it is also an additional optimization pressure, so it can help or hurt depending on weighting and regime.

## Augmentation Strategy

This repo uses deterministic, split-safe online augmentation rather than materializing augmented images on disk.

High-level properties:

- source splits remain disjoint
- augmentation is deterministic per split/source/repeat/view/epoch combination
- magnitudes are sampled from clipped Gaussian bands
- augmentation is applied online

The augmentation stack is deliberately broad because the intended deployment domain is messy real-world waste imagery. The repo currently includes:

- random resized crop
- horizontal flip
- mild vertical flip
- translation
- rotation
- scale
- shear
- perspective distortion
- brightness
- contrast
- saturation
- hue shift
- gamma
- sharpness
- JPEG compression
- Gaussian blur
- motion blur
- defocus blur
- Gaussian noise
- channel shift
- grayscale mixing
- illumination gradient
- shadow overlay
- specular glare
- cutout
- border truncation / framing loss
- resolution degradation via downsample-upsample
- stain / smudge overlay

The intent is not just “more augmentation.” It is to cover realistic nuisance factors for waste images:

- blur
- compression
- framing issues
- dirty surfaces
- lighting variation
- cluttered acquisition conditions

## Sampling Strategy

The launcher scripts always pass:

- `--weighted-sampling`

So the day-to-day default repo behavior is class-rebalancing at sampling time, not plain natural-frequency sampling.

This matters because the dataset is imbalanced, especially with `other` dominating.

## Early Stopping Strategy

The repo now uses stage-specific patience settings:

- `supcon_early_stopping_patience = 10`
- `head_early_stopping_patience = 10`
- `stage_early_stopping_patience = 10`
- `early_stopping_min_delta = 1e-4`

Validation cadence:

- `eval_every_epochs = 0.5`

So patience is measured in validation windows, not epochs.

At the current default `batch_size = 512` and effective train size `629,072`:

- train steps per epoch = `1229`
- one validation window = `1024 / 1229 ≈ 83.3%` of an epoch

That means each patience unit is a fairly large chunk of training.

## Checkpointing and Resume

The repo uses two checkpoint concepts.

### `last.pt`

Saved at stable boundaries:

- after completed validation windows
- at key stage/phase transitions

Use:

- normal resume
- best-known stable restart point

### `step_last.pt`

Saved after every train step.

Use:

- emergency resume when the process dies before the next validation-boundary save

This was added because long runs on constrained local hardware can die mid-window, and losing almost an entire validation interval is too expensive.

### Resume Modes

The trainer supports:

- `latest`
- `global_best`
- `phase_best`

And also supports restarting from a specified phase via:

- `resume_phase_index`
- `resume_phase_name`

So the repo can do both:

- exact-ish continuation from the latest training state
- experimental restart from a chosen best checkpoint and chosen phase

## Logging and Outputs

The repo writes JSONL logs and model artifacts.

Typical outputs:

- `logs/*.log.jsonl`
- `Results/.../last.pt`
- `Results/.../step_last.pt`
- `Results/.../best.pt`
- `Results/.../metrics.json`
- `Results/.../validation_metrics.json`
- `Results/.../test_metrics.json`

The logs are structured and event-based. They track:

- run start
- dataset schedule
- train-step summaries
- validation start/finish
- early stopping decisions
- final evaluation

This repo is therefore designed for post-hoc debugging and training forensics, not just final-model export.

## Memory and Stability Design

Because this project is running on a limited local machine, the repo now contains explicit logic to reduce memory-overlap crashes at phase boundaries.

Current behavior includes:

- per-step emergency checkpointing
- explicit cleanup at train/val/test/stage transitions
- explicit dataloader worker teardown at those boundaries
- CUDA cache cleanup and IPC cleanup
- Python garbage collection on transitions

This exists because the observed failure mode in real runs was:

- host RAM/shared-memory OOM
- especially near validation startup

So the repo is not only a training pipeline; it is also a stability-engineered local experimentation harness.

## How the Repo Is Intended to Be Used

There are two main usage modes.

### Full Experiment

Use the default pipeline to test the complete design:

- SupCon
- cross-entropy
- progressive unfreezing
- mixed precision
- SAM by default

### Controlled Ablation

Turn off selected components to isolate value:

- `--skip-supcon`
- `--optimizer adamw`
- plain vs Gabor launcher
- phase-specific resume/restart

The repo is therefore built for comparative experimentation, not just one locked training recipe.

## Main Experimental Questions This Repo Encodes

From the code and its current shape, the real research questions are:

1. Is a compact EfficientNet-B0 strong enough for this waste task on a 6 GB GPU?
2. Does deterministic heavy augmentation improve robustness without wrecking clean validation behavior?
3. Does SupCon help the final classifier, or is direct classifier fine-tuning enough?
4. Does SAM help generalization here, or only slow things down?
5. Does the Gabor front-end add value beyond a plain CNN baseline?
6. How much progressive unfreezing is actually helpful before validation saturates?
7. Can confidence-aware metrics and penalties produce a model that is not only correct, but decisively correct?

## Current Practical Defaults

As of the current code, the default training profile is aggressive:

- `batch_size = 512`
- `num_workers = 4`
- `prefetch_factor = PyTorch default`
- `precision = mixed`
- `optimizer = sam`
- `augment_repeats = 16`
- `eval_every_epochs = 0.5`
- `confidence_threshold = 0.80`

This means the repo currently prioritizes:

- high-throughput local runs
- strong regularization
- resumability
- detailed logging

over minimalism.

## Important Caveats

The repo is strong technically, but not every enabled idea is guaranteed to help final accuracy on this exact dataset.

Examples:

- SupCon may help, do nothing, or hurt
- SAM may help, do nothing, or hurt
- the confidence penalty may improve decisiveness or may over-constrain optimization
- deeper progressive unfreezing may plateau or degrade validation performance

So this repo should be understood as:

- a serious experimentation framework
- not a proof that every enabled option is globally optimal

## File Map

Primary files:

- [README.md](/home/anurag-basistha/Projects/ESD/README.md)
- [REPO_BLUEPRINT.md](/home/anurag-basistha/Projects/ESD/REPO_BLUEPRINT.md)
- [scripts/metric_learning_pipeline.py](/home/anurag-basistha/Projects/ESD/scripts/metric_learning_pipeline.py)
- [scripts/train_efficientnet_b0_progressive.py](/home/anurag-basistha/Projects/ESD/scripts/train_efficientnet_b0_progressive.py)
- [scripts/train_efficientnet_b0_gabor_progressive.py](/home/anurag-basistha/Projects/ESD/scripts/train_efficientnet_b0_gabor_progressive.py)
- [run_non_gabor.sh](/home/anurag-basistha/Projects/ESD/run_non_gabor.sh)
- [run_gabor.sh](/home/anurag-basistha/Projects/ESD/run_gabor.sh)
- [scripts/create_dataset_splits.py](/home/anurag-basistha/Projects/ESD/scripts/create_dataset_splits.py)
- [scripts/reorganize_dataset.py](/home/anurag-basistha/Projects/ESD/scripts/reorganize_dataset.py)

## Bottom Line

This repository is a compact but fairly sophisticated waste-classification experimentation stack. Its design combines:

- compact pretrained CNN transfer learning
- deterministic heavy augmentation
- optional contrastive warmup
- cross-entropy metric classification
- progressive fine-tuning
- resume and checkpoint safety
- local-hardware-aware stability fixes

The repo’s real goal is not just to train one model. It is to let you systematically answer which combination of:

- plain vs Gabor
- SupCon vs no SupCon
- SAM vs AdamW
- shallow vs deep unfreezing
- confidence-aware vs plain classification

actually produces the best waste classifier under your real hardware and dataset constraints.
