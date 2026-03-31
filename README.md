# ESD Waste Classification Project

## Overview

This repository contains:

- the curated waste-classification dataset archive tracked with Git LFS
- the extracted dataset in a train/val/test folder layout
- CUDA-only PyTorch training code for two matched image-classification experiments:
  - plain EfficientNet-B0
  - EfficientNet-B0 with a Gabor front-end

The current training pipeline uses:

- pretrained `efficientnet_b0`
- supervised contrastive pretraining (`SupCon`)
- cross-entropy classification
- SAM (`Sharpness-Aware Minimization`)
- AdamW
- warmup + cosine decay scheduling
- progressive unfreezing with early stopping

Loss-selection note:

- cross-entropy is now the mandatory classifier loss everywhere in this repo
- the removed margin-loss branch lost on both metrics that matter here
- head-only comparison result:
  - raw test accuracy: `0.852909` for cross-entropy vs `0.799546`
  - average confidence on correct test predictions: `0.90157` for cross-entropy vs `0.51333`

## Current Dataset State

The dataset was reorganized earlier in this project from its original class layout.

Current classes:

- `metal`
- `organic`
- `other`
- `paper`

Important class-history note:

- the old `plastic` class was merged into `other`
- metallic content from the old `other` class was split into its own `metal` class

So the project currently has **4 classes**, not 3.

## Split Counts

Current split counts on disk:

| Split | Total | metal | organic | other | paper |
|---|---:|---:|---:|---:|---:|
| train | 39,317 | 2,494 | 6,305 | 21,107 | 9,411 |
| val | 11,234 | 713 | 1,801 | 6,031 | 2,689 |
| test | 5,616 | 356 | 901 | 3,015 | 1,344 |

Total images across all splits: **56,167**

## Repository Layout

- [Dataset_Final.zip](/home/anurag-basistha/Projects/ESD/Dataset_Final.zip): dataset archive, tracked with Git LFS
- [Dataset_Final](/home/anurag-basistha/Projects/ESD/Dataset_Final): extracted dataset used locally for training
- [SMART_DUSTBIN_PLATFORM_SPEC.md](/home/anurag-basistha/Projects/ESD/SMART_DUSTBIN_PLATFORM_SPEC.md): formal mobile + backend + edge deployment specification for the smart dustbin system
- [scripts/metric_learning_pipeline.py](/home/anurag-basistha/Projects/ESD/scripts/metric_learning_pipeline.py): shared training pipeline
- [scripts/train_efficientnet_b0_progressive.py](/home/anurag-basistha/Projects/ESD/scripts/train_efficientnet_b0_progressive.py): plain EfficientNet-B0 trainer
- [scripts/train_efficientnet_b0_gabor_progressive.py](/home/anurag-basistha/Projects/ESD/scripts/train_efficientnet_b0_gabor_progressive.py): Gabor variant
- [run_non_gabor.sh](/home/anurag-basistha/Projects/ESD/run_non_gabor.sh): shell launcher for the plain model
- [run_gabor.sh](/home/anurag-basistha/Projects/ESD/run_gabor.sh): shell launcher for the Gabor model
- [requirements-cu128.txt](/home/anurag-basistha/Projects/ESD/requirements-cu128.txt): CUDA-indexed Python requirements
- [PYTORCH_SETUP.md](/home/anurag-basistha/Projects/ESD/PYTORCH_SETUP.md): setup notes

Git / storage notes:

- [Dataset_Final.zip](/home/anurag-basistha/Projects/ESD/Dataset_Final.zip) is tracked via Git LFS
- the extracted dataset folder is intentionally ignored in Git
- generated `Results/` and `logs/` folders are ignored in Git

## Environment

This repo is set up for:

- Linux
- Python virtual environment
- CUDA-enabled PyTorch wheels
- NVIDIA RTX 3050 6 GB target hardware

The project uses the local `.venv` created by:

```bash
chmod +x scripts/setup_venv_cuda.sh
./scripts/setup_venv_cuda.sh .venv
```

## Current Model Variants

### 1. Plain Model

- pretrained `efficientnet_b0`
- no custom front-end before the backbone
- serves as the control / baseline arm

### 2. Gabor Model

- same pretrained `efficientnet_b0` backbone
- fixed multi-orientation Gabor filter bank on grayscale input
- responses concatenated with RGB channels
- learnable `1x1` adapter projects the fused tensor back to 3 channels
- then the same EfficientNet-B0 backbone is used

This is an ablation setup:

- only one model gets the Gabor front-end
- the other stays plain EfficientNet-B0

## Backbone Details

The current backbone is `efficientnet_b0`.

Approximate stock size:

- about `5.29M` parameters in the stock backbone+classifier

Compared with the previous abandoned large setup:

- old `efficientnet_v2_l` wrapper was about `119M` parameters
- `efficientnet_b0` is far more realistic for a 6 GB RTX 3050

### EfficientNet-B0 Stage Widths / Depths

Stage structure:

1. Stem:
   - `3 -> 32`
   - `3x3`
   - stride `2`
2. Stage 1:
   - `1` MBConv block
   - output width `16`
3. Stage 2:
   - `2` MBConv blocks
   - output width `24`
   - first block downsamples
4. Stage 3:
   - `2` MBConv blocks
   - output width `40`
   - first block downsamples
5. Stage 4:
   - `3` MBConv blocks
   - output width `80`
   - first block downsamples
6. Stage 5:
   - `3` MBConv blocks
   - output width `112`
7. Stage 6:
   - `4` MBConv blocks
   - output width `192`
   - first block downsamples
8. Stage 7:
   - `1` MBConv block
   - output width `320`
9. Head conv:
   - `320 -> 1280`
   - `1x1`

## Important Clarification: 512 Is Not Class Count

The model is **not** a “512-class model”.

What `512` means in this repo:

- the backbone pooled feature vector is projected into a `512`-dimensional embedding space
- this embedding is used for:
  - SupCon pretraining
  - cross-entropy classification

Current classification logic:

- backbone output width: `1280`
- embedding layer: `1280 -> 512`
- SupCon projection head: `512 -> 256 -> 256`
- cross-entropy classification head: `512 -> num_classes`

Because the current dataset has 4 classes, the final classifier is effectively:

- `512 -> 4`

So the model learns 4-way classification from a 512-dimensional embedding, not 512 classes.

## Training Pipeline

The shared training logic lives in [metric_learning_pipeline.py](/home/anurag-basistha/Projects/ESD/scripts/metric_learning_pipeline.py).

Current training flow:

1. Load pretrained `efficientnet_b0`
2. Build the plain or Gabor front-end
3. Replace the stock classifier with:
   - embedding head
   - projection head
   - classifier head
4. Optionally run supervised contrastive pretraining (`SupCon`)
5. Early-stop SupCon on validation loss with patience `20`
6. Restore best SupCon weights
7. Switch to cross-entropy classification
8. Train head-only first
9. Progressively unfreeze the backbone from the tail in chunks of `20` parameter-bearing modules
10. Early-stop every phase with patience `20`
11. Restore best phase weights before moving to the next phase
12. Evaluate on validation and test
13. Save checkpoints and metrics

## Optimization

Current optimization stack:

- base optimizer: `AdamW`
- sharpness-aware wrapper: `SAM`
- scheduler: linear warmup + cosine decay
- CUDA AMP autocast enabled during training

Current defaults:

- `image-size = 224`
- `batch-size = 512`
- `num-workers = 4`
- `prefetch-factor = PyTorch default`
- `augment-repeats = 16`
- `supcon-epochs = 200`
- `head-epochs = 200`
- `stage-epochs = 200`
- `unfreeze-chunk-size = 20`
- `supcon-early-stopping-patience = 10`
- `head-early-stopping-patience = 10`
- `stage-early-stopping-patience = 10`
- `eval-every-train-steps = 1024`
- `warmup-steps = 1024`
- `log-every-steps = 100`

Practical note:

- `200` is a hard cap, not a claim that learning is impossible beyond 200
- early stopping can stop earlier if validation does not improve for 10 consecutive validation windows in the relevant stage
- `--skip-supcon` skips the contrastive stage and starts directly with classifier fine-tuning
- if a matching `Results/.../last.pt` exists, the trainer auto-resumes from that checkpoint

## Augmentation Strategy

The project uses deterministic, split-safe online augmentation.

What is guaranteed:

- `train`, `val`, and `test` stay source-disjoint
- no original image crosses split boundaries
- each source image is expanded into exactly `16` generated variants by default
- each variant uses its own deterministic random seed based on:
  - split
  - source image index
  - repeat index
  - view index for SupCon
- augmentation magnitudes are now sampled from clipped Gaussian distributions centered on task-appropriate safe values for each transform
- the default Gaussian span is `2.0`, controlled by `--augment-gaussian-sigmas`

This means:

- 16 generated variants are produced per source image
- those variants are independent from one another in parameter sampling
- the same source image cannot leak across splits

Current split behavior:

- `train`: augmented, `16x`
- `val`: clean preprocessing only, no augmentation
- `test`: augmented, `16x`

Augmentation definitions live in:

- [scripts/metric_learning_pipeline.py](/home/anurag-basistha/Projects/ESD/scripts/metric_learning_pipeline.py)

Image-level augmentations:

| Augmentation | Current behavior | Safe Gaussian band | Hard clip | Mean / notes |
|---|---|---|---|---|
| Random resized crop retained area | Always applied | `0.70-1.00` | `0.55-1.00` | mean `0.85` |
| Random resized crop aspect ratio | Always applied, log-space | `0.80-1.25` | `0.70-1.35` | centered at `1.0` |
| Horizontal flip | Probabilistic | n/a | n/a | probability `0.50` |
| Vertical flip | Probabilistic | n/a | n/a | probability `0.05` |
| Translation X/Y | Always applied through affine | `+/-0.10` | `+/-0.15` | symmetric around `0` |
| Rotation | Always applied through affine | `+/-15 deg` | `+/-25 deg` | symmetric around `0` |
| Scale | Always applied through affine | `0.85-1.15` | `0.75-1.25` | mean `1.0` |
| Shear X/Y | Always applied through affine | `+/-8 deg` | `+/-12 deg` | symmetric around `0` |
| Perspective distortion | Always applied | `0.05-0.15` | `0.00-0.20` | mean `0.10` |
| Border truncation retained fraction | Probabilistic | `0.85-1.00` | `0.70-1.00` | probability `0.20`, mean `0.93` |
| Border truncation aspect ratio | Applied when truncation triggers, log-space | `0.85-1.15` | `0.70-1.30` | centered at `1.0` |
| Brightness | Always applied | `0.80-1.20` | `0.65-1.35` | mean `1.0` |
| Contrast | Always applied | `0.80-1.25` | `0.70-1.40` | mean `1.0` |
| Saturation | Always applied | `0.80-1.25` | `0.70-1.35` | mean `1.0` |
| Hue | Always applied | `+/-0.04` | `+/-0.08` | symmetric around `0` |
| Gamma | Always applied | `0.85-1.20` | `0.75-1.30` | mean `1.0` |
| Sharpness | Always applied | `0.70-1.40` | `0.50-1.80` | mean `1.0` |
| JPEG quality | Always applied | `50-100` | `35-100` | mean `75` |
| Resolution degradation scale | Probabilistic downsample-upsample | `0.45-0.80` | `0.25-1.00` | probability `0.20`, mean `0.60` |
| Gaussian blur sigma | Applied if sigma `> 0.05` | `0.00-1.00` | `0.00-1.80` | mean `0.50` |
| Motion blur length | Probabilistic | `3-9 px` | `1-15 px` | probability `0.18`, mean `5.5`, random direction |
| Defocus blur radius | Probabilistic | `0.8-1.8 px` | `0.0-3.0 px` | probability `0.18`, mean `1.2` |
| Gaussian noise std | Always sampled | `0.00-0.02` | `0.00-0.06` | mean `0.01` |
| Channel shift per channel | Always applied | `+/-0.02` | `+/-0.05` | symmetric around `0` |
| Grayscale mix | Always applied | `0.00-0.08` | `0.00-0.15` | mean `0.04` |
| Illumination gradient endpoints | Always applied | `0.80-1.05` | `0.70-1.10` | mean `0.95` |
| Shadow overlay darkness | Probabilistic | `0.82-0.95` | `0.70-1.00` | probability `0.20`, mean `0.90` |
| Shadow midpoint | Applied when shadow triggers | `0.35-0.65` | `0.15-0.85` | mean `0.50` |
| Shadow softness | Applied when shadow triggers | `0.08-0.18` | `0.03-0.30` | mean `0.12` |
| Specular glare center X/Y | Probabilistic | `-0.4 to 0.4` | `-0.8 to 0.8` | probability `0.25`, mean `0.0` |
| Specular glare sigma X | Applied when glare triggers | `0.08-0.18` | `0.03-0.30` | mean `0.12` |
| Specular glare sigma Y | Applied when glare triggers | `0.05-0.14` | `0.02-0.22` | mean `0.09` |
| Specular glare intensity | Applied when glare triggers | `0.10-0.22` | `0.00-0.35` | mean `0.16` |
| Specular glare color scale R | Applied when glare triggers | `0.95-1.05` | `0.90-1.10` | mean `1.00` |
| Specular glare color scale G | Applied when glare triggers | `0.97-1.08` | `0.90-1.12` | mean `1.02` |
| Specular glare color scale B | Applied when glare triggers | `1.00-1.12` | `0.92-1.18` | mean `1.06` |
| Smudge center X/Y | Probabilistic | `-0.45 to 0.45` | `-0.85 to 0.85` | probability `0.18`, mean `0.0` |
| Smudge sigma X | Applied when smudge triggers | `0.10-0.22` | `0.04-0.35` | mean `0.15` |
| Smudge sigma Y | Applied when smudge triggers | `0.07-0.18` | `0.03-0.30` | mean `0.12` |
| Smudge opacity | Applied when smudge triggers | `0.05-0.14` | `0.00-0.25` | mean `0.09` |
| Smudge stain color R | Applied when smudge triggers | `0.32-0.55` | `0.20-0.65` | mean `0.43` |
| Smudge stain color G | Applied when smudge triggers | `0.28-0.46` | `0.18-0.56` | mean `0.36` |
| Smudge stain color B | Applied when smudge triggers | `0.18-0.34` | `0.10-0.44` | mean `0.24` |
| Cutout total occlusion fraction | Always sampled | `0.08-0.15` | `0.02-0.25` | mean `0.11`, 1 to 3 holes |
| Cutout hole aspect ratio | Applied per hole | `0.80-1.25` | `0.70-1.35` | centered at `1.0` |
| Cutout fill color | Applied per hole | n/a | `0.0-1.0` | Gaussian around `0.5` with std `0.25` |

Precision note:

- every generated image passes through the full implemented augmentation pipeline
- transform magnitudes are not sampled uniformly anymore; they are drawn from clipped Gaussians centered on the safe values and clipped at the hard ceilings
- some transform parameters may still land near identity
- some transforms are probabilistic, so this is not the same as “every transform always strongly applied”

## Logging

The training pipeline writes structured JSONL logs.

Current behavior:

- a run-start event is written at launch
- every 100th optimizer step is logged by default
- per-epoch / per-phase summaries are logged
- early-stopping events are logged
- a final run-finished event is logged

Step-level logs include:

- stage
- phase name when applicable
- epoch
- epoch step
- global optimizer step
- batch size
- cumulative samples seen
- running loss
- running thresholded and raw accuracy for cross-entropy training
- current learning rates

Default log paths:

- plain model:
  - `logs/efficientnet_b0_metric_learning.log.jsonl`
- Gabor model:
  - `logs/efficientnet_b0_gabor_metric_learning.log.jsonl`

## Saved Outputs

Each run writes artifacts under `Results/...`.

Plain model output directory:

- `Results/efficientnet_b0_metric_learning`

Gabor model output directory:

- `Results/efficientnet_b0_gabor_metric_learning`

Files saved per run:

- `best.pt`
- `last.pt`
- `metrics.json`
- `validation_metrics.json`
- `test_metrics.json`

## Evaluation Metrics

The current pipeline saves standard multiclass classification metrics including:

- loss
- top-1 accuracy
- top-3 accuracy
- balanced accuracy
- macro precision
- weighted precision
- macro recall
- weighted recall
- macro F1
- weighted F1
- per-class precision
- per-class recall
- per-class specificity
- per-class F1
- confusion matrix
- one-vs-rest ROC-AUC
- one-vs-rest PR-AUC
- Cohen’s kappa
- multiclass Matthews correlation coefficient

## Commands

### Recommended Plain Run

From the repo root:

```bash
cd /home/anurag-basistha/Projects/ESD
./run_non_gabor.sh
```

Equivalent explicit command:

```bash
cd /home/anurag-basistha/Projects/ESD
source .venv/bin/activate
python scripts/train_efficientnet_b0_progressive.py \
  --dataset-root Dataset_Final \
  --weighted-sampling \
  --log-every-steps 100
```

### Gabor Run

```bash
cd /home/anurag-basistha/Projects/ESD
./run_gabor.sh
```

## CLI Flags

The training scripts [run_non_gabor.sh](/home/anurag-basistha/Projects/ESD/run_non_gabor.sh) and [run_gabor.sh](/home/anurag-basistha/Projects/ESD/run_gabor.sh) forward all extra flags to the shared parser in [metric_learning_pipeline.py](/home/anurag-basistha/Projects/ESD/scripts/metric_learning_pipeline.py).

Core data and loading:

- `--dataset-root`
  - dataset root directory
  - default: `Dataset_Final`
- `--image-size`
  - resized square input resolution
  - default: `224`
- `--batch-size`
  - per-step batch size
  - default: `512`
- `--num-workers`
  - dataloader worker count
  - default: `4`
- `--prefetch-factor`
  - dataloader batches queued ahead per worker
  - default: PyTorch default when omitted

Augmentation:

- `--augment-repeats`
  - deterministic augmentation-bank size for `train` and `test`
  - default: `16`
- `--augment-gaussian-sigmas`
  - controls how tightly augmentation magnitudes cluster around the safe band
  - default: `2.0`

Embedding and projection:

- `--embedding-dim`
  - cross-entropy embedding dimension
  - default: `512`
- `--projection-dim`
  - SupCon projection head dimension
  - default: `256`

Training stages:

- `--supcon-epochs`
  - hard cap for the SupCon stage
  - default: `200`
- `--supcon-unfreeze-backbone-modules`
  - number of backbone leaf modules trainable during SupCon
  - default: `0`
- `--skip-supcon`
  - skip contrastive pretraining and start directly with cross-entropy
- `--head-epochs`
  - hard cap for head-only cross-entropy training
  - default: `200`
- `--stage-epochs`
  - hard cap for each progressive unfreeze phase
  - default: `200`
- `--unfreeze-chunk-size`
  - how many backbone leaf modules are added per progressive phase
  - default: `20`
- `--max-progressive-phases`
  - optional limit on number of progressive cross-entropy phases
  - default: `0` meaning no extra limit
- `--progressive-phase-global-gating`
  - optional hard stop that prevents opening more backbone layers when a completed progressive phase fails to beat the global best checkpoint on the selected classifier early-stopping metric
  - default: disabled
  - use this only as an explicit experiment flag, not as the standard training behavior

Loss and classifier settings:

- `--supcon-temperature`
  - SupCon temperature
  - default: `0.07`
- `--classifier-margin`
  - cross-entropy angular margin
  - default: `0.35`
- `--classifier-scale`
  - cross-entropy logit scale
  - default: `30.0`
- `--label-smoothing`
  - cross-entropy label smoothing for cross-entropy
  - default: `0.0`
- `--confidence-threshold`
  - accuracy threshold; predictions below this confidence are counted as incorrect in cross-entropy metrics and final classification metrics
  - default: `0.80`

Optimization:

- `--supcon-head-lr`
  - SupCon head learning rate
  - default: `3e-4`
- `--supcon-backbone-lr`
  - SupCon backbone learning rate
  - default: `1e-4`
- `--head-lr`
  - classifier head learning rate
  - default: `1e-3`
- `--backbone-lr`
  - cross-entropy backbone learning rate
  - default: `1e-4`
- `--weight-decay`
  - weight decay
  - default: `1e-4`
- `--optimizer`
  - optimizer choice
  - choices: `sam`, `adamw`
  - default: `sam`
- `--precision`
  - model/input precision mode currently supported by the trainer
  - choices: `mixed`, `32`, `64`
  - default: `mixed`
- `--adam-beta1`
  - Adam or AdamW beta1
  - default: `0.9`
- `--adam-beta2`
  - Adam or AdamW beta2
  - default: `0.999`
- `--sam-rho`
  - SAM neighborhood radius when `--optimizer sam`
  - default: `0.05`
- `--grad-accum-steps`
  - gradient accumulation steps
  - default: `1`
  - current SAM path requires `1`

Sampling, initialization, and outputs:

- `--weighted-sampling`
  - enables class-balanced weighted sampling for `train`
- `--weights`
  - torchvision backbone initialization
  - choices: `default`, `none`
  - default: `default`
- `--output-dir`
  - run artifact directory
  - default: `Results/metric_learning_experiment`
- `--log-file`
  - JSONL log path
  - default: `logs/metric_learning_experiment.log.jsonl`
- `--seed`
  - random seed
  - default: `42`

Resume controls:

- `--resume-checkpoint`
  - optional explicit checkpoint path to load instead of auto-detecting `last.pt`
- `--resume-mode`
  - resume behavior
  - choices: `latest`, `global_best`, `phase_best`
  - default: `latest`
- `--resume-phase-index`
  - force restart from a specific cross-entropy phase index
  - default: `0` meaning infer from checkpoint
- `--resume-phase-name`
  - force restart from a specific cross-entropy phase name
  - default: empty

Logging, evaluation cadence, and stopping:

- `--max-train-batches`
  - optional train batch cap for smoke tests
  - default: `0` meaning no cap
- `--max-eval-batches`
  - optional eval batch cap for smoke tests
  - default: `0` meaning no cap
- `--log-every-steps`
  - train-step log cadence
  - default: `100`
- `--log-eval-every-steps`
  - eval-step log cadence
  - default: `1000`
- `--eval-every-train-steps`
  - run validation every N training steps
  - default: `1024`
- `--supcon-early-stopping-patience`
  - SupCon-stage validation patience in validation windows
  - default: `10`
- `--head-early-stopping-patience`
  - head-only cross-entropy validation patience in validation windows
  - default: `10`
- `--stage-early-stopping-patience`
  - progressive unfreeze cross-entropy validation patience in validation windows
  - default: `10`
- `--early-stopping-min-delta`
  - minimum improvement threshold
  - default: `1e-4`
- `--warmup-epochs`
  - warmup expressed in epochs if `--warmup-steps` is zero
  - default: `0`
- `--warmup-steps`
  - explicit LR warmup in optimizer steps
  - default: `1024`

Gabor-only flags:

- `--gabor-kernel-size`
  - default: `15`
- `--gabor-orientations`
  - default: `8`
- `--gabor-wavelengths`
  - comma-separated wavelengths
  - default: `4.0,8.0`
- `--gabor-sigma`
  - default: `4.5`
- `--gabor-gamma`
  - default: `0.6`

## Project History Summary

This repo went through these major steps:

1. Initial dataset audit:
   - full dataset scan
   - metadata validation
   - image integrity checks
   - extension mismatch detection
   - EXIF/GPS review
2. Dataset cleanup:
   - normalized filenames
   - corrected extension/content mismatches
   - stripped EXIF/GPS metadata
3. Class reorganization:
   - merged `plastic` into `other`
   - split metallic content out into `metal`
   - regenerated dataset metadata
4. Dataset splitting:
   - stratified `train/val/test` split at `70/20/10`
   - generated split-specific metadata files
5. PyTorch environment setup:
   - Linux venv
   - CUDA-compatible PyTorch install
6. Training pipeline evolution:
   - initial generic classifier path
   - replaced with EfficientNet-based progressive fine-tuning
   - added matched Gabor ablation
   - added SupCon pretraining
   - added cross-entropy
   - added SAM
   - added richer evaluation
   - then rewired from `efficientnet_v2_l` to `efficientnet_b0` for RTX 3050 practicality
7. Logging improvements:
   - structured run logs
   - step-level logging every 100 optimizer steps

## Current Git State

The repo is Git-initialized, and the dataset archive is tracked using Git LFS:

- [Dataset_Final.zip](/home/anurag-basistha/Projects/ESD/Dataset_Final.zip) -> Git LFS
- extracted dataset folder -> ignored
- `Results/` -> Git LFS tracked when added
- `logs/` -> Git LFS tracked when added

## Known Caveats

- The dataset currently has 4 classes, not 3.
- There is no separate `plastic` class anymore.
- The Gabor model may help on texture-sensitive distinctions, but it is not guaranteed to outperform the plain model.
- The 16 augmentation variants are deterministic per source image, not freshly resampled every epoch.
- Validation is clean and unaugmented, while test keeps the split-safe augmentation wrapper.
- `efficientnet_b0` is much more practical than the previously attempted `efficientnet_v2_l`, but training can still be long because the pipeline includes SupCon, SAM, cross-entropy, and progressive unfreezing.

## Recommended Mental Model

Think of the current plain model as:

`image -> EfficientNet-B0 backbone -> 1280 features -> 512 embedding -> cross-entropy(4 classes)`

And the Gabor model as:

`image -> Gabor front-end -> EfficientNet-B0 backbone -> 1280 features -> 512 embedding -> cross-entropy(4 classes)`
