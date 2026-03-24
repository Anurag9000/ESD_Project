# PyTorch Setup

This project is set up for Linux with a Python virtual environment and CUDA-enabled PyTorch wheels.

## Create the venv and install CUDA builds

```bash
chmod +x scripts/setup_venv_cuda.sh
./scripts/setup_venv_cuda.sh .venv
```

The installer uses [requirements-cu128.txt](/home/anurag-basistha/Projects/ESD/requirements-cu128.txt), which points `pip` at the official CUDA 12.8 PyTorch wheel index.

## Training Variants

There are two matched training scripts:
- [train_efficientnet_b0_progressive.py](/home/anurag-basistha/Projects/ESD/scripts/train_efficientnet_b0_progressive.py): plain pretrained `efficientnet_b0`
- [train_efficientnet_b0_gabor_progressive.py](/home/anurag-basistha/Projects/ESD/scripts/train_efficientnet_b0_gabor_progressive.py): the same EfficientNet backbone, but with a fixed multi-orientation Gabor front-end and a learnable adapter before the backbone

Both scripts share the same pipeline implementation in [metric_learning_pipeline.py](/home/anurag-basistha/Projects/ESD/scripts/metric_learning_pipeline.py). The plain model is the control arm; only the Gabor variant gets the extra texture front-end.

## Training Pipeline

The full pipeline is:
1. Initialize a pretrained `efficientnet_b0` backbone.
2. Run supervised contrastive pretraining on the embedding head.
3. Early-stop the SupCon phase on validation SupCon loss with patience `20`.
4. Restore the best SupCon checkpoint.
5. Replace classification training with ArcFace metric learning.
6. Train the ArcFace head first with the backbone frozen.
7. Progressively unfreeze the backbone from the end in chunks, default `20` parameter-bearing modules at a time.
8. For every ArcFace phase, early-stop on validation loss with patience `20`.
9. Restore the best phase checkpoint before moving to the next phase.
10. Evaluate the best overall model on validation and test.

## Optimization

The trainers use:
- `AdamW` as the base optimizer
- `SAM` (Sharpness-Aware Minimization) for flatter minima
- warmup + cosine decay scheduling
- AMP autocast on CUDA

This does not guarantee the global minimum, but it is a strong practical setup for robust fine-tuning.

## Augmentation Strategy

All three splits use deterministic, split-safe online augmentation:
- `train`, `val`, and `test` remain source-disjoint because they still read from separate folders
- each source image is expanded into `20` deterministic augmented variants by default with `--augment-repeats 20`
- the same original file never crosses split boundaries
- each variant is keyed by split, source index, and repeat index, which keeps the probability of duplicate augmentations low

The augmentation stack includes only image-classification-safe transforms:
- random resized crop
- horizontal and vertical flip
- affine transforms
- perspective distortion
- brightness, contrast, saturation, hue
- gamma and sharpness perturbation
- blur and additive noise
- JPEG compression
- channel shift and grayscale mixing
- illumination gradient overlays
- shadow overlays
- specular glare overlays
- cutout / coarse occlusion

ArcFace training also uses conservative batch mixing:
- `MixUp`
- `CutMix`

## Outputs

Each run writes result artifacts to its output directory under `Results/...`:
- `best.pt`
- `last.pt`
- `metrics.json`
- `validation_metrics.json`
- `test_metrics.json`

Each run also writes a structured JSONL training log under `logs/...` with:
- run start metadata
- every 100th optimizer step by default via `--log-every-steps 100`
- phase-by-phase epoch summaries
- early-stopping events
- final validation/test summary

Saved metrics include the standard classification set used in model comparison:
- loss
- top-1 accuracy
- top-3 accuracy
- balanced accuracy
- macro and weighted precision
- macro and weighted recall
- macro and weighted F1
- per-class precision, recall, specificity, and F1
- confusion matrix
- one-vs-rest ROC-AUC
- one-vs-rest PR-AUC
- Cohen's kappa
- multiclass Matthews correlation coefficient

## Standard Fine-Tuner

```bash
source .venv/bin/activate
python scripts/train_efficientnet_b0_progressive.py \
  --dataset-root Dataset_Final \
  --batch-size 4 \
  --num-workers 4 \
  --augment-repeats 20 \
  --supcon-epochs 200 \
  --head-epochs 200 \
  --stage-epochs 200 \
  --unfreeze-chunk-size 20 \
  --log-every-steps 100 \
  --early-stopping-patience 20 \
  --weighted-sampling
```

## Gabor Fine-Tuner

```bash
source .venv/bin/activate
python scripts/train_efficientnet_b0_gabor_progressive.py \
  --dataset-root Dataset_Final \
  --batch-size 4 \
  --num-workers 4 \
  --augment-repeats 20 \
  --supcon-epochs 200 \
  --head-epochs 200 \
  --stage-epochs 200 \
  --unfreeze-chunk-size 20 \
  --log-every-steps 100 \
  --early-stopping-patience 20 \
  --weighted-sampling
```

## Notes

- Both scripts default to pretrained ImageNet weights with `EfficientNet_B0_Weights.DEFAULT`.
- ArcFace was chosen over AdaFace because AdaFace is mainly useful for quality-varying face embeddings, while ArcFace is the cleaner metric-learning choice for generic multiclass material/object classification.
- The current dataset no longer has a separate `plastic` class; it was merged into `other`.
- The Gabor variant may help if texture cues matter, but it can also become a front-end bottleneck. That is why it is kept as a separate ablation instead of being forced into both models.
- The dataset structure expected by the scripts is:
  - `Dataset_Final/train/<class>`
  - `Dataset_Final/val/<class>`
  - `Dataset_Final/test/<class>`
