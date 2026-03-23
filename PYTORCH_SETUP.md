# PyTorch Setup

This project is set up for Linux with a Python virtual environment and CUDA-enabled PyTorch wheels.

## Create the venv and install CUDA builds

```bash
chmod +x scripts/setup_venv_cuda.sh
./scripts/setup_venv_cuda.sh .venv
```

The installer uses [requirements-cu128.txt](/home/anurag-basistha/Projects/ESD/requirements-cu128.txt), which points `pip` at the official CUDA 12.8 PyTorch wheel index.

## Progressive Fine-Tuning

Both training scripts use the largest pretrained EfficientNet available in the installed `torchvision`: `efficientnet_v2_l`.

The fine-tuning schedule is progressive:
- phase 1: train only the classification head
- later phases: unfreeze the backbone from the end in chunks, default `20` parameter-bearing modules at a time

## Standard Fine-Tuner

```bash
source .venv/bin/activate
python scripts/train_efficientnet_v2l_progressive.py \
  --dataset-root Dataset_Final \
  --batch-size 4 \
  --num-workers 8 \
  --head-epochs 2 \
  --stage-epochs 1 \
  --unfreeze-chunk-size 20 \
  --weighted-sampling
```

## Gabor Fine-Tuner

```bash
source .venv/bin/activate
python scripts/train_efficientnet_v2l_gabor_progressive.py \
  --dataset-root Dataset_Final \
  --batch-size 4 \
  --num-workers 8 \
  --head-epochs 2 \
  --stage-epochs 1 \
  --unfreeze-chunk-size 20 \
  --weighted-sampling
```

## Notes

- Both scripts default to pretrained ImageNet weights with `EfficientNet_V2_L_Weights.DEFAULT`.
- The Gabor variant adds a fixed multi-orientation Gabor bank plus a learnable adapter before the pretrained backbone.
- The scripts use `cuda` automatically when `torch.cuda.is_available()` is true.
- On an RTX 3050 6GB, lower `--image-size` or `--batch-size` if you hit out-of-memory.
- The dataset structure expected by the scripts is:
  - `Dataset_Final/train/<class>`
  - `Dataset_Final/val/<class>`
  - `Dataset_Final/test/<class>`
