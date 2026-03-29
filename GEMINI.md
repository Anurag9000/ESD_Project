# GEMINI.md - Project Context & Instructions

## Project Overview
**ESD (Electronic Smart Dustbin)** is an integrated waste-classification platform. It combines edge-based AI classification with a mobile monitoring ecosystem. The project is split into two primary domains:
1.  **Machine Learning (Python/PyTorch):** Training a robust 4-class classifier (`metal`, `organic`, `other`, `paper`) optimized for an NVIDIA RTX 3050 (6GB VRAM). It uses EfficientNet-B0 with an optional Gabor filter front-end for enhanced texture feature extraction.
2.  **Android Application (Kotlin/Compose):** A mobile dashboard located in `SmartBin_Android/` for visualizing bin locations and waste analytics in real-time.

## Technical Architecture
### Machine Learning Pipeline
- **Backbone:** `efficientnet_b0` (5.29M parameters).
- **Techniques:** 
    - **SupCon:** Optional supervised contrastive pre-training to improve embedding space geometry.
    - **SAM (Sharpness-Aware Minimization):** Used for better generalization.
    - **Progressive Unfreezing:** Gradual fine-tuning of the backbone in chunks (default 20 modules).
    - **Augmentation:** Deterministic, split-safe online augmentation with 16 variants per image.
- **Classes:** 4 classes (`metal`, `organic`, `other`, `paper`). Note: `plastic` was merged into `other`.

### Android Application
- **Stack:** Kotlin, Jetpack Compose, Android Studio.
- **Features:** Google Maps/OpenStreetMap integration for bin tracking, analytics charts (Pie/Bar), and real-time WebSocket updates for bin events.

## Directory Structure
- `scripts/`: Contains the core ML logic.
    - `metric_learning_pipeline.py`: Shared training logic, augmentation, and evaluation.
    - `train_efficientnet_b0_progressive.py`: Trainer for the baseline model.
    - `train_efficientnet_b0_gabor_progressive.py`: Trainer for the Gabor-augmented model.
- `SmartBin_Android/`: The Android project root.
- `Dataset_Final/`: Local dataset directory (ignored by git, but structure is `train/val/test`).
- `Results/`: Directory for model checkpoints (`best.pt`, `last.pt`) and metrics.
- `logs/`: Structured JSONL logs for training runs.

## Key Commands
### Environment Setup
```bash
chmod +x scripts/setup_venv_cuda.sh
./scripts/setup_venv_cuda.sh .venv
source .venv/bin/activate
```

### Training Models
- **Plain Model:** `./run_non_gabor.sh`
- **Gabor Model:** `./run_gabor.sh`
- **Manual Execution (Plain):**
  ```bash
  python scripts/train_efficientnet_b0_progressive.py --dataset-root Dataset_Final --weighted-sampling
  ```

### Dataset Management
- `python scripts/reorganize_dataset.py`: Cleans and restructures raw data.
- `python scripts/create_dataset_splits.py`: Performs stratified 70/20/10 splitting.

## Development Conventions
- **Python:** Use `AdamW` or `SAM` for optimization. Follow the progressive unfreezing pattern defined in `metric_learning_pipeline.py`.
- **Android:** Prefer Jetpack Compose for UI. Maintain modularity between `data`, `ui`, and `domain` layers.
- **Logging:** All training events should be logged to the JSONL structured logs for consistency.
- **Testing:** Ensure `val` split remains clean (unaugmented) while `test` uses the 16x augmentation bank for robustness verification.

## Core Documentation
- `README.md`: High-level overview and dataset stats.
- `REPO_BLUEPRINT.md`: Detailed training philosophy and hyperparameter rationale.
- `SMART_DUSTBIN_PLATFORM_SPEC.md`: Full system architecture including Edge/Pi4 and FastAPI backend.
- `PYTORCH_SETUP.md`: Environment and CUDA troubleshooting.
