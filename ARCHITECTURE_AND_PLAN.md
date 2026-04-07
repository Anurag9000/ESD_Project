# ESD Platform: Technical Architecture and Training Orchestration

This document defines the current architectural specifications and training methodologies for the ESD classification system.

## 1. Model Specifications

The classification engine utilizes a highly efficient feature extractor tuned for industrial material identification.

- **Architecture:** EfficientNet-B0.
- **Parameters:** ~5.3 Million.
- **Precision:** FP16 Mixed Precision.
- **Classification Head:** 1280-dimension Global Average Pooling (GAP) followed by a 15-node linear classifier.
- **Optimization Strategy:** Sharpness-Aware Minimization (SAM) combined with AdamW weight decay regularization.

---

## 2. Multi-Stage Training Pipeline

The system progresses through four deterministic phases to achieve maximal convergence and categorical separation.

### Phase I: Supervised Contrastive Pre-training (SupCon)
- **Objective:** Optimize the embedding space by pulling positive samples (same class) together and pushing negative samples apart.
- **Outcome:** A robust feature representation that precedes the traditional classification training.

### Phase II: Progressive Backbone Unfreezing
- **Process:** The model begins with a frozen backbone. It iteratively unfreezes 20-module slices, moving from the classification head toward the input layer.
- **Goal:** Stabilize the head before fine-tuning higher-order spatial features.

### Phase III: Recursive Validation Loss Minimization
- **Mechanism:** Iteratively trains the model until the validation loss improvement drops below the 1e-4 threshold.
- **Automation:** Upon each plateau, the system restores the best state, halves the learning rate, and restarts the iteration.

### Phase IV: Recursive Validation Accuracy Refinement
- **Final Polish:** Specifically targets the maximization of categorical precision by refining the model based on raw validation accuracy.

---

## 3. Data Robustness and Visualization

### Deterministic Augmentation
Every training sample undergoes a 16x split-safe online augmentation cycle, simulating shadows, glares, and lens distortions found in physical dustbin environments.

### Step-Wise Analytics
- **Confusion Matrices:** Automated PNG generation at the end of every "Best Phase" save.
- **CSV Metrics:** Exhaustive logging of every training step and validation pass into `train_metrics.csv` and `val_metrics.csv`.
- **Checkpointing:** High-frequency saving of `step_last.pt` and `best_phase_X.pt` to ensure zero data loss during training.
