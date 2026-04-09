# ESD Platform: Technical Architecture and Training Orchestration

This document defines the current architectural specifications and training methodologies for the ESD classification system.

## 1. Model Specifications

The classification engine utilizes a highly efficient feature extractor tuned for industrial material identification.

- **Architecture:** EfficientNet-B0.
- **Parameters:** ~5.3 Million.
- **Precision:** FP16 Mixed Precision.
- **Optimization Strategy:** AdamW or SAM.
- **Class Balancing:** **Weighted Random Sampling** is utilized by default. This balances class exposure in expectation across training and reduces bias toward high-volume categories like `organic`.

---

## 2. Multi-Stage Training Pipeline

### Phase I: Supervised Contrastive Pre-training (SupCon)
- **Objective:** Optimize the embedding space clusters.
- **State:** Backbone is frozen by default to preserve ImageNet-derived spatial features.

### Phase II: Progressive 20-Module Unfreezing
- **Process:** The system iteratively unfreezes 20-module slices of the backbone. 
- **Validation Frequency:** Recommended at 0.1 (10%) of the epoch to capture peak performance in high-volume datasets.
- **Phase Rejection:** Enabled by default. If an unfreezing phase results in a validation loss worse than the previous global best, the phase is rejected, weights are restored to the safe state, and the pipeline proceeds to the next slice.

### Phase III: Recursive Refinement (Loss & Acc)
- **Mechanism:** Automatic learning rate halving upon validation plateau.
- **End-State:** Deployment-ready `best.pt` with maximized categorical separation.

---

## 3. Dynamic Inference and Display

### Class Mapping (Custom Heads)
The platform supports **Dynamic Class Merging** at both training and inference time. This allows the model to learn from all 15 granular classes while grouping them into logical UI categories (e.g., merging all plastic variants into one `plastic` head) using LogSumExp aggregation to preserve probability integrity.

---

## 4. Integrity and Visualization
- **Step-Wise Analytics:** Automated confusion matrix PNGs are generated every time a new "Phase Best" is captured.
- **Exhaustive Logging:** Every training step and validation pass is recorded in synchronized CSV files for post-run audit.

---

## 5. Deferred Work

- **Grad-CAM / Weak Localization TODO:** Add classifier-side localization output for inference-time visualization. The intended first step is Grad-CAM heatmap generation with an approximate bounding box extracted from the hottest connected region. This is explicitly deferred and is not part of the current classifier training or deployment path.
