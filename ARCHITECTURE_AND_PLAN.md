# ESD Platform: Technical Architecture and Training Orchestration

This document defines the current architectural specifications and training methodologies for the ESD classification system.

## 1. Model Specifications

The classification engine utilizes a highly efficient feature extractor tuned for industrial material identification.

- **Architecture:** EfficientNet-B0.
- **Parameters:** ~5.3 Million.
- **Precision:** FP16 Mixed Precision.
- **Optimization Strategy:** AdamW or SAM.
- **Output Classes:** **10** (battery, clothes, ewaste, glass, hard_plastic, metal, organic, paper, shoes, soft_plastic).
- **Class Balancing:** **Balanced per-batch class cycling** is utilized by default. Each batch is made as class-uniform as possible, and each class is sampled from shuffled per-class source-image queues before repeating items.

---

## 2. Class Taxonomy (10 Classes)

| Index | Class Name | Description |
| :--- | :--- | :--- |
| 0 | `battery` | Electrochemical cells and hazardous power storage |
| 1 | `clothes` | Textiles, apparel, woven synthetic/natural fabrics |
| 2 | `ewaste` | Electronic components, PCBs, peripheral hardware |
| 3 | `glass` | Silica-based containers (clear and pigmented) |
| 4 | `hard_plastic` | High-density polymers, rigid containers, bottles |
| 5 | `metal` | Ferrous/non-ferrous metals, aluminum, alloys |
| 6 | `organic` | Biodegradable matter, food residuals, vegetation |
| 7 | `paper` | Cellulose flat material: office print, newsprint, cardboard |
| 8 | `shoes` | Footwear, rubber/leather-based apparel |
| 9 | `soft_plastic` | Flexible films, bags, thin polymer sheets |

---

## 3. Multi-Stage Training Pipeline

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

## 4. Dynamic Inference and Display

### Class Mapping (Custom Heads)
The platform supports **Dynamic Class Merging** at both training and inference time. This allows the model to learn from all 10 granular classes while optionally grouping them into logical UI categories using LogSumExp aggregation to preserve probability integrity.

> **Note:** The legacy `plastic` alias merging `soft_plastic + hard_plastic` has been removed. Both are now fully independent first-class categories.

---

## 5. Integrity and Visualization
- **Step-Wise Analytics:** Automated confusion matrix PNGs are generated every time a new "Phase Best" is captured.
- **Exhaustive Logging:** Every training step and validation pass is recorded in synchronized CSV files for post-run audit.

---

## 6. Deferred Work

- **Grad-CAM / Weak Localization TODO:** Add classifier-side localization output for inference-time visualization. The intended first step is Grad-CAM heatmap generation with an approximate bounding box extracted from the hottest connected region. This is explicitly deferred and is not part of the current classifier training or deployment path.
