# ESD Platform: Technical Architecture and Training Orchestration

This document defines the current architectural specifications and training methodologies for the ESD classification system.

## 1. Model Specifications

- **Architecture:** EfficientNet-B0
- **Parameters:** ~5.3 Million
- **Precision:** FP16 Mixed Precision
- **Optimization Strategy:** AdamW or SAM
- **Output Classes:** **8**
- **Class Balancing:** Balanced per-batch class cycling (default). Each batch is made as class-uniform as possible, sampling from shuffled per-class queues before repeating.

---

## 2. Class Taxonomy (8 Classes — Ground Reality)

| Index | Class Name | Image Count | Description |
| :--- | :--- | :--- | :--- |
| 0 | `clothes` | 40,295 | Textiles, apparel, woven fabrics |
| 1 | `ewaste` | 2,147 | Electronic components, PCBs, hardware |
| 2 | `glass` | 9,997 | Silica-based containers (clear/pigmented) |
| 3 | `hard_plastic` | 15,297 | Rigid polymers, containers, bottles |
| 4 | `metal` | 55,628 | Ferrous/non-ferrous metals, aluminum |
| 5 | `organic` | 168,439 | Biodegradable: food, vegetation |
| 6 | `paper` | 11,126 | Cellulose flat material, cardboard |
| 7 | `soft_plastic` | 5,079 | Flexible films, bags, thin sheets |

> **Note:** `battery` and `shoes` have been permanently eliminated. 100% of shoe images were sub-200px thumbnails (zero training value). Only 29 battery images survived the 200px resolution floor — insufficient for any meaningful learning.

---

## 3. Multi-Stage Training Pipeline

### Phase I: Supervised Contrastive Pre-training (SupCon)
- **Objective:** Optimize embedding space clusters.
- **State:** Backbone frozen by default to preserve ImageNet-derived spatial features.

### Phase II: Progressive 20-Module Unfreezing
- **Process:** Iteratively unfreezes 20-module slices of the backbone.
- **Validation Frequency:** 0.1 (10%) of epoch to capture peak performance.
- **Phase Rejection:** If a phase degrades validation loss, weights are restored and the pipeline proceeds to the next slice.

### Phase III: Recursive Refinement
- **Mechanism:** Automatic learning rate halving upon validation plateau.
- **End-State:** Deployment-ready `best.pt` with maximized categorical separation.

---

## 4. Resolution Quality Policy
- **Minimum Image Size:** 200×200px (strictly enforced on physical disk)
- **Rationale:** EfficientNet-B0 targets 224×224. Any image below 200px requires >1.12x upscaling, which is safe. Images previously at 60-136px (all shoes, most batteries) required 2-4x upscaling, producing hallucinated texture artifacts that caused the model to learn thumbnail blur patterns instead of material features.

---

## 5. Integrity and Visualization
- **Confusion Matrix:** Auto-generated PNG on every new phase best.
- **Exhaustive Logging:** Every training step recorded in synchronized CSV files.

---

## 6. Deferred Work
- **Grad-CAM Localization:** Classifier-side heatmap generation with approximate bounding box extraction. Explicitly deferred.
