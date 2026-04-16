# ESD Platform: Technical Architecture and Training Orchestration

This document defines the current architectural specifications and training methodologies for the ESD classification system.

## 1. Model Specifications

- **Architecture:** EfficientNet-B0
- **Parameters:** ~5.3 Million (backbone: 4.0M, head: 1.3M)
- **Precision:** FP16 Mixed Precision via `torch.amp`
- **Optimization:** AdamW with per-stage learning rate groups
- **Output Classes:** **8**

---

## 2. Class Taxonomy (8 Classes — Ground Reality)

| Index | Class Name    | Image Count | Description                              |
| :---- | :------------ | :---------- | :--------------------------------------- |
| 0     | `clothes`     | 40,295      | Textiles, apparel, woven fabrics         |
| 1     | `ewaste`      | 2,147       | Electronic components, PCBs, hardware    |
| 2     | `glass`       | 9,997       | Silica-based containers (clear/pigmented)|
| 3     | `hard_plastic`| 15,297      | Rigid polymers, containers, bottles      |
| 4     | `metal`       | 55,628      | Ferrous/non-ferrous metals, aluminum     |
| 5     | `organic`     | 168,439     | Biodegradable: food, vegetation          |
| 6     | `paper`       | 11,126      | Cellulose flat material, cardboard       |
| 7     | `soft_plastic`| 5,079       | Flexible films, bags, thin sheets        |

> **Eliminated:** `battery` (only 29 images survived the 200px floor) and `shoes` (100% sub-200px thumbnails, zero training value).

---

## 3. Eight-Stage Training Pipeline

The pipeline follows the research-validated principle: **Contrastive representation learning first, then supervised fine-tuning on top.** This maximises inter-class separation in the embedding space before any cross-entropy decision boundaries are drawn.

### Stage 1 — SupCon Head Warm-up (Frozen Backbone)
| Parameter            | Value  |
| :------------------- | :----- |
| **Trainable:**       | `embedding` (1280→128) + `embedding_norm` + `projection_head` (128→128) |
| **Frozen:**          | Entire EfficientNet-B0 backbone (130 leaf modules, 4.0M params) |
| **Loss:**            | Supervised Contrastive (SupCon, temperature=0.07) |
| **Head LR:**         | `3e-3` |
| **Backbone LR:**     | `0.0` (frozen) |
| **Duration:**        | `supcon_head_epochs=5` max |
| **Stopping:**        | Early stopping patience=5 on SupCon val_loss |
| **Goal:**            | Orient the randomly-initialised projection head into a stable contrastive attractor before touching backbone weights |

### Stage 2 — SupCon Last-20 Modules
| Parameter            | Value  |
| :------------------- | :----- |
| **Trainable:**       | Top 20 semantic leaf modules + SupCon head |
| **Loss:**            | Supervised Contrastive (SupCon) |
| **Head LR:**         | `3e-3` |
| **Backbone LR:**     | `5e-5` |
| **Stopping:**        | Early stopping patience=5 on SupCon val_loss |
| **Goal:**            | Start adapting only the semantic tail while the frozen core stays intact |

### Stage 3 — SupCon Last-40 Modules
| Parameter            | Value  |
| :------------------- | :----- |
| **Trainable:**       | Top 40 leaf modules (of 130 total) + SupCon head |
| **Frozen forever:**  | First 90 leaf modules (stem, stages 0–5: edges, textures, patterns) = **928,416 params** |
| **Loss:**            | Supervised Contrastive (SupCon) |
| **Head LR:**         | `3e-3` |
| **Backbone LR:**     | `5e-5` |
| **Stopping:**        | Early stopping patience=5 on SupCon val_loss |
| **Goal:**            | Adapt the high-level semantic representations specifically to distinguish waste categories (ewaste vs soft_plastic vs organic etc) while keeping universally-useful low-level encoders intact |

### Stage 4 — CE Head Warm-up (Backbone Re-frozen)
| Parameter            | Value  |
| :------------------- | :----- |
| **Trainable:**       | `ce_head` (128→8) + `embedding` + `embedding_norm` |
| **Frozen:**          | Entire backbone (all 130 leaf modules) + projection_head |
| **Loss:**            | Cross-Entropy (label_smoothing=0.0 default) |
| **Head LR:**         | `1e-3` |
| **Backbone LR:**     | `0.0` (frozen) |
| **Duration:**        | `head_epochs=5` epochs max, patience=5 |
| **Goal:**            | **Critical safety gate.** The `ce_head` is randomly initialised. Without this warm-up, random CE gradients would backpropagate into the carefully arranged contrastive embedding space and corrupt it. This phase stabilises the CE hyperplanes between the tight SupCon clusters before any backbone gradients flow. |

### Stage 5 — CE Last-20 Modules
| Parameter            | Value  |
| :------------------- | :----- |
| **Trainable:**       | Top 20 semantic leaf modules + CE head |
| **Loss:**            | Cross-Entropy |
| **Head LR:**         | `1e-4` after exponential decay from the `1e-3` head-only phase |
| **Backbone LR:**     | `1e-5` |
| **Stopping:**        | Early stopping patience=5 on `val_loss` |
| **Phase rejection:** | If a phase fails to beat the global best, its weights are NOT used to initialise the next phase (global best is restored) |
| **Goal:**            | Begin supervised backbone adaptation while preserving the frozen core |

### Stage 6 — CE Last-40 Modules
| Parameter            | Value  |
| :------------------- | :----- |
| **Trainable:**       | Top 40 semantic leaf modules + CE head |
| **Loss:**            | Cross-Entropy |
| **Head LR:**         | `1e-5` |
| **Backbone LR:**     | `1e-5` |
| **Stopping:**        | Early stopping patience=5 on `val_loss` |
| **Phase rejection:** | Same global-best gate as Stage 5 |
| **Goal:**            | Final supervised semantic alignment before recursive cleanup |

### Stage 7 — Recursive val_loss Refinement
- Runs `run_recursive_refinement.py` with `metric=val_loss`, `threshold=1e-4`.
- Automatically halves learning rates on plateau and repeats until loss stops improving.
- Produces `accepted_best.pt`.

### Stage 8 — Recursive val_raw_acc Refinement
- Same mechanism as Stage 5 but optimises for raw validation accuracy.
- LRs bootstrapped from Stage 5 final state (halved further).
- Produces final deployment-ready `accepted_best.pt`.

---

## 4. Backbone Module Map (EfficientNet-B0)

| Top-level Module | Type                  | Params     | SupCon Stage | Notes |
| :--------------- | :-------------------- | :--------- | :----------- | :---- |
| `features[0]`    | Conv2dNormActivation  | 928        | ❄️ FROZEN    | Stem — raw pixels → edges |
| `features[1]`    | Sequential (MBConv)   | 1,448      | ❄️ FROZEN    | Basic textures |
| `features[2]`    | Sequential (MBConv)   | 16,714     | ❄️ FROZEN    | Texture gradients |
| `features[3]`    | Sequential (MBConv)   | 46,640     | ❄️ FROZEN    | Patterns |
| `features[4]`    | Sequential (MBConv)   | 242,930    | ❄️ FROZEN    | Object parts |
| `features[5]`    | Sequential (MBConv)   | 543,148    | ❄️ FROZEN    | Mid-level semantics |
| `features[6]`    | Sequential (MBConv×4) | 2,026,348  | 🔥 UNFROZEN  | High-level semantics (richest) |
| `features[7]`    | Sequential (MBConv)   | 717,232    | 🔥 UNFROZEN  | Abstract waste features |
| `features[8]`    | Conv2dNormActivation  | 412,160    | 🔥 UNFROZEN  | Final projection conv |

> The 90-module freeze boundary was selected by diagnostic analysis of the full 130-leaf-module tree. It permanently protects universally-useful visual primitives (edges, textures, patterns) that would degrade if touched by task-specific gradients.

---

## 5. Checkpointing Strategy

| Event | File | Location |
| :-- | :-- | :-- |
| Every step | `step_last.pt` | `Results/<run>/progressive/` |
| Every epoch | `last.pt` | `Results/<run>/progressive/` |
| SupCon global best | `supcon_best.pt` | `Results/<run>/progressive/` |
| Final progressive best | `best.pt` | `Results/<run>/progressive/` |
| Phase-best snapshot | `best.pt` | `Results/<run>/progressive/phases/<phase_name>/` |
| Phase-best confusion matrix | `best_confusion_matrix.png` | `Results/<run>/progressive/phases/<phase_name>/` |
| Per-epoch visual audit | `visualizations/<epoch_label>/...` | `Results/<run>/progressive/phases/<phase_name>/` |

Resume is fully automatic: re-running `./run_training.sh --batch-size 224` detects the most recent run stamp and injects `--resume-checkpoint step_last.pt` automatically.

---

## 6. Dataset and Resolution Policy

- **Corpus:** WSS-308K — 308,008 verified images, all ≥200×200px
- **Minimum Size:** 200px (strictly enforced on physical disk)
- **Split Ratios:** 70% train / 20% val / 10% test
- **Augmentation:** 16× deterministic split-safe augmentation (train only; val/test use raw images)
- **Class Balancing:** Balanced per-batch class cycling is the default and mandatory production path.
- **Interpretability Audit:** At run start and after every completed epoch, the pipeline generates clean no-augmentation test-set visualizations:
  global t-SNE, per-class t-SNE highlights, all-layer activation summaries, and a full test atlas.
- **Augmentation types:** Random crops, flips, colour jitter, Gaussian blur, shadow, glare, motion blur, defocus, resolution degradation, truncation, smudging

---

## 7. Automated Validation Artefacts

- **Confusion Matrix:** Auto-generated PNG on every new phase-best checkpoint. Filename: `phase_N_best_confusion_matrix.png`
- **Training CSV:** `train_metrics.csv` — every training step logged
- **Validation CSV:** `val_metrics.csv` — every validation window logged
- **JSONL Log:** Full structured event stream, one JSON object per event

---

## 8. Deferred Work

- **Grad-CAM Localisation:** Classifier-side heatmap generation with approximate bounding box extraction. Explicitly deferred to post-training.
