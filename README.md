# Electronic Smart Dustbin (ESD) Platform

The Electronic Smart Dustbin (ESD) platform is an industrial-scale ecosystem for automated waste classification and real-time fleet monitoring. It integrates an advanced machine learning pipeline with a reactive Android dashboard to provide a verified "Edge-to-Cloud" infrastructure.

## 1. System Components

### Machine Learning Engine
- **Backbone:** EfficientNet-B0 (5.3M Parameters).
- **Corpus:** 691,015 verified images (WSS-691K, post-decontamination).
- **Taxonomy:** **10 material classes** — battery, clothes, ewaste, glass, hard_plastic, metal, organic, paper, shoes, soft_plastic.
- **Orchestration:** Multi-phase training involving SupCon pre-training, 20-module Progressive Unfreezing, and Recursive Metric Refinement.
- **Balancing:** Automatic **balanced per-batch class cycling** enabled by default to keep batches as class-uniform as possible while exploring new source images before repeats.

### SmartBin Android Fleet Dashboard
- **Framework:** Native Kotlin, Jetpack Compose, Material 3.
- **Geospatial:** OpenStreetMap (OSM) via MapLibre.
- **Real-Time:** WebSocket-based event streaming for instantaneous updates.
- **Analytics:** Professional time-series and composition charting (Vico).

---

## 2. Platform Summary

| Metric | Specification |
| :--- | :--- |
| **Current Taxonomy** | **10 material classes** (battery, clothes, ewaste, glass, hard_plastic, metal, organic, paper, shoes, soft_plastic) |
| **Total Images** | **691,015 (post-decontamination)** |
| **Class Balancing** | **Balanced Per-Batch Cycling (Default)** |
| **Unfreeze Step** | **20 modules (Default)** |
| **Optimization** | AdamW (Base) or SAM (Sharpness-Aware Minimization) |
| **Training Precision** | Mixed Precision (FP16) via `torch.amp` |
| **Mobile State** | Clean Architecture / MVVM / Hilt |

---

## 3. Documentation Index
- **`ARCHITECTURE_AND_PLAN.md`**: Technical specification of the model and training pipeline including the 10-class index table.
- **`DATASET_SPECIFICATION.md`**: Detailed breakdown of the 691K image corpus and class definitions post-decontamination.
- **`PYTORCH_SETUP.md`**: Environment configuration and execution manual.
- **`SmartBin_Android/docs/`**: Mobile-specific architectural and product specifications.

## 4. Execution Manual

### Standard Production Training
```bash
# Execute the full lifecycle with 10% validation steps
# Classes auto-detected from Dataset_Final/ subfolders
./run_training.sh \
  --eval-every-epochs 0.1 \
  --optimizer adamw
```

### Android Dashboard
```bash
cd SmartBin_Android
./gradlew :app:assembleDebug
```
