# Electronic Smart Dustbin (ESD) Platform

The Electronic Smart Dustbin (ESD) platform is an industrial-scale ecosystem for automated waste classification and real-time fleet monitoring. It integrates an advanced machine learning pipeline with a reactive Android dashboard to provide a verified "Edge-to-Cloud" infrastructure.

## 1. System Components

### Machine Learning Engine
- **Backbone:** EfficientNet-B0 (5.3M Parameters).
- **Corpus:** 1,045,679 verified images (WSS-1.04M).
- **Taxonomy:** 15 granular classes, dynamically mappable to custom classification heads.
- **Orchestration:** Multi-phase training involving SupCon pre-training, 20-module Progressive Unfreezing, and Recursive Metric Refinement.
- **Balancing:** Automatic **Weighted Random Sampling** enabled by default to handle severe class imbalances.

### SmartBin Android Fleet Dashboard
- **Framework:** Native Kotlin, Jetpack Compose, Material 3.
- **Geospatial:** OpenStreetMap (OSM) via MapLibre.
- **Real-Time:** WebSocket-based event streaming for instantaneous updates.
- **Analytics:** Professional time-series and composition charting (Vico).

---

## 2. Platform Summary

| Metric | Specification |
| :--- | :--- |
| **Current Taxonomy** | 15 granular material classes (battery, ceramic, cardboard, etc.) |
| **Class Balancing** | **Weighted Random Sampling (Default: ON)** |
| **Unfreeze Step** | **20 modules (Default)** |
| **Optimization** | AdamW (Base) or SAM (Sharpness-Aware Minimization) |
| **Training Precision** | Mixed Precision (FP16) via `torch.amp` |
| **Mobile State** | Clean Architecture / MVVM / Hilt |

---

## 3. Documentation Index
- **`ARCHITECTURE_AND_PLAN.md`**: Technical specification of the model and training pipeline.
- **`DATASET_SPECIFICATION.md`**: Detailed breakdown of the 1.04M image corpus and class definitions.
- **`PYTORCH_SETUP.md`**: Environment configuration and execution manual.
- **`SmartBin_Android/docs/`**: Mobile-specific architectural and product specifications.

## 4. Execution Manual

### Standard Production Training
```bash
# Execute the full lifecycle with 10% validation steps and merged plastics
./run_training.sh \
  --eval-every-epochs 0.1 \
  --optimizer adamw \
  --class-mapping '{"plastic": ["plastic", "hard_plastic", "rigid_plastic", "soft_plastic"]}'
```

### Android Dashboard
```bash
cd SmartBin_Android
./gradlew :app:assembleDebug
```
