# Electronic Smart Dustbin (ESD) Platform

The Electronic Smart Dustbin (ESD) platform is an industrial-scale ecosystem for automated waste classification and real-time fleet monitoring. It integrates an advanced machine learning pipeline with a reactive Android dashboard to provide a verified "Edge-to-Cloud" infrastructure.

## 1. System Components

### Machine Learning Engine
- **Backbone:** EfficientNet-B0 (5.3M Parameters).
- **Corpus:** 1,045,679 verified images (WSS-1.04M).
- **Taxonomy:** 15 non-overlapping material classes.
- **Orchestration:** Multi-phase training involving SupCon pre-training, Progressive Unfreezing, and Recursive Metric Refinement.

### SmartBin Android Fleet Dashboard
- **Framework:** Native Kotlin, Jetpack Compose, Material 3.
- **Geospatial:** OpenStreetMap (OSM) via MapLibre.
- **Real-Time:** WebSocket-based event streaming for instantaneous updates.
- **Analytics:** Professional time-series and composition charting (Vico).

---

## 2. Platform Summary

| Metric | Specification |
| :--- | :--- |
| **Current Taxonomy** | battery, cardboard, ceramic, clothes, ewaste, glass, hard_plastic, medical, metal, organic, paper, plastic, rigid_plastic, shoes, soft_plastic |
| **Dataset Size** | 1,045,679 verified images |
| **Model Architecture** | EfficientNet-B0 |
| **Optimization** | AdamW + SAM (Sharpness-Aware Minimization) |
| **Training Precision** | Mixed Precision (FP16) via `torch.amp` |
| **Mobile State** | Clean Architecture / MVVM / Hilt |

---

## 3. Documentation Index
- **`ARCHITECTURE_AND_PLAN.md`**: Technical specification of the model and training pipeline.
- **`DATASET_SPECIFICATION.md`**: Detailed breakdown of the 1.04M image corpus and class definitions.
- **`PYTORCH_SETUP.md`**: Environment configuration and execution manual.
- **`SmartBin_Android/docs/`**: Mobile-specific architectural and product specifications.

## 4. Execution Manual

### Training Pipeline
```bash
./scripts/setup_venv_cuda.sh .venv
source .venv/bin/activate
./run_full_training_pipeline.sh
```

### Android Dashboard
```bash
cd SmartBin_Android
./gradlew :app:assembleDebug
```
