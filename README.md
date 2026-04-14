# Electronic Smart Dustbin (ESD) Platform

The Electronic Smart Dustbin (ESD) platform is an industrial-scale ecosystem for automated waste classification and real-time fleet monitoring. It integrates an advanced machine learning pipeline with a reactive Android dashboard to provide a verified "Edge-to-Cloud" infrastructure.

## 1. System Components

### Machine Learning Engine
- **Backbone:** EfficientNet-B0 (5.3M Parameters)
- **Corpus:** 308,008 verified images (WSS-308K, post-decontamination + 200px resolution floor)
- **Taxonomy:** **8 material classes** — clothes, ewaste, glass, hard_plastic, metal, organic, paper, soft_plastic
- **Orchestration:** Multi-phase training: SupCon pre-training → 20-module Progressive Unfreezing → Recursive Metric Refinement
- **Balancing:** Balanced per-batch class cycling (default)

### SmartBin Android Fleet Dashboard
- **Framework:** Native Kotlin, Jetpack Compose, Material 3
- **Geospatial:** OpenStreetMap (OSM) via MapLibre
- **Real-Time:** WebSocket-based event streaming
- **Analytics:** Professional time-series and composition charting (Vico)

---

## 2. Platform Summary

| Metric | Specification |
| :--- | :--- |
| **Current Taxonomy** | **8 material classes** (clothes, ewaste, glass, hard_plastic, metal, organic, paper, soft_plastic) |
| **Total Images** | **308,008 verified (≥200px floor, post-decontamination)** |
| **Class Balancing** | Balanced Per-Batch Cycling (Default) |
| **Unfreeze Step** | 20 modules (Default) |
| **Optimization** | AdamW (Base) or SAM |
| **Training Precision** | Mixed Precision (FP16) via `torch.amp` |
| **Mobile State** | Clean Architecture / MVVM / Hilt |

---

## 3. Documentation Index
- **`ARCHITECTURE_AND_PLAN.md`**: Model specs, 8-class index table, training pipeline, resolution policy.
- **`DATASET_SPECIFICATION.md`**: 308K corpus breakdown per class with decontamination history.
- **`PYTORCH_SETUP.md`**: Environment configuration and execution manual.
- **`SmartBin_Android/docs/`**: Mobile-specific architectural and product specifications.

## 4. Execution Manual

### Standard Production Training
```bash
# Execute the full lifecycle with 10% validation steps
# Classes auto-detected from Dataset_Final/ subfolders (8 classes)
./run_training.sh \
  --eval-every-epochs 0.1 \
  --optimizer adamw
```

### Android Dashboard
```bash
cd SmartBin_Android
./gradlew :app:assembleDebug
```
