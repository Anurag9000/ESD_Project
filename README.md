# Electronic Smart Dustbin (ESD) Platform

The Electronic Smart Dustbin (ESD) platform is an industrial-scale ecosystem for automated waste classification and real-time fleet monitoring. It integrates an advanced machine learning pipeline with a reactive Android dashboard to provide a verified "Edge-to-Cloud" infrastructure.

## 1. System Components

### Machine Learning Engine
- **Backbone:** Configurable; default ConvNeXt V2 Nano FCMAE
- **Corpus:** 304,258 verified images (WSS-304K, post-decontamination + 224px resolution floor)
- **Taxonomy:** **8 material classes** — clothes, ewaste, glass, hard_plastic, metal, organic, paper, soft_plastic
- **Orchestration:** 8-stage pipeline: SupCon Head → SupCon Last-20 → SupCon Last-40 → CE Head → CE Last-20 → CE Last-40 → Recursive val_loss → Recursive val_raw_acc
- **Balancing:** Balanced per-batch class cycling (default in all training scripts)
- **Visual Audit:** Startup + end-of-epoch clean test-set visualizations, plus optional Grad-CAM and calibration plots

### SmartBin Android Fleet Dashboard
- **Framework:** Native Kotlin, Jetpack Compose, Material 3
- **Geospatial:** OpenStreetMap (OSM) via MapLibre
- **Real-Time:** WebSocket-based event streaming
- **Analytics:** Professional time-series and composition charting (Vico)

---

## 2. Platform Summary

| Metric               | Specification                                                                 |
| :------------------- | :---------------------------------------------------------------------------- |
| **Current Taxonomy** | **8 material classes** (clothes, ewaste, glass, hard_plastic, metal, organic, paper, soft_plastic) |
| **Total Images**     | **304,258 verified (≥224px floor, post-decontamination)**                     |
| **Class Balancing**  | Balanced per-batch class cycling                                              |
| **Unfreeze Step**    | 20 leaf modules per SupCon and CE progressive phase                           |
| **Optimization**     | AdamW + warmup-cosine decay (default)                                         |
| **Training Precision**| Mixed Precision (FP16) via `torch.amp`                                       |
| **Mobile State**     | Clean Architecture / MVVM / Hilt                                              |

---

## 3. Documentation Index
- **`ARCHITECTURE_AND_PLAN.md`**: Full staged pipeline specification, per-stage LRs, backbone module map, checkpointing strategy.
- **`DATASET_SPECIFICATION.md`**: 304K corpus breakdown per class with decontamination history.
- **`PYTORCH_SETUP.md`**: Environment configuration and execution manual.
- **`scripts/evaluate_external_holdout.py`**: No-augmentation evaluation on a genuinely unseen dataset root.
- **`scripts/gradcam_classifier.py`**: Class-specific Grad-CAM overlays for trained checkpoints.
- **`SmartBin_Android/docs/`**: Mobile-specific architectural and product specifications.

## 4. Execution Manual

### Standard Production Training (Fresh Run)
```bash
cd /home/anurag-basistha/Projects/ESD
source .venv/bin/activate

# Launches the full staged lifecycle automatically.
# Defaults now include:
# - eval every 0.01 epoch
# - patience 5 across SupCon, CE head, CE stages, and recursive stages
# - startup + per-epoch clean test-set visualizations
./run_training.sh --backbone convnextv2_nano --num-workers 2 --prefetch-factor 1
```

### Resume After Interruption (Exact Same Command)
```bash
cd /home/anurag-basistha/Projects/ESD
source .venv/bin/activate

# Automatically detects the most recent run, finds step_last.pt or last.pt,
# and resumes from the exact last training step.
./run_training.sh --backbone convnextv2_nano --num-workers 2 --prefetch-factor 1
```

### Android Dashboard
```bash
cd SmartBin_Android
./gradlew :app:assembleDebug
```
