# Electronic Smart Dustbin (ESD) Platform

The Electronic Smart Dustbin (ESD) platform is an industrial-scale ecosystem for automated waste classification and real-time fleet monitoring. It integrates an advanced machine learning pipeline with a reactive Android dashboard to provide a verified "Edge-to-Cloud" infrastructure.

## 1. System Components

### Machine Learning Engine
- **Backbone:** Configurable; default ConvNeXt V2 Nano FCMAE
- **Corpus:** 304,258 verified physical images on disk; the next-run logical training taxonomy keeps only the 3 supervised classes
- **Taxonomy:** **3 logical training classes** — organic, metal, paper
- **Orchestration:** 8-stage pipeline: SupCon Head → SupCon Last-20 → SupCon full tail after frozen core → CE Head → CE Last-20 → CE full tail after frozen core → Recursive val_loss → Recursive val_raw_acc, with the same frozen 40-module stem/core preserved through recursive refinement
- **Logging:** Pure accuracy plus per-class accuracy and per-class average confidence; thresholded accuracy is not printed in live logs.
- **Balancing:** Balanced per-batch class cycling (default in all training scripts)
- **Visual Audit:** Startup + end-of-epoch fixed-tint test-set visualizations, plus optional Grad-CAM and calibration plots
- **Phase 0 Reconstruction Audit:** Saved MIM checkpoints can be rendered into original / masked / reconstruction previews with `scripts/visualize_phase0_reconstruction.py`

### SmartBin Android Fleet Dashboard
- **Framework:** Native Kotlin, Jetpack Compose, Material 3
- **Geospatial:** OpenStreetMap (OSM) via MapLibre
- **Real-Time:** WebSocket-based event streaming
- **Analytics:** Professional time-series and composition charting (Vico)

---

## 2. Platform Summary

| Metric               | Specification                                                                 |
| :------------------- | :---------------------------------------------------------------------------- |
| **Current Taxonomy** | **3 logical training classes** (organic, metal, paper) |
| **Total Images**     | **Trainable logical samples are projected from the physical dataset into the 3-class head** |
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
# - Phase 0 MIM trains the full backbone with the same balanced class sampler and an effective 256 batch size via 128-image physical batches plus 2-step accumulation, then SupCon/CE re-freeze the earliest 40 leaf modules
# - Dataset_Final train/val/test are rendered through the same fixed Pi-camera magenta tint by default; no stochastic augmentations remain
# - validation triggered by train-step patience
# - patience 3 across SupCon, CE head, CE stages, and recursive stages
# - automatic same-command resume from the incomplete phase's own step_last.pt or last.pt
./run_training.sh --phase0-mim --backbone convnextv2_nano --num-workers 2 --prefetch-factor 1
```

### Resume After Interruption (Exact Same Command)
```bash
cd /home/anurag-basistha/Projects/ESD
source .venv/bin/activate

# Same command as fresh launch. Completed phases are skipped; the incomplete
# phase resumes from its own step_last.pt or last.pt.
./run_training.sh --phase0-mim --backbone convnextv2_nano --num-workers 2 --prefetch-factor 1
```

### End-of-Run Test Report
- After the recursive `val_loss` and `val_raw_acc` stages complete, the pipeline runs one protected final test evaluation on the accepted checkpoint.
- The final test bundle now includes `confmat_counts_test.csv`, `confmat_rate_pct_test.csv`, `classification_report_test.csv`, `test_confusion_matrix.png`, `test_reliability_diagram.png`, `test_confidence_histogram.png`, `metrics.json`, and `summary.json`.
- The reported metric bundle includes loss, raw accuracy, top-1/top-3/top-5 accuracy, macro/weighted precision, recall, F1, balanced accuracy, macro/weighted ROC AUC OVR, macro/weighted PR AUC OVR, Cohen's kappa, MCC, ECE, MCE, Brier score, and NLL.

### Inspect Phase 0 Reconstructions
```bash
cd /home/anurag-basistha/Projects/ESD
source .venv/bin/activate

python scripts/visualize_phase0_reconstruction.py \
  --checkpoint Results/<run>/phase0_mim/best.pt \
  --dataset-root Dataset_Final \
  --split train
```
This writes a preview PNG next to the checkpoint by default. Use `--split val` or `--split test` if you want to inspect the saved model on a different split.

### Android Dashboard
```bash
cd SmartBin_Android
./gradlew :app:assembleDebug
```
