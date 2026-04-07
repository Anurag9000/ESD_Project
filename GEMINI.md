# GEMINI.md - Platform Context and Operational Mandates

This file provides the primary context for AI agents operating within the Electronic Smart Dustbin (ESD) repository. It indexes the authoritative documentation and defines the technical boundaries of the platform.

## 1. Documentation Index (Source of Truth)
- **`README.md`**: Top-level platform summary, taxonomy counts (1.04M images), and primary execution commands.
- **`ARCHITECTURE_AND_PLAN.md`**: Exhaustive detail on the EfficientNet-B0 backbone, progressive unfreezing, and recursive refinement phases.
- **`DATASET_RESEARCH_AND_REMAP.md`**: Provenance of all 20+ dataset sources and the deterministic remapping logic for the 15-class taxonomy.
- **`PYTORCH_SETUP.md`**: CUDA environment requirements and Python virtual environment lifecycle.
- **`SmartBin_Android/docs/`**: Mobile-specific specifications for the fleet monitoring dashboard.

## 2. Technical Core
- **Backbone:** `efficientnet_b0` (5.3M params).
- **Taxonomy:** 15 classes (Industrial scale).
- **Training Strategy:** SupCon pre-training -> Progressive Layer Unfreezing -> Recursive Loss Refinement -> Recursive Accuracy Refinement.
- **Integrity Gate:** `Dataset_Final/dataset_metadata.json` is the sole source of truth for the image corpus.

## 3. Dataset Integrity Mandates (Zero-Mistake Standard)
- **Purification:** `cardboard` must never overlap with `paper`. `shoes` must remain isolated from `clothes`.
- **Synchronization:** Any physical file addition or deletion must be reflected immediately in `dataset_metadata.json` via the audit scripts.
- **Audit Tool:** Use `scripts/metric_learning_pipeline.py` for automated validation and confusion matrix generation.

## 4. Mobile Dashboard Constraints
- **Mapping:** Strict usage of **OpenStreetMap (OSM)** via MapLibre to ensure billing-free operation.
- **Real-Time:** Connectivity via WebSockets for instantaneous fleet updates.
- **Architecture:** Clean Architecture with Hilt and Jetpack Compose.

## 5. Development Conventions
- **Optimization:** Prefer **SAM** or **AdamW** with cosine annealing.
- **Hardware:** RTX 3050 (6GB VRAM) target.
- **Logging:** Every training phase must generate exhaustive CSV metrics and phase-best confusion matrices.
