# GEMINI.md - Platform Mandates and Technical Context

This file serves as the definitive context for AI agents operating within the ESD repository. It indexes the technical truth of the current platform state.

## 1. Source of Truth (Technical Docs)
- **`README.md`**: Platform summary and execution commands.
- **`ARCHITECTURE_AND_PLAN.md`**: Model specs and multi-phase training orchestration.
- **`DATASET_SPECIFICATION.md`**: Definitive 1.04M image counts and 15-class taxonomy.
- **`PYTORCH_SETUP.md`**: Environment and runtime configuration.

## 2. Definitive State
- **Core Model:** EfficientNet-B0 (5.3M params).
- **Core Dataset:** WSS-308K (308,008 verified images, ≥200px resolution floor).
- **Taxonomy:** 8 Material Classes — clothes, ewaste, glass, hard_plastic, metal, organic, paper, soft_plastic. NO battery (purged, insufficient count post-200px filter), NO shoes (purged, 100% sub-200px thumbnails), NO plastic, NO cardboard (merged into paper), NO medical, NO ceramic.
- **Mobile Stack:** Jetpack Compose, OSM (MapLibre), WebSockets.

## 3. Mandatory Constraints
- **Class Balancing:** **Weighted Random Sampling is MANDATORY** for all training runs to address class imbalance.
- **Unfreezing:** Progressive unfreezing must occur in **20-module increments** to maintain feature stability.
- **Zero-Mistake Data:** 1:1 metadata-to-physical sync via `dataset_metadata.json`.
- **Billing-Free:** Strictly use OpenStreetMap for all geospatial features.
- **Automated Validation:** Every training phase must generate a confusion matrix PNG and exhaustive CSV metrics.
- **Precision:** Mixed Precision (FP16) via `torch.amp` is mandatory for efficiency.
