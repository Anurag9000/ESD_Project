# GEMINI.md - Platform Mandates and Technical Context

This file serves as the definitive context for AI agents operating within the ESD repository. It indexes the technical truth of the current platform state.

## 1. Source of Truth (Technical Docs)
- **`README.md`**: Platform summary and execution commands.
- **`ARCHITECTURE_AND_PLAN.md`**: Model specs and multi-phase training orchestration.
- **`DATASET_SPECIFICATION.md`**: Definitive 1.04M image counts and 15-class taxonomy.
- **`PYTORCH_SETUP.md`**: Environment and runtime configuration.

## 2. Definitive State
- **Core Model:** EfficientNet-B0 (5.3M params).
- **Core Dataset:** WSS-1.04M (1,045,679 verified images).
- **Taxonomy:** 15 Material Classes (Industrial Standard).
- **Mobile Stack:** Jetpack Compose, OSM (MapLibre), WebSockets.

## 3. Mandatory Constraints
- **Zero-Mistake Data:** 1:1 metadata-to-physical sync via `dataset_metadata.json`.
- **Billing-Free:** Strictly use OpenStreetMap for all geospatial features.
- **Automated Validation:** Every training phase must generate a confusion matrix PNG and exhaustive CSV metrics.
- **Precision:** Mixed Precision (FP16) via `torch.amp` is mandatory for efficiency.
