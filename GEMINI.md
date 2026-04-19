# GEMINI.md - Platform Mandates and Technical Context

This file serves as the definitive context for AI agents operating within the ESD repository. It indexes the technical truth of the current platform state.

## 1. Source of Truth (Technical Docs)
- **`README.md`**: Platform summary and execution commands.
- **`ARCHITECTURE_AND_PLAN.md`**: Model specs and multi-phase training orchestration.
- **`DATASET_SPECIFICATION.md`**: Definitive corpus counts and 6-class logical training taxonomy.
- **`PYTORCH_SETUP.md`**: Environment and runtime configuration.

## 2. Definitive State
- **Core Model:** Configurable backbone; default ConvNeXt V2 Nano FCMAE.
- **Core Dataset:** WSS-304K physical corpus (304,258 verified images, ≥224px resolution floor); 299,818 samples participate in default training after taxonomy projection.
- **Taxonomy:** 6 logical training classes — clothes, glass, metal, organic, paper, plastic. `ewaste` is excluded at load time. `hard_plastic` and `soft_plastic` are merged into `plastic`. NO battery, NO shoes, NO cardboard (merged into paper), NO medical, NO ceramic.
- **Mobile Stack:** Jetpack Compose, OSM (MapLibre), WebSockets.

## 3. Mandatory Constraints
- **Class Balancing:** **Weighted Random Sampling is MANDATORY** for all training runs to address class imbalance.
- **Unfreezing:** Progressive unfreezing must occur in **20-module increments** to maintain feature stability.
- **Zero-Mistake Data:** 1:1 metadata-to-physical sync via `dataset_metadata.json`.
- **Billing-Free:** Strictly use OpenStreetMap for all geospatial features.
- **Automated Validation:** Every training phase must generate a confusion matrix PNG and exhaustive CSV metrics.
- **Precision:** Mixed Precision (FP16) via `torch.amp` is mandatory for efficiency.
