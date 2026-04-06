# Project ESD: Industrial Waste Classification Baseline (WSS-1.04M)

## Overview
Project ESD provides a high-fidelity, dynamic waste classification system utilizing a PyTorch-driven training pipeline and a Jetpack Compose Android application. The system is designed for a runtime-configurable taxonomy, where a subset of trained classes is selected for monitoring, and the remainder are autonomously aggregated into a standardized `Other` category.

## Technical Architecture

### 1. Training Pipeline
The core classification model is based on the EfficientNet-B0 architecture, optimized for deployment on mobile and edge devices.
- **Progressive Training:** Implements iterative unfreezing of model layers, transitioning from the classification head through backbone slices to preserve pretrained feature integrity while adapting to specialized waste domains.
- **Recursive Refinement:** Employs a multi-stage recursive refinement process. 
    - **Stage A (Val Loss):** Iteratively optimizes for validation loss minimization with early stopping and automatic learning rate halving upon convergence.
    - **Stage B (Val Accuracy):** Further refines the model targeting validation raw accuracy, ensuring maximal categorical separation.
- **Optimizer:** AdamW with weight decay and a cosine annealing scheduler.

### 2. SmartBin Android Application
A modern, reactive mobile interface for real-time waste stream monitoring.
- **Framework:** Jetpack Compose, Material 3.
- **State Management:** Unidirectional Data Flow (UDF) powered by Kotlin Coroutines and StateFlow.
- **Networking:** WebSocket integration for low-latency event streaming from the SmartBin hardware.

---

## Dataset Integration and Taxonomy

The current baseline, **WSS-1.04M**, integrates approximately 1.04 million images across a unified taxonomy. All external sources were normalized and remapped according to the following authoritative material-first definitions:

### Standardized Taxonomy
- **`battery`**: Electrochemical cells and related hazardous power storage.
- **`clothes`**: Textiles, apparel, and woven synthetic/natural fabrics.
- **`ewaste`**: Consumer electronics, circuit boards, and peripheral hardware.
- **`glass`**: Silica-based containers, including clear and pigmented variants.
- **`medical`**: Specialized clinical waste and personal protective equipment (PPE).
- **`metal`**: Ferrous and non-ferrous scrap, aluminum canisters, and industrial alloys.
- **`organic`**: Biodegradable matter, food residuals, and yard waste.
- **`paper`**: Cellulose-based materials, including high-grade paper and corrugated cardboard.
- **`plastic`**: Comprehensive polymer category, currently encompassing rigid, hard, and soft variants.
- **`shoes`**: Specialized footwear category.

### Authoritative Mapping Registry
| Source Dataset | Original Class | Target Mapping |
| :--- | :--- | :--- |
| **RealWaste** | Cardboard | `paper` |
| | Food Organics / Vegetation | `organic` |
| | Textile Trash | `clothes` |
| | Miscellaneous Trash | `trash` |
| **TrashNet** | Cardboard | `paper` |
| **TACO** | Various (Material-based) | Direct mapping to `plastic`, `metal`, `glass`, etc. |
| **TrashBox** | E-waste / Medical | `ewaste` / `medical` |
| **Recycle_Net** | Glass / Paper / Plastic / Metal | Standardized material mapping |
| **CompostNet** | Cardboard | `paper` (Verified for material purity) |
| **Wastevision** | E-waste / Medical | `ewaste` / `medical` |

---

## Execution Manual

### Model Training
The training process is orchestrated via standardized shell scripts to ensure reproducibility across environments.
```bash
# Execute the full progressive and recursive pipeline
./run_training.sh
```

### Android Development
Ensure the Android SDK and JDK 21 are configured.
```bash
cd SmartBin_Android
./gradlew :app:assembleDebug :app:testDebugUnitTest
```

## Maintenance & Integrity
- **Purge Policy:** All legacy Gabor-filter logic and experimental "Scourge" artifacts have been permanently removed to favor the production-grade baseline.
- **Taxonomy Drift:** Any additions to `Dataset_Final/` are dynamically detected by the training engine.
