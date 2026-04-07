# Electronic Smart Dustbin (ESD) Platform

## Project Summary
The Electronic Smart Dustbin (ESD) platform is a production-grade ecosystem for automated waste classification and real-time fleet monitoring. It integrates a high-performance machine learning pipeline with a native Android dashboard to provide a seamless "Edge-to-Cloud" experience.

The project is defined by its massive, purified dataset of over **1.04 million images** and a multi-stage training orchestration designed for maximal categorical separation and generalization.

---

## Technical Architecture

### 1. Machine Learning Pipeline (Python/PyTorch)
The core classification engine utilizes the **EfficientNet-B0** architecture, selected for its optimal balance of parameter efficiency and feature representation.

#### Training Methodology
- **Supervised Contrastive Learning (SupCon):** An optional pre-training phase that optimizes the embedding space by pulling similar classes together and pushing dissimilar ones apart before classification training begins.
- **Progressive Unfreezing:** To preserve pretrained features, the pipeline iteratively unfreezes the backbone in chunks (defaulting to 20-module slices), transitioning from the classification head down through the feature extractor.
- **Recursive Refinement Suite:**
    - **Loss Optimization:** Iteratively reduces validation loss through learning rate halving and early stopping.
    - **Accuracy Optimization:** A final refinement stage targeting validation raw accuracy to maximize real-world performance.
- **Advanced Optimization:** Employs **AdamW** and **Sharpness-Aware Minimization (SAM)** to find flatter minima, enhancing the model's ability to generalize to unseen waste items.
- **Deterministic Augmentation:** A 16x split-safe online augmentation bank (shadows, glares, rotations) ensures the model is robust against environmental variances in physical dustbins.

### 2. SmartBin Android Application (Kotlin/Compose)
A native monitoring interface built for city-scale fleet management.
- **Fleet Mapping:** Utilizes **OpenStreetMap (OSM)** via MapLibre for a billing-free, high-performance geographic view of all active bins.
- **Real-Time Synchronicity:** Integrated with a WebSocket/SSE backend to provide "instant-pulse" visual feedback when a waste event occurs at the edge.
- **Deep Analytics:** Features professional-grade charts (Vico) with custom **Seasonal Filtering** (Q1: Feb-Apr, Q2: May-Jul, etc.) and multi-bin aggregation logic.
- **Architecture:** Follows **Clean Architecture** and **MVVM**, utilizing Hilt for dependency injection and StateFlow for reactive UI updates.

---

## The ESD Dataset (WSS-1.04M)

The platform is powered by the **WSS-1.04M** corpus, containing **1,045,679 verified images** across 15 synchronized classes. This dataset is the result of an exhaustive normalization and purification process from over 20 primary and academic sources.

### Standardized Taxonomy & Counts
| Class | Image Count | Description |
| :--- | :--- | :--- |
| **organic** | 391,545 | Biodegradable waste, food scraps, and vegetation. |
| **metal** | 134,510 | Aluminum cans, ferrous scrap, and industrial metal. |
| **clothes** | 104,417 | Textiles, apparel, and woven fabrics. |
| **hard_plastic** | 67,352 | High-density polymers and rigid containers. |
| **paper** | 65,583 | Non-corrugated cellulose, office paper, and newsprint. |
| **cardboard** | 64,050 | Corrugated shipping materials and heavy cartons. |
| **glass** | 61,296 | Silica-based containers (clear and pigmented). |
| **plastic** | 42,494 | General polymer waste and mixed plastics. |
| **soft_plastic** | 38,346 | Flexible films, bags, and thin polymer sheets. |
| **shoes** | 37,294 | Dedicated footwear and leather goods category. |
| **battery** | 11,695 | Electrochemical cells and power storage units. |
| **medical** | 8,422 | Clinical waste and personal protective equipment. |
| **ewaste** | 7,910 | Electronic components, PCBs, and peripherals. |
| **rigid_plastic** | 7,600 | Specialized high-rigidity industrial polymers. |
| **ceramic** | 3,173 | Clay-based materials and pottery shards. |

### Integration & Purity
The dataset was constructed by merging and remapping several authoritative sources, including **TrashNet**, **TACO**, **RealWaste**, **WasteVision**, and **TrashBox**. 

**Purification Highlights:**
- **Cardboard/Paper Separation:** Exhaustive keyword and visual audit to ensure all corrugated materials are strictly in `cardboard` and high-grade cellulose is in `paper`.
- **Footwear Isolation:** Segregated `shoes` from general `clothes` and `organic` (leather) to prevent texture confusion.
- **Metadata Synchronization:** 100% 1:1 physical-to-logical synchronization verified via automated audit scripts.

---

## Deployment & Execution

### Training Environment
```bash
# Setup the CUDA-enabled environment
./scripts/setup_venv_cuda.sh .venv
source .venv/bin/activate

# Execute the full progressive and recursive pipeline
./run_full_training_pipeline.sh
```

### Dashboard Development
```bash
cd SmartBin_Android
./gradlew :app:assembleDebug
```

## System Integrity
Every component of this repository—from the data metadata to the model saving logic—has been exhaustively verified. The pipeline generates automated confusion matrices and exhaustive metrics CSVs at every step boundary to ensure complete transparency in model evolution.
