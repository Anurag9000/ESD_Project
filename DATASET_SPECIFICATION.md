# ESD Platform: Dataset Specification (WSS-1.04M)

This document provides the exhaustive technical specification for the current image corpus utilized by the ESD classification engine.

## 1. Corpus Summary
- **Total Verified Images:** 1,045,679
- **Class Taxonomy:** 15 Categorical Material Classes
- **Data Format:** Normalized RGB (224x224 input target)
- **Integrity Standard:** 1:1 physical-to-metadata synchronization via `Dataset_Final/dataset_metadata.json`.

---

## 2. Definitive Taxonomy and Counts

The dataset is partitioned into 15 non-overlapping classes based on material composition and industrial segregation requirements.

| Class | Image Count | Industrial Definition |
| :--- | :--- | :--- |
| **organic** | 391,545 | Biodegradable matter, food residuals, and vegetation. |
| **metal** | 134,510 | Ferrous and non-ferrous metals, aluminum canisters, alloys. |
| **clothes** | 104,417 | Textiles, apparel, and woven synthetic/natural fabrics. |
| **hard_plastic** | 67,352 | High-density polymers and rigid plastic containers. |
| **paper** | 65,583 | Cellulose-based materials (non-corrugated), office/newsprint. |
| **cardboard** | 64,050 | Corrugated materials, heavy cartons, and shipping containers. |
| **glass** | 61,296 | Silica-based containers including clear and pigmented variants. |
| **plastic** | 42,494 | General polymer waste and mixed plastic variants. |
| **soft_plastic** | 38,346 | Flexible films, bags, and thin polymer sheets. |
| **shoes** | 37,294 | Specialized footwear and rubber/leather-based apparel. |
| **battery** | 11,695 | Electrochemical cells and hazardous power storage units. |
| **medical** | 8,422 | Clinical waste and personal protective equipment (PPE). |
| **ewaste** | 7,910 | Electronic components, PCBs, and peripheral hardware. |
| **rigid_plastic** | 7,600 | Specialized high-rigidity industrial polymers. |
| **ceramic** | 3,173 | Clay-based materials and pottery residuals. |

---

## 3. Data Integrity and Synchronization
The ESD platform enforces a **Zero-Mistake Data Environment**:
- **Purity:** Every class is mutually exclusive. Visual audits ensure no overlap between texture-similar categories (e.g., `cardboard` vs. `paper`).
- **Metadata:** All training operations draw strictly from the verified `dataset_metadata.json` registry.
- **Physical Structure:** Files are organized into a flat class-root directory: `Dataset_Final/<class_name>/<image_name>.jpg`.
- **LFS Tracking:** Large-scale archives are tracked via Git LFS to ensure repository performance and data permanence.
