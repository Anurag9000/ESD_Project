# ESD Platform: Dataset Specification (WSS-691K)

This document provides the exhaustive technical specification for the current image corpus utilized by the ESD classification engine post-decontamination.

## 1. Corpus Summary
- **Total Verified Images:** 691,015
- **Class Taxonomy:** 10 Categorical Material Classes
- **Data Format:** Normalized RGB (224x224 input target)
- **Integrity Standard:** 1:1 physical-to-metadata synchronization via `Dataset_Final/dataset_metadata.json`.

---

## 2. Definitive Taxonomy and Counts (Post-Decontamination)

The dataset is partitioned into **10 non-overlapping classes** based on material composition and industrial segregation requirements. The `plastic`, `cardboard`, `medical`, and `ceramic` classes have been eliminated:
- **`plastic`** → dissolved; images redistributed to `hard_plastic` or `soft_plastic`.
- **`cardboard`** → merged into `paper` (both are cellulose-based flat waste).
- **`medical`** → purged entirely (zero valid training images found).
- **`ceramic`** → purged entirely (previously decontaminated).

| Class | Image Count | Industrial Definition |
| :--- | :--- | :--- |
| **organic** | 367,253 | Biodegradable matter, food residuals, and vegetation. |
| **hard_plastic** | 149,382 | High-density polymers, rigid containers, and bottles. |
| **clothes** | 60,939 | Textiles, apparel, and woven synthetic/natural fabrics. |
| **metal** | 56,654 | Ferrous and non-ferrous metals, aluminum canisters, alloys. |
| **shoes** | 24,115 | Specialized footwear and rubber/leather-based apparel. |
| **glass** | 9,997 | Silica-based containers including clear and pigmented variants. |
| **paper** | 13,108 | Cellulose-based flat material: office print, newsprint, and corrugated cardboard. |
| **soft_plastic** | 6,213 | Flexible films, bags, and thin polymer sheets. |
| **ewaste** | 2,430 | Electronic components, PCBs, and peripheral hardware. |
| **battery** | 924 | Electrochemical cells and hazardous power storage units. |

**TOTAL: 691,015 images**

---

## 3. Data Integrity and Synchronization
The ESD platform enforces a **Zero-Mistake Data Environment**:
- **Purity:** Every class is mutually exclusive. Visual audits using `scripts/audit_dataset_by_source.py` ensure no overlap between texture-similar categories.
- **Metadata:** All training operations draw strictly from the verified `dataset_metadata.json` registry.
- **Physical Structure:** Files are organized into a flat class-root directory: `Dataset_Final/<class_name>/<image_name>.jpg`.
- **Decontamination Log:** All purged source batches (e.g., `realwaste`, `openrecycle`, `zerowaste`, `sumn2u`, `trashbox`) have been physically deleted from disk and removed from metadata.
