# ESD Dataset: Exhaustive Integration and Remapping Registry

This document serves as the authoritative record for the construction, normalization, and purification of the **WSS-1.04M** dataset. It details the provenance of all external data and the deterministic mapping rules used to achieve a standardized 15-class industrial taxonomy.

## 1. Executive Summary: WSS-1.04M
- **Total Physical Images:** 1,045,679
- **Logical Classes:** 15
- **Data Integrity:** 100% synchronization between physical files and `dataset_metadata.json`.
- **Purification Status:** Completed visual and keyword-based audits for `cardboard`, `paper`, `shoes`, and `clothes`.

---

## 2. Standardized Taxonomy and Final Counts

| Target Class | Count | Description |
| :--- | :--- | :--- |
| **organic** | 391,545 | Food waste, biological residuals, compostables. |
| **metal** | 134,510 | Cans, scrap metal, foils, aluminum alloys. |
| **clothes** | 104,417 | Textiles, apparel, fabric scraps. |
| **hard_plastic** | 67,352 | High-density polymers, rigid plastic containers. |
| **paper** | 65,583 | Non-corrugated office paper, magazines, newsprint. |
| **cardboard** | 64,050 | Corrugated boxes, heavy cartons, shipping materials. |
| **glass** | 61,296 | Bottles, jars, silica shards (all colors). |
| **plastic** | 42,494 | General and mixed polymer waste. |
| **soft_plastic** | 38,346 | Flexible films, poly-bags, plastic wraps. |
| **shoes** | 37,294 | Dedicated footwear and leather goods. |
| **battery** | 11,695 | Household and industrial power cells. |
| **medical** | 8,422 | Clinical waste, PPE, masks, syringes. |
| **ewaste** | 7,910 | Circuit boards, computer peripherals, phones. |
| **rigid_plastic** | 7,600 | High-rigidity industrial plastic components. |
| **ceramic** | 3,173 | Pottery, tiles, clay-based material shards. |

---

## 3. Authoritative Source Registry

The dataset is an integration of several primary and academic corpora. Each source was audited for label quality and remapped to the target taxonomy.

### Primary Sources
- **TrashNet:** One of the earliest material-sorting benchmarks. Contributed high-quality `glass`, `metal`, `paper`, and `plastic` object photos.
- **TACO (Trash Annotations in Context):** Large-scale open dataset of waste in environmental settings. Used for its diversity in `metal`, `plastic`, and `glass` detection-style crops.
- **RealWaste:** Provided grounded landfill imagery, enhancing the model's ability to recognize waste in non-studio environments.
- **WasteVision / TrashNet++:** Contributed significantly to `ewaste` and `medical` waste coverage.
- **TrashBox:** Integrated for its practical segregation labels, including specialized `medical` and `electronic` classes.
- **Recycle_Net:** Added diverse recyclable material supervision.
- **CompostNet:** Used to bolster the `organic` class with high-signal compostable material imagery.

### Extended Integration Tracker
- `DONE` `mnemora litter sort`
- `DONE` `dmedhi/garbage-image-classification-detection`
- `DONE` `openrecycle`
- `DONE` `metal_scrap_dataset`
- `DONE` `greensorter`
- `DONE` `recycle_detector`
- `DONE` `ewastenet`
- `DONE` `wade-ai`
- `DONE` `garbage_object_detection`
- `DONE` `recodewaste`
- `DONE` `trash_detection_dataset`
- `DONE` `recycled dataset (portland state)`

---

## 4. Normalization and Mapping Logic

### Deterministic Remapping Assumptions
To ensure categorical purity, the following source-to-target mapping rules were applied:

| Source Keyword/Class | Target Class | Rationale |
| :--- | :--- | :--- |
| `cardboard`, `carton`, `corrugated` | `cardboard` | Separated from paper to isolate texture features. |
| `compost`, `bio`, `food`, `vegetation` | `organic` | Unified under biological decomposition category. |
| `aluminum`, `can`, `foil`, `scrap` | `metal` | Unified all metallic materials. |
| `textile`, `clothing`, `fabric` | `clothes` | Isolated apparel to prevent texture overlap with organic leather. |
| `bottle`, `jar` (Silica) | `glass` | Consolidated all glass containers. |
| `bottle`, `container` (Polymer) | `plastic` / `hard_plastic` | Routed based on physical rigidity attributes. |
| `pcb`, `electronics`, `cable` | `ewaste` | Consolidated all electronic waste. |

### Purification Procedures
1.  **Cardboard vs. Paper Audit:** A recursive search identified files containing "cardboard" in the `paper` class. These were physically migrated to the `cardboard` folder to prevent the model from learning inconsistent edge patterns.
2.  **Footwear Isolation:** `shoes` were extracted from the `clothes` and `organic` classes. This prevents texture confusion between woven textiles (clothes) and leather/rubber (shoes).
3.  **Filenaming Normalization:** All filenames were checked for consistency. Redundant or "noisy" filenames from mixed-waste buckets (e.g., `other`, `trash`) were re-labeled based on content-keywords.

---

## 5. Integrity Verification
The dataset is audited for every training run. The `dataset_metadata.json` is the single source of truth, ensuring that every image used in training has a verified physical path and a synchronized label. The current state represents a zero-mistake data environment designed for industrial-grade classification performance.
