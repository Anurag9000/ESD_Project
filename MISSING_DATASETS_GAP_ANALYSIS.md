# ESD Platform: Missing Datasets Gap Analysis

This document identifies the primary large-scale waste datasets that are **not** currently integrated into the **WSS-308K** corpus. Integration of these sources would expand the corpus to ~620,000 images and provide industry-leading granularity.

---

## 1. The High-Priority Targets

### 1.1. Chinese Household Waste (ModelScope)
- **Total Images:** ~150,000
- **Primary Classes (4):** Recyclable, Kitchen, Hazardous, Other Waste.
- **Granular Classes (265):** Deep hierarchy including specialized items like specific fruit types, pharmaceutical packaging, and diverse plastic polymers.
- **Value:** Adds massive diversity for "In-the-Wild" domestic sorting.

### 1.2. Huawei Cloud Garbage Classification (competition)
- **Total Images:** ~100,000
- **Total Classes:** **400**
- **Classes Summary:**
    - Recyclable (Disposable cups, bottles, cans, paper, etc.)
    - Hazardous (Batteries, lamps, thermometers, drugs)
    - Kitchen/Organic (Food waste, bones, peels)
    - Other (Cigarette butts, dust, ceramic)
- **Value:** Provides the most granular classification mapping in the world.

### 1.3. TrashBox (ResearchGate/GitHub)
- **Total Images:** 17,785
- **Class Breakdown:**
    - **Medical Waste:** 2,010 images (Syringes, gloves, masks)
    - **E-Waste:** 2,883 images (PCBs, laptops, smartphones)
    - **Plastic:** 2,669 images
    - **Paper:** 2,695 images
    - **Metal:** 2,586 images
    - **Glass:** 2,528 images
    - **Cardboard:** 2,414 images
- **Value:** Critical for bootstrapping the currently thin `ewaste` class and adding a `medical` segment.

---

## 2. Specialized Industrial/Environmental Targets

### 2.1. ZeroWaste (CVPR 2022)
- **Total Images:** ~12,000 (4,503 fully annotated, 6,212 unlabeled)
- **Class Breakdown:**
    - Soft Plastic
    - Rigid Plastic
    - Cardboard
    - Metal
- **Value:** Essential for **Industrial Sorting Line** perspectives (cluttered, overlapping objects on conveyor belts).

### 2.2. Waste_pictures (Kaggle)
- **Total Images:** ~24,000
- **Total Classes:** 34
- **Value:** Fine-grained textures for specific packaging types (PET, HDPE, Aluminum vs. Steel).

### 2.3. CODD (2024 Benchmark)
- **Total Images:** ~10,000
- **Class Breakdown:**
    - Brick
    - Concrete
    - Timber
    - Rebar
    - Plastic Tube
- **Value:** Expands the model taxonomy to **Construction & Demolition (C&D)** waste—a massive sector currently missing from our material list.

---

## 3. Remote/Contextual Targets

| Dataset | Scale | Specialty |
| :--- | :--- | :--- |
| **TrashCan 1.1** | 7,212 | Underwater/Oceanic debris and marine life. |
| **UAVVaste** | 772 | Drone-based overhead detection (3,716 annotations). |
| **SpectralWaste** | 852 | Multimodal (RGB + Hyperspectral) for sorting automation. |

---

## 4. Integration Roadmap
To maintain the **Zero-Mistake Data Environment**, any future integration of these sets must follow the **Multi-Stage Clean** protocol:
1.  Verify Image Resolution (Floor: 200px).
2.  De-duplicate against the existing 308K corpus.
3.  Harmonize Taxonomy (Map 400 Huawei classes to our 8 core material classes).
4.  Verify Metadata (1:1 JSON sync via `dataset_metadata.json`).
