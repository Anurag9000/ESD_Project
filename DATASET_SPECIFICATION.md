# ESD Platform: Dataset Specification (WSS-308K)

This document provides the exhaustive technical specification for the current image corpus after rigorous multi-stage decontamination and resolution quality enforcement.

## 1. Corpus Summary
- **Total Verified Images:** 308,008
- **Class Taxonomy:** 8 Material Classes
- **Data Format:** All images verified ≥ 200×200px (high-fidelity, no thumbnail upscaling)
- **Integrity Standard:** 1:1 physical-to-metadata synchronization via `Dataset_Final/dataset_metadata.json`.

---

## 2. Definitive Taxonomy and Counts

The dataset is partitioned into **8 non-overlapping, high-resolution classes**. Eliminated classes:
- **`battery`** → purged (only 29 images survived the 200px threshold, insufficient for training)
- **`shoes`** → purged (100% of images were sub-200px thumbnails; zero training value)
- **`plastic`** → dissolved into `hard_plastic` and `soft_plastic`
- **`cardboard`** → merged into `paper`
- **`medical`** → purged (zero valid training images)
- **`ceramic`** → purged (previously decontaminated)

| # | Class | Image Count | Industrial Definition |
| :--- | :--- | :--- | :--- |
| 0 | **organic** | 168,439 | Biodegradable matter, food residuals, vegetation |
| 1 | **metal** | 55,628 | Ferrous/non-ferrous metals, aluminum, alloys |
| 2 | **clothes** | 40,295 | Textiles, apparel, woven synthetic/natural fabrics |
| 3 | **hard_plastic** | 15,297 | High-density polymers, rigid containers, bottles |
| 4 | **paper** | 11,126 | Cellulose flat material: office print, newsprint, corrugated cardboard |
| 5 | **glass** | 9,997 | Silica-based containers (clear and pigmented) |
| 6 | **soft_plastic** | 5,079 | Flexible films, bags, thin polymer sheets |
| 7 | **ewaste** | 2,147 | Electronic components, PCBs, peripheral hardware |

**TOTAL: 308,008 images**

---

## 3. Data Integrity and Synchronization
The ESD platform enforces a **Zero-Mistake Data Environment**:
- **Resolution Floor:** Every image has been physically checked to verify it is ≥ 200×200px. Images below this threshold were permanently deleted, eliminating all "shortcut learning" from upscaled thumbnails.
- **Purity:** Every class is mutually exclusive. Visual audits using `scripts/audit_dataset_by_source.py` (randomized on every run) confirm no class contamination.
- **Metadata:** All training operations draw strictly from the verified `dataset_metadata.json` registry.
- **Physical Structure:** `Dataset_Final/<class_name>/<image_name>.ext`

---

## 4. Source Attribution and Provenance

The **WSS-308K** corpus is a curated meta-dataset synthesized from over 40 distinct industrial and academic sub-collections. The following primary sources are cited and acknowledged:

### 4.1. Core Academic Baselines
- **TrashNet (Stanford University):** The original waste classification baseline; provided fundamental visual patterns for glass, paper, and metal.
- **TrashBox (ResearchGate/GitHub):** Integrated multi-class trash corpus with detailed plastic subclasses; now mapped into `paper`, `ewaste`, `glass`, `metal`, `hard_plastic`, and `soft_plastic`.
- **TACO (Trash Annotations in Context):** Source for high-fidelity images of litter in diverse, complex out-of-door environments.
- **Recycleye Waste-v1:** Public research segment of the Recycleye industrial sorting challenge (Glass, Paper, Metal, Plastic).
- **Garbage Classification (Kaggle/Yang):** 6-class fundamental material distribution.

### 4.2. Domain-Specific Aggregations
- **Clothing & Textiles:** Aggregated from the **Zalando Store Crawl**, **Biodegradable Fabrics Dataset**, and **E-commerce Clothing Attributes** collections.
- **E-Waste:** Sourced from the **EWasteNet** research dataset (PCB classification) and custom e-waste vision scrapings.
- **Organic Matter:** Derived from the **CompostNet** and **Biological Waste** vision projects.

### 4.3. Industrial Synthetic Enhancements
To improve robustness against deformation and rare perspectives, the following synthetic sets were incorporated:
- **Greensorter Automated Sorting Set:** (Metal, Plastic, Glass).
- **Bottle-Synthetic-Images:** Targeted glass container variations.
- **Tin-and-Steel-Cans-Synthetic:** High-variance metal geometry.
- **Mechanical-Parts-Boltnut:** Fine-grained metal texture detection.

A full, per-image source registry is maintained in `Dataset_Final/dataset_metadata.json`. Use `scripts/audit_dataset_by_source.py` to generate randomized visual audits for any of these specific provenance sources.
