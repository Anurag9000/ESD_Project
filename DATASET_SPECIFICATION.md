# ESD Platform: Dataset Specification (WSS-304K)

This document provides the exhaustive technical specification for the current image corpus after rigorous multi-stage decontamination and resolution quality enforcement.

## 1. Corpus Summary
- **Total Verified Physical Images:** 304,258
- **Default Trainable Logical Images:** 299,818 (`ewaste` is excluded at load time)
- **Class Taxonomy:** 6 logical training classes
- **Data Format:** All images verified ≥ 224×224px (high-fidelity, no thumbnail upscaling)
- **Integrity Standard:** 1:1 physical-to-metadata synchronization via `Dataset_Final/dataset_metadata.json`.

---

## 2. Definitive Taxonomy and Counts

The physical dataset remains partitioned in the original folder layout, but the training loader projects it into **6 logical classes**. Eliminated or remapped classes:
- **`battery`** → purged (only 29 images survived the 224px threshold, insufficient for training)
- **`shoes`** → purged (100% of images were sub-224px thumbnails; zero training value)
- **`ewaste`** → excluded from all default training/evaluation splits
- **`hard_plastic` + `soft_plastic`** → merged into logical `plastic`
- **`cardboard`** → merged into `paper`
- **`medical`** → purged (zero valid training images)
- **`ceramic`** → purged (previously decontaminated)

| # | Class | Image Count | Industrial Definition |
| :--- | :--- | :--- | :--- |
| 0 | **clothes** | 40,295 | Textiles, apparel, woven synthetic/natural fabrics |
| 1 | **glass** | 12,055 | Silica-based containers (clear and pigmented) |
| 2 | **metal** | 57,795 | Ferrous/non-ferrous metals, aluminum, alloys |
| 3 | **organic** | 152,745 | Biodegradable matter, food residuals, vegetation |
| 4 | **paper** | 14,270 | Cellulose flat material: office print, newsprint, corrugated cardboard |
| 5 | **plastic** | 22,658 | Combined rigid and flexible polymer waste (`hard_plastic` + `soft_plastic`) |

**TRAINABLE TOTAL: 299,818 images**

`ewaste` remains on disk as a physical archive folder with 4,440 images but is ignored by the training/evaluation dataset builders.

---

## 3. Data Integrity and Synchronization
The ESD platform enforces a **Zero-Mistake Data Environment**:
- **Resolution Floor:** Every image has been physically checked to verify it is ≥ 224×224px. Images below this threshold were permanently deleted, eliminating all "shortcut learning" from upscaled thumbnails.
- **Purity:** Every class is mutually exclusive. Visual audits using `scripts/audit_dataset_by_source.py` (randomized on every run) confirm no class contamination.
- **Metadata:** All training operations draw strictly from the verified `dataset_metadata.json` registry.
- **Physical Structure:** `Dataset_Final/<class_name>/<image_name>.ext`

---

## 4. Source Attribution and Provenance

The **WSS-304K** corpus is a curated meta-dataset synthesized from over 40 distinct industrial and academic sub-collections. The following primary sources are cited and acknowledged:

### 4.1. Core Academic Baselines
- **TrashNet (Stanford University):** The original waste classification baseline; provided fundamental visual patterns for glass, paper, and metal.
- **TrashBox (ResearchGate/GitHub):** Integrated multi-class trash corpus with detailed plastic subclasses; now mapped into `paper`, `glass`, `metal`, and logical `plastic`; e-waste sources are skipped for default training.
- **TACO (Trash Annotations in Context):** Source for high-fidelity images of litter in diverse, complex out-of-door environments.
- **Recycleye Waste-v1:** Public research segment of the Recycleye industrial sorting challenge (Glass, Paper, Metal, Plastic).
- **Garbage Classification (Kaggle/Yang):** 6-class fundamental material distribution.

### 4.2. Domain-Specific Aggregations
- **Clothing & Textiles:** Aggregated from the **Zalando Store Crawl**, **Biodegradable Fabrics Dataset**, and **E-commerce Clothing Attributes** collections.
- **E-Waste:** Retained only as a physical archive; excluded from the default trainable taxonomy.
- **Organic Matter:** Derived from the **CompostNet** and **Biological Waste** vision projects.

### 4.3. Industrial Synthetic Enhancements
To improve robustness against deformation and rare perspectives, the following synthetic sets were incorporated:
- **Greensorter Automated Sorting Set:** (Metal, Plastic, Glass).
- **Bottle-Synthetic-Images:** Targeted glass container variations.
- **Tin-and-Steel-Cans-Synthetic:** High-variance metal geometry.
- **Mechanical-Parts-Boltnut:** Fine-grained metal texture detection.

A full, per-image source registry is maintained in `Dataset_Final/dataset_metadata.json`. Use `scripts/audit_dataset_by_source.py` to generate randomized visual audits for any of these specific provenance sources.
