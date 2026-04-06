# GEMINI.md - Project Context & Instructions

This repository, **ESD**, contains a dual-component system for waste classification: a dynamic PyTorch-based training pipeline and a companion Android application (**SmartBin**) that utilizes the trained models with a runtime-configurable taxonomy.

## Project Overview

### 1. Training Pipeline (Python/PyTorch)
A class-dynamic training system designed for EfficientNet-B0. It uses a flat dataset structure where classes are inferred from directory names at runtime.
- **Core Technology:** Python 3.x, PyTorch, EfficientNet-B0, AdamW.
- **Architecture:** Progressive fine-tuning (head to backbone) followed by recursive refinement for both loss and raw accuracy.
- **Key Philosophy:** Train on all available classes; collapse unselected classes into an "Other" category only at runtime/evaluation.

### 2. SmartBin Android App (Kotlin/Compose)
A Jetpack Compose-based application that provides a UI for waste management and monitoring.
- **Core Technology:** Kotlin, Jetpack Compose, Material 3, Hilt (DI), Coroutines/Flow, MapLibre (Maps), Vico (Charts).
- **Architecture:** Clean Architecture with MVVM and Unidirectional Data Flow (UDF).
- **Runtime Behavior:** Users select $n-1$ classes from the trained catalog; all others are merged into a fixed "Other" bucket.
- **Real-time:** Uses WebSockets for live waste event streaming.

---

## Directory Structure

- `Dataset_Final/`: The flat root directory for training data (e.g., `Dataset_Final/plastic/*.jpg`).
- `scripts/`: Python utilities for training, evaluation, and dataset management.
- `SmartBin_Android/`: The complete Android Studio project.
- `.tooling/`: Contains a stub backend (`smartbin_stub_backend.py`) for local app testing.
- `Results/` & `logs/`: Local artifacts generated during training (git-ignored).

---

## Building and Running

### Training Pipeline
**Prerequisites:** NVIDIA GPU, CUDA-enabled PyTorch environment.

1. **Environment Setup:**
   ```bash
   ./scripts/setup_venv_cuda.sh .venv
   source .venv/bin/activate
   ```
2. **Execute Full Training:**
   ```bash
   # Sets performance mode and runs progressive + recursive training
   ./run_training.sh
   ```
3. **Evaluation with Class Collapsing:**
   ```bash
   python scripts/evaluate_saved_classifier.py \
     --checkpoint Results/<run>/rawacc_refine/accepted_best.pt \
     --selected-class metal --selected-class organic --selected-class paper
   ```

### Android Application
**Prerequisites:** Android SDK, JDK 21 (located in `JAVA_HOME`).

1. **Build and Test:**
   ```bash
   cd SmartBin_Android
   ./gradlew :app:assembleDebug :app:testDebugUnitTest :app:connectedDebugAndroidTest
   ```
2. **Install to Device:**
   ```bash
   adb install -r SmartBin_Android/artifacts/SmartBin-debug.apk
   ```
3. **Local Testing (Stub Backend):**
   ```bash
   python .tooling/smartbin_stub_backend.py
   # In another terminal:
   adb reverse tcp:8000 tcp:8000
   ```

---

## Development Conventions

### Python / Machine Learning
- **Dataset Contract:** Always use the flat folder structure in `Dataset_Final/`. Scripts should infer classes dynamically using `os.listdir` or similar.
- **Backbone:** Standardized on `efficientnet_b0`. Avoid introducing Gabor-based or fixed-taxonomy logic.
- **Checkpoints:** The pipeline produces progressive and recursive checkpoints. The `accepted_best.pt` in the final refinement stage is typically the production target.

### Android / Kotlin
- **Clean Architecture:** Keep business logic in `domain/usecase` and ensure `domain` has no dependencies on Android frameworks.
- **State Management:** Use `StateFlow` in ViewModels and represent UI state as a single immutable data class where possible.
- **Dependency Injection:** All repositories and use cases must be provided via Hilt modules in the `di` package.
- **Maps:** Use MapLibre for map rendering. Ensure API keys or local tile server configs are handled securely.
- **Testing:** New features require Unit Tests in `test/` and, if UI-related, Instrumented Tests in `androidTest/`.

### Shared Catalog
- The file `SmartBin_Android/app/src/main/assets/waste_class_catalog.json` is the single source of truth for the classes the app "knows" about. It must be kept in sync with the folders in `Dataset_Final/`.

---

## Key Files for Reference
- `README.md`: Top-level overview and hardware verification status.
- `PYTORCH_SETUP.md`: Detailed environment and training workflow notes.
- `SmartBin_Android/docs/architecture.md`: Deep dive into the Android app's internal structure.
- `scripts/metric_learning_pipeline.py`: Main logic for the training orchestration.
- `DATASET_RESEARCH_AND_REMAP.md`: Dataset state, survey, and mapping history.
