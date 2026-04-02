# ESD

This repo now has two main parts:

- a dynamic non-Gabor waste-classification training pipeline
- a SmartBin Android app that lets the user choose runtime classes and collapses all unselected trained classes into `Other`

## Documentation Map

- [README.md](/home/anurag-basistha/Projects/ESD/README.md)
  - top-level repo overview
  - current training philosophy
  - Android app status and APK location
- [PYTORCH_SETUP.md](/home/anurag-basistha/Projects/ESD/PYTORCH_SETUP.md)
  - training environment
  - current flat-dataset contract
  - run entry points
  - evaluation workflow
- [DATASET_RESEARCH_AND_REMAP.md](/home/anurag-basistha/Projects/ESD/DATASET_RESEARCH_AND_REMAP.md)
  - current local dataset state
  - old `other` remap history
  - external dataset survey
  - overlap and acquisition notes

## Current Repo State

- training is class-dynamic and no longer tied to a fixed small taxonomy
- the Android app is class-dynamic at runtime and keeps an explicit fixed `Other` bucket
- the checked-in Android APK is a tested debug artifact
- there is no production backend implementation in this repo

## Current Dataset / Training Philosophy

Training is no longer built around a fixed small class list.

- The dataset is stored as a flat folder tree under [Dataset_Final](/home/anurag-basistha/Projects/ESD/Dataset_Final).
- Training infers classes from folder names at runtime.
- The model trains on every available class separately.
- At runtime or evaluation time, the caller can choose which explicit classes to keep.
- Every trained class not selected by the caller is merged into runtime `Other`.

Current dataset classes on disk:

- `battery`
- `clothes`
- `ewaste`
- `glass`
- `metal`
- `organic`
- `paper`
- `plastic`
- `shoes`
- `trash`

## Training Entry Points

Primary training scripts:

- [run_non_gabor.sh](/home/anurag-basistha/Projects/ESD/run_non_gabor.sh)
- [run_non_gabor_full_model_loss_then_rawacc_recursive.sh](/home/anurag-basistha/Projects/ESD/run_non_gabor_full_model_loss_then_rawacc_recursive.sh)
- [scripts/train_efficientnet_b0_progressive.py](/home/anurag-basistha/Projects/ESD/scripts/train_efficientnet_b0_progressive.py)
- [scripts/metric_learning_pipeline.py](/home/anurag-basistha/Projects/ESD/scripts/metric_learning_pipeline.py)
- [scripts/run_recursive_refinement.py](/home/anurag-basistha/Projects/ESD/scripts/run_recursive_refinement.py)
- [scripts/evaluate_saved_classifier.py](/home/anurag-basistha/Projects/ESD/scripts/evaluate_saved_classifier.py)

Default training batch size across the retained path is `224`.

The master non-Gabor run now does all three stages in sequence:

1. progressive fine-tuning from the head through the backbone
2. recursive loss refinement with LR halving
3. recursive raw-accuracy refinement with LR halving

Run it with:

```bash
cd /home/anurag-basistha/Projects/ESD
powerprofilesctl set performance
./run_non_gabor.sh
```

## Android App

Android project:

- [SmartBin_Android](/home/anurag-basistha/Projects/ESD/SmartBin_Android)

Stable APK checked into this repo:

- [SmartBin-debug.apk](/home/anurag-basistha/Projects/ESD/SmartBin_Android/artifacts/SmartBin-debug.apk)

### App Runtime Behavior

The app no longer hardcodes a fixed waste taxonomy.

- On first launch, the user picks how many runtime classes they want. Default is `4`.
- The user explicitly chooses `n-1` classes from the trained class catalog.
- The final runtime class is fixed to `Other`.
- All trained classes not explicitly selected are merged into `Other`.
- The chosen runtime configuration is persisted and reused on relaunch.

The raw class catalog used by the app lives in:

- [waste_class_catalog.json](/home/anurag-basistha/Projects/ESD/SmartBin_Android/app/src/main/assets/waste_class_catalog.json)

### Verified App Testing

Build / automated verification completed:

- `./gradlew :app:assembleDebug`
- `./gradlew :app:testDebugUnitTest`
- `./gradlew :app:connectedDebugAndroidTest`

Hardware verified on:

- physical phone: `vivo 1933`
- emulator: `emulator-5554`

Frontend flows manually verified on hardware:

- first-run class configuration gate
- default `4` class setup
- save and continue flow
- map shell rendering
- locality selection
- mock event trigger and visible UI state update
- analytics screen state update
- classes screen persistence and restored ordering
- live-mode connection state against a local backend

### Backend / Integration Scope

There is no production backend implementation in this repo.

For live integration verification, the Android app was tested against a local protocol-compatible stub backend:

- [.tooling/smartbin_stub_backend.py](/home/anurag-basistha/Projects/ESD/.tooling/smartbin_stub_backend.py)

That stub exposes:

- `GET /bins`
- `GET /bins/{bin_id}`
- `GET /events`
- `POST /events`
- `WS /events/stream`

For physical-device live testing, `adb reverse tcp:8000 tcp:8000` was used so the phone could hit `127.0.0.1:8000`.

### Build Commands

```bash
cd /home/anurag-basistha/Projects/ESD/SmartBin_Android
export JAVA_HOME=/opt/android-studio/jbr
export PATH="$JAVA_HOME/bin:$HOME/Android/Sdk/platform-tools:$PATH"
./gradlew :app:assembleDebug :app:testDebugUnitTest :app:connectedDebugAndroidTest
```

The checked-in APK can then be installed with:

```bash
adb install -r /home/anurag-basistha/Projects/ESD/SmartBin_Android/artifacts/SmartBin-debug.apk
```

## Repo Notes

- [DATASET_RESEARCH_AND_REMAP.md](/home/anurag-basistha/Projects/ESD/DATASET_RESEARCH_AND_REMAP.md) contains the external waste-dataset survey and remap notes.
- [PYTORCH_SETUP.md](/home/anurag-basistha/Projects/ESD/PYTORCH_SETUP.md) contains the CUDA PyTorch environment notes.

Generated training outputs such as `Results/` and `logs/` remain local/ignored unless explicitly retained.
