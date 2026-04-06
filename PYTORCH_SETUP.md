# PyTorch Development Environment and Workflow

## Environment Configuration

The development environment is optimized for a Linux-based architecture with NVIDIA GPU acceleration.

### 1. Virtual Environment Initialization
Initialization of the Python virtual environment and CUDA-specific PyTorch distribution is automated.
```bash
# Initialize venv with CUDA 12.8 wheel indices
./scripts/setup_venv_cuda.sh .venv
source .venv/bin/activate
```

### 2. Dependency Management
Standardized requirements are maintained in `requirements-cu128.txt`, including `torch`, `torchvision`, and `huggingface_hub`.

---

## Dataset Definition and Contract

The training pipeline implements a **Material-Oriented Dynamic Taxonomy**.
- **Dataset Root:** `Dataset_Final/`
- **Structure:** `Dataset_Final/<class_name>/<image_files>`
- **Dynamic Inference:** At runtime, the class list is derived from the immediate subdirectory names of the root.

---

## Model Training Lifecycle

The training lifecycle is divided into three distinct phases to maximize feature extraction and model precision.

### Phase 1: Progressive Fine-Tuning
The system begins with a frozen EfficientNet-B0 backbone, incrementally unfreezing backbone layers in discrete slices while decaying the learning rate. This ensures the classification head is stabilized before higher-order features are specialized.

### Phase 2: Recursive Validation Loss Refinement
Utilizes the `run_recursive_refinement.py` module to iteratively train the model until the validation loss improvement falls below the defined `threshold`. Each successful iteration halves the head and backbone learning rates for the subsequent run.

### Phase 3: Recursive Validation Accuracy Refinement
The final refinement stage prioritizes validation raw accuracy. This ensures that the categorical boundaries are fine-tuned for maximal separation across the entire taxonomy.

---

## Pipeline Execution

Primary Entry Point:
```bash
# Sets performance power profile and initiates training sequence
./run_training.sh
```

Post-Training Evaluation:
The evaluation suite supports **Runtime Taxonomy Collapsing**, allowing the user to evaluate performance for any arbitrary subset of classes while treating others as the residual `Other` class.
```bash
# Evaluate model against a selected subset of material classes
python scripts/evaluate_saved_classifier.py \
  --checkpoint Results/accepted_best.pt \
  --selected-class metal \
  --selected-class organic \
  --selected-class paper
```

---

## Verification and Standards
All scripts must adhere to strict type-hinting and professional documentation standards. Legacy artifacts, including Gabor-specific logic and experimental search wordlists, must be rigorously excluded from the production-grade baseline.
