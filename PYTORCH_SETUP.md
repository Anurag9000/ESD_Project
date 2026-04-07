# PyTorch Environment and Execution Manual

## 1. Environment Setup

The ESD platform is optimized for CUDA-accelerated Linux environments.

### Initialization
```bash
# Initialize Python venv with CUDA 12.8 wheel indices
chmod +x scripts/setup_venv_cuda.sh
./scripts/setup_venv_cuda.sh .venv
source .venv/bin/activate
```

---

## 2. Training Execution

The training process is automated via deterministic shell scripts.

### Full Pipeline Orchestration
Executes SupCon pre-training, Progressive Unfreezing, and Recursive Refinement (Loss + Accuracy).
```bash
./run_full_training_pipeline.sh
```

### Manual Execution (Standardized Params)
```bash
python scripts/train_efficientnet_b0_progressive.py \
  --dataset-root Dataset_Final \
  --batch-size 224 \
  --precision mixed \
  --optimizer sam
```

---

## 3. Evaluation and Verification

### Model Evaluation
Supports runtime taxonomy collapsing for focused performance analysis.
```bash
python scripts/evaluate_saved_classifier.py \
  --checkpoint Results/best.pt \
  --selected-class metal \
  --selected-class organic \
  --selected-class paper
```

### Data Audit
Verifies the 1:1 synchronization between physical files and logical metadata.
```bash
python scripts/audit_dataset_integrity.py
```
