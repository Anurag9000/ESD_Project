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

The training process is automated via deterministic shell scripts and now defaults to:
- balanced per-batch class cycling for train, val, test, and SupCon loaders
- validation triggered by train-step patience (`train_loss_validation_patience=500`, `validation_patience=1000`)
- patience `1` across SupCon, CE head, CE progressive, and recursive refinement
- warmup + cosine decay annealing
- startup and phase-end clean test-set visual audits

### Full Pipeline Orchestration
Executes the staged pipeline:
1. `supcon_head_only`
2. `supcon_last_10_modules`
3. `supcon_last_20_modules`
4. `supcon_last_30_modules`
5. `...`
6. `ce_head_only`
7. `ce_last_10_modules`
8. `ce_last_20_modules`
9. `ce_last_30_modules`
10. `...`
11. recursive `val_loss` cleanup
12. recursive `val_raw_acc` refinement

Each phase writes into its own folder under `Results/<run>/progressive/phases/<phase_name>/`.
```bash
./run_training.sh --phase0-mim --backbone femto --num-workers 2 --prefetch-factor 1
```

### Manual Execution (Standardized Params)
```bash
./run_training.sh --phase0-mim --backbone femto --num-workers 2 --prefetch-factor 1
```

### Default Hyperparameters
| Parameter | Value |
| :-- | :-- |
| `batch_size` | `320` |
| `optimizer` | `adamw` |
| `scheduler` | `warmup + cosine decay` |
| `sampling_strategy` | `balanced` |
| `supcon_head_lr` | `3e-3` |
| `supcon_backbone_lr` | `5e-5` |
| `head_lr` | `1e-3` |
| `backbone_lr` | `1e-5` |
| `weight_decay` | `1e-4` |
| `label_smoothing` | `0.1` |
| `warmup_steps` | `1024` |
| `unfreeze_chunk_size` | `10` |
| `supcon_unfreeze_backbone_modules` | `40` |
| `ce_max_unfreeze_modules` | `40` |
| `train_loss_validation_patience` | `500` |
| `validation_patience` | `1000` |
| `supcon_early_stopping_patience` | `1` |
| `head_early_stopping_patience` | `1` |
| `stage_early_stopping_patience` | `1` |

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
