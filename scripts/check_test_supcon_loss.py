#!/usr/bin/env python3

import argparse
import torch
import sys
from pathlib import Path
from tqdm import tqdm

# Ensure we can import from the scripts directory if running from ESD root
sys.path.append(str(Path(__file__).parent))

from metric_learning_pipeline import (
    MetricLearningEfficientNetB0,
    build_datasets,
    DeterministicSupConDataset,
    make_loader,
    make_weighted_sampler,
    evaluate_supcon,
    SupConLoss,
    model_dtype_for_args
)

def main():
    parser = argparse.ArgumentParser(description="One-off script to check SupCon loss on the Test set.")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint (e.g., step_last.pt)")
    parser.add_argument("--dataset-root", default="Dataset_Final", help="Path to the dataset root")
    parser.add_argument("--batch-size", type=int, default=224, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return 1

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract original args used during training
    if 'args' not in checkpoint:
        print("Error: Checkpoint does not contain 'args' metadata.")
        return 1
        
    ckpt_args_dict = checkpoint['args']
    if isinstance(ckpt_args_dict, dict):
        ckpt_args = argparse.Namespace(**ckpt_args_dict)
    else:
        ckpt_args = ckpt_args_dict # Already a namespace

    # Override relevant paths and batch size for this evaluation
    ckpt_args.dataset_root = args.dataset_root
    ckpt_args.batch_size = args.batch_size
    
    # Force Weighted Sampling to ensure the loss scale matches the training/val progress you've been seeing
    print("Using Weighted Random Sampler for consistent loss scaling...")
    ckpt_args.sampling_strategy = "weighted" 

    print("Initializing datasets...")
    # build_datasets returns (train, val, test, supcon_train, supcon_val)
    _, _, test_dataset, _, _ = build_datasets(ckpt_args)
    
    # Wrap in SupCon dataset handler (handles double-views)
    supcon_test_dataset = DeterministicSupConDataset(test_dataset)
    
    # Create the sampler
    sampler = make_weighted_sampler(
        supcon_test_dataset, 
        test_dataset.classes, 
        supcon_test_dataset.target_for_index, 
        ckpt_args.seed
    )
    
    loader = make_loader(
        supcon_test_dataset, 
        ckpt_args.batch_size, 
        num_workers=args.num_workers, 
        prefetch_factor=2, 
        shuffle=False, 
        sampler=sampler
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model with training configuration
    model = MetricLearningEfficientNetB0(
        num_classes=len(test_dataset.classes),
        weights_mode="none", # Load weights from checkpoint, not defaults
        embedding_dim=ckpt_args.embedding_dim,
        projection_dim=ckpt_args.projection_dim,
        args=ckpt_args
    ).to(device=device, dtype=model_dtype_for_args(ckpt_args))
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    criterion = SupConLoss(ckpt_args.supcon_temperature)
    
    print(f"\nStarting Test Set Evaluation ({len(loader)} batches)...")
    # Redirect logs to a safe temporary location
    temp_log_dir = Path("Results/temp_verification")
    temp_log_dir.mkdir(parents=True, exist_ok=True)
    
    loss = evaluate_supcon(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        max_batches=0,
        log_path=temp_log_dir / "test_verification.log.jsonl",
        log_every_eval_steps=10,
        stage="test_verification",
        split="test",
        args=ckpt_args
    )
    
    print(f"\n{'='*40}")
    print(f"FINAL TEST SUPCON LOSS: {loss:.6f}")
    print(f"{'='*40}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
