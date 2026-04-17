#!/usr/bin/env python3
"""
exports the `best.pt` checkpoint to a fully standalone TorchScript model (`best_scripted.pt`)
This model does not require the original source code or class definitions to be imported.
"""

import torch
import argparse
import sys
from pathlib import Path

# Add project root to path so we can import the pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.metric_learning_pipeline import MetricLearningEfficientNetB0, model_dtype_for_args

class InferenceESDModel(torch.nn.Module):
    """
    A minimal wrapper that encapsulates to the exact Inference sequence.
    This hides all training-specific layers like projection_head which aren't used in inference.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # 1. Encode image to metrics embedding (normalized)
        emb = self.model.encode(x, normalize=True)
        # 2. Classify the embedding
        return self.model.classify(emb)

def main():
    ckpt_path = Path("Results/convnextv2_nano_master_run/loss_cleanup/best.pt")
    if not ckpt_path.exists():
        print(f"Error: Could not find {ckpt_path}", file=sys.stderr)
        return 1
        
    print(f"Loading {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    
    ckpt_args = ckpt.get("args", {})
    num_classes = len(class_names)
    embedding_dim = int(ckpt_args.get("embedding_dim", 128))
    projection_dim = int(ckpt_args.get("projection_dim", 128))
    image_size = int(ckpt_args.get("image_size", 224))
    
    model_args = argparse.Namespace(precision=ckpt_args.get("precision", "mixed"))
    
    print("Instantiating original custom architecture...")
    model = MetricLearningEfficientNetB0(
        num_classes=num_classes,
        weights_mode="none", 
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        args=model_args
    )
    
    # Load weights
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print("Wrapping model for inference-only trace...")
    wrapper = InferenceESDModel(model)
    wrapper.eval()
    
    print("Tracing to TorchScript...")
    # TorchScript tracing requires an example input tensor
    example_input = torch.randn(1, 3, image_size, image_size)
    traced_model = torch.jit.trace(wrapper, example_input)
    
    out_path = ckpt_path.parent / "best_scripted.pt"
    traced_model.save(str(out_path))
    
    print("\n✅ SUCCESS!")
    print(f"Saved standalone TorchScript model to: {out_path}")
    print("This file can be loaded directly using `torch.jit.load()` without needing any class definitions.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
