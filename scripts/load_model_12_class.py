import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

class SmartBinModel(nn.Module):
    """
    Custom Metric Learning EfficientNet-B0 architecture used in the ESD project.
    """
    def __init__(self, num_classes=12, embedding_dim=128):
        super().__init__()
        # 1. Base Backbone (EfficientNet-B0)
        self.backbone = efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features  # 1280
        
        # 2. Metric Learning Embedding Head
        self.embedding = nn.Linear(in_features, embedding_dim, bias=False)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        
        # 3. Supervised Contrastive Projection (Needed to load the full state_dict seamlessly)
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # 4. Final Classification Head
        self.ce_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.embedding(x)
        x = self.embedding_norm(x)
        x = F.normalize(x, dim=1)
        
        return self.ce_head(x)

if __name__ == "__main__":
    # ======== HOW TO LOAD ========
    # Initialize with 12 classes
    model_12class = SmartBinModel(num_classes=12)

    # Note: Update this path to where your checkpoint actually is!
    checkpoint_path = "../Results/efficientnet_b0_master_run/loss_cleanup/best.pt"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_12class.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model_12class.eval()
        print("✅ Current 12-Class Model Loaded Perfectly!")
    except FileNotFoundError:
        print(f"⚠️ Placeholder script ran! Please update checkpoint_path to your real .pt file.")
