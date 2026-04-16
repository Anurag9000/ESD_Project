#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.append(str(Path(__file__).parent))

from metric_learning_pipeline import (
    MetricLearningEfficientNetB0,
    build_datasets,
    model_dtype_for_args,
    IMAGENET_MEAN,
    IMAGENET_STD
)

class ActivationHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()

def save_tsne_plot(embeddings, labels, class_names, output_path, title_suffix=""):
    print(f"Running t-SNE on {len(embeddings)} samples (Stratified)...")
    # Using PCA initialization + Barnes-Hut for faster large-scale t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', angle=0.5)
    embed_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(14, 12))
    sns.scatterplot(
        x=embed_2d[:, 0], y=embed_2d[:, 1],
        hue=[class_names[l] for l in labels],
        palette='tab10',
        legend='full',
        alpha=0.4,
        s=10,
        edgecolor='none'
    )
    plt.title(f"t-SNE Geometry: Entire Dataset Representation {title_suffix}")
    plt.xlabel("Semantic Dimension 1")
    plt.ylabel("Semantic Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Global t-SNE plot saved to {output_path}")

def generate_global_activations(model, loader, device, output_dir, class_names, limit=2000):
    print(f"Computing Global Activation Averages across {limit} samples...")
    model.eval()
    
    target_layers = {
        "Low-Level (Edges)": model.backbone.features[0],
        "Mid-Level (Textures)": model.backbone.features[4],
        "High-Level (Semantic)": model.backbone.features[8]
    }
    
    hooks = {name: ActivationHook(layer) for name, layer in target_layers.items()}
    global_maps = {name: None for name in target_layers.keys()}
    
    count = 0
    with torch.no_grad():
        for images, _ in tqdm(loader, total=limit // loader.batch_size, desc="Aggregating activations"):
            images = images.to(device)
            _ = model.encode(images)
            
            for name, hook in hooks.items():
                # Mean over batch + channels
                batch_map = hook.features.mean(dim=(0, 1)).cpu().numpy()
                if global_maps[name] is None:
                    global_maps[name] = batch_map
                else:
                    global_maps[name] += batch_map
            
            count += len(images)
            if count >= limit:
                break

    # Normalize and Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (name, gmap) in enumerate(global_maps.items()):
        amap = gmap / (count // loader.batch_size) 
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        
        im = axes[idx].imshow(amap, cmap='inferno')
        axes[idx].set_title(f"Global Mean: {name}\n({count} samples)")
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    plt.suptitle("Common Neural Pathways: Dataset-Wide Activation Signatures", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / "global_activation_signatures.png", dpi=200)
    plt.close()
                
    for hook in hooks.values():
        hook.close()
    print(f"Global activation signatures saved to {output_dir}")

def generate_activation_maps(model, loader, device, output_dir, class_names, num_samples=5):
    print(f"Generating Individual Activation Samples for {output_dir.name}...")
    model.eval()
    
    target_layers = {
        "Stem": model.backbone.features[0],
        "Mid": model.backbone.features[4],
        "Tail": model.backbone.features[8]
    }
    
    hooks = {name: ActivationHook(layer) for name, layer in target_layers.items()}
    
    samples_found = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            _ = model.encode(images)
            
            for i in range(min(len(images), num_samples - samples_found)):
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Original Image
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
                img = np.clip(img, 0, 1)
                axes[0].imshow(img)
                axes[0].set_title(f"Target: {class_names[targets[i]]}")
                axes[0].axis('off')
                
                for idx, (name, hook) in enumerate(hooks.items(), start=1):
                    # Take mean across channels
                    amap = hook.features[i].mean(dim=0).cpu().numpy()
                    # Normalize for visualization
                    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
                    
                    im = axes[idx].imshow(amap, cmap='magma')
                    axes[idx].set_title(f"Layer: {name}")
                    axes[idx].axis('off')
                    plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                plt.savefig(output_dir / f"sample_activation_{samples_found + i}.png")
                plt.close()
            
            samples_found += len(images)
            if samples_found >= num_samples:
                break
                
    for hook in hooks.values():
        hook.close()
    print(f"Individual activation samples saved to {output_dir}")

def generate_classification_atlas(model, loader, device, output_path, class_names, samples_per_class=5):
    print("Generating Classification Atlas...")
    model.eval()
    
    results = {c: [] for c in range(len(class_names))}
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Collecting atlas samples"):
            images = images.to(device)
            embeddings = model.encode(images)
            logits = model.classify(embeddings)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            for i in range(len(images)):
                pred = int(predictions[i])
                if len(results[pred]) < samples_per_class:
                    img = images[i].cpu().permute(1, 2, 0).numpy()
                    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    results[pred].append({
                        "img": img,
                        "target": int(targets[i]),
                        "conf": float(confidences[i])
                    })
            
            if all(len(v) >= samples_per_class for v in results.values()):
                break

    # Create Grid
    rows = len(class_names)
    cols = samples_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if c < len(results[r]):
                data = results[r][c]
                ax.imshow(data["img"])
                color = "green" if data["target"] == r else "red"
                ax.set_title(f"P:{class_names[r]} ({data['conf']:.2f})\nActual: {class_names[data['target']]}", color=color, fontsize=8)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Classification atlas saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--output-dir", default="Results/visualizations")
    parser.add_argument("--sample-limit", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = argparse.Namespace(**checkpoint['args'])
    ckpt_args.dataset_root = args.dataset_root
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, val_dataset, test_dataset, _, _ = build_datasets(ckpt_args)
    dataset = val_dataset if args.split == "val" else test_dataset
    class_names = dataset.classes
    
    model = MetricLearningEfficientNetB0(
        num_classes=len(class_names),
        weights_mode="none",
        embedding_dim=ckpt_args.embedding_dim,
        projection_dim=ckpt_args.projection_dim,
        args=ckpt_args
    ).to(device=device, dtype=model_dtype_for_args(ckpt_args))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create a simple non-shuffled loader for visualization
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 1. t-SNE (Global Stratified)
    all_embeddings = []
    all_labels = []
    print(f"Extracting Global embeddings for t-SNE (limit {args.sample_limit})...")
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(loader, total=args.sample_limit // args.batch_size):
            images = images.to(device)
            embeddings = model.encode(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(targets.numpy())
            if len(all_labels) >= args.sample_limit:
                break
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)[:args.sample_limit]
    all_labels = all_labels[:args.sample_limit]
    save_tsne_plot(all_embeddings, all_labels, class_names, output_dir / "global_tsne_projection.png", title_suffix=f"({args.split} split)")
    
    # 2. Global Activation Signatures
    generate_global_activations(model, loader, device, output_dir, class_names, limit=max(2000, args.sample_limit // 2))
    
    # 3. Random Sample activations for local verification
    generate_activation_maps(model, loader, device, output_dir, class_names)
    
    # 4. Atlas
    generate_classification_atlas(model, loader, device, output_dir / "global_classification_atlas.png", class_names)

if __name__ == "__main__":
    main()
