#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from metric_learning_pipeline import (
    MetricLearningEfficientNetB0,
    evaluation_tensor_from_image,
    model_dtype_for_args,
    save_json,
)

warnings.filterwarnings("ignore", message="Palette images with Transparency", category=UserWarning)


class GradCAMHook:
    def __init__(self, module: torch.nn.Module) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = module.register_forward_hook(self._capture_activations)
        self._backward_handle = module.register_full_backward_hook(self._capture_gradients)

    def _capture_activations(self, module, inputs, output) -> None:  # noqa: ANN001
        del module, inputs
        self.activations = output

    def _capture_gradients(self, module, grad_input, grad_output) -> None:  # noqa: ANN001
        del module, grad_input
        self.gradients = grad_output[0]

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for a saved waste classifier checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--class-name", action="append", default=[], help="Limit overlays to these class names.")
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], argparse.Namespace]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict):
        raise ValueError("Checkpoint is missing args metadata.")
    return checkpoint, argparse.Namespace(**checkpoint_args)


def select_target_module(backbone: torch.nn.Module) -> torch.nn.Module:
    if hasattr(backbone, "stages"):
        stages = list(getattr(backbone, "stages"))
        if stages:
            last_stage = stages[-1]
            if hasattr(last_stage, "blocks") and len(getattr(last_stage, "blocks")) > 0:
                return last_stage.blocks[-1]
            return last_stage
    if hasattr(backbone, "blocks"):
        blocks = list(getattr(backbone, "blocks"))
        if blocks:
            return blocks[-1]
    if hasattr(backbone, "features"):
        features = getattr(backbone, "features")
        if isinstance(features, torch.nn.Sequential) and len(features) > 0:
            return features[-1]
    leaf_modules = [module for module in backbone.modules() if not list(module.children())]
    if not leaf_modules:
        raise ValueError("Could not identify a Grad-CAM target layer.")
    return leaf_modules[-1]


def to_overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heat = np.asarray(heatmap, dtype=np.float32)
    heat = np.clip(heat, 0.0, 1.0)
    heat_rgb = np.stack([heat, np.zeros_like(heat), 1.0 - heat], axis=-1)
    overlay = np.clip(0.58 * rgb + 0.42 * heat_rgb, 0.0, 1.0)
    return Image.fromarray((overlay * 255.0).astype(np.uint8))


def generate_gradcam(
    model: MetricLearningEfficientNetB0,
    target_module: torch.nn.Module,
    image_path: Path,
    class_index: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Image.Image, np.ndarray, int]:
    hook = GradCAMHook(target_module)
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = evaluation_tensor_from_image(image, image_size).unsqueeze(0).to(device=device, dtype=dtype)
        input_tensor.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        logits = model.classify(model.encode(input_tensor), labels=None)
        score = logits[0, class_index]
        score.backward()

        if hook.activations is None or hook.gradients is None:
            raise RuntimeError(f"No activations captured for {image_path}")

        activations = hook.activations.detach()
        gradients = hook.gradients.detach()
        if activations.ndim == 4:
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations).sum(dim=1, keepdim=False))
            cam = cam[0]
        else:
            cam = torch.relu(activations[0])
        cam = cam - cam.min()
        cam = cam / (cam.max().clamp_min(1e-12))
        cam_np = cam.detach().cpu().numpy()
        cam_img = Image.fromarray((cam_np * 255.0).astype(np.uint8)).resize(image.size)
        overlay = to_overlay(image, np.asarray(cam_img, dtype=np.float32) / 255.0)
        return overlay, cam_np, int(torch.argmax(logits, dim=1).item())
    finally:
        hook.close()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, checkpoint_args = load_checkpoint(checkpoint_path)
    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint_args.get("image_size", 224))
    weights_mode = str(checkpoint_args.get("weights", "default"))
    backbone_name = str(checkpoint_args.get("backbone", "efficientnet_b0"))
    embedding_dim = int(checkpoint_args.get("embedding_dim", 512))
    projection_dim = int(checkpoint_args.get("projection_dim", 256))
    precision = str(checkpoint_args.get("precision", "mixed"))
    model_args = argparse.Namespace(precision=precision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = model_dtype_for_args(model_args)

    model = MetricLearningEfficientNetB0(
        num_classes=len(class_names),
        weights_mode=weights_mode,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        args=model_args,
        backbone_name=backbone_name,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = datasets.ImageFolder(str(dataset_root))
    class_to_index = {name: idx for idx, name in enumerate(dataset.classes)}
    selected_classes = [class_name for class_name in args.class_name if class_name in class_to_index] or list(dataset.classes)
    rng = random.Random(args.seed)
    target_module = select_target_module(model.backbone)

    manifest = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "selected_classes": selected_classes,
        "samples_per_class": int(args.samples_per_class),
        "backbone": backbone_name,
    }
    save_json(output_dir / "gradcam_manifest.json", manifest)

    summary: dict[str, Any] = {"classes": {}, "checkpoint": str(checkpoint_path)}
    for class_name in selected_classes:
        class_index = class_to_index[class_name]
        class_samples = [sample for sample in dataset.samples if sample[1] == class_index]
        if not class_samples:
            continue
        chosen_samples = class_samples if args.samples_per_class <= 0 else rng.sample(class_samples, min(args.samples_per_class, len(class_samples)))
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        overlays: list[str] = []
        for sample_index, (sample_path, _) in enumerate(chosen_samples, start=1):
            overlay, _, predicted_index = generate_gradcam(
                model=model,
                target_module=target_module,
                image_path=Path(sample_path),
                class_index=class_index,
                image_size=image_size,
                device=device,
                dtype=dtype,
            )
            predicted_name = dataset.classes[predicted_index] if 0 <= predicted_index < len(dataset.classes) else str(predicted_index)
            filename = f"{sample_index:02d}_{Path(sample_path).stem}_pred_{predicted_name}.png"
            overlay.save(class_dir / filename)
            overlays.append(filename)
        summary["classes"][class_name] = {"samples": len(overlays), "files": overlays}

    save_json(output_dir / "gradcam_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
