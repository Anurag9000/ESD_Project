#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import io
import json
import logging
import math
import os
import random
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

try:
    import timm
except ImportError:  # pragma: no cover - surfaced explicitly when backbone creation is attempted
    timm = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_BACKBONE_NAME = "atto"
DEFAULT_BATCH_SIZE = 320
# The repo-wide training taxonomy is now strictly 3 classes.
# Physical folders stay untouched; all datasets are projected into this order.
TRAINING_CLASS_ORDER = ("organic", "metal", "paper")
TRAINING_CLASS_MAPPING: dict[str, list[str]] = {}
TRAINING_EXCLUDED_CLASSES = frozenset({"clothes", "ewaste", "glass", "hard_plastic", "soft_plastic", "plastic"})
LEGACY_8_CLASS_TAXONOMY = (
    "clothes",
    "ewaste",
    "glass",
    "hard_plastic",
    "metal",
    "organic",
    "paper",
    "soft_plastic",
)
SPLIT_OFFSETS = {"train": 0, "val": 10_000_000, "test": 20_000_000}
SHADOW_PROBABILITY = 0.0
GLARE_PROBABILITY = 0.0
CAMERA_COLOR_CAST_PROBABILITY = 1.0
CAMERA_COLOR_CAST_STRENGTH = 0.50
CAMERA_COLOR_CAST_EVAL = True
MOTION_BLUR_PROBABILITY = 0.0
DEFOCUS_BLUR_PROBABILITY = 0.0
RESOLUTION_DEGRADE_PROBABILITY = 0.0
TRUNCATION_PROBABILITY = 0.0
SMUDGE_PROBABILITY = 0.0
PALETTE_TRANSPARENCY_WARNING = "Palette images with Transparency expressed in bytes should be converted to RGBA images"
RUNTIME_BAD_SAMPLE_CLEANUP_LOG = "runtime_bad_sample_cleanup.jsonl"
RUNTIME_BAD_SAMPLE_CLEANUP_LOCK = ".runtime_bad_sample_cleanup.lock"

warnings.filterwarnings("ignore", message=PALETTE_TRANSPARENCY_WARNING, category=UserWarning)
logging.basicConfig(level=logging.WARNING)


@dataclass
class PhaseSpec:
    name: str
    unfrozen_backbone_modules: int


@dataclass(frozen=True)
class BackboneSpec:
    pretrained_name: str
    scratch_name: str
    description: str


BACKBONE_REGISTRY: dict[str, BackboneSpec] = {
    "femto": BackboneSpec(
        pretrained_name="convnextv2_femto.fcmae_ft_in1k",
        scratch_name="convnextv2_femto",
        description="ConvNeXt V2 Femto FCMAE fine-tuned on IN1K",
    ),
    "convnextv2_nano": BackboneSpec(
        pretrained_name="convnextv2_nano.fcmae_ft_in22k_in1k",
        scratch_name="convnextv2_nano",
        description="ConvNeXt V2 Nano FCMAE fine-tuned on IN22K + IN1K",
    ),
    "convnextv2_tiny": BackboneSpec(
        pretrained_name="convnextv2_tiny.fcmae",
        scratch_name="convnextv2_tiny",
        description="ConvNeXt V2 Tiny FCMAE",
    ),
    "convnextv2_pico": BackboneSpec(
        pretrained_name="convnextv2_pico.fcmae_ft_in1k",
        scratch_name="convnextv2_pico",
        description="ConvNeXt V2 Pico FCMAE fine-tuned on IN1K",
    ),
    "atto": BackboneSpec(
        pretrained_name="convnextv2_atto.fcmae_ft_in1k",
        scratch_name="convnextv2_atto",
        description="ConvNeXt V2 Atto FCMAE fine-tuned on IN1K",
    ),
    "efficientnetv2_s": BackboneSpec(
        pretrained_name="efficientnetv2_s",
        scratch_name="efficientnetv2_s",
        description="EfficientNetV2-S",
    ),
}


def ensure_timm_available() -> None:
    if timm is None:
        raise ImportError(
            "timm is required for backbone selection. Install the repo dependencies from requirements-cu128.txt."
        )


def create_backbone(backbone_name: str, weights_mode: str) -> nn.Module:
    ensure_timm_available()
    pretrained = weights_mode == "default"
    spec = BACKBONE_REGISTRY.get(backbone_name)
    if spec is not None:
        model_name = spec.pretrained_name if pretrained else spec.scratch_name
        try:
            return timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception:
            if pretrained:
                return timm.create_model(spec.scratch_name, pretrained=False, num_classes=0)
            raise
    try:
        return timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
    except Exception:
        if pretrained:
            return timm.create_model(backbone_name, pretrained=False, num_classes=0)
        raise


def parse_json_class_mapping(spec: str | None) -> dict[str, list[str]]:
    if not spec:
        return {}
    parsed = json.loads(spec)
    if not isinstance(parsed, dict):
        raise ValueError("--class-mapping must be a JSON object mapping target class to source class list.")
    mapping: dict[str, list[str]] = {}
    for target, sources in parsed.items():
        if not isinstance(target, str) or not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
            raise ValueError("--class-mapping must look like '{\"target\": [\"source_a\", \"source_b\"]}'.")
        mapping[target] = list(sources)
    return mapping


def normalize_class_name(name: str) -> str:
    return name.strip().lower()


def enforced_training_class_mapping(custom_mapping: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Return the repo-wide training taxonomy mapping.

    Physical dataset folders stay untouched, but every training/evaluation split
    is projected into this logical taxonomy before sampling/model construction:
    the current repo keeps only the 3 supervised classes organic / metal / paper.
    """
    merged: dict[str, list[str]] = {target: list(sources) for target, sources in TRAINING_CLASS_MAPPING.items()}
    for target, sources in (custom_mapping or {}).items():
        normalized_target = normalize_class_name(target)
        if normalized_target in TRAINING_EXCLUDED_CLASSES:
            continue
        target_sources = merged.setdefault(normalized_target, [])
        for source in sources:
            normalized_source = normalize_class_name(source)
            if normalized_source not in TRAINING_EXCLUDED_CLASSES and normalized_source not in target_sources:
                target_sources.append(normalized_source)
    return merged


def project_class_name_to_training_taxonomy(
    source_name: str,
    class_mapping: dict[str, list[str]] | None = None,
) -> str | None:
    normalized_source = normalize_class_name(source_name)
    if normalized_source in TRAINING_EXCLUDED_CLASSES:
        return None
    effective_mapping = enforced_training_class_mapping(class_mapping)
    for target, sources in effective_mapping.items():
        if normalized_source in [normalize_class_name(source) for source in sources]:
            normalized_target = normalize_class_name(target)
            return normalized_target if normalized_target in TRAINING_CLASS_ORDER else None
    return normalized_source if normalized_source in TRAINING_CLASS_ORDER else None


def project_samples_to_training_taxonomy(
    source_classes: list[str],
    samples: list[tuple[str, int]],
    class_mapping: dict[str, list[str]] | None = None,
) -> tuple[list[str], dict[str, int], list[tuple[str, int]], dict[str, Any]]:
    """Drop excluded source classes and remap folders into logical train classes."""
    effective_mapping = enforced_training_class_mapping(class_mapping)

    remapped_names: list[str] = []
    dropped_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for _, old_target in samples:
        source_name = source_classes[int(old_target)]
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
        target_name = project_class_name_to_training_taxonomy(source_name, effective_mapping)
        if target_name is None:
            dropped_counts[source_name] = dropped_counts.get(source_name, 0) + 1
            continue
        remapped_names.append(target_name)

    # Keep a fixed head size even when a split is missing one of the classes.
    new_classes = list(TRAINING_CLASS_ORDER)
    new_class_to_idx = {class_name: index for index, class_name in enumerate(new_classes)}

    new_samples: list[tuple[str, int]] = []
    for path, old_target in samples:
        source_name = source_classes[int(old_target)]
        target_name = project_class_name_to_training_taxonomy(source_name, effective_mapping)
        if target_name is None or target_name not in new_class_to_idx:
            continue
        new_samples.append((path, new_class_to_idx[target_name]))

    metadata = {
        "logical_class_order": list(TRAINING_CLASS_ORDER),
        "class_mapping": effective_mapping,
        "excluded_classes": sorted(TRAINING_EXCLUDED_CLASSES),
        "source_classes": list(source_classes),
        "source_counts": source_counts,
        "dropped_counts": dropped_counts,
        "logical_classes": new_classes,
        "logical_sample_count": len(new_samples),
    }
    return new_classes, new_class_to_idx, new_samples, metadata


def clone_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def adapt_checkpoint_state_dict_to_training_taxonomy(
    state_dict: dict[str, torch.Tensor],
    source_class_names: list[str],
    target_class_names: list[str],
    *,
    class_mapping: dict[str, list[str]] | None = None,
    head_prefix: str = "ce_head",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Rewrite a checkpoint head to match the repo's current logical taxonomy.

    This keeps all backbone / embedding weights unchanged and only remaps the
    final classification head. Exact one-to-one classes are copied directly.
    Merged classes use a bias-weighted average of the legacy source rows, with
    the merged bias set to logsumexp of the source biases.
    """
    source_class_names = list(source_class_names)
    target_class_names = list(target_class_names)
    effective_mapping = enforced_training_class_mapping(class_mapping)
    projected_source_targets = {
        source_name: project_class_name_to_training_taxonomy(source_name, effective_mapping)
        for source_name in source_class_names
    }

    if source_class_names == target_class_names:
        return clone_state_dict(state_dict), {
            "applied": False,
            "reason": "source_and_target_taxonomies_match",
            "source_class_names": source_class_names,
            "target_class_names": target_class_names,
        }

    weight_key = f"{head_prefix}.weight"
    bias_key = f"{head_prefix}.bias"
    if weight_key not in state_dict:
        raise ValueError(f"Checkpoint state_dict is missing {weight_key!r}.")
    old_weight = state_dict[weight_key]
    old_bias = state_dict.get(bias_key)
    if old_weight.ndim != 2:
        raise ValueError(f"Expected {weight_key!r} to be a 2D tensor, got shape {tuple(old_weight.shape)}.")
    if old_weight.shape[0] != len(source_class_names):
        raise ValueError(
            f"Cannot adapt checkpoint head: {weight_key} has {old_weight.shape[0]} rows but "
            f"{len(source_class_names)} source classes were supplied."
        )
    if old_bias is not None and old_bias.shape[0] != len(source_class_names):
        raise ValueError(
            f"Cannot adapt checkpoint head: {bias_key} has {old_bias.shape[0]} rows but "
            f"{len(source_class_names)} source classes were supplied."
        )

    target_to_source_indices: dict[str, list[int]] = {
        target_name: [
            index
            for index, source_name in enumerate(source_class_names)
            if projected_source_targets.get(source_name) == target_name
        ]
        for target_name in target_class_names
    }
    missing_targets = [target for target, indices in target_to_source_indices.items() if not indices]
    if missing_targets:
        raise ValueError(
            "Could not adapt checkpoint head to the current taxonomy because these target classes have no source "
            f"counterparts: {missing_targets}. Source classes: {source_class_names}"
        )

    adapted = clone_state_dict(state_dict)
    adapted_weight = old_weight.new_zeros((len(target_class_names), old_weight.shape[1]))
    adapted_bias = old_bias.new_zeros(len(target_class_names)) if old_bias is not None else None
    merged_sources: dict[str, list[str]] = {}

    for target_index, target_name in enumerate(target_class_names):
        source_indices = target_to_source_indices[target_name]
        source_names = [source_class_names[index] for index in source_indices]
        if len(source_indices) == 1:
            source_index = source_indices[0]
            adapted_weight[target_index] = old_weight[source_index]
            if adapted_bias is not None and old_bias is not None:
                adapted_bias[target_index] = old_bias[source_index]
            continue

        source_weight_rows = old_weight[source_indices].float()
        if old_bias is not None:
            source_bias_rows = old_bias[source_indices].float()
            mixing_weights = torch.softmax(source_bias_rows, dim=0).unsqueeze(1)
            merged_weight = (mixing_weights * source_weight_rows).sum(dim=0)
            merged_bias = torch.logsumexp(source_bias_rows, dim=0)
            adapted_bias[target_index] = merged_bias.to(dtype=old_bias.dtype)
        else:
            merged_weight = source_weight_rows.mean(dim=0)
        adapted_weight[target_index] = merged_weight.to(dtype=old_weight.dtype)
        merged_sources[target_name] = source_names

    adapted[weight_key] = adapted_weight
    if adapted_bias is not None:
        adapted[bias_key] = adapted_bias

    return adapted, {
        "applied": True,
        "reason": "taxonomy_projection",
        "source_class_names": source_class_names,
        "target_class_names": target_class_names,
        "merged_sources": merged_sources,
        "excluded_source_classes": sorted(
            source_name
            for source_name in source_class_names
            if projected_source_targets.get(source_name) is None
        ),
    }


class DeterministicAugmentedImageFolder(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        root: Path | None,
        image_size: int,
        augment_repeats: int,
        split_name: str,
        seed: int,
        gaussian_sigmas: float,
        camera_color_cast_probability: float = CAMERA_COLOR_CAST_PROBABILITY,
        camera_color_cast_strength: float = CAMERA_COLOR_CAST_STRENGTH,
        camera_color_cast_eval: bool = CAMERA_COLOR_CAST_EVAL,
        apply_augmentation: bool = True,
        *,
        base_dataset: datasets.ImageFolder | None = None,
        samples: list[tuple[str, int]] | None = None,
        class_mapping: dict[str, list[str]] | None = None,
        dataset_root: Path | None = None,
        enable_runtime_bad_sample_cleanup: bool = False,
    ) -> None:
        if base_dataset is None:
            if root is None:
                raise ValueError("Either `root` or `base_dataset` must be provided.")
            base_dataset = datasets.ImageFolder(root)
        self.base_dataset = base_dataset
        self.image_size = image_size
        self.augment_repeats = max(1, int(augment_repeats))
        self.split_name = split_name
        self.seed = seed
        self.gaussian_sigmas = gaussian_sigmas
        self.camera_color_cast_probability = float(camera_color_cast_probability)
        self.camera_color_cast_strength = float(camera_color_cast_strength)
        self.camera_color_cast_eval = bool(camera_color_cast_eval)
        self.apply_augmentation = apply_augmentation
        # Train-time views may use deterministic-seeded random crop/flip augmentation.
        # Validation/test stay deterministic apart from the fixed tint.
        self.stochastic_augmentation = bool(apply_augmentation)
        self.dataset_root = dataset_root
        self.enable_runtime_bad_sample_cleanup = enable_runtime_bad_sample_cleanup
        self.disabled_paths: set[str] = set()
        self._draw_counter = 0
        
        # Original metadata
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx
        self.samples = list(self.base_dataset.samples if samples is None else samples)
        
        self.apply_class_mapping(class_mapping or {})
            
        self.targets = [target for _, target in self.samples]
        self.current_epoch = 0

    def apply_class_mapping(self, class_mapping: dict[str, list[str]]) -> None:
        new_classes, new_class_to_idx, new_samples, metadata = project_samples_to_training_taxonomy(
            list(self.classes),
            list(self.samples),
            class_mapping,
        )
        self.classes = new_classes
        self.class_to_idx = new_class_to_idx
        self.samples = new_samples
        self.taxonomy_metadata = metadata

    def __len__(self) -> int:
        return len(self.samples) * self.augment_repeats

    def source_count(self) -> int:
        return len(self.samples)

    def source_target_for_index(self, index: int) -> int:
        return self.targets[index]

    def target_for_index(self, index: int) -> int:
        source_index = index % len(self.samples)
        return self.targets[source_index]

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(0, int(epoch))

    def variant_for_source(self, source_index: int) -> int:
        if self.augment_repeats <= 1:
            return 0
        variant_rng = random.Random(
            self.seed * 1_000_003
            + source_index * 9_973
            + self.current_epoch * 104_729
        )
        return variant_rng.randrange(self.augment_repeats)

    def _seed_for_variant(self, source_index: int, variant_index: int, view_offset: int = 0) -> int:
        return (
            self.seed * 1_000_003
            + SPLIT_OFFSETS[self.split_name]
            + source_index * 9_973
            + variant_index * 99_991
            + self.current_epoch * 104_729
            + view_offset * 1_299_721
        )

    def _runtime_augmentation_seed(
        self,
        source_index: int,
        variant_index: int,
        view_offset: int = 0,
        attempt: int = 0,
    ) -> int:
        self._draw_counter += 1
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        entropy = int.from_bytes(os.urandom(8), byteorder="little", signed=False)
        return (
            entropy
            ^ self._seed_for_variant(source_index, variant_index, view_offset)
            ^ ((attempt + 1) * 15_485_863)
            ^ ((self._draw_counter + 1) * 32_452_843)
            ^ ((os.getpid() & 0xFFFFFFFF) << 1)
            ^ ((worker_id + 2) << 17)
        ) & ((1 << 63) - 1)

    def _fallback_index_for_target(self, source_index: int, target: int, attempt: int) -> int | None:
        same_class_indices = [
            i
            for i, (candidate_path, candidate_target) in enumerate(self.samples)
            if candidate_target == target and candidate_path not in self.disabled_paths
        ]
        if not same_class_indices:
            return None
        fallback_rng = random.Random(source_index + self.seed + self.current_epoch + attempt)
        return fallback_rng.choice(same_class_indices)

    def _cleanup_bad_sample(self, path: str, target: int, attempt: int, error: Exception) -> None:
        self.disabled_paths.add(path)
        cleanup_event: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": "runtime_bad_sample_detected",
            "split": self.split_name,
            "attempt": attempt,
            "path": path,
            "target_index": int(target),
            "target_class": self.classes[target] if 0 <= target < len(self.classes) else None,
            "error_type": type(error).__name__,
            "error": str(error),
            "cleanup_enabled": bool(self.enable_runtime_bad_sample_cleanup),
        }

        if self.enable_runtime_bad_sample_cleanup and self.dataset_root is not None:
            file_removed = delete_bad_sample_file(Path(path))
            metadata_removed = remove_bad_sample_from_metadata(self.dataset_root, Path(path))
            cleanup_event["file_removed"] = file_removed
            cleanup_event["metadata_removed"] = metadata_removed
            append_jsonl(self.dataset_root / RUNTIME_BAD_SAMPLE_CLEANUP_LOG, cleanup_event)
            console_line = format_console_event(cleanup_event)
            if console_line:
                print(console_line, flush=True)

        logging.warning(
            "Attempt %s/100: Skipping corrupted image %s: %s",
            attempt,
            path,
            error,
        )

    def load_augmented(self, source_index: int, variant_index: int, view_offset: int = 0) -> tuple[torch.Tensor, int]:
        max_attempts = 100
        attempt = 0
        current_source_index = source_index
        
        while attempt < max_attempts:
            path, target = self.samples[current_source_index]
            if path in self.disabled_paths:
                attempt += 1
                next_index = self._fallback_index_for_target(source_index, target, attempt)
                if next_index is None:
                    break
                current_source_index = next_index
                continue
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=PALETTE_TRANSPARENCY_WARNING, category=UserWarning)
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    raw_image = self.base_dataset.loader(path)
                # Handle images with transparency or palettes properly before preprocessing.
                if raw_image.mode in ("RGBA", "P"):
                    image = raw_image.convert("RGB")
                else:
                    image = raw_image.convert("RGB")

                if self.apply_augmentation:
                    aug_rng = random.Random(
                        self._runtime_augmentation_seed(
                            current_source_index,
                            variant_index,
                            view_offset=view_offset,
                            attempt=attempt,
                        )
                    )
                    tensor = training_tensor_from_image(
                        image,
                        self.image_size,
                        aug_rng,
                        self.gaussian_sigmas,
                        camera_color_cast_probability=self.camera_color_cast_probability,
                        camera_color_cast_strength=self.camera_color_cast_strength,
                    )
                else:
                    tensor = evaluation_tensor_from_image(
                        image,
                        self.image_size,
                        rng=None,
                        camera_color_cast_eval=self.camera_color_cast_eval,
                        camera_color_cast_strength=self.camera_color_cast_strength,
                    )
                return tensor, target
                
            except Exception as e:
                attempt += 1
                self._cleanup_bad_sample(path, target, attempt, e)
                next_index = self._fallback_index_for_target(source_index, target, attempt)
                if next_index is None:
                    break
                current_source_index = next_index

        raise RuntimeError(f"Could not find a valid image for class index {target} after {max_attempts} attempts.")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        source_index = index % len(self.samples)
        variant_index = index // len(self.samples)
        return self.load_augmented(source_index, variant_index)


class DeterministicSupConDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(self, base_dataset: DeterministicAugmentedImageFolder) -> None:
        self.base_dataset = base_dataset
        self.classes = base_dataset.classes
        self.class_to_idx = base_dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_epoch(self, epoch: int) -> None:
        self.base_dataset.set_epoch(epoch)

    def target_for_index(self, index: int) -> int:
        return self.base_dataset.target_for_index(index)

    def source_count(self) -> int:
        return self.base_dataset.source_count()

    def source_target_for_index(self, index: int) -> int:
        return self.base_dataset.source_target_for_index(index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        source_index = index % self.base_dataset.source_count()
        variant_index = index // self.base_dataset.source_count()
        view_one, target = self.base_dataset.load_augmented(source_index, variant_index, view_offset=0)
        view_two, _ = self.base_dataset.load_augmented(source_index, variant_index, view_offset=1)
        return view_one, view_two, target


def sample_gaussian_clipped(rng: random.Random, mean: float, std: float, low: float, high: float) -> float:
    del rng, std
    return min(max(mean, low), high)


def sample_safe_range(
    rng: random.Random,
    safe_low: float,
    safe_high: float,
    hard_low: float,
    hard_high: float,
    gaussian_sigmas: float,
    *,
    mean: float | None = None,
) -> float:
    del rng, hard_low, hard_high, gaussian_sigmas
    if mean is None:
        mean = (safe_low + safe_high) / 2.0
    return min(max(mean, safe_low), safe_high)


def sample_symmetric(
    rng: random.Random,
    safe_abs: float,
    hard_abs: float,
    gaussian_sigmas: float,
    *,
    mean: float = 0.0,
) -> float:
    del rng, safe_abs, hard_abs, gaussian_sigmas
    return mean


def sample_log_safe_ratio(
    rng: random.Random,
    safe_low: float,
    safe_high: float,
    hard_low: float,
    hard_high: float,
    gaussian_sigmas: float,
) -> float:
    del rng, hard_low, hard_high, gaussian_sigmas
    return math.sqrt(max(safe_low * safe_high, 1e-12))


def random_resized_crop(image: Image.Image, image_size: int, rng: random.Random, gaussian_sigmas: float) -> Image.Image:
    del image_size, gaussian_sigmas
    width, height = image.size
    if width <= 1 or height <= 1:
        return image

    scale = sample_safe_range(rng, 0.70, 1.00, 0.70, 1.00, 0.0)
    crop_width = max(1, min(width, int(round(width * scale))))
    crop_height = max(1, min(height, int(round(height * scale))))
    if crop_width >= width and crop_height >= height:
        return image

    max_left = max(0, width - crop_width)
    max_top = max(0, height - crop_height)
    left = rng.randint(0, max_left) if max_left > 0 else 0
    top = rng.randint(0, max_top) if max_top > 0 else 0
    return image.crop((left, top, left + crop_width, top + crop_height))


def random_perspective(image: Image.Image, rng: random.Random, gaussian_sigmas: float) -> Image.Image:
    del rng, gaussian_sigmas
    return image


def apply_resolution_degradation(image: Image.Image, rng: random.Random, gaussian_sigmas: float) -> Image.Image:
    del rng, gaussian_sigmas
    return image


def apply_border_truncation(image: Image.Image, rng: random.Random, gaussian_sigmas: float) -> Image.Image:
    del rng, gaussian_sigmas
    return image


def jpeg_compress(image: Image.Image, rng: random.Random, gaussian_sigmas: float) -> Image.Image:
    del rng, gaussian_sigmas
    return image


def motion_blur_kernel(kernel_size: int, angle_degrees: float) -> torch.Tensor:
    base = Image.new("F", (kernel_size, kernel_size), 0.0)
    center = kernel_size // 2
    for x in range(kernel_size):
        base.putpixel((x, center), 1.0)
    rotated = base.rotate(angle_degrees, resample=Image.Resampling.BILINEAR)
    kernel = torch.from_numpy(np.array(rotated, dtype=np.float32))
    kernel = kernel / max(float(kernel.sum()), 1e-6)
    return kernel


def apply_motion_blur(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    del rng, gaussian_sigmas
    return tensor


def defocus_blur_kernel(radius: float) -> torch.Tensor:
    radius = max(radius, 0.5)
    kernel_size = max(3, int(math.ceil(radius * 2.0)) * 2 + 1)
    center = kernel_size // 2
    yy, xx = torch.meshgrid(
        torch.arange(kernel_size, dtype=torch.float32),
        torch.arange(kernel_size, dtype=torch.float32),
        indexing="ij",
    )
    dist = torch.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    kernel = (dist <= radius).float()
    kernel = kernel / max(float(kernel.sum()), 1e-6)
    return kernel


def apply_defocus_blur(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    del rng, gaussian_sigmas
    return tensor


def apply_gaussian_noise(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    del rng, gaussian_sigmas
    return tensor


def apply_channel_shift(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    del rng, gaussian_sigmas
    return tensor


def apply_grayscale_mix(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    del rng, gaussian_sigmas
    return tensor


def apply_illumination_gradient(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    del rng, gaussian_sigmas
    return tensor


def apply_smudge_overlay(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    del rng, gaussian_sigmas
    return tensor


def apply_shadow_overlay(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float, shadow_prob: float) -> torch.Tensor:
    del rng, gaussian_sigmas, shadow_prob
    return tensor


def apply_specular_glare(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float, glare_prob: float) -> torch.Tensor:
    del rng, gaussian_sigmas, glare_prob
    return tensor


def apply_cutout(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    del rng, gaussian_sigmas
    return tensor


def apply_random_flips(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return image


def training_tensor_from_image(
    image: Image.Image,
    image_size: int,
    rng: random.Random,
    gaussian_sigmas: float,
    *,
    camera_color_cast_probability: float = CAMERA_COLOR_CAST_PROBABILITY,
    camera_color_cast_strength: float = CAMERA_COLOR_CAST_STRENGTH,
) -> torch.Tensor:
    image = random_resized_crop(image, image_size, rng, gaussian_sigmas)
    image = apply_random_flips(image, rng)
    image = resize_with_letterbox(image, image_size)
    tensor = TF.to_tensor(image).clamp(0.0, 1.0)
    tensor = apply_camera_color_cast(
        tensor,
        random.Random(0),
        gaussian_sigmas=0.0,
        probability=1.0 if camera_color_cast_probability > 0 else 0.0,
        strength=camera_color_cast_strength,
    )
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def apply_camera_color_cast(
    tensor: torch.Tensor,
    rng: random.Random,
    gaussian_sigmas: float,
    probability: float,
    strength: float,
) -> torch.Tensor:
    """Apply the repo-wide fixed Raspberry Pi pink tint."""
    _ = rng, gaussian_sigmas, probability
    strength = max(0.0, float(strength))
    if strength <= 0.0:
        return tensor

    red_gain = 1.0 + 0.70 * strength
    green_gain = 1.0 - 0.35 * strength
    blue_gain = 1.0 + 0.62 * strength
    gains = torch.tensor([red_gain, green_gain, blue_gain], dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    tensor = tensor * gains

    wash_opacity = 0.16 * strength
    wash_color = torch.tensor([1.0, 0.72, 0.98], dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return tensor * (1.0 - wash_opacity) + wash_color * wash_opacity


def resize_with_letterbox(image: Image.Image, image_size: int, *, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Resize while preserving aspect ratio, then pad to a square canvas."""
    target = int(image_size)
    if target <= 0:
        raise ValueError("image_size must be positive")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("input image must have positive dimensions")

    scale = min(target / float(width), target / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target, target), fill)
    offset_x = (target - new_width) // 2
    offset_y = (target - new_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def augmented_tensor_from_image(
    image: Image.Image,
    image_size: int,
    rng: random.Random,
    gaussian_sigmas: float,
    *,
    camera_color_cast_probability: float = CAMERA_COLOR_CAST_PROBABILITY,
    camera_color_cast_strength: float = CAMERA_COLOR_CAST_STRENGTH,
) -> torch.Tensor:
    _ = rng, gaussian_sigmas, camera_color_cast_probability
    image = resize_with_letterbox(image, image_size)
    tensor = TF.to_tensor(image).clamp(0.0, 1.0)
    tensor = apply_camera_color_cast(
        tensor,
        random.Random(0),
        gaussian_sigmas=0.0,
        probability=1.0,
        strength=camera_color_cast_strength,
    )
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def evaluation_tensor_from_image(
    image: Image.Image,
    image_size: int,
    *,
    rng: random.Random | None = None,
    camera_color_cast_eval: bool = False,
    camera_color_cast_strength: float = CAMERA_COLOR_CAST_STRENGTH,
) -> torch.Tensor:
    image = resize_with_letterbox(image, image_size)
    tensor = TF.to_tensor(image).clamp(0.0, 1.0)
    _ = rng, camera_color_cast_eval
    tensor = apply_camera_color_cast(
        tensor,
        random.Random(0),
        gaussian_sigmas=0.0,
        probability=1.0,
        strength=camera_color_cast_strength,
    ).clamp(0.0, 1.0)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class MetricLearningEfficientNetB0(nn.Module):
    def __init__(
        self,
        num_classes: int,
        weights_mode: str,
        embedding_dim: int,
        projection_dim: int,
        args: argparse.Namespace,
        backbone_name: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name or getattr(args, "backbone", DEFAULT_BACKBONE_NAME)
        backbone = create_backbone(self.backbone_name, weights_mode)
        self.front_end = nn.Identity()
        self.backbone = backbone
        self.in_features = int(getattr(backbone, "num_features", 0) or 0)
        if self.in_features <= 0:
            image_size = int(getattr(args, "image_size", 224))
            with torch.no_grad():
                dummy = torch.zeros(1, 3, image_size, image_size)
                features = self.backbone.forward_features(dummy) if hasattr(self.backbone, "forward_features") else self.backbone(dummy)
                if isinstance(features, (list, tuple)):
                    features = features[-1]
                if features.ndim == 4:
                    features = F.adaptive_avg_pool2d(features, 1)
                self.in_features = int(torch.flatten(features, 1).shape[1])
        self.dropout_p = float(getattr(backbone, "drop_rate", 0.0) or 0.0)
        self.gradient_checkpointing = True
        if self.gradient_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            try:
                self.backbone.set_grad_checkpointing(True)
            except TypeError:
                self.backbone.set_grad_checkpointing(enable=True)
        self.checkpoint_segments = 4
        self.embedding = nn.Linear(self.in_features, embedding_dim, bias=False)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.ce_head = nn.Linear(embedding_dim, num_classes)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.front_end(x)
        if hasattr(self.backbone, "forward_features"):
            x = self.backbone.forward_features(x)
        else:
            x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        if x.ndim == 4:
            x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def encode(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.forward_backbone(x)
        x = self.embedding(x)
        x = self.embedding_norm(x)
        if normalize:
            x = F.normalize(x, dim=1)
        return x

    def supcon_projection(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection_head(embeddings), dim=1)

    def classify(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        del labels
        return self.ce_head(embeddings)


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, views, dim]")
        batch_size, view_count, feature_dim = features.shape
        del feature_dim

        features = F.normalize(features, dim=2)
        flat_features = features.reshape(batch_size * view_count, -1)
        flat_labels = labels.repeat_interleave(view_count)

        logits = torch.matmul(flat_features, flat_features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(batch_size * view_count, device=features.device, dtype=torch.bool)
        positive_mask = flat_labels.unsqueeze(0).eq(flat_labels.unsqueeze(1)) & ~self_mask
        exp_logits = torch.exp(logits) * (~self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_counts = positive_mask.sum(dim=1)
        valid = positive_counts > 0
        if not valid.any():
            return flat_features.new_tensor(0.0)

        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_counts.clamp_min(1)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


def supcon_contrastive_metrics(
    proj_one: torch.Tensor,
    proj_two: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float | int]:
    with torch.no_grad():
        proj_one = F.normalize(proj_one.detach().float(), dim=1)
        proj_two = F.normalize(proj_two.detach().float(), dim=1)
        labels = labels.detach()

        view_pair_cosine = (proj_one * proj_two).sum(dim=1).mean()
        features = torch.stack([proj_one, proj_two], dim=1)
        batch_size, view_count, feature_dim = features.shape
        del feature_dim

        flat_features = features.reshape(batch_size * view_count, -1)
        flat_labels = labels.repeat_interleave(view_count)
        similarities = torch.matmul(flat_features, flat_features.T)
        self_mask = torch.eye(batch_size * view_count, device=similarities.device, dtype=torch.bool)
        positive_mask = flat_labels.unsqueeze(0).eq(flat_labels.unsqueeze(1)) & ~self_mask
        negative_mask = ~flat_labels.unsqueeze(0).eq(flat_labels.unsqueeze(1))

        positive_values = similarities[positive_mask]
        negative_values = similarities[negative_mask]
        positive_mean = positive_values.mean() if positive_values.numel() else similarities.new_tensor(float("nan"))
        negative_mean = negative_values.mean() if negative_values.numel() else similarities.new_tensor(float("nan"))
        margin = positive_mean - negative_mean

    return {
        "same_image_view_cosine": float(view_pair_cosine.item()),
        "same_class_positive_cosine": float(positive_mean.item()),
        "different_class_negative_cosine": float(negative_mean.item()),
        "positive_negative_cosine_margin": float(margin.item()),
        "positive_pair_count": int(positive_values.numel()),
        "negative_pair_count": int(negative_values.numel()),
    }


class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs) -> None:
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                self.state.setdefault(parameter, {})
                self.state[parameter]["old_p"] = parameter.data.clone()
                e_w = (torch.pow(parameter, 2) if group["adaptive"] else 1.0) * parameter.grad * scale.to(parameter)
                parameter.add_(e_w)
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.data = self.state[parameter]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("Use first_step and second_step for SAM.")

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if group["adaptive"]:
                    norms.append((parameter.abs() * parameter.grad).norm(p=2))
                else:
                    norms.append(parameter.grad.norm(p=2))
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)

    def state_dict(self):
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "sam_state": super().state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        super().load_state_dict(state_dict["sam_state"])


def parse_wavelengths(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_counts(dataset: DeterministicAugmentedImageFolder) -> dict[str, int]:
    counts = {name: 0 for name in dataset.classes}
    for target in dataset.targets:
        counts[dataset.classes[target]] += 1
    return counts


def effective_class_counts(dataset: DeterministicAugmentedImageFolder) -> dict[str, int]:
    base_counts = class_counts(dataset)
    return {name: count * dataset.augment_repeats for name, count in base_counts.items()}


class DeterministicEpochSampler(Sampler[int]):
    def __init__(
        self,
        num_samples: int,
        seed: int,
        *,
        shuffle: bool,
        weights: torch.Tensor | None = None,
        replacement: bool = True,
    ) -> None:
        self.num_samples = num_samples
        self.seed = seed
        self.shuffle = shuffle
        self.weights = weights
        self.replacement = replacement
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    def set_start_index(self, start_index: int) -> None:
        self.start_index = min(max(0, int(start_index)), self.num_samples)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.weights is not None:
            indices = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=generator).tolist()
        elif self.shuffle:
            indices = torch.randperm(self.num_samples, generator=generator).tolist()
        else:
            indices = list(range(self.num_samples))
        return iter(indices[self.start_index :])

    def __len__(self) -> int:
        return max(0, self.num_samples - self.start_index)


class BalancedClassEpochSampler(Sampler[int]):
    def __init__(
        self,
        classes: list[str],
        class_source_indices: dict[int, list[int]],
        source_count: int,
        augment_repeats: int,
        batch_size: int,
        seed: int,
    ) -> None:
        self.classes = list(classes)
        self.class_source_indices = {int(key): list(value) for key, value in class_source_indices.items()}
        self.source_count = int(source_count)
        self.augment_repeats = max(1, int(augment_repeats))
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0
        if not self.classes:
            raise ValueError("Balanced sampling requires at least one class.")
        if self.batch_size < len(self.classes):
            raise ValueError(
                "Balanced sampling requires --batch-size to be at least the number of classes "
                "so every batch can include every class."
            )
        self.samples_per_class = self.batch_size // len(self.classes)
        self.remainder_samples = self.batch_size % len(self.classes)
        class_counts = [len(self.class_source_indices.get(class_index, [])) for class_index in range(len(self.classes))]
        missing_classes = [self.classes[class_index] for class_index, count in enumerate(class_counts) if count <= 0]
        if missing_classes:
            raise ValueError(f"Balanced sampling requires at least one sample in every class; missing: {missing_classes}")
        # One epoch should cover the full source corpus, so we size the epoch by
        # the largest class and let smaller classes repeat only as needed to keep
        # batches class-balanced.
        self.num_batches = max(math.ceil(count / self.samples_per_class) for count in class_counts)
        if self.num_batches <= 0:
            raise ValueError(
                "Balanced sampling cannot form a full batch with the available class counts. "
                f"Need at least {self.samples_per_class} samples per class, got {class_counts}."
            )
        self.num_samples = self.num_batches * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    def set_start_index(self, start_index: int) -> None:
        self.start_index = min(max(0, int(start_index)), self.num_samples)

    def __iter__(self):
        if self.num_samples <= 0:
            return iter(())

        rng = random.Random(self.seed + self.epoch)
        class_order = list(range(len(self.classes)))
        rng.shuffle(class_order)
        per_class_streams: dict[int, list[int]] = {}
        for class_index in class_order:
            source_indices = list(self.class_source_indices.get(class_index, []))
            if not source_indices:
                continue
            rng.shuffle(source_indices)
            target_len = self.num_batches * (self.samples_per_class + (1 if self.remainder_samples > 0 else 0))
            stream: list[int] = []
            while len(stream) < target_len:
                stream.extend(source_indices)
                rng.shuffle(source_indices)
            per_class_streams[class_index] = stream[:target_len]

        flat_indices: list[int] = []
        stream_offsets = {class_index: 0 for class_index in per_class_streams}
        for batch_index in range(self.num_batches):
            batch_class_order = class_order[batch_index % len(class_order) :] + class_order[: batch_index % len(class_order)]
            extra_classes = set(batch_class_order[: self.remainder_samples])
            for class_index in class_order:
                stream = per_class_streams.get(class_index)
                if stream is None:
                    continue
                batch_quota = self.samples_per_class + (1 if class_index in extra_classes else 0)
                start = stream_offsets[class_index]
                end = start + batch_quota
                for offset, source_index in enumerate(stream[start:end]):
                    variant_index = rng.randrange(self.augment_repeats)
                    flat_indices.append(source_index + (variant_index * self.source_count))
                stream_offsets[class_index] = end
        return iter(flat_indices[self.start_index :])

    def __len__(self) -> int:
        return max(0, self.num_samples - self.start_index)


def make_weighted_sampler(dataset: Dataset, classes: list[str], target_fn, seed: int) -> DeterministicEpochSampler:
    counts = {name: 0 for name in classes}
    for index in range(len(dataset)):
        counts[classes[target_fn(index)]] += 1
    missing_classes = [name for name, count in counts.items() if count <= 0]
    if missing_classes:
        raise ValueError(f"Weighted sampling requires at least one sample in every class; missing: {missing_classes}")
    weights = [1.0 / counts[classes[target_fn(index)]] for index in range(len(dataset))]
    return DeterministicEpochSampler(
        len(weights),
        seed,
        shuffle=False,
        weights=torch.as_tensor(weights, dtype=torch.double),
        replacement=False,
    )


def make_epoch_sampler(dataset: Dataset, seed: int, shuffle: bool) -> DeterministicEpochSampler:
    return DeterministicEpochSampler(len(dataset), seed, shuffle=shuffle)


def make_balanced_sampler(dataset: Dataset, classes: list[str], batch_size: int, seed: int) -> BalancedClassEpochSampler:
    if not hasattr(dataset, "source_count") or not hasattr(dataset, "source_target_for_index"):
        raise TypeError("Balanced sampling requires dataset.source_count() and dataset.source_target_for_index().")
    class_source_indices: dict[int, list[int]] = {index: [] for index in range(len(classes))}
    source_count = int(dataset.source_count())
    for source_index in range(source_count):
        class_source_indices[int(dataset.source_target_for_index(source_index))].append(source_index)
    augment_repeats = int(getattr(dataset, "augment_repeats", 1))
    return BalancedClassEpochSampler(
        classes,
        class_source_indices,
        source_count=source_count,
        augment_repeats=augment_repeats,
        batch_size=batch_size,
        seed=seed,
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    shuffle: bool,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def build_datasets(
    args: argparse.Namespace,
) -> tuple[
    DeterministicAugmentedImageFolder,
    DeterministicAugmentedImageFolder,
    DeterministicAugmentedImageFolder,
    DeterministicSupConDataset,
    DeterministicSupConDataset,
]:
    class_mapping = parse_json_class_mapping(getattr(args, "class_mapping", ""))
    root = Path(args.dataset_root)
    if getattr(args, "train_only", False):
        base_dataset = datasets.ImageFolder(root)
        new_classes, new_class_to_idx, new_samples, taxonomy_metadata = project_samples_to_training_taxonomy(
            list(base_dataset.classes),
            list(base_dataset.samples),
            class_mapping,
        )
        base_dataset.classes = new_classes
        base_dataset.class_to_idx = new_class_to_idx
        base_dataset.samples = new_samples
        base_dataset.targets = [sample[1] for sample in base_dataset.samples]

        manifest = {
            "dataset_root": str(root),
            "split_mode": "train_only",
            "seed": int(args.seed),
            "split_ratios": {"train": 1.0, "val": 0.0, "test": 0.0},
            "class_names": list(base_dataset.classes),
            "taxonomy": taxonomy_metadata,
            "source_samples": len(base_dataset.samples),
            "train_samples": len(base_dataset.samples),
            "val_samples": 0,
            "test_samples": 0,
        }
        manifest_output_dir = Path(getattr(args, "output_dir", "")) if getattr(args, "output_dir", "") else None
        if manifest_output_dir is not None:
            manifest_output_dir.mkdir(parents=True, exist_ok=True)
            save_json(manifest_output_dir / "auto_split_manifest.json", manifest)

        train_dataset = DeterministicAugmentedImageFolder(
            None,
            args.image_size,
            args.augment_repeats,
            "train",
            args.seed,
            args.augment_gaussian_sigmas,
            args.camera_color_cast_probability,
            args.camera_color_cast_strength,
            args.camera_color_cast_eval,
            apply_augmentation=True,
            base_dataset=base_dataset,
            samples=list(base_dataset.samples),
            dataset_root=root,
            enable_runtime_bad_sample_cleanup=args.runtime_bad_sample_cleanup,
        )
        val_dataset = DeterministicAugmentedImageFolder(
            None,
            args.image_size,
            1,
            "val",
            args.seed,
            args.augment_gaussian_sigmas,
            args.camera_color_cast_probability,
            args.camera_color_cast_strength,
            args.camera_color_cast_eval,
            apply_augmentation=False,
            base_dataset=base_dataset,
            samples=[],
            dataset_root=root,
            enable_runtime_bad_sample_cleanup=args.runtime_bad_sample_cleanup,
        )
        test_dataset = DeterministicAugmentedImageFolder(
            None,
            args.image_size,
            1,
            "test",
            args.seed,
            args.augment_gaussian_sigmas,
            args.camera_color_cast_probability,
            args.camera_color_cast_strength,
            args.camera_color_cast_eval,
            apply_augmentation=False,
            base_dataset=base_dataset,
            samples=[],
            dataset_root=root,
            enable_runtime_bad_sample_cleanup=args.runtime_bad_sample_cleanup,
        )
        return (
            train_dataset,
            val_dataset,
            test_dataset,
            DeterministicSupConDataset(train_dataset),
            DeterministicSupConDataset(val_dataset),
        )
    if has_explicit_split_layout(root):
        train_dataset = DeterministicAugmentedImageFolder(
            root / "train",
            args.image_size,
            args.augment_repeats,
            "train",
            args.seed,
            args.augment_gaussian_sigmas,
            args.camera_color_cast_probability,
            args.camera_color_cast_strength,
            args.camera_color_cast_eval,
            apply_augmentation=True,
            class_mapping=class_mapping,
            dataset_root=root,
            enable_runtime_bad_sample_cleanup=args.runtime_bad_sample_cleanup,
        )
        val_dataset = DeterministicAugmentedImageFolder(
            root / "val",
            args.image_size,
            1,
            "val",
            args.seed,
            args.augment_gaussian_sigmas,
            args.camera_color_cast_probability,
            args.camera_color_cast_strength,
            args.camera_color_cast_eval,
            apply_augmentation=False,
            class_mapping=class_mapping,
            dataset_root=root,
            enable_runtime_bad_sample_cleanup=args.runtime_bad_sample_cleanup,
        )
        test_dataset = DeterministicAugmentedImageFolder(
            root / "test",
            args.image_size,
            1,  # no repeat multiplier on test — evaluate each image exactly once
            "test",
            args.seed,
            args.augment_gaussian_sigmas,
            args.camera_color_cast_probability,
            args.camera_color_cast_strength,
            args.camera_color_cast_eval,
            apply_augmentation=False,  # deterministic eval path; only fixed tint is applied
            class_mapping=class_mapping,
            dataset_root=root,
            enable_runtime_bad_sample_cleanup=args.runtime_bad_sample_cleanup,
        )
        if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
            raise ValueError("Class folders differ across train/val/test")
    else:
        # Pass mapping through to auto-split logic too
        train_dataset, val_dataset, test_dataset = build_auto_split_datasets(root, args, class_mapping=class_mapping)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        DeterministicSupConDataset(train_dataset),
        DeterministicSupConDataset(val_dataset),
    )


def has_explicit_split_layout(root: Path) -> bool:
    return all((root / split_name).is_dir() for split_name in ("train", "val", "test"))


def parse_auto_split_ratios(spec: str) -> tuple[float, float, float]:
    parts = [segment.strip() for segment in spec.split(",") if segment.strip()]
    if len(parts) != 3:
        raise ValueError("--auto-split-ratios must contain exactly three comma-separated values.")
    values = [float(part) for part in parts]
    if any(value < 0.0 for value in values):
        raise ValueError("--auto-split-ratios values must be non-negative.")
    if not any(value > 0.0 for value in values):
        raise ValueError("--auto-split-ratios must contain at least one positive value.")
    total = sum(values)
    return (values[0] / total, values[1] / total, values[2] / total)


def allocate_split_counts(sample_count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if sample_count <= 0:
        return (0, 0, 0)
    minimums = [0, 0, 0]
    if sample_count >= 1:
        minimums[0] = 1
    if sample_count >= 2:
        minimums[1] = 1
    if sample_count >= 3:
        minimums[2] = 1
    remaining = sample_count - sum(minimums)
    if remaining <= 0:
        return tuple(minimums)
    raw = [remaining * ratio for ratio in ratios]
    extras = [int(math.floor(value)) for value in raw]
    shortfall = remaining - sum(extras)
    remainders = sorted(
        ((raw[index] - extras[index], index) for index in range(3)),
        reverse=True,
    )
    for _, index in remainders[:shortfall]:
        extras[index] += 1
    return tuple(minimums[index] + extras[index] for index in range(3))


def build_auto_split_datasets(
    root: Path,
    args: argparse.Namespace,
    class_mapping: dict[str, list[str]] | None = None,
) -> tuple[DeterministicAugmentedImageFolder, DeterministicAugmentedImageFolder, DeterministicAugmentedImageFolder]:
    ratios = parse_auto_split_ratios(args.auto_split_ratios)
    base_dataset = datasets.ImageFolder(root)
    new_classes, new_class_to_idx, new_samples, taxonomy_metadata = project_samples_to_training_taxonomy(
        list(base_dataset.classes),
        list(base_dataset.samples),
        class_mapping,
    )
    base_dataset.classes = new_classes
    base_dataset.class_to_idx = new_class_to_idx
    base_dataset.samples = new_samples
    base_dataset.targets = [sample[1] for sample in base_dataset.samples]

    # ── Source-level stratified split ────────────────────────────────────────
    # Within each class, group images by their semantic source prefix (derived
    # from filename). Each source batch is split independently at the ratio
    # boundary, so every source contributes its proportional share of images
    # to train, val, and test. This prevents any single dominant source from
    # flooding val/test while others are under-represented.
    import re as _re

    def _source_prefix(path: str) -> str:
        stem = Path(path).stem
        parts = _re.split(r'[^a-zA-Z0-9]', stem)
        for p in parts:
            word = _re.sub(r'\d+', '', p)
            if word:
                return word.lower()
        return stem[:8].lower()

    by_class: dict[int, dict[str, list[tuple[str, int]]]] = {
        index: {} for index in range(len(base_dataset.classes))
    }
    for path, target in base_dataset.samples:
        src = _source_prefix(path)
        by_class[int(target)].setdefault(src, []).append((path, int(target)))

    rng = random.Random(args.seed)
    train_samples: list[tuple[str, int]] = []
    val_samples: list[tuple[str, int]] = []
    test_samples: list[tuple[str, int]] = []
    split_counts: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for class_index, sources in by_class.items():
        class_name = base_dataset.classes[class_index]
        cls_train, cls_val, cls_test = 0, 0, 0
        for src_name, src_samples in sorted(sources.items()):
            shuffled = list(src_samples)
            rng.shuffle(shuffled)
            tr_n, va_n, te_n = allocate_split_counts(len(shuffled), ratios)
            train_samples.extend(shuffled[:tr_n])
            val_samples.extend(shuffled[tr_n : tr_n + va_n])
            test_samples.extend(shuffled[tr_n + va_n : tr_n + va_n + te_n])
            cls_train += tr_n
            cls_val   += va_n
            cls_test  += te_n
        split_counts["train"][class_name] = cls_train
        split_counts["val"][class_name]   = cls_val
        split_counts["test"][class_name]  = cls_test

    manifest = {
        "dataset_root": str(root),
        "split_mode": "source_stratified_within_class",
        "seed": int(args.seed),
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "class_names": list(base_dataset.classes),
        "taxonomy": taxonomy_metadata,
        "split_counts": split_counts,
        "source_samples": len(base_dataset.samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
    }
    manifest_output_dir = Path(getattr(args, "output_dir", "")) if getattr(args, "output_dir", "") else None
    if manifest_output_dir is not None:
        manifest_output_dir.mkdir(parents=True, exist_ok=True)
        save_json(manifest_output_dir / "auto_split_manifest.json", manifest)

    train_dataset = DeterministicAugmentedImageFolder(
        None,
        args.image_size,
        args.augment_repeats,
        "train",
        args.seed,
        args.augment_gaussian_sigmas,
        args.camera_color_cast_probability,
        args.camera_color_cast_strength,
        args.camera_color_cast_eval,
        apply_augmentation=True,
        base_dataset=base_dataset,
        samples=train_samples,
        dataset_root=root,
        enable_runtime_bad_sample_cleanup=args.runtime_bad_sample_cleanup,
    )
    val_dataset = DeterministicAugmentedImageFolder(
        None,
        args.image_size,
        1,
        "val",
        args.seed,
        args.augment_gaussian_sigmas,
        args.camera_color_cast_probability,
        args.camera_color_cast_strength,
        args.camera_color_cast_eval,
        apply_augmentation=False,
        base_dataset=base_dataset,
        samples=val_samples,
        dataset_root=root,
        enable_runtime_bad_sample_cleanup=args.runtime_bad_sample_cleanup,
    )
    test_dataset = DeterministicAugmentedImageFolder(
        None,
        args.image_size,
        1,  # no repeat multiplier on test — evaluate each image exactly once
        "test",
        args.seed,
        args.augment_gaussian_sigmas,
        args.camera_color_cast_probability,
        args.camera_color_cast_strength,
        args.camera_color_cast_eval,
        apply_augmentation=False,  # deterministic eval path; only fixed tint is applied
        base_dataset=base_dataset,
        samples=test_samples,
        dataset_root=root,
        enable_runtime_bad_sample_cleanup=args.runtime_bad_sample_cleanup,
    )
    return train_dataset, val_dataset, test_dataset


def build_supcon_phase_plan(total_modules: int, args: argparse.Namespace) -> list[PhaseSpec]:
    """Build the ordered list of SupCon phases.

    The SupCon stage now mirrors the classifier stage design:
    - head-only warm-up with a fully frozen backbone
    - progressive semantic-tail unfreezing in fixed chunks
    - permanently frozen low-level core modules
    """
    phases: list[PhaseSpec] = []
    phases.append(PhaseSpec(name="supcon_head_only", unfrozen_backbone_modules=0))

    effective_max = effective_unfrozen_backbone_cap(total_modules, args, "supcon")
    count = 0
    phase_index = 0
    while count < effective_max:
        count = min(effective_max, count + args.unfreeze_chunk_size)
        phase_index += 1
        phases.append(PhaseSpec(name=f"supcon_last_{count}_modules", unfrozen_backbone_modules=count))
    return phases


def build_classifier_phase_plan(total_modules: int, args: argparse.Namespace) -> list[PhaseSpec]:
    """Build the ordered list of classifier training phases.

    The permanent freeze boundary is enforced here: CE never unfreezes into
    the first `frozen_core_backbone_modules` leaf modules. Optional legacy caps
    can reduce the trainable tail further, but cannot thaw the frozen core.
    """
    phases: list[PhaseSpec] = []
    if args.classifier_train_mode == "full_model":
        effective_max = effective_unfrozen_backbone_cap(total_modules, args, "ce")
        phases.append(PhaseSpec(name="ce_full_model", unfrozen_backbone_modules=effective_max))
        return phases
    # ── Stage 3: CE head warm-up (backbone fully frozen) ─────────────────────
    # embedding + embedding_norm + ce_head are trainable; all backbone frozen.
    # This prevents random ce_head gradients from corrupting contrastive features.
    phases.append(PhaseSpec(name="ce_head_only", unfrozen_backbone_modules=0))
    # ── Stage 4: CE progressive unfreezing ────────────────────────────────────
    # HARD BOUNDARY: never thaw the frozen core. Only the semantic tail beyond
    # frozen_core_backbone_modules is eligible for CE adaptation.
    effective_max = effective_unfrozen_backbone_cap(total_modules, args, "ce")
    count = 0
    phase_index = 0
    while count < effective_max:
        count = min(effective_max, count + args.unfreeze_chunk_size)
        phase_index += 1
        phases.append(PhaseSpec(name=f"ce_last_{count}_modules", unfrozen_backbone_modules=count))
    return phases


def effective_unfrozen_backbone_cap(total_modules: int, args: argparse.Namespace, stage: str) -> int:
    frozen_core = min(total_modules, max(0, int(getattr(args, "frozen_core_backbone_modules", 40))))
    default_tail = max(0, total_modules - frozen_core)
    if stage == "supcon":
        stage_cap = getattr(args, "supcon_unfreeze_backbone_modules", None)
        shared_cap = getattr(args, "ce_max_unfreeze_modules", None)
        caps = [default_tail]
        if stage_cap is not None:
            caps.append(int(stage_cap))
        if shared_cap is not None:
            caps.append(int(shared_cap))
        return max(0, min(total_modules, *caps))
    if stage == "ce":
        stage_cap = getattr(args, "ce_max_unfreeze_modules", None)
        caps = [default_tail]
        if stage_cap is not None:
            caps.append(int(stage_cap))
        return max(0, min(total_modules, *caps))
    raise ValueError(f"Unknown backbone unfreeze stage: {stage}")


def backbone_leaf_modules(model: MetricLearningEfficientNetB0) -> list[tuple[str, nn.Module]]:
    modules: list[tuple[str, nn.Module]] = []
    for name, module in model.backbone.named_modules():
        if not name:
            continue
        if list(module.children()):
            continue
        if any(True for _ in module.parameters(recurse=False)):
            modules.append((f"backbone.{name}", module))
    return modules


def set_trainability_for_supcon(
    model: MetricLearningEfficientNetB0,
    backbone_modules: list[tuple[str, nn.Module]],
    unfrozen_backbone_modules: int,
) -> list[str]:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.embedding.parameters():
        parameter.requires_grad = True
    for parameter in model.projection_head.parameters():
        parameter.requires_grad = True
    for parameter in model.embedding_norm.parameters():
        parameter.requires_grad = True
    for parameter in model.ce_head.parameters():
        parameter.requires_grad = False
    if not isinstance(model.front_end, nn.Identity):
        for param in model.front_end.parameters():
            param.requires_grad = True

    if unfrozen_backbone_modules <= 0:
        assert_backbone_trainability_boundary(backbone_modules, 0, context="supcon")
        return []
    thawed = backbone_modules[-unfrozen_backbone_modules:]
    for _, module in thawed:
        for parameter in module.parameters(recurse=True):
            parameter.requires_grad = True
    assert_backbone_trainability_boundary(backbone_modules, unfrozen_backbone_modules, context="supcon")
    return [name for name, _ in thawed]


def set_trainability_for_classifier(
    model: MetricLearningEfficientNetB0,
    backbone_modules: list[tuple[str, nn.Module]],
    unfrozen_backbone_modules: int,
) -> list[str]:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.embedding.parameters():
        parameter.requires_grad = True
    for parameter in model.embedding_norm.parameters():
        parameter.requires_grad = True
    for parameter in model.ce_head.parameters():
        parameter.requires_grad = True
    for parameter in model.projection_head.parameters():
        parameter.requires_grad = False
    if not isinstance(model.front_end, nn.Identity):
        for param in model.front_end.parameters():
            param.requires_grad = True

    if unfrozen_backbone_modules <= 0:
        assert_backbone_trainability_boundary(backbone_modules, 0, context="classifier")
        return []
    thawed = backbone_modules[-unfrozen_backbone_modules:]
    for _, module in thawed:
        for parameter in module.parameters(recurse=True):
            parameter.requires_grad = True
    assert_backbone_trainability_boundary(backbone_modules, unfrozen_backbone_modules, context="classifier")
    return [name for name, _ in thawed]


def assert_backbone_trainability_boundary(
    backbone_modules: list[tuple[str, nn.Module]],
    unfrozen_backbone_modules: int,
    *,
    context: str,
) -> None:
    effective_unfrozen = min(max(0, int(unfrozen_backbone_modules)), len(backbone_modules))
    trainable_start = len(backbone_modules) - effective_unfrozen
    for index, (name, module) in enumerate(backbone_modules):
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        is_tail_module = index >= trainable_start
        has_trainable_param = any(parameter.requires_grad for parameter in params)
        has_frozen_param = any(not parameter.requires_grad for parameter in params)
        if is_tail_module and has_frozen_param:
            raise RuntimeError(
                f"{context} freeze-boundary violation: expected tail module {name} to be trainable."
            )
        if not is_tail_module and has_trainable_param:
            raise RuntimeError(
                f"{context} freeze-boundary violation: expected core module {name} to stay frozen."
            )


def freeze_frozen_batchnorms(backbone_modules: list[tuple[str, nn.Module]]) -> None:
    for _, module in backbone_modules:
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        if any(parameter.requires_grad for parameter in params):
            continue
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def build_sam_optimizer(
    parameter_groups: list[dict[str, Any]],
    rho: float,
    weight_decay: float,
    adam_betas: tuple[float, float],
) -> SAM:
    return SAM(
        parameter_groups,
        AdamW,
        rho=rho,
        adaptive=False,
        weight_decay=weight_decay,
        betas=adam_betas,
    )


def build_base_optimizer(
    parameter_groups: list[dict[str, Any]],
    weight_decay: float,
    adam_betas: tuple[float, float],
) -> Optimizer:
    sanitized_groups = []
    for group in parameter_groups:
        sanitized = dict(group)
        sanitized.pop("rho", None)
        sanitized_groups.append(sanitized)
    return AdamW(sanitized_groups, weight_decay=weight_decay, betas=adam_betas)


def build_optimizer_from_groups(parameter_groups: list[dict[str, Any]], args: argparse.Namespace) -> Optimizer:
    if args.optimizer == "sam":
        return build_sam_optimizer(parameter_groups, args.sam_rho, args.weight_decay, (args.adam_beta1, args.adam_beta2))
    return build_base_optimizer(parameter_groups, args.weight_decay, (args.adam_beta1, args.adam_beta2))


def build_supcon_optimizer(
    model: MetricLearningEfficientNetB0,
    args: argparse.Namespace,
    *,
    head_lr: float | None = None,
    backbone_lr: float | None = None,
) -> Optimizer:
    head_params = []
    if not isinstance(model.front_end, nn.Identity):
        head_params.extend(parameter for parameter in model.front_end.parameters() if parameter.requires_grad)
    head_params.extend(parameter for parameter in model.embedding.parameters() if parameter.requires_grad)
    head_params.extend(parameter for parameter in model.embedding_norm.parameters() if parameter.requires_grad)
    head_params.extend(parameter for parameter in model.projection_head.parameters() if parameter.requires_grad)
    head_ids = {id(parameter) for parameter in head_params}
    backbone_params = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    param_groups = []
    head_group_lr = float(args.supcon_head_lr if head_lr is None else head_lr)
    backbone_group_lr = float(args.supcon_backbone_lr if backbone_lr is None else backbone_lr)
    if head_params:
        param_groups.append({"params": head_params, "lr": head_group_lr, "rho": args.sam_rho})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_group_lr, "rho": args.sam_rho})
    return build_optimizer_from_groups(param_groups, args)


def classifier_phase_learning_rates(
    args: argparse.Namespace,
    *,
    unfrozen_backbone_modules: int,
    total_backbone_modules: int,
) -> tuple[float, float]:
    backbone_lr = float(args.backbone_lr)
    head_lr = float(args.head_lr)
    if args.classifier_train_mode != "progressive" or total_backbone_modules <= 0 or unfrozen_backbone_modules <= 0:
        return head_lr, backbone_lr

    # Use the actual trainable tail size after the frozen-core boundary as the
    # denominator, not total_backbone_modules. That keeps the final progressive
    # phase aligned with the real unfreeze ceiling, so head LR decays fully.
    effective_total = effective_unfrozen_backbone_cap(total_backbone_modules, args, "ce")
    if effective_total <= 0:
        return head_lr, backbone_lr

    # As progressively more of the backbone is opened, apply rigorous exponential decay
    # to the head LR toward backbone_lr. This ensures the final phase does not over-update
    # an already-adapted classifier head, stabilising it before recursive refinement.
    progress = min(1.0, max(0.0, float(unfrozen_backbone_modules) / float(effective_total)))
    effective_head_lr = head_lr * ((backbone_lr / head_lr) ** progress)
    return max(backbone_lr, effective_head_lr), backbone_lr


def build_classifier_optimizer(
    model: MetricLearningEfficientNetB0,
    args: argparse.Namespace,
    *,
    head_lr: float | None = None,
    backbone_lr: float | None = None,
) -> Optimizer:
    head_group_lr = float(args.head_lr if head_lr is None else head_lr)
    backbone_group_lr = float(args.backbone_lr if backbone_lr is None else backbone_lr)
    head_params = []
    if not isinstance(model.front_end, nn.Identity):
        head_params.extend(parameter for parameter in model.front_end.parameters() if parameter.requires_grad)
    head_params.extend(parameter for parameter in model.embedding.parameters() if parameter.requires_grad)
    head_params.extend(parameter for parameter in model.embedding_norm.parameters() if parameter.requires_grad)
    head_params.extend(parameter for parameter in model.ce_head.parameters() if parameter.requires_grad)
    head_ids = {id(parameter) for parameter in head_params}
    backbone_params = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": head_group_lr, "rho": args.sam_rho})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_group_lr, "rho": args.sam_rho})
    return build_optimizer_from_groups(param_groups, args)


def base_optimizer_for_scheduler(optimizer: Optimizer) -> Optimizer:
    return optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer


def optimizer_learning_rates(optimizer: Optimizer) -> list[float]:
    return [group["lr"] for group in base_optimizer_for_scheduler(optimizer).param_groups]


def model_dtype_for_args(args: argparse.Namespace) -> torch.dtype:
    return torch.float64 if str(args.precision) == "64" else torch.float32


def json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return json_safe_value(vars(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else json_safe_value(tensor.tolist())
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return json_safe_value(vars(value))
    return str(value)


def autocast_enabled(device: torch.device, args: argparse.Namespace) -> bool:
    return device.type == "cuda" and str(args.precision) == "mixed"


def build_grad_scaler(device: torch.device, args: argparse.Namespace) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=autocast_enabled(device, args))


def move_images_to_device(images: torch.Tensor, device: torch.device, args: argparse.Namespace) -> torch.Tensor:
    return images.to(device=device, dtype=model_dtype_for_args(args), non_blocking=True)


class WarmupCosineScheduler:
    def __init__(self, optimizer: Optimizer, steps_per_epoch: int, warmup_epochs: int, warmup_steps: int = 0) -> None:
        self.optimizer = optimizer
        self.steps_per_epoch = max(1, steps_per_epoch)
        requested_warmup_steps = max(0, int(warmup_steps)) if warmup_steps > 0 else max(0, warmup_epochs * self.steps_per_epoch)
        self.warmup_steps = requested_warmup_steps
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_index = 0

    def _factor(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        return 1.0

    def step(self) -> None:
        factor = self._factor(self.step_index)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor
        self.step_index += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "steps_per_epoch": self.steps_per_epoch,
            "warmup_steps": self.warmup_steps,
            "base_lrs": self.base_lrs,
            "step_index": self.step_index,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.steps_per_epoch = int(state_dict.get("steps_per_epoch", self.steps_per_epoch))
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.base_lrs = [float(value) for value in state_dict["base_lrs"]]
        self.step_index = int(state_dict["step_index"])


def set_scheduler_base_lrs(scheduler: WarmupCosineScheduler, base_lrs: list[float]) -> None:
    scheduler.base_lrs = [float(value) for value in base_lrs]
    if scheduler.step_index <= 0:
        current_factor = 1.0
    else:
        current_factor = scheduler._factor(scheduler.step_index - 1)
    for group, base_lr in zip(scheduler.optimizer.param_groups, scheduler.base_lrs):
        group["lr"] = base_lr * current_factor


def build_scheduler(
    optimizer: Optimizer,
    steps_per_epoch: int,
    warmup_epochs: int,
    warmup_steps: int = 0,
) -> WarmupCosineScheduler:
    return WarmupCosineScheduler(
        optimizer,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=warmup_epochs,
        warmup_steps=warmup_steps,
    )


def steps_per_epoch_for_dataset(dataset: Dataset, batch_size: int, max_batches: int) -> int:
    steps = math.ceil(len(dataset) / max(1, batch_size))
    return min(steps, max_batches) if max_batches > 0 else steps


def steps_per_epoch_for_sampler(sampler: Sampler[int] | None, dataset: Dataset, batch_size: int, max_batches: int) -> int:
    sample_count = len(sampler) if sampler is not None else len(dataset)
    steps = math.ceil(sample_count / max(1, batch_size))
    return min(steps, max_batches) if max_batches > 0 else steps


def infer_phase_train_loss_baseline(
    history: list[dict[str, Any]],
    stage: str,
    phase_name: str | None,
) -> float:
    for row in reversed(history):
        if row.get("stage") != stage:
            continue
        if phase_name is not None and row.get("phase_name") != phase_name:
            continue
        for key in ("window_train_loss", "epoch_running_train_loss", "running_loss"):
            value = row.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return float("inf")


def limited_batches(loader: DataLoader, max_batches: int):
    if max_batches <= 0:
        yield from loader
        return
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        yield batch


def improved_classifier_selection_metric(
    selection_metric: str,
    best_loss: float,
    best_raw_acc: float,
    val_loss: float,
    val_raw_acc: float,
    min_delta: float,
) -> bool:
    if selection_metric == "val_raw_acc":
        return val_raw_acc > best_raw_acc + min_delta
    return val_loss < best_loss - min_delta


def train_supcon_epoch(
    model: MetricLearningEfficientNetB0,
    loader: DataLoader,
    criterion: SupConLoss,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    backbone_modules: list[tuple[str, nn.Module]],
    max_batches: int,
    epoch: int,
    log_path: Path,
    log_every_steps: int,
    train_progress: dict[str, int],
    args: argparse.Namespace,
    phase_index: int,
    phase_name: str,
) -> float:
    model.train()
    freeze_frozen_batchnorms(backbone_modules)
    total_loss = 0.0
    total_seen = 0
    total_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    progress = tqdm(enumerate(limited_batches(loader, max_batches), start=1), total=total_batches, leave=False)

    for step_in_epoch, (view_one, view_two, labels) in progress:
        view_one = move_images_to_device(view_one, device, args)
        view_two = move_images_to_device(view_two, device, args)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
            emb_one = model.encode(view_one)
            emb_two = model.encode(view_two)
            proj_one = model.supcon_projection(emb_one)
            proj_two = model.supcon_projection(emb_two)
            stacked = torch.stack([proj_one, proj_two], dim=1)
            loss = criterion(stacked, labels)
        if isinstance(optimizer, SAM):
            scaler.scale(loss).backward()
            scaler.unscale_(base_optimizer_for_scheduler(optimizer))
            optimizer.first_step(zero_grad=True)

            with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
                emb_one = model.encode(view_one)
                emb_two = model.encode(view_two)
                proj_one = model.supcon_projection(emb_one)
                proj_two = model.supcon_projection(emb_two)
                stacked = torch.stack([proj_one, proj_two], dim=1)
                second_loss = criterion(stacked, labels)
            scaler.scale(second_loss).backward()
            scaler.unscale_(base_optimizer_for_scheduler(optimizer))
            optimizer.second_step(zero_grad=True)
            scaler.update()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            second_loss = loss
        scheduler.step()

        batch_size = labels.size(0)
        batch_contrastive_metrics = supcon_contrastive_metrics(proj_one, proj_two, labels)
        total_loss += second_loss.item() * batch_size
        total_seen += batch_size
        train_progress["global_train_step"] += 1
        train_progress["global_source_samples_seen"] += batch_size
        progress.set_postfix(
            loss=f"{second_loss.item():.4f}",
            margin=f"{batch_contrastive_metrics['positive_negative_cosine_margin']:.4f}",
        )

        if train_progress["global_train_step"] % log_every_steps == 0:
            event = {
                "event": "train_step",
                "stage": "supcon",
                "phase_index": phase_index,
                "phase_name": phase_name,
                "epoch": epoch,
                "epoch_step": step_in_epoch,
                "global_train_step": train_progress["global_train_step"],
                "batch_size": batch_size,
                "epoch_source_samples_seen": total_seen,
                "global_source_samples_seen": train_progress["global_source_samples_seen"],
                "loss": float(second_loss.item()),
                **batch_contrastive_metrics,
                "learning_rates": optimizer_learning_rates(optimizer),
            }
            log_json_event(log_path, event)

    return total_loss / max(1, total_seen)


def train_supcon_steps(
    model: MetricLearningEfficientNetB0,
    batch_iterator,
    step_limit: int,
    criterion: SupConLoss,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    backbone_modules: list[tuple[str, nn.Module]],
    epoch: int,
    epoch_step_offset: int,
    log_path: Path,
    log_every_steps: int,
    train_progress: dict[str, int],
    step_checkpoint_path: Path,
    class_names: list[str],
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
    phase_index: int,
    phase_name: str,
    step_checkpoint_payload: dict[str, Any] | None = None,
    step_resume_payload: dict[str, Any] | None = None,
) -> tuple[float, int, int]:
    model.train()
    freeze_frozen_batchnorms(backbone_modules)
    total_loss = 0.0
    total_seen = 0
    steps_done = 0
    if step_checkpoint_payload is None:
        step_checkpoint_payload = {}
    if step_resume_payload is None:
        step_resume_payload = {}

    for local_step in range(step_limit):
        try:
            view_one, view_two, labels = next(batch_iterator)
        except StopIteration:
            break

        step_in_epoch = epoch_step_offset + local_step + 1
        view_one = move_images_to_device(view_one, device, args)
        view_two = move_images_to_device(view_two, device, args)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
            emb_one = model.encode(view_one)
            emb_two = model.encode(view_two)
            proj_one = model.supcon_projection(emb_one)
            proj_two = model.supcon_projection(emb_two)
            stacked = torch.stack([proj_one, proj_two], dim=1)
            loss = criterion(stacked, labels)
        if isinstance(optimizer, SAM):
            scaler.scale(loss).backward()
            scaler.unscale_(base_optimizer_for_scheduler(optimizer))
            optimizer.first_step(zero_grad=True)

            with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
                emb_one = model.encode(view_one)
                emb_two = model.encode(view_two)
                proj_one = model.supcon_projection(emb_one)
                proj_two = model.supcon_projection(emb_two)
                stacked = torch.stack([proj_one, proj_two], dim=1)
                second_loss = criterion(stacked, labels)
            scaler.scale(second_loss).backward()
            scaler.unscale_(base_optimizer_for_scheduler(optimizer))
            optimizer.second_step(zero_grad=True)
            scaler.update()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            second_loss = loss
        scheduler.step()

        step_train_loss = float(second_loss.item())
        best_train_loss = float(step_checkpoint_payload.get("phase_train_loss_best", float("inf")))
        train_loss_wait = int(step_checkpoint_payload.get("phase_train_loss_wait", 0))
        validation_wait = int(step_checkpoint_payload.get("phase_validation_wait", 0))
        if step_train_loss < best_train_loss - args.early_stopping_min_delta:
            best_train_loss = step_train_loss
            train_loss_wait = 0
        else:
            train_loss_wait += 1
        validation_wait += 1
        validation_due = False
        validation_reason: str | None = None
        if train_loss_wait >= args.train_loss_validation_patience:
            validation_due = True
            validation_reason = "train_loss_plateau"
            train_loss_wait = 0
        elif validation_wait >= args.validation_patience:
            validation_due = True
            validation_reason = "validation_stale"
        step_checkpoint_payload["phase_train_loss_best"] = best_train_loss
        step_checkpoint_payload["phase_train_loss_wait"] = train_loss_wait
        step_checkpoint_payload["phase_validation_wait"] = validation_wait
        step_checkpoint_payload["validation_due"] = validation_due
        step_checkpoint_payload["validation_reason"] = validation_reason
        step_resume_payload["phase_train_loss_best"] = best_train_loss
        step_resume_payload["phase_train_loss_wait"] = train_loss_wait
        step_resume_payload["phase_validation_wait"] = validation_wait
        step_resume_payload["validation_due"] = validation_due
        step_resume_payload["validation_reason"] = validation_reason

        batch_size = labels.size(0)
        batch_contrastive_metrics = supcon_contrastive_metrics(proj_one, proj_two, labels)
        step_checkpoint_payload["last_batch_loss"] = float(second_loss.item())
        step_checkpoint_payload["last_batch_contrastive_metrics"] = batch_contrastive_metrics
        step_resume_payload["last_batch_loss"] = float(second_loss.item())
        step_resume_payload["last_batch_contrastive_metrics"] = batch_contrastive_metrics
        total_loss += second_loss.item() * batch_size
        total_seen += batch_size
        steps_done += 1
        train_progress["global_train_step"] += 1
        train_progress["global_source_samples_seen"] += batch_size
        save_step_checkpoint(
            path=step_checkpoint_path,
            model=model,
            class_names=class_names,
            class_to_idx=class_to_idx,
            args=args,
            train_progress=train_progress,
            stage="supcon",
            epoch=epoch,
            epoch_step=step_in_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            phase_index=phase_index,
            phase_name=phase_name,
            extra_payload=step_checkpoint_payload,
            extra_resume=step_resume_payload,
        )

        if train_progress["global_train_step"] % log_every_steps == 0:
            event = {
                "event": "train_step",
                "stage": "supcon",
                "phase_index": phase_index,
                "phase_name": phase_name,
                "epoch": epoch,
                "epoch_step": step_in_epoch,
                "global_train_step": train_progress["global_train_step"],
                "batch_size": batch_size,
                "epoch_source_samples_seen": total_seen,
                "global_source_samples_seen": train_progress["global_source_samples_seen"],
                "loss": float(second_loss.item()),
                **batch_contrastive_metrics,
                "learning_rates": optimizer_learning_rates(optimizer),
                "phase_train_loss_best": best_train_loss,
                "phase_train_loss_wait": train_loss_wait,
                "phase_validation_wait": validation_wait,
                "validation_due": validation_due,
                "validation_reason": validation_reason,
            }
            log_json_event(log_path, event)

        if validation_due:
            break

    return total_loss / max(1, total_seen), steps_done, total_seen


def evaluate_supcon(
    model: MetricLearningEfficientNetB0,
    loader: DataLoader,
    criterion: SupConLoss,
    device: torch.device,
    max_batches: int,
    log_path: Path,
    log_every_eval_steps: int,
    stage: str,
    split: str,
    eval_context: dict[str, object] | None = None,
    args: argparse.Namespace | None = None,
) -> tuple[float, dict[str, dict[str, float | None] | float]]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    metric_sums = {
        "same_image_view_cosine": 0.0,
        "same_class_positive_cosine": 0.0,
        "different_class_negative_cosine": 0.0,
        "positive_negative_cosine_margin": 0.0,
    }
    total_positive_pairs = 0
    total_negative_pairs = 0
    eval_context = dict(eval_context or {})
    
    # Calculate total for tqdm
    try:
        total = max_batches if max_batches > 0 else len(loader)
    except (TypeError, AttributeError):
        total = None

    with torch.no_grad():
        for eval_step, (view_one, view_two, labels) in enumerate(tqdm(limited_batches(loader, max_batches), total=total, desc=f"Evaluating {split}", leave=False), start=1):
            view_one = move_images_to_device(view_one, device, args) if args is not None else view_one.to(device, non_blocking=True)
            view_two = move_images_to_device(view_two, device, args) if args is not None else view_two.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=args is not None and autocast_enabled(device, args)):
                emb_one = model.encode(view_one)
                emb_two = model.encode(view_two)
                proj_one = model.supcon_projection(emb_one)
                proj_two = model.supcon_projection(emb_two)
                loss = criterion(torch.stack([proj_one, proj_two], dim=1), labels)
            batch_contrastive_metrics = supcon_contrastive_metrics(proj_one, proj_two, labels)
            total_loss += loss.item() * labels.size(0)
            total_seen += labels.size(0)
            for key in metric_sums:
                metric_sums[key] += float(batch_contrastive_metrics[key]) * labels.size(0)
            total_positive_pairs += int(batch_contrastive_metrics["positive_pair_count"])
            total_negative_pairs += int(batch_contrastive_metrics["negative_pair_count"])
            if eval_step % log_every_eval_steps == 0:
                log_json_event(
                    log_path,
                    {
                        "event": "eval_step",
                        "stage": stage,
                        "split": split,
                        "eval_step": eval_step,
                        "batch_size": labels.size(0),
                        "samples_seen": total_seen,
                        "loss": float(loss.item()),
                        **batch_contrastive_metrics,
                        **eval_context,
                    },
                )
    contrastive_metrics = {key: value / max(1, total_seen) for key, value in metric_sums.items()}
    contrastive_metrics["positive_pair_count"] = total_positive_pairs
    contrastive_metrics["negative_pair_count"] = total_negative_pairs
    return total_loss / max(1, total_seen), {
        "contrastive_metrics": contrastive_metrics,
    }


def classifier_loss(
    model: MetricLearningEfficientNetB0,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = model.classify(embeddings, labels)
    plain_logits = model.classify(embeddings, labels=None)
    return classifier_loss_from_logits(logits, plain_logits, labels, args)


def classifier_loss_from_logits(
    logits: torch.Tensor,
    plain_logits: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    per_sample_loss = F.cross_entropy(
        logits,
        labels,
        reduction="none",
        label_smoothing=args.label_smoothing,
    )
    class_loss_weight_map = getattr(args, "class_loss_weight_map_resolved", {}) or {}
    if class_loss_weight_map:
        sample_weights = torch.ones_like(per_sample_loss)
        for class_index, weight in class_loss_weight_map.items():
            sample_weights = torch.where(labels == int(class_index), sample_weights.new_tensor(float(weight)), sample_weights)
        per_sample_loss = per_sample_loss * sample_weights
    base_loss = per_sample_loss.mean()
    probabilities = torch.softmax(plain_logits, dim=1)
    true_class_confidence = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    predictions = plain_logits.argmax(dim=1)
    correct_mask = predictions == labels
    if correct_mask.any():
        confidence_gap = (args.confidence_threshold - true_class_confidence).clamp(min=0.0)
        confidence_penalty = confidence_gap[correct_mask].square().mean()
    else:
        confidence_penalty = logits.new_zeros(())
    targeted_confusion_penalties = getattr(args, "targeted_confusion_penalties_resolved", []) or []
    targeted_penalty = logits.new_zeros(())
    if targeted_confusion_penalties:
        for penalty_spec in targeted_confusion_penalties:
            true_index = int(penalty_spec["true_index"])
            predicted_index = int(penalty_spec["predicted_index"])
            weight = float(penalty_spec["weight"])
            true_mask = labels == true_index
            if true_mask.any():
                targeted_penalty = targeted_penalty + probabilities[true_mask, predicted_index].mean() * weight
    total_loss = base_loss + (args.confidence_gap_penalty_weight * confidence_penalty) + targeted_penalty
    return total_loss, base_loss, confidence_penalty, targeted_penalty


def per_class_accuracy_and_confidence_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: list[str],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    if logits.numel() == 0 or targets.numel() == 0 or not class_names:
        return {}, {}
    probabilities = torch.softmax(logits.detach(), dim=1)
    confidences = probabilities.max(dim=1).values.detach().cpu()
    predictions = logits.detach().argmax(dim=1).cpu()
    targets_cpu = targets.detach().cpu()
    per_class_accuracy: dict[str, float | None] = {}
    per_class_confidence: dict[str, float | None] = {}
    for class_index, class_name in enumerate(class_names):
        class_mask = targets_cpu == class_index
        if class_mask.any():
            per_class_accuracy[class_name] = float((predictions[class_mask] == targets_cpu[class_mask]).float().mean().item())
            per_class_confidence[class_name] = float(confidences[class_mask].mean().item())
        else:
            per_class_accuracy[class_name] = None
            per_class_confidence[class_name] = None
    return per_class_accuracy, per_class_confidence


def format_per_class_confidence(per_class_confidence: dict[str, float | None] | None) -> str | None:
    if not per_class_confidence:
        return None
    parts = []
    for class_name, value in per_class_confidence.items():
        if isinstance(value, (int, float)):
            parts.append(f"{class_name}:{float(value):.4f}")
    return "{ " + ", ".join(parts) + " }" if parts else None


def train_classifier_epoch(
    model: MetricLearningEfficientNetB0,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    backbone_modules: list[tuple[str, nn.Module]],
    max_batches: int,
    epoch: int,
    phase_index: int,
    phase_name: str,
    log_path: Path,
    log_every_steps: int,
    train_progress: dict[str, int],
    args: argparse.Namespace,
) -> tuple[float, float]:
    model.train()
    freeze_frozen_batchnorms(backbone_modules)
    total_loss = 0.0
    total_correct = 0.0
    total_base_loss = 0.0
    total_confidence_gap = 0.0
    total_targeted_penalty = 0.0
    total_seen = 0
    total_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    progress = tqdm(enumerate(limited_batches(loader, max_batches), start=1), total=total_batches, leave=False)

    for step_in_epoch, (images, labels) in progress:
        images = move_images_to_device(images, device, args)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
            embeddings = model.encode(images)
            loss, base_loss, confidence_penalty, targeted_penalty = classifier_loss(model, embeddings, labels, criterion, args)
        if isinstance(optimizer, SAM):
            scaler.scale(loss).backward()
            scaler.unscale_(base_optimizer_for_scheduler(optimizer))
            optimizer.first_step(zero_grad=True)

            with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
                embeddings = model.encode(images)
                second_loss, second_base_loss, second_confidence_penalty, second_targeted_penalty = classifier_loss(
                    model, embeddings, labels, criterion, args
                )
            scaler.scale(second_loss).backward()
            scaler.unscale_(base_optimizer_for_scheduler(optimizer))
            optimizer.second_step(zero_grad=True)
            scaler.update()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            second_loss = loss
            second_base_loss = base_loss
            second_confidence_penalty = confidence_penalty
            second_targeted_penalty = targeted_penalty
        scheduler.step()

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
                eval_logits = model.classify(model.encode(images), labels=None)
            per_class_accuracy, per_class_avg_confidence = per_class_accuracy_and_confidence_from_logits(
                eval_logits,
                labels,
                class_names,
            )

        batch_size = labels.size(0)
        batch_predictions = eval_logits.detach().argmax(dim=1)
        batch_acc = float((batch_predictions == labels).float().mean().item())
        step_checkpoint_payload["last_batch_loss"] = float(second_loss.item())
        step_checkpoint_payload["last_batch_acc"] = batch_acc
        step_checkpoint_payload["last_batch_per_class_accuracy"] = per_class_accuracy
        step_checkpoint_payload["last_batch_per_class_avg_confidence"] = per_class_avg_confidence
        step_resume_payload["last_batch_loss"] = float(second_loss.item())
        step_resume_payload["last_batch_acc"] = batch_acc
        step_resume_payload["last_batch_per_class_accuracy"] = per_class_accuracy
        step_resume_payload["last_batch_per_class_avg_confidence"] = per_class_avg_confidence
        total_loss += second_loss.item() * batch_size
        total_base_loss += second_base_loss.item() * batch_size
        total_confidence_gap += second_confidence_penalty.item() * batch_size
        total_targeted_penalty += second_targeted_penalty.item() * batch_size
        total_correct += batch_acc * batch_size
        total_seen += batch_size
        train_progress["global_train_step"] += 1
        train_progress["global_source_samples_seen"] += batch_size
        progress.set_postfix(
            loss=f"{second_loss.item():.4f}",
            acc=f"{batch_acc:.4f}",
        )

        if train_progress["global_train_step"] % log_every_steps == 0:
            event = {
                "event": "train_step",
                "stage": "classifier",
                "phase_index": phase_index,
                "phase_name": phase_name,
                "epoch": epoch,
                "epoch_step": step_in_epoch,
                "global_train_step": train_progress["global_train_step"],
                "batch_size": batch_size,
                "epoch_source_samples_seen": total_seen,
                "global_source_samples_seen": train_progress["global_source_samples_seen"],
                "loss": float(second_loss.item()),
                "base_loss": float(second_base_loss.item()),
                "confidence_gap_penalty": float(second_confidence_penalty.item()),
                "targeted_confusion_penalty": float(second_targeted_penalty.item()),
                "acc": batch_acc,
                "per_class_accuracy": per_class_accuracy,
                "per_class_avg_confidence": per_class_avg_confidence,
                "learning_rates": optimizer_learning_rates(optimizer),
            }
            log_json_event(log_path, event)

    return total_loss / max(1, total_seen), total_correct / max(1, total_seen)


def train_classifier_steps(
    model: MetricLearningEfficientNetB0,
    batch_iterator,
    step_limit: int,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    backbone_modules: list[tuple[str, nn.Module]],
    epoch: int,
    epoch_step_offset: int,
    phase_index: int,
    phase_name: str,
    log_path: Path,
    log_every_steps: int,
    train_progress: dict[str, int],
    step_checkpoint_path: Path,
    class_names: list[str],
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
    step_checkpoint_payload: dict[str, Any] | None = None,
    step_resume_payload: dict[str, Any] | None = None,
) -> tuple[float, float, int, int]:
    model.train()
    freeze_frozen_batchnorms(backbone_modules)
    total_loss = 0.0
    total_correct = 0.0
    total_base_loss = 0.0
    total_confidence_gap = 0.0
    total_targeted_penalty = 0.0
    total_seen = 0
    steps_done = 0
    if step_checkpoint_payload is None:
        step_checkpoint_payload = {}
    if step_resume_payload is None:
        step_resume_payload = {}

    for local_step in range(step_limit):
        try:
            images, labels = next(batch_iterator)
        except StopIteration:
            break

        step_in_epoch = epoch_step_offset + local_step + 1
        images = move_images_to_device(images, device, args)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
            embeddings = model.encode(images)
            loss, base_loss, confidence_penalty, targeted_penalty = classifier_loss(model, embeddings, labels, criterion, args)
        if isinstance(optimizer, SAM):
            scaler.scale(loss).backward()
            scaler.unscale_(base_optimizer_for_scheduler(optimizer))
            optimizer.first_step(zero_grad=True)

            with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
                embeddings = model.encode(images)
                second_loss, second_base_loss, second_confidence_penalty, second_targeted_penalty = classifier_loss(
                    model, embeddings, labels, criterion, args
                )
            scaler.scale(second_loss).backward()
            scaler.unscale_(base_optimizer_for_scheduler(optimizer))
            optimizer.second_step(zero_grad=True)
            scaler.update()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            second_loss = loss
            second_base_loss = base_loss
            second_confidence_penalty = confidence_penalty
            second_targeted_penalty = targeted_penalty
        scheduler.step()

        step_train_loss = float(second_loss.item())
        best_train_loss = float(step_checkpoint_payload.get("phase_train_loss_best", float("inf")))
        train_loss_wait = int(step_checkpoint_payload.get("phase_train_loss_wait", 0))
        validation_wait = int(step_checkpoint_payload.get("phase_validation_wait", 0))
        if step_train_loss < best_train_loss - args.early_stopping_min_delta:
            best_train_loss = step_train_loss
            train_loss_wait = 0
        else:
            train_loss_wait += 1
        validation_wait += 1
        validation_due = False
        validation_reason: str | None = None
        if train_loss_wait >= args.train_loss_validation_patience:
            validation_due = True
            validation_reason = "train_loss_plateau"
            train_loss_wait = 0
        elif validation_wait >= args.validation_patience:
            validation_due = True
            validation_reason = "validation_stale"
        step_checkpoint_payload["phase_train_loss_best"] = best_train_loss
        step_checkpoint_payload["phase_train_loss_wait"] = train_loss_wait
        step_checkpoint_payload["phase_validation_wait"] = validation_wait
        step_checkpoint_payload["validation_due"] = validation_due
        step_checkpoint_payload["validation_reason"] = validation_reason
        step_resume_payload["phase_train_loss_best"] = best_train_loss
        step_resume_payload["phase_train_loss_wait"] = train_loss_wait
        step_resume_payload["phase_validation_wait"] = validation_wait
        step_resume_payload["validation_due"] = validation_due
        step_resume_payload["validation_reason"] = validation_reason

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
                eval_logits = model.classify(model.encode(images), labels=None)
            per_class_accuracy, per_class_avg_confidence = per_class_accuracy_and_confidence_from_logits(
                eval_logits,
                labels,
                class_names,
            )

        batch_size = labels.size(0)
        batch_predictions = eval_logits.detach().argmax(dim=1)
        batch_acc = float((batch_predictions == labels).float().mean().item())
        total_loss += second_loss.item() * batch_size
        total_base_loss += second_base_loss.item() * batch_size
        total_confidence_gap += second_confidence_penalty.item() * batch_size
        total_targeted_penalty += second_targeted_penalty.item() * batch_size
        total_correct += batch_acc * batch_size
        total_seen += batch_size
        steps_done += 1
        train_progress["global_train_step"] += 1
        train_progress["global_source_samples_seen"] += batch_size
        save_step_checkpoint(
            path=step_checkpoint_path,
            model=model,
            class_names=class_names,
            class_to_idx=class_to_idx,
            args=args,
            train_progress=train_progress,
            stage="classifier",
            epoch=epoch,
            epoch_step=step_in_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            phase_index=phase_index,
            phase_name=phase_name,
            extra_payload=step_checkpoint_payload,
            extra_resume=step_resume_payload,
        )

        if train_progress["global_train_step"] % log_every_steps == 0:
            event = {
                "event": "train_step",
                "stage": "classifier",
                "phase_index": phase_index,
                "phase_name": phase_name,
                "epoch": epoch,
                "epoch_step": step_in_epoch,
                "global_train_step": train_progress["global_train_step"],
                "batch_size": batch_size,
                "epoch_source_samples_seen": total_seen,
                "global_source_samples_seen": train_progress["global_source_samples_seen"],
                "loss": float(second_loss.item()),
                "base_loss": float(second_base_loss.item()),
                "confidence_gap_penalty": float(second_confidence_penalty.item()),
                "targeted_confusion_penalty": float(second_targeted_penalty.item()),
                "acc": batch_acc,
                "per_class_accuracy": per_class_accuracy,
                "per_class_avg_confidence": per_class_avg_confidence,
                "learning_rates": optimizer_learning_rates(optimizer),
                "phase_train_loss_best": best_train_loss,
                "phase_train_loss_wait": train_loss_wait,
                "phase_validation_wait": validation_wait,
                "validation_due": validation_due,
                "validation_reason": validation_reason,
            }
            log_json_event(log_path, event)

        if validation_due:
            break

    return (
        total_loss / max(1, total_seen),
        total_correct / max(1, total_seen),
        steps_done,
        total_seen,
    )


def evaluate_classifier(
    model: MetricLearningEfficientNetB0,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
    log_path: Path,
    log_every_eval_steps: int,
    stage: str,
    split: str,
    eval_context: dict[str, object] | None = None,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    total_correct = 0.0
    all_logits = []
    all_targets = []
    eval_context = dict(eval_context or {})
    total_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    with torch.no_grad():
        progress = tqdm(
            enumerate(limited_batches(loader, max_batches), start=1),
            total=total_batches,
            desc=f"eval/{split}",
            leave=False,
        )
        for eval_step, (images, labels) in progress:
            images = move_images_to_device(images, device, args) if args is not None else images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=args is not None and autocast_enabled(device, args)):
                embeddings = model.encode(images)
                margin_logits = model.classify(embeddings, labels)
                logits = model.classify(embeddings, labels=None)
                loss, _, _, _ = classifier_loss_from_logits(margin_logits, logits, labels, args)
            
            batch_predictions = logits.detach().argmax(dim=1)
            batch_acc = float((batch_predictions == labels).float().mean().item())
            batch_per_class_accuracy, batch_per_class_avg_confidence = per_class_accuracy_and_confidence_from_logits(
                logits,
                labels,
                getattr(loader.dataset, "classes", []),
            )
            
            total_loss += loss.item() * labels.size(0)
            total_seen += labels.size(0)
            total_correct += batch_acc * labels.size(0)
            
            all_logits.append(logits.detach().cpu().float().numpy())
            all_targets.append(labels.detach().cpu().numpy())
            
            progress.set_postfix(step=eval_step, loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")
            if eval_step % log_every_eval_steps == 0:
                log_json_event(
                    log_path,
                    {
                        "event": "eval_step",
                        "stage": stage,
                        "split": split,
                        "eval_step": eval_step,
                        "batch_size": labels.size(0),
                        "samples_seen": total_seen,
                        "loss": float(loss.item()),
                        "acc": batch_acc,
                        "per_class_accuracy": batch_per_class_accuracy,
                        "per_class_avg_confidence": batch_per_class_avg_confidence,
                        **eval_context,
                    },
                )
    
    logits_concat = np.concatenate(all_logits, axis=0)
    targets_concat = np.concatenate(all_targets, axis=0)
    metrics = compute_classification_metrics(logits_concat, targets_concat, getattr(loader.dataset, "classes", []), args.confidence_threshold)
    metrics["loss"] = total_loss / max(1, total_seen)
    return metrics


def collect_logits_and_labels(
    model: MetricLearningEfficientNetB0,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    log_path: Path,
    log_every_eval_steps: int,
    criterion: nn.Module | None,
    split: str,
    args: argparse.Namespace | None = None,
    stage: str = "checkpoint_evaluation",
    phase_name: str | None = None,
    eval_context: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    eval_args = args or argparse.Namespace(
        precision="mixed",
        label_smoothing=0.0,
        confidence_threshold=0.0,
        confidence_gap_penalty_weight=0.0,
        class_loss_weight_map_resolved={},
        targeted_confusion_penalties_resolved=[],
    )
    eval_context = dict(eval_context or {})
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    total_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    with torch.no_grad():
        total_seen = 0
        progress = tqdm(
            enumerate(limited_batches(loader, max_batches), start=1),
            total=total_batches,
            desc=f"eval/{split}",
            leave=False,
        )
        for eval_step, (images, labels) in progress:
            images = move_images_to_device(images, device, eval_args)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=autocast_enabled(device, eval_args)):
                embeddings = model.encode(images)
                if criterion is not None:
                    margin_logits = model.classify(embeddings, labels)
                    logits = model.classify(embeddings, labels=None)
                    loss, _, _, _ = classifier_loss_from_logits(margin_logits, logits, labels, eval_args)
                else:
                    logits = model.classify(embeddings, labels=None)
                    loss = None
            
            batch_predictions = logits.detach().argmax(dim=1)
            batch_acc = float((batch_predictions == labels).float().mean().item())
            batch_per_class_accuracy, batch_per_class_avg_confidence = per_class_accuracy_and_confidence_from_logits(
                logits,
                labels,
                getattr(loader.dataset, "classes", []),
            )
            
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())
            
            total_seen += labels.size(0)
            
            postfix = {"step": eval_step, "acc": f"{batch_acc:.4f}"}
            if loss is not None:
                postfix["loss"] = f"{loss.item():.4f}"
            progress.set_postfix(**postfix)
            
            if eval_step % log_every_eval_steps == 0:
                event = {
                    "event": "eval_step",
                    "stage": stage,
                    "split": split,
                    "eval_step": eval_step,
                    "batch_size": labels.size(0),
                    "samples_seen": total_seen,
                    "acc": batch_acc,
                    "per_class_accuracy": batch_per_class_accuracy,
                    "per_class_avg_confidence": batch_per_class_avg_confidence,
                    **eval_context,
                }
                if phase_name is not None:
                    event["phase_name"] = phase_name
                if loss is not None:
                    event["loss"] = float(loss.item())
                log_json_event(
                    log_path,
                    event,
                )
    logits = torch.cat(logits_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    return logits, labels


def confusion_matrix_from_predictions(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def binary_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = y_true.astype(np.int64)
    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[start:end] = average_rank
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = original_ranks[y_true == 1].sum()
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = y_true.astype(np.int64)
    positives = int(y_true.sum())
    if positives == 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    true_positives = np.cumsum(y_sorted)
    false_positives = np.cumsum(1 - y_sorted)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / positives
    ap = 0.0
    previous_recall = 0.0
    for idx, label in enumerate(y_sorted):
        if label == 1:
            ap += precision[idx] * (recall[idx] - previous_recall)
            previous_recall = recall[idx]
    return float(ap)


def top_k_accuracy(logits: np.ndarray, targets: np.ndarray, k: int) -> float:
    k = min(k, logits.shape[1])
    topk = np.argpartition(logits, -k, axis=1)[:, -k:]
    matches = np.any(topk == targets[:, None], axis=1)
    return float(matches.mean())


def macro_weighted(values: list[float | None], weights: list[float]) -> tuple[float | None, float | None]:
    valid_pairs = [(value, weight) for value, weight in zip(values, weights) if value is not None]
    if not valid_pairs:
        return None, None
    macro = float(sum(value for value, _ in valid_pairs) / len(valid_pairs))
    weight_sum = sum(weight for _, weight in valid_pairs)
    weighted = float(sum(value * weight for value, weight in valid_pairs) / max(weight_sum, 1e-12))
    return macro, weighted


def compute_classification_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    confidence_threshold: float,
) -> dict[str, Any]:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = logits.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    num_classes = len(class_names)
    support = np.bincount(targets, minlength=num_classes)
    cm = confusion_matrix_from_predictions(targets, predictions, num_classes)
    if logits.size > 0 and targets.size > 0:
        ce_loss = float(
            F.cross_entropy(
                torch.from_numpy(logits).float(),
                torch.from_numpy(targets).long(),
            ).item()
        )
    else:
        ce_loss = 0.0

    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    pr_aucs = []
    total = int(cm.sum())
    for class_index, class_name in enumerate(class_names):
        tp = float(cm[class_index, class_index])
        fp = float(cm[:, class_index].sum() - tp)
        fn = float(cm[class_index, :].sum() - tp)
        tn = float(total - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        one_vs_rest = (targets == class_index).astype(np.int64)
        roc_auc = binary_roc_auc_score(one_vs_rest, probabilities[:, class_index])
        pr_auc = binary_average_precision(one_vs_rest, probabilities[:, class_index])
        per_class[class_name] = {
            "support": int(support[class_index]),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "roc_auc_ovr": roc_auc,
            "pr_auc_ovr": pr_auc,
        }
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_scores.append(float(f1))
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)

    raw_accuracy = float((predictions == targets).mean())
    accuracy = raw_accuracy
    balanced_accuracy = float(sum(recalls) / num_classes)
    macro_roc_auc, weighted_roc_auc = macro_weighted(roc_aucs, support.tolist())
    macro_pr_auc, weighted_pr_auc = macro_weighted(pr_aucs, support.tolist())
    weighted_precision = float(np.average(precisions, weights=support)) if support.sum() > 0 else 0.0
    weighted_recall = float(np.average(recalls, weights=support)) if support.sum() > 0 else 0.0
    weighted_f1 = float(np.average(f1_scores, weights=support)) if support.sum() > 0 else 0.0
    top1 = accuracy
    top3 = top_k_accuracy(logits, targets, 3)
    top5 = top_k_accuracy(logits, targets, min(5, num_classes)) if num_classes > 0 else 0.0

    total_samples = float(cm.sum())
    t_sum = cm.sum(axis=1).astype(np.float64)
    p_sum = cm.sum(axis=0).astype(np.float64)
    correct = float(np.trace(cm))
    expected_accuracy = float((t_sum * p_sum).sum() / max(total_samples * total_samples, 1.0))
    cohen_kappa = (accuracy - expected_accuracy) / (1.0 - expected_accuracy) if expected_accuracy < 1.0 else 0.0
    mcc_num = correct * total_samples - float(np.dot(t_sum, p_sum))
    mcc_den = math.sqrt(
        max(total_samples * total_samples - float(np.dot(t_sum, t_sum)), 0.0)
        * max(total_samples * total_samples - float(np.dot(p_sum, p_sum)), 0.0)
    )
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0
    calibration = compute_calibration_metrics(probabilities, targets)
    per_class_accuracy: dict[str, float | None] = {}
    per_class_avg_confidence: dict[str, float | None] = {}
    for class_index, class_name in enumerate(class_names):
        class_mask = targets == class_index
        if class_mask.any():
            per_class_accuracy[class_name] = float((predictions[class_mask] == targets[class_mask]).mean())
            per_class_avg_confidence[class_name] = float(confidence[class_mask].mean())
        else:
            per_class_accuracy[class_name] = None
            per_class_avg_confidence[class_name] = None

    return {
        "num_classes": num_classes,
        "num_samples": int(total_samples),
        "raw_accuracy": raw_accuracy,
        "accuracy": accuracy,
        "loss": float(ce_loss),
        "cross_entropy_loss": float(ce_loss),
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "top5_accuracy": top5,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": float(sum(precisions) / num_classes),
        "macro_recall": float(sum(recalls) / num_classes),
        "macro_f1": float(sum(f1_scores) / num_classes),
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_roc_auc_ovr": macro_roc_auc,
        "weighted_roc_auc_ovr": weighted_roc_auc,
        "macro_pr_auc_ovr": macro_pr_auc,
        "weighted_pr_auc_ovr": weighted_pr_auc,
        "cohen_kappa": float(cohen_kappa),
        "mcc": float(mcc),
        "per_class_accuracy": per_class_accuracy,
        "per_class_avg_confidence": per_class_avg_confidence,
        "calibration": calibration,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def compute_calibration_metrics(probabilities: np.ndarray, targets: np.ndarray, num_bins: int = 15) -> dict[str, Any]:
    if probabilities.size == 0 or targets.size == 0:
        return {
            "num_bins": int(num_bins),
            "expected_calibration_error": 0.0,
            "maximum_calibration_error": 0.0,
            "brier_score": 0.0,
            "negative_log_likelihood": 0.0,
            "average_confidence": 0.0,
            "confidence_std": 0.0,
            "bins": [],
        }

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == targets).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.clip(np.digitize(confidences, bin_edges[1:-1], right=False), 0, num_bins - 1)

    bins: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    sample_count = float(len(confidences))
    for bin_index in range(num_bins):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        lower = float(bin_edges[bin_index])
        upper = float(bin_edges[bin_index + 1])
        midpoint = 0.5 * (lower + upper)
        if count == 0:
            bins.append(
                {
                    "bin_index": bin_index,
                    "lower_edge": lower,
                    "upper_edge": upper,
                    "midpoint": midpoint,
                    "count": 0,
                    "accuracy": None,
                    "confidence": None,
                    "gap": None,
                }
            )
            continue
        bin_accuracy = float(correctness[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        gap = abs(bin_accuracy - bin_confidence)
        ece += (count / sample_count) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "bin_index": bin_index,
                "lower_edge": lower,
                "upper_edge": upper,
                "midpoint": midpoint,
                "count": count,
                "accuracy": bin_accuracy,
                "confidence": bin_confidence,
                "gap": gap,
            }
        )

    target_one_hot = np.zeros_like(probabilities, dtype=np.float64)
    target_one_hot[np.arange(len(targets)), targets] = 1.0
    brier_score = float(np.mean(np.sum((probabilities - target_one_hot) ** 2, axis=1)))
    nll = float(-np.mean(np.log(np.take_along_axis(probabilities, targets[:, None], axis=1).squeeze(1) + 1e-12)))
    return {
        "num_bins": int(num_bins),
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "brier_score": brier_score,
        "negative_log_likelihood": nll,
        "average_confidence": float(confidences.mean()),
        "confidence_std": float(confidences.std()),
        "bins": bins,
    }


def save_reliability_diagram(path: Path, calibration: dict[str, Any], title: str) -> None:
    bins = calibration.get("bins", [])
    if not bins:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.text(0.5, 0.5, "No calibration data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    valid_bins = [bin_item for bin_item in bins if bin_item.get("count", 0) > 0]
    if not valid_bins:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.text(0.5, 0.5, "No calibration data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    centers = [float(item["midpoint"]) for item in bins]
    accuracies = [float(item["accuracy"]) if item["accuracy"] is not None else 0.0 for item in bins]
    confidences = [float(item["confidence"]) if item["confidence"] is not None else 0.0 for item in bins]
    counts = [int(item["count"]) for item in bins]
    max_count = max(counts) if counts else 1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.2, label="Perfect calibration")
    widths = 1.0 / max(1, len(bins))
    ax.bar(
        centers,
        accuracies,
        width=widths,
        color="#4C78A8",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
        label="Accuracy",
    )
    ax.plot(centers, confidences, marker="o", color="#F58518", linewidth=2.0, label="Confidence")
    for center, accuracy, confidence, count in zip(centers, accuracies, confidences, counts):
        if count <= 0:
            continue
        ax.text(
            center,
            max(accuracy, confidence) + 0.02,
            str(count),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_confidence_histogram(path: Path, probabilities: np.ndarray, title: str) -> None:
    if probabilities.size == 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No confidence data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    confidence = probabilities.max(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidence, bins=20, range=(0.0, 1.0), color="#54A24B", alpha=0.85, edgecolor="white")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Sample count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_csv(path: Path, confusion_matrix: np.ndarray, class_names: list[str], percent: bool = False) -> None:
    matrix = np.asarray(confusion_matrix, dtype=np.float64)
    if matrix.size == 0:
        path.write_text("", encoding="utf-8")
        return
    rows = [["true_class\\pred_class", *class_names]]
    for row_index, class_name in enumerate(class_names):
        row_total = float(matrix[row_index].sum())
        if percent:
            values = [f"{(matrix[row_index, col_index] / row_total * 100.0):.3f}" if row_total > 0 else "0.000" for col_index in range(len(class_names))]
        else:
            values = [str(int(matrix[row_index, col_index])) for col_index in range(len(class_names))]
        rows.append([class_name, *values])
    path.write_text("\n".join(",".join(row) for row in rows), encoding="utf-8")


def save_classification_report_csv(path: Path, metrics: dict[str, Any], class_names: list[str]) -> None:
    per_class = metrics.get("per_class", {}) or {}
    per_class_accuracy = metrics.get("per_class_accuracy", {}) or {}
    per_class_confidence = metrics.get("per_class_avg_confidence", {}) or {}
    rows = [[
        "class",
        "support",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc_ovr",
        "pr_auc_ovr",
        "accuracy",
        "avg_confidence",
    ]]
    for class_name in class_names:
        stats = per_class.get(class_name, {}) or {}
        rows.append([
            class_name,
            str(stats.get("support", "")),
            f"{float(stats.get('precision', 0.0)):.6f}",
            f"{float(stats.get('recall', 0.0)):.6f}",
            f"{float(stats.get('specificity', 0.0)):.6f}",
            f"{float(stats.get('f1', 0.0)):.6f}",
            "" if stats.get("roc_auc_ovr") is None else f"{float(stats['roc_auc_ovr']):.6f}",
            "" if stats.get("pr_auc_ovr") is None else f"{float(stats['pr_auc_ovr']):.6f}",
            "" if per_class_accuracy.get(class_name) is None else f"{float(per_class_accuracy[class_name]):.6f}",
            "" if per_class_confidence.get(class_name) is None else f"{float(per_class_confidence[class_name]):.6f}",
        ])
    path.write_text("\n".join(",".join(row) for row in rows), encoding="utf-8")


def compute_correct_confidence_by_class(
    logits: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = logits.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    per_class: dict[str, Any] = {}
    all_correct_confidences: list[float] = []
    for class_index, class_name in enumerate(class_names):
        class_mask = targets == class_index
        correct_mask = class_mask & (predictions == targets)
        correct_confidences = confidence[correct_mask]
        class_confidences = confidence[class_mask]
        all_correct_confidences.extend(correct_confidences.tolist())
        per_class[class_name] = {
            "support": int(class_mask.sum()),
            "correct_count": int(correct_mask.sum()),
            "average_confidence_on_correct_predictions": (
                float(correct_confidences.mean()) if correct_confidences.size > 0 else None
            ),
            "average_confidence_all_samples_of_class": (
                float(class_confidences.mean()) if class_confidences.size > 0 else None
            ),
        }

    return {
        "num_classes": len(class_names),
        "num_samples": int(targets.shape[0]),
        "overall_average_confidence_on_correct_predictions": (
            float(np.mean(all_correct_confidences)) if all_correct_confidences else None
        ),
        "per_class": per_class,
    }


def save_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe_value(payload), handle, indent=2)
        handle.write("\n")


def save_confusion_matrix_plot(
    path: Path,
    confusion_matrix: np.ndarray,
    class_names: list[str],
    title: str,
) -> None:
    matrix = np.asarray(confusion_matrix, dtype=np.int64)
    num_classes = len(class_names)
    cell_size = 108
    left_margin = 190
    top_margin = 140
    right_margin = 40
    bottom_margin = 70
    width = left_margin + num_classes * cell_size + right_margin
    height = top_margin + num_classes * cell_size + bottom_margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = font
    max_count = int(matrix.max()) if matrix.size > 0 else 0

    def cell_fill(value: int) -> tuple[int, int, int]:
        if max_count <= 0:
            return (245, 247, 250)
        intensity = value / max_count
        light = np.array([245, 247, 250], dtype=np.float64)
        dark = np.array([27, 94, 154], dtype=np.float64)
        blended = light + (dark - light) * intensity
        return tuple(int(round(channel)) for channel in blended)

    def text_size(text: str) -> tuple[int, int]:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]

    title_box = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_box[2] - title_box[0]
    draw.text(((width - title_width) / 2, 24), title, fill="black", font=title_font)
    draw.text((left_margin + (num_classes * cell_size) / 2 - 48, 64), "Predicted", fill="black", font=font)
    draw.text((26, top_margin + (num_classes * cell_size) / 2 - 6), "True", fill="black", font=font)

    for class_index, class_name in enumerate(class_names):
        x0 = left_margin + class_index * cell_size
        y0 = top_margin + class_index * cell_size
        label_width, label_height = text_size(class_name)
        draw.text(
            (x0 + (cell_size - label_width) / 2, top_margin - label_height - 18),
            class_name,
            fill="black",
            font=font,
        )
        draw.text(
            (left_margin - label_width - 16, y0 + (cell_size - label_height) / 2),
            class_name,
            fill="black",
            font=font,
        )

    for row_index in range(num_classes):
        row_sum = int(matrix[row_index].sum())
        for col_index in range(num_classes):
            value = int(matrix[row_index, col_index])
            x0 = left_margin + col_index * cell_size
            y0 = top_margin + row_index * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            fill = cell_fill(value)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(190, 196, 204), width=1)
            percent = (100.0 * value / row_sum) if row_sum > 0 else 0.0
            count_text = str(value)
            pct_text = f"{percent:.1f}%"
            count_width, count_height = text_size(count_text)
            pct_width, pct_height = text_size(pct_text)
            text_color = "white" if value > 0.55 * max(max_count, 1) else "black"
            draw.text(
                (x0 + (cell_size - count_width) / 2, y0 + cell_size / 2 - count_height - 6),
                count_text,
                fill=text_color,
                font=font,
            )
            draw.text(
                (x0 + (cell_size - pct_width) / 2, y0 + cell_size / 2 + 4),
                pct_text,
                fill=text_color,
                font=font,
            )

    image.save(path)


def append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump(json_safe_value(payload), handle)
        handle.write("\n")


def delete_bad_sample_file(path: Path) -> bool:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        return False
    return True


def remove_bad_sample_from_metadata(dataset_root: Path, sample_path: Path) -> int:
    metadata_path = dataset_root / "dataset_metadata.json"
    if not metadata_path.exists():
        return 0

    lock_path = dataset_root / RUNTIME_BAD_SAMPLE_CLEANUP_LOCK
    normalized_targets = {sample_path.as_posix()}
    try:
        relative_path = sample_path.relative_to(dataset_root).as_posix()
        normalized_targets.add(relative_path)
        normalized_targets.add(f"{dataset_root.name}/{relative_path}")
    except ValueError:
        pass

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            records = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                return 0
            filtered = [
                record
                for record in records
                if str(record.get("file_path", "")).replace(os.sep, "/") not in normalized_targets
            ]
            removed = len(records) - len(filtered)
            if removed > 0:
                temp_path = metadata_path.with_suffix(".json.tmp")
                temp_path.write_text(json.dumps(filtered, ensure_ascii=True, indent=2), encoding="utf-8")
                temp_path.replace(metadata_path)
            return removed
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def format_console_event(payload: dict[str, Any]) -> str | None:
    event = payload.get("event")
    timestamp = payload.get("timestamp", "")
    phase_name = payload.get("phase_name")
    phase_piece = f" phase_name={phase_name}" if phase_name is not None else ""
    if event == "run_started":
        return (
            f"[{timestamp}] run_started model={payload.get('model_name')} "
            f"output_dir={payload.get('output_dir')}"
        )
    if event == "dataset_schedule":
        return (
            f"[{timestamp}] dataset_schedule train_steps={payload.get('train_steps_per_epoch')} "
            f"val_steps={payload.get('val_steps_per_eval')} test_steps={payload.get('test_steps_per_eval')}"
        )
    if event == "train_step":
        per_class_accuracy = payload.get("per_class_accuracy")
        per_class_accuracy_text = format_per_class_confidence(per_class_accuracy) if isinstance(per_class_accuracy, dict) else None
        classwise_confidence = payload.get("per_class_avg_confidence")
        classwise_confidence_text = format_per_class_confidence(classwise_confidence) if isinstance(classwise_confidence, dict) else None
        pieces = [
            f"[{timestamp}] train_step",
            f"stage={payload.get('stage')}",
            f"phase_name={phase_name}" if phase_name is not None else None,
            f"epoch={payload.get('epoch')}",
            f"epoch_step={payload.get('epoch_step')}",
            f"global_step={payload.get('global_train_step')}",
            f"loss={payload.get('loss'):.6f}" if isinstance(payload.get("loss"), (int, float)) else None,
        ]
        if isinstance(payload.get("acc"), (int, float)):
            pieces.append(f"acc={payload.get('acc'):.6f}")
        if per_class_accuracy_text:
            pieces.append(f"per_class_accuracy={per_class_accuracy_text}")
        if classwise_confidence_text:
            pieces.append(f"per_class_avg_confidence={classwise_confidence_text}")
        for key in (
            "same_image_view_cosine",
            "same_class_positive_cosine",
            "different_class_negative_cosine",
            "positive_negative_cosine_margin",
        ):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                pieces.append(f"{key}={value:.6f}")
        learning_rates = payload.get("learning_rates")
        if learning_rates:
            pieces.append("lr=" + ",".join(f"{lr:.8g}" for lr in learning_rates))
        return " ".join(piece for piece in pieces if piece)
    if event in {"validation_finished", "final_evaluation_finished", "resume_initial_val_finished"}:
        per_class_accuracy = payload.get("per_class_accuracy")
        per_class_accuracy_text = format_per_class_confidence(per_class_accuracy) if isinstance(per_class_accuracy, dict) else None
        classwise_confidence = payload.get("per_class_avg_confidence")
        classwise_confidence_text = format_per_class_confidence(classwise_confidence) if isinstance(classwise_confidence, dict) else None
        pieces = [
            f"[{timestamp}] {event}",
            f"stage={payload.get('stage')}",
            f"phase_name={phase_name}" if phase_name is not None else None,
            f"epoch={payload.get('epoch')}",
            f"loss={payload.get('loss'):.6f}" if isinstance(payload.get("loss"), (int, float)) else None,
            f"accuracy={payload.get('accuracy'):.6f}" if isinstance(payload.get("accuracy"), (int, float)) else None,
            f"val_loss={payload.get('val_loss'):.6f}" if isinstance(payload.get("val_loss"), (int, float)) else None,
        ]
        if per_class_accuracy_text:
            pieces.append(f"per_class_accuracy={per_class_accuracy_text}")
        if classwise_confidence_text:
            pieces.append(f"per_class_avg_confidence={classwise_confidence_text}")
        for key in (
            "same_image_view_cosine",
            "same_class_positive_cosine",
            "different_class_negative_cosine",
            "positive_negative_cosine_margin",
        ):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                pieces.append(f"{key}={value:.6f}")
        return " ".join(piece for piece in pieces if piece)
    if event == "validation_started":
        pieces = [
            f"[{timestamp}] validation_started",
            f"stage={payload.get('stage')}",
            f"phase_name={phase_name}" if phase_name is not None else None,
            f"epoch={payload.get('epoch_in_phase')}",
            f"validation_index={payload.get('validation_index')}",
            f"epoch_step={payload.get('epoch_step')}",
            f"eval_batches={payload.get('eval_batches')}",
        ]
        return " ".join(piece for piece in pieces if piece)
    if event == "final_evaluation_started":
        pieces = [
            f"[{timestamp}] final_evaluation_started",
            f"stage={payload.get('stage')}",
            f"phase_name={phase_name}" if phase_name is not None else None,
            f"split={payload.get('split')}",
            f"eval_batches={payload.get('eval_batches')}",
        ]
        return " ".join(piece for piece in pieces if piece)
    if event == "eval_step":
        per_class_accuracy = payload.get("per_class_accuracy")
        per_class_accuracy_text = format_per_class_confidence(per_class_accuracy) if isinstance(per_class_accuracy, dict) else None
        classwise_confidence = payload.get("per_class_avg_confidence")
        classwise_confidence_text = format_per_class_confidence(classwise_confidence) if isinstance(classwise_confidence, dict) else None
        pieces = [
            f"[{timestamp}] eval_step",
            f"stage={payload.get('stage')}",
            f"split={payload.get('split')}",
            f"phase_name={phase_name}" if phase_name is not None else None,
            f"step={payload.get('eval_step')}",
            f"loss={payload.get('loss'):.6f}" if isinstance(payload.get("loss"), (int, float)) else None,
            f"acc={payload.get('acc'):.6f}" if isinstance(payload.get("acc"), (int, float)) else None,
        ]
        if per_class_accuracy_text:
            pieces.append(f"per_class_accuracy={per_class_accuracy_text}")
        if classwise_confidence_text:
            pieces.append(f"per_class_avg_confidence={classwise_confidence_text}")
        for key in (
            "same_image_view_cosine",
            "same_class_positive_cosine",
            "different_class_negative_cosine",
            "positive_negative_cosine_margin",
        ):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                pieces.append(f"{key}={value:.6f}")
        return " ".join(piece for piece in pieces if piece)
    if event == "resume_initial_val_pass":
        return f"[{timestamp}] resume_initial_val_pass stage={payload.get('stage')}{phase_piece} starting verification..."
    if event == "phase_visualization_started":
        return (
            f"[{timestamp}] phase_visualization_started stage={payload.get('stage')}"
            f"{phase_piece} output_dir={payload.get('output_dir')}"
        )
    if event == "phase_visualization_failed":
        return (
            f"[{timestamp}] phase_visualization_failed stage={payload.get('stage')}"
            f"{phase_piece} error={payload.get('error')}"
        )
    if event == "phase_visualization_finished":
        return (
            f"[{timestamp}] phase_visualization_finished stage={payload.get('stage')}"
            f"{phase_piece} output_dir={payload.get('output_dir')}"
        )
    if event == "phase_global_best_comparison":
        return (
            f"[{timestamp}] phase_global_best_comparison stage={payload.get('stage')}"
            f"{phase_piece} improved_global_best={payload.get('phase_improved_global_best')}"
        )
    if event == "phase_rejected_for_next_initialization":
        return (
            f"[{timestamp}] phase_rejected_for_next_initialization stage={payload.get('stage')}"
            f"{phase_piece} reason={payload.get('reason')}"
        )
    if event == "next_phase_initialization_selected":
        return (
            f"[{timestamp}] next_phase_initialization_selected stage={payload.get('stage')}"
            f"{phase_piece} next_phase_init_source={payload.get('next_phase_init_source')}"
        )
    if event == "runtime_bad_sample_detected":
        return (
            f"[{timestamp}] runtime_bad_sample_detected split={payload.get('split')} "
            f"path={payload.get('path')} removed={payload.get('file_removed')} "
            f"metadata_removed={payload.get('metadata_removed')} error={payload.get('error_type')}: {payload.get('error')}"
        )
    return None





import csv
from datetime import datetime


def append_to_csv(csv_path: Path, data: dict[str, Any]) -> None:
    flat_data: dict[str, Any] = {}
    for k, v in data.items():
        safe_value = json_safe_value(v)
        flat_data[k] = json.dumps(safe_value) if isinstance(safe_value, (list, dict)) else safe_value
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_data.keys()), extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerow(flat_data)
        return

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
        existing_fields = list(reader.fieldnames or [])

    merged_fields = list(existing_fields)
    for key in flat_data.keys():
        if key not in merged_fields:
            merged_fields.append(key)

    if merged_fields == existing_fields:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore", restval="")
            writer.writerow(flat_data)
        return

    existing_rows.append(flat_data)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
    tmp_path.replace(csv_path)


def log_json_event(path: Path, payload: dict[str, Any]) -> None:
    payload = json_safe_value(payload)
    if "timestamp" not in payload:
        payload = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), **payload}
    
    # Log to CSV
    csv_name = "train_metrics.csv"
    split = payload.get("split")
    event = payload.get("event")
    if event == "train_step":
        csv_name = "train_metrics.csv"
    elif split == "val" and event in {"validation_finished", "eval_step", "final_evaluation_finished"}:
        csv_name = "val_metrics.csv"
    elif split == "test" and event in {"final_test_evaluation_finished", "eval_step"}:
        csv_name = "test_metrics.csv"
    append_to_csv(path.parent / csv_name, payload)
    
    # Log to JSONL
    append_jsonl(path, payload)
    console_line = format_console_event(payload)
    if console_line:
        print(console_line, flush=True)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_training_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    torch.save(payload, path)


def save_supcon_best_checkpoint(
    path: Path,
    *,
    model_state_dict: dict[str, torch.Tensor],
    class_names: list[str],
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    train_progress: dict[str, Any],
    best_val_loss: float,
    best_val_acc: float,
    best_val_raw_acc: float,
    best_classifier_state: dict[str, torch.Tensor],
    supcon_best_state: dict[str, torch.Tensor],
    supcon_best_loss: float,
    supcon_best_epoch: int,
    supcon_wait: int,
    augmentation_epoch_cursor: int,
) -> None:
    payload = {
        "model_state_dict": model_state_dict,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "args": vars(args),
        "history": history,
        "train_progress": train_progress,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_raw_acc": best_val_raw_acc,
        "best_classifier_state": best_classifier_state,
        "supcon_best_state": supcon_best_state,
        "supcon_best_loss": supcon_best_loss,
        "supcon_best_epoch": supcon_best_epoch,
        "supcon_wait": supcon_wait,
        "augmentation_epoch_cursor": augmentation_epoch_cursor,
        "resume": {
            "stage": "supcon",
            "epoch": supcon_best_epoch,
            "epoch_step_completed": 0,
            "validation_index": 0,
        },
    }
    torch.save(payload, path)


def maybe_run_phase_visualizations(
    *,
    checkpoint_path: Path,
    dataset_root: str,
    output_dir: Path,
    phase_label: str,
    args: argparse.Namespace,
    log_path: Path,
    reason: str,
) -> None:
    if not getattr(args, "epoch_visualizations", False):
        return
    safe_label = (
        str(phase_label)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
    )
    target_dir = output_dir / "visualizations" / safe_label
    command = [
        sys.executable,
        str(Path(__file__).with_name("visualize_epoch.py")),
        "--checkpoint",
        str(checkpoint_path),
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(target_dir),
        "--epoch",
        str(phase_label),
        "--batch-size",
        str(args.epoch_visualization_batch_size),
        "--num-workers",
        str(args.epoch_visualization_num_workers),
        "--umap-thumb-size",
        str(args.epoch_visualization_umap_thumb_size),
        "--umap-max-samples",
        str(args.epoch_visualization_umap_max_samples),
        "--umap-thumbnail-limit",
        str(args.epoch_visualization_umap_thumbnail_limit),
    ]
    log_json_event(
        log_path,
        {
            "event": "phase_visualization_started",
            "reason": reason,
            "phase_label": str(phase_label),
            "output_dir": str(target_dir),
            "checkpoint": str(checkpoint_path),
        },
    )
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        log_json_event(
            log_path,
            {
                "event": "phase_visualization_failed",
                "reason": reason,
                "phase_label": str(phase_label),
                "output_dir": str(target_dir),
                "checkpoint": str(checkpoint_path),
                "returncode": int(exc.returncode),
            },
        )
    else:
        log_json_event(
            log_path,
            {
                "event": "phase_visualization_finished",
                "reason": reason,
                "phase_label": str(phase_label),
                "output_dir": str(target_dir),
                "checkpoint": str(checkpoint_path),
            },
        )


def phase_artifact_dir(output_dir: Path, phase_name: str) -> Path:
    safe_name = (
        str(phase_name)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
    )
    return output_dir / "phases" / safe_name


def save_step_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: list[str],
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
    train_progress: dict[str, int],
    stage: str,
    epoch: int,
    epoch_step: int,
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    phase_index: int | None = None,
    phase_name: str | None = None,
    extra_payload: dict[str, Any] | None = None,
    extra_resume: dict[str, Any] | None = None,
) -> None:
    resume_payload: dict[str, Any] = {
        "stage": stage,
        "epoch": epoch,
        "epoch_step_completed": epoch_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    if phase_index is not None:
        resume_payload["phase_index"] = phase_index
    if phase_name is not None:
        resume_payload["phase_name"] = phase_name
    if extra_resume:
        resume_payload.update(extra_resume)
    payload = {
        "model_state_dict": cpu_state_dict(model),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "args": vars(args),
        "train_progress": dict(train_progress),
        "resume": resume_payload,
    }
    if extra_payload:
        payload.update(extra_payload)
    torch.save(payload, path)


def shutdown_loader_workers(loader: DataLoader | None) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass
    try:
        loader._iterator = None
    except Exception:
        pass


def release_training_memory(device: torch.device, *loaders: DataLoader | None) -> None:
    for loader in loaders:
        shutdown_loader_workers(loader)
    gc.collect()
    if device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
        except RuntimeError:
            pass
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass


def load_resume_checkpoint(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        return torch.load(path, map_location="cpu"), None
    except Exception as exc:
        corrupt_path = path.with_name(f"{path.name}.corrupt")
        suffix_index = 1
        while corrupt_path.exists():
            corrupt_path = path.with_name(f"{path.name}.corrupt.{suffix_index}")
            suffix_index += 1
        path.rename(corrupt_path)
        return None, f"{type(exc).__name__}: {exc}. Corrupted checkpoint moved to {corrupt_path.name}"


def path_with_timestamp(path: Path, timestamp: str) -> Path:
    suffixes = "".join(path.suffixes)
    if suffixes:
        base_name = path.name[: -len(suffixes)]
        file_name = f"{base_name}_{timestamp}{suffixes}"
    else:
        file_name = f"{path.name}_{timestamp}"
    return path.with_name(file_name)


def unique_run_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir.with_name(f"{base_dir.name}_{timestamp}")
    suffix_index = 1
    while candidate.exists():
        candidate = base_dir.with_name(f"{base_dir.name}_{timestamp}_{suffix_index}")
        suffix_index += 1
    return candidate


def resolve_run_paths(
    requested_output_dir: Path,
    requested_log_path: Path,
    requested_resume_path: Path,
    resume_checkpoint: dict[str, Any] | None,
    resume_warning: str | None,
) -> tuple[Path, Path, Path, Path | None]:
    output_dir = requested_output_dir
    log_path = requested_log_path
    checkpoint_path = output_dir / "last.pt"
    step_checkpoint_path = output_dir / "step_last.pt"

    output_dir_occupied = requested_output_dir.exists() and any(requested_output_dir.iterdir())
    log_path_occupied = requested_log_path.exists() and requested_log_path.stat().st_size > 0
    would_overwrite_existing_run = output_dir_occupied or log_path_occupied

    if resume_checkpoint is not None or not would_overwrite_existing_run:
        return output_dir, checkpoint_path, step_checkpoint_path, log_path

    timestamped_output_dir = unique_run_output_dir(requested_output_dir)
    timestamp = timestamped_output_dir.name.removeprefix(f"{requested_output_dir.name}_")
    timestamped_log_path = path_with_timestamp(requested_log_path, timestamp)
    timestamped_output_dir.mkdir(parents=True, exist_ok=True)
    timestamped_log_path.parent.mkdir(parents=True, exist_ok=True)
    return (
        timestamped_output_dir,
        timestamped_output_dir / "last.pt",
        timestamped_output_dir / "step_last.pt",
        timestamped_log_path,
    )


def checkpoint_state_for_mode(resume_checkpoint: dict[str, Any], resume_mode: str) -> dict[str, torch.Tensor]:
    if resume_mode == "global_best":
        return resume_checkpoint.get("best_classifier_state", resume_checkpoint["model_state_dict"])
    if resume_mode == "phase_best":
        return resume_checkpoint.get(
            "phase_best_state",
            resume_checkpoint.get("best_classifier_state", resume_checkpoint["model_state_dict"]),
        )
    return resume_checkpoint["model_state_dict"]


def extract_state_dict_from_checkpoint(candidate: Any) -> dict[str, torch.Tensor]:
    """Extract a tensor-only state_dict from a checkpoint-like object.

    The Phase 0 MIM stage stores either a raw encoder state_dict or a wrapper
    checkpoint containing `encoder_state_dict`. The main trainer only needs the
    tensor map, so this helper normalizes both forms.
    """
    if isinstance(candidate, dict):
        for key in ("encoder_state_dict", "backbone_state_dict", "model_state_dict"):
            value = candidate.get(key)
            if isinstance(value, dict):
                return value
        if candidate and all(torch.is_tensor(value) for value in candidate.values()):
            return candidate  # raw state_dict
    raise ValueError("Could not extract a tensor-only state_dict from the supplied checkpoint.")


def resolve_phase_start_index(
    args: argparse.Namespace,
    phases: list[PhaseSpec],
    resume_state: dict[str, Any],
) -> int:
    if args.resume_phase_name:
        for index, phase in enumerate(phases, start=1):
            if phase.name == args.resume_phase_name:
                return index
        raise ValueError(f"--resume-phase-name {args.resume_phase_name!r} did not match any phase")
    if args.resume_phase_index > 0:
        if args.resume_phase_index > len(phases):
            raise ValueError(f"--resume-phase-index must be <= {len(phases)}")
        return args.resume_phase_index
    if resume_state.get("stage") == "classifier":
        return int(resume_state.get("phase_index", 1))
    return 1


def parse_class_loss_weight_specs(specs: list[str], class_to_idx: dict[str, int]) -> dict[int, float]:
    resolved: dict[int, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--class-loss-weight must be NAME=WEIGHT, got {spec!r}")
        class_name, weight_text = spec.split("=", 1)
        class_name = class_name.strip()
        if class_name not in class_to_idx:
            raise ValueError(f"--class-loss-weight referenced unknown class {class_name!r}")
        weight = float(weight_text)
        if weight <= 0:
            raise ValueError(f"--class-loss-weight requires positive weight, got {weight} for {class_name!r}")
        resolved[int(class_to_idx[class_name])] = weight
    return resolved


def parse_targeted_confusion_penalty_specs(specs: list[str], class_to_idx: dict[str, int]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for spec in specs:
        parts = [part.strip() for part in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(
                "--targeted-confusion-penalty must be TRUE_CLASS:PREDICTED_CLASS:WEIGHT, "
                f"got {spec!r}"
            )
        true_name, predicted_name, weight_text = parts
        if true_name not in class_to_idx:
            raise ValueError(f"--targeted-confusion-penalty referenced unknown true class {true_name!r}")
        if predicted_name not in class_to_idx:
            raise ValueError(f"--targeted-confusion-penalty referenced unknown predicted class {predicted_name!r}")
        weight = float(weight_text)
        if weight <= 0:
            raise ValueError(f"--targeted-confusion-penalty requires positive weight, got {weight} for {spec!r}")
        resolved.append(
            {
                "true_class": true_name,
                "predicted_class": predicted_name,
                "true_index": int(class_to_idx[true_name]),
                "predicted_index": int(class_to_idx[predicted_name]),
                "weight": weight,
            }
        )
    return resolved


def build_parser() -> argparse.ArgumentParser:
    description = (
        "Configurable backbone staged training pipeline: "
        "(1) SupCon head-only warm-up [frozen backbone] → "
        "(2) SupCon progressive backbone unfreezing [10 leaf modules/step, capped semantic tail] → "
        "(3) CE head-only warm-up [backbone re-frozen] → "
        "(4) CE progressive backbone unfreezing [10 leaf modules/step, same frozen-core boundary] → "
        "(5) Recursive val_loss refinement → "
        "(6) Recursive val_raw_acc refinement. "
        "Checkpoints saved at every step, validation, and phase transition. "
        "Phase-end clean test-set visual audits, calibration reports, and optional Grad-CAM exports are available."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--unfreeze-chunk-size", type=int, default=10)
    parser.add_argument("--skip-supcon", action="store_true")
    parser.add_argument("--classifier-train-mode", choices=("progressive", "full_model"), default="progressive")
    parser.add_argument("--classifier-early-stopping-metric", choices=("val_loss", "val_raw_acc"), default="val_loss")
    parser.add_argument(
        "--reject-current-phase-on-global-miss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, a completed progressive phase that fails to beat the global best checkpoint "
            "on the selected classifier early-stopping metric is not used to initialize the next phase. "
            "Use --no-reject-current-phase-on-global-miss to disable the rejection gate."
        ),
    )
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    # SupCon stages:
    # - head-only warm-up uses 3e-3 to rapidly organise the embedding/projection space
    # - progressive semantic-tail tuning uses 5e-5 on the backbone to avoid disturbing the frozen core
    parser.add_argument("--supcon-head-lr", type=float, default=3e-3)
    parser.add_argument("--supcon-backbone-lr", type=float, default=5e-5)
    # CE stages:
    # - head-only warm-up uses 1e-3 to initialize the classifier boundary quickly
    # - progressive CE tuning uses 1e-5 on the backbone; head LR decays toward backbone LR
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE_NAME)
    parser.add_argument("--optimizer", choices=["sam", "adamw"], default="adamw")
    parser.add_argument("--precision", choices=("mixed", "32", "64"), default="mixed")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    # label_smoothing remains enabled for the imbalanced 3-class logical setup.
    # The repo projects every physical dataset into organic / metal / paper.
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--confidence-gap-penalty-weight", type=float, default=0.0)
    parser.add_argument("--class-loss-weight", action="append", default=[])
    parser.add_argument("--targeted-confusion-penalty", action="append", default=[])
    parser.add_argument(
        "--train-loss-validation-patience",
        type=int,
        default=500,
        help="Trigger validation after this many consecutive train steps without a new best train loss in the current phase.",
    )
    parser.add_argument(
        "--validation-patience",
        type=int,
        default=1000,
        help="Force validation after this many consecutive train steps since the last validation checkpoint in the current phase.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("balanced", "weighted", "shuffle"),
        default="balanced",
        help="Sampling strategy for train and SupCon loaders. Defaults to balanced per-batch class cycling.",
    )
    parser.add_argument(
        "--weighted-sampling",
        action="store_const",
        const="weighted",
        dest="sampling_strategy",
        help="Legacy alias: use class-balanced weighted random sampling with replacement.",
    )
    parser.add_argument(
        "--no-weighted-sampling",
        action="store_const",
        const="shuffle",
        dest="sampling_strategy",
        help="Legacy alias: disable class balancing and use shuffled source-order sampling.",
    )
    parser.add_argument("--weights", choices=("default", "none"), default="default")
    parser.add_argument(
        "--phase0-encoder-checkpoint",
        default="",
        help=(
            "Optional Phase 0 masked-image-modeling encoder checkpoint. "
            "When set and no resume checkpoint is active, the backbone is seeded "
            "from this encoder state before SupCon/CE training begins."
        ),
    )
    parser.add_argument(
        "--augment-repeats",
        type=int,
        default=1,
        help="Legacy compatibility knob retained for parser stability. Fixed-tint preprocessing remains deterministic.",
    )
    parser.add_argument(
        "--augment-gaussian-sigmas",
        type=float,
        default=1.0,
        help="Legacy compatibility knob retained for parser stability. Fixed-tint preprocessing remains deterministic.",
    )
    parser.add_argument(
        "--camera-color-cast-probability",
        type=float,
        default=CAMERA_COLOR_CAST_PROBABILITY,
        help=(
            "Fixed Raspberry Pi style magenta/pink white-balance cast applied to every image. "
            "No stochastic augmentations remain."
        ),
    )
    parser.add_argument(
        "--camera-color-cast-strength",
        type=float,
        default=CAMERA_COLOR_CAST_STRENGTH,
        help="Strength of the fixed magenta/pink camera color-cast applied to every image. Default is 0.50.",
    )
    parser.add_argument(
        "--camera-color-cast-eval",
        action=argparse.BooleanOptionalAction,
        default=CAMERA_COLOR_CAST_EVAL,
        help=(
            "Render validation/test Dataset_Final samples through the same fixed Pi-camera "
            "magenta/pink cast used for training. External holdout images use the same fixed cast."
        ),
    )
    # SupCon and CE both respect the same permanent frozen-core boundary by default.
    # Only the semantic tail after this frozen core is eligible for unfreezing.
    parser.add_argument(
        "--frozen-core-backbone-modules",
        type=int,
        default=40,
        help=(
            "Number of earliest backbone leaf modules that stay frozen in every SupCon, CE, "
            "and recursive phase. Default=40; this always preserves the earliest backbone leaves "
            "and only opens the semantic tail for training."
        ),
    )
    parser.add_argument("--supcon-unfreeze-backbone-modules", type=int, default=None)
    parser.add_argument(
        "--ce-max-unfreeze-modules",
        type=int,
        default=None,
        help=(
            "Optional extra cap on backbone leaf modules unfrozen during CE training. "
            "When unset, CE may train only the tail left after --frozen-core-backbone-modules. "
            "This cap can reduce the trainable tail, but cannot thaw the frozen core."
        ),
    )
    parser.add_argument("--output-dir", default="Results/metric_learning_experiment")
    parser.add_argument("--log-file", default="logs/metric_learning_experiment.log.jsonl")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train on the entire dataset with train-loss plateau stopping and skip validation/test evaluation.",
    )
    parser.add_argument(
        "--class-mapping",
        type=str,
        default="",
        help=(
            "Optional extra JSON training-time class merging. The repo always enforces "
            "The repo always enforces the 3-class logical taxonomy (organic, metal, paper) before this is applied."
        ),
    )
    parser.add_argument("--auto-split-ratios", default="0.9,0.05,0.05")
    parser.add_argument("--resume-mode", choices=("latest", "global_best", "phase_best"), default="latest")
    parser.add_argument("--resume-phase-index", type=int, default=0)
    parser.add_argument("--resume-phase-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=1)
    parser.add_argument("--log-eval-every-steps", type=int, default=1)
    parser.add_argument(
        "--epoch-visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate test-set visualizations at startup and after every completed phase. "
            "Uses the clean no-augmentation test split."
        ),
    )
    parser.add_argument("--epoch-visualization-batch-size", type=int, default=128)
    parser.add_argument("--epoch-visualization-num-workers", type=int, default=2)
    parser.add_argument(
        "--epoch-visualization-umap-thumb-size",
        type=int,
        default=48,
        help="Thumbnail size for the UMAP thumbnail map.",
    )
    parser.add_argument(
        "--epoch-visualization-umap-max-samples",
        type=int,
        default=0,
        help="Optional cap for UMAP samples. Use 0 for the full test set.",
    )
    parser.add_argument(
        "--epoch-visualization-umap-thumbnail-limit",
        type=int,
        default=0,
        help="Maximum number of thumbnails rendered on the UMAP map. Use 0 to render all thumbnails.",
    )
    parser.add_argument(
        "--runtime-bad-sample-cleanup",
        action="store_true",
        help=(
            "Temporary opt-in cleanup path: when a sample fails to load during training, "
            "delete the file immediately and remove its entry from dataset_metadata.json."
        ),
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--supcon-early-stopping-patience", type=int, default=3)
    parser.add_argument("--head-early-stopping-patience", type=int, default=3)
    parser.add_argument("--stage-early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=1024)
    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        default=False,
        help=(
            "Run a final test-set evaluation pass at the end of training. "
            "OFF by default — the test set is a sacred holdout and must never "
            "be touched during any phase of training or phase hand-offs. "
            "Enable only when you explicitly want a final one-time test report. "
            "A test pass can also be run at any time via evaluate_saved_classifier.py."
        ),
    )
    return parser



def resolve_runtime_selected_classes(
    class_names: list[str],
    selected_classes: list[str] | None,
) -> list[str]:
    if not selected_classes:
        return list(class_names)
    selected: list[str] = []
    seen: set[str] = set()
    unknown: list[str] = []
    available = set(class_names)
    for class_name in selected_classes:
        if class_name in seen:
            continue
        if class_name not in available:
            unknown.append(class_name)
            continue
        selected.append(class_name)
        seen.add(class_name)
    if unknown:
        raise ValueError(f"Unknown runtime-selected classes: {unknown}")
    if not selected:
        raise ValueError("At least one runtime-selected class must match the trained class names.")
    return selected


def collapse_logits_and_targets_to_runtime_classes(
    logits: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    selected_classes: list[str] | None = None,
    class_mapping: dict[str, list[str]] | None = None,
    other_label: str = "other",
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """
    Collapses granular model classes into a smaller set of runtime classes for UI display.
    Supports simple selection (subsetting) or explicit merging (grouping).
    """
    if class_mapping:
        # 1. Explicit Merging Mode
        collapsed_class_names = []
        collapsed_columns = []
        target_map = {}
        
        # Track which source classes have been mapped to avoid duplicates
        mapped_source_indices = set()
        
        for custom_name, source_names in class_mapping.items():
            source_indices = [class_names.index(name) for name in source_names if name in class_names]
            if not source_indices:
                continue
                
            # Aggregate Logits: LogSumExp for correct probability preservation
            group_logits = logits[:, source_indices]
            group_max = np.max(group_logits, axis=1, keepdims=True)
            group_lse = group_max + np.log(np.exp(group_logits - group_max).sum(axis=1, keepdims=True))
            
            collapsed_columns.append(group_lse)
            new_index = len(collapsed_class_names)
            collapsed_class_names.append(custom_name)
            
            for idx in source_indices:
                target_map[idx] = new_index
                mapped_source_indices.add(idx)
        
        # Handle residual 'Other' if any classes weren't mapped
        residual_indices = [i for i in range(len(class_names)) if i not in mapped_source_indices]
        if residual_indices:
            other_logits = logits[:, residual_indices]
            other_max = np.max(other_logits, axis=1, keepdims=True)
            other_lse = other_max + np.log(np.exp(other_logits - other_max).sum(axis=1, keepdims=True))
            
            collapsed_columns.append(other_lse)
            other_index = len(collapsed_class_names)
            collapsed_class_names.append(other_label)
            for idx in residual_indices:
                target_map[idx] = other_index

        final_logits = np.concatenate(collapsed_columns, axis=1)
        final_targets = np.asarray([target_map.get(int(t), -1) for t in targets], dtype=np.int64)
        
        return final_logits, final_targets, collapsed_class_names, {
            "mode": "explicit_merge",
            "mapping": class_mapping,
            "collapse_applied": True
        }

    # 2. Simple Subsetting Mode (Original behavior)
    resolved_selected = resolve_runtime_selected_classes(class_names, selected_classes)
    if not selected_classes:
        return (
            logits,
            targets,
            list(class_names),
            {
                "selected_classes": list(class_names),
                "merged_other_source_classes": [],
                "other_label": other_label,
                "collapse_applied": False,
            },
        )

    selected_indices = [class_names.index(name) for name in resolved_selected]
    merged_indices = [index for index, name in enumerate(class_names) if name not in resolved_selected]
    collapsed_class_names = list(resolved_selected)
    collapsed_columns = [logits[:, index : index + 1] for index in selected_indices]

    target_map = {original_index: new_index for new_index, original_index in enumerate(selected_indices)}
    if merged_indices:
        merged_logits = logits[:, merged_indices]
        merged_max = np.max(merged_logits, axis=1, keepdims=True)
        merged_logsumexp = merged_max + np.log(np.exp(merged_logits - merged_max).sum(axis=1, keepdims=True))
        collapsed_columns.append(merged_logsumexp)
        other_index = len(collapsed_class_names)
        collapsed_class_names.append(other_label)
        for original_index in merged_indices:
            target_map[original_index] = other_index

    collapsed_logits = np.concatenate(collapsed_columns, axis=1)
    collapsed_targets = np.asarray([target_map[int(target)] for target in targets], dtype=np.int64)
    return (
        collapsed_logits,
        collapsed_targets,
        collapsed_class_names,
        {
            "selected_classes": list(resolved_selected),
            "merged_other_source_classes": [class_names[index] for index in merged_indices],
            "other_label": other_label,
            "collapse_applied": True,
        },
    )


def run_experiment(args: argparse.Namespace) -> int:
    if args.optimizer == "sam" and args.grad_accum_steps != 1:
        raise ValueError("SAM support in this trainer requires --grad-accum-steps 1")
    if args.unfreeze_chunk_size < 1:
        raise ValueError("--unfreeze-chunk-size must be >= 1")
    if args.augment_repeats < 1:
        raise ValueError("--augment-repeats must be >= 1")
    if args.augment_gaussian_sigmas <= 0:
        raise ValueError("--augment-gaussian-sigmas must be > 0")
    if not (0.0 <= args.camera_color_cast_probability <= 1.0):
        raise ValueError("--camera-color-cast-probability must be between 0 and 1")
    if args.camera_color_cast_strength < 0:
        raise ValueError("--camera-color-cast-strength must be >= 0")
    if args.log_every_steps < 1:
        raise ValueError("--log-every-steps must be >= 1")
    if args.log_eval_every_steps < 1:
        raise ValueError("--log-eval-every-steps must be >= 1")
    if args.train_loss_validation_patience < 1:
        raise ValueError("--train-loss-validation-patience must be >= 1")
    if args.validation_patience < 1:
        raise ValueError("--validation-patience must be >= 1")
    if args.confidence_threshold < 0.0 or args.confidence_threshold > 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1")
    if args.supcon_early_stopping_patience < 1:
        raise ValueError("--supcon-early-stopping-patience must be >= 1")
    if args.head_early_stopping_patience < 1:
        raise ValueError("--head-early-stopping-patience must be >= 1")
    if args.stage_early_stopping_patience < 1:
        raise ValueError("--stage-early-stopping-patience must be >= 1")
    if args.frozen_core_backbone_modules < 0:
        raise ValueError("--frozen-core-backbone-modules must be >= 0")
    if args.supcon_unfreeze_backbone_modules is not None and args.supcon_unfreeze_backbone_modules < 0:
        raise ValueError("--supcon-unfreeze-backbone-modules must be >= 0 when set")
    if args.ce_max_unfreeze_modules is not None and args.ce_max_unfreeze_modules < 0:
        raise ValueError("--ce-max-unfreeze-modules must be >= 0 when set")
    if (
        args.supcon_unfreeze_backbone_modules is not None
        and args.ce_max_unfreeze_modules is not None
        and args.supcon_unfreeze_backbone_modules > args.ce_max_unfreeze_modules
    ):
        raise ValueError(
            "--supcon-unfreeze-backbone-modules must be <= --ce-max-unfreeze-modules "
            "so the permanent frozen-core boundary is respected across the full pipeline."
        )

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    if str(args.precision) in {"32", "mixed"}:
        torch.set_float32_matmul_precision("high")
    model_name = f"{args.backbone}_metric_learning"

    requested_output_dir = Path(args.output_dir)
    requested_log_path = Path(args.log_file)
    requested_checkpoint_path = requested_output_dir / "last.pt"
    resume_path = Path(args.resume_checkpoint) if args.resume_checkpoint else requested_checkpoint_path
    resume_checkpoint, resume_warning = load_resume_checkpoint(resume_path)
    output_dir, checkpoint_path, step_checkpoint_path, log_path = resolve_run_paths(
        requested_output_dir=requested_output_dir,
        requested_log_path=requested_log_path,
        requested_resume_path=resume_path,
        resume_checkpoint=resume_checkpoint,
        resume_warning=resume_warning,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    auto_forked_run = output_dir != requested_output_dir or log_path != requested_log_path
    args.output_dir = str(output_dir)
    args.log_file = str(log_path)
    log_json_event(
        log_path,
        {
            "event": "run_resumed" if resume_checkpoint is not None else "run_started",
            "model_name": model_name,
            "output_dir": str(output_dir),
            "log_file": str(log_path),
            "resume_path": str(resume_path),
            "args": vars(args),
        },
    )
    if auto_forked_run:
        log_json_event(
            log_path,
            {
                "event": "run_paths_forked",
                "reason": "existing_outputs_preserved",
                "requested_output_dir": str(requested_output_dir),
                "actual_output_dir": str(output_dir),
                "requested_log_file": str(requested_log_path),
                "actual_log_file": str(log_path),
                "requested_resume_path": str(resume_path),
            },
        )
    if resume_warning is not None:
        log_json_event(log_path, {"event": "resume_checkpoint_ignored", "message": resume_warning})

    train_dataset, val_dataset, test_dataset, supcon_train_dataset, supcon_val_dataset = build_datasets(args)
    args.class_names_resolved = train_dataset.classes
    args.training_class_order = list(TRAINING_CLASS_ORDER)
    args.training_class_mapping = enforced_training_class_mapping(parse_json_class_mapping(getattr(args, "class_mapping", "")))
    args.training_excluded_classes = sorted(TRAINING_EXCLUDED_CLASSES)
    args.class_loss_weight_map_resolved = parse_class_loss_weight_specs(list(args.class_loss_weight), train_dataset.class_to_idx)
    args.targeted_confusion_penalties_resolved = parse_targeted_confusion_penalty_specs(
        list(args.targeted_confusion_penalty),
        train_dataset.class_to_idx,
    )
    if args.class_loss_weight_map_resolved or args.targeted_confusion_penalties_resolved:
        log_json_event(
            log_path,
            {
                "event": "classifier_loss_configuration",
                "class_loss_weight_map": {
                    train_dataset.classes[int(class_index)]: float(weight)
                    for class_index, weight in args.class_loss_weight_map_resolved.items()
                },
                "targeted_confusion_penalties": args.targeted_confusion_penalties_resolved,
            },
        )

    if args.sampling_strategy == "balanced":
        train_sampler = make_balanced_sampler(train_dataset, train_dataset.classes, args.batch_size, args.seed + 101)
        supcon_sampler = make_balanced_sampler(supcon_train_dataset, train_dataset.classes, args.batch_size, args.seed + 202)
        val_sampler = make_epoch_sampler(val_dataset, args.seed + 303, shuffle=False)
        supcon_val_sampler = make_epoch_sampler(supcon_val_dataset, args.seed + 404, shuffle=False)
        test_sampler = make_epoch_sampler(test_dataset, args.seed + 505, shuffle=False)
    elif args.sampling_strategy == "weighted":
        train_sampler = make_weighted_sampler(train_dataset, train_dataset.classes, train_dataset.target_for_index, args.seed + 101)
        supcon_sampler = make_weighted_sampler(supcon_train_dataset, train_dataset.classes, supcon_train_dataset.target_for_index, args.seed + 202)
        val_sampler = make_epoch_sampler(val_dataset, args.seed + 303, shuffle=False)
        supcon_val_sampler = make_epoch_sampler(supcon_val_dataset, args.seed + 404, shuffle=False)
        test_sampler = make_epoch_sampler(test_dataset, args.seed + 505, shuffle=False)
    else:
        train_sampler = make_epoch_sampler(train_dataset, args.seed + 101, shuffle=True)
        supcon_sampler = make_epoch_sampler(supcon_train_dataset, args.seed + 202, shuffle=True)
        val_sampler = make_epoch_sampler(val_dataset, args.seed + 303, shuffle=False)
        supcon_val_sampler = make_epoch_sampler(supcon_val_dataset, args.seed + 404, shuffle=False)
        test_sampler = make_epoch_sampler(test_dataset, args.seed + 505, shuffle=False)

    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, args.prefetch_factor, shuffle=False, sampler=train_sampler)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, args.prefetch_factor, shuffle=False, sampler=val_sampler)
    test_loader = make_loader(test_dataset, args.batch_size, args.num_workers, args.prefetch_factor, shuffle=False, sampler=test_sampler)
    supcon_train_loader = make_loader(
        supcon_train_dataset, args.batch_size, args.num_workers, args.prefetch_factor, shuffle=False, sampler=supcon_sampler
    )
    supcon_val_loader = make_loader(supcon_val_dataset, args.batch_size, args.num_workers, args.prefetch_factor, shuffle=False, sampler=supcon_val_sampler)
    log_json_event(
        log_path,
        {
            "event": "dataset_schedule",
            "source_train_count": train_dataset.source_count(),
            "source_val_count": val_dataset.source_count(),
            "source_test_count": test_dataset.source_count(),
            "augmentation_bank_train_count": len(train_dataset),
            "augmentation_bank_val_count": len(val_dataset),
            "augmentation_bank_test_count": len(test_dataset),
            "train_samples_per_epoch": len(train_sampler),
            "val_samples_per_eval": len(val_sampler),
            "test_samples_per_eval": len(test_sampler),
            "train_steps_per_epoch": len(train_loader),
            "supcon_steps_per_epoch": len(supcon_train_loader),
            "val_steps_per_eval": len(val_loader),
            "test_steps_per_eval": len(test_loader),
            "sampling_strategy": args.sampling_strategy,
            "class_names": train_dataset.classes,
            "class_to_idx": train_dataset.class_to_idx,
            "training_class_order": list(TRAINING_CLASS_ORDER),
            "training_class_mapping": args.training_class_mapping,
            "training_excluded_classes": args.training_excluded_classes,
            "train_taxonomy": getattr(train_dataset, "taxonomy_metadata", {}),
            "val_taxonomy": getattr(val_dataset, "taxonomy_metadata", {}),
            "test_taxonomy": getattr(test_dataset, "taxonomy_metadata", {}),
            "train_loss_validation_patience": args.train_loss_validation_patience,
            "validation_patience": args.validation_patience,
            "supcon_early_stopping_patience": args.supcon_early_stopping_patience,
            "head_early_stopping_patience": args.head_early_stopping_patience,
            "stage_early_stopping_patience": args.stage_early_stopping_patience,
            "warmup_epochs": args.warmup_epochs,
            "warmup_steps": args.warmup_steps,
        },
    )
    val_eval_batches = min(len(supcon_val_loader), args.max_eval_batches) if args.max_eval_batches > 0 else len(supcon_val_loader)
    test_eval_batches = min(len(test_loader), args.max_eval_batches) if args.max_eval_batches > 0 else len(test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = model_dtype_for_args(args)
    effective_weights_mode = args.weights
    if resume_checkpoint is None and getattr(args, "phase0_encoder_checkpoint", ""):
        # Phase 1+ must start from the exported Phase 0 encoder, not from a timm
        # pretrained checkpoint that would then be overwritten.
        effective_weights_mode = "none"
        log_json_event(
            log_path,
            {
                "event": "phase0_encoder_checkpoint_forced_scratch_init",
                "phase0_encoder_checkpoint": str(args.phase0_encoder_checkpoint),
                "backbone_name": args.backbone,
                "weights_mode": args.weights,
            },
        )
    model = MetricLearningEfficientNetB0(
        num_classes=len(train_dataset.classes),
        weights_mode=effective_weights_mode,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        args=args,
        backbone_name=args.backbone,
    ).to(device=device, dtype=model_dtype)
    if resume_checkpoint is not None:
        resume_state_dict = checkpoint_state_for_mode(resume_checkpoint, args.resume_mode)
        source_class_names = list(resume_checkpoint.get("class_names", []))
        target_class_names = list(train_dataset.classes)
        if source_class_names and source_class_names != target_class_names:
            resume_state_dict, adaptation_report = adapt_checkpoint_state_dict_to_training_taxonomy(
                resume_state_dict,
                source_class_names,
                target_class_names,
                class_mapping=args.training_class_mapping,
            )
            log_json_event(
                log_path,
                {
                    "event": "checkpoint_taxonomy_adapted",
                    "resume_mode": args.resume_mode,
                    "source_class_names": source_class_names,
                    "target_class_names": target_class_names,
                    "adaptation": adaptation_report,
                },
            )
        model.load_state_dict(resume_state_dict)
    elif getattr(args, "phase0_encoder_checkpoint", ""):
        phase0_path = Path(str(args.phase0_encoder_checkpoint))
        if not phase0_path.exists():
            raise FileNotFoundError(f"Phase 0 encoder checkpoint does not exist: {phase0_path}")
        phase0_checkpoint = load_resume_checkpoint(phase0_path)[0]
        if phase0_checkpoint is None:
            raise ValueError(f"Could not load Phase 0 encoder checkpoint: {phase0_path}")
        phase0_state_dict = extract_state_dict_from_checkpoint(phase0_checkpoint)
        model.backbone.load_state_dict(phase0_state_dict, strict=True)
        log_json_event(
            log_path,
            {
                "event": "phase0_encoder_checkpoint_loaded",
                "phase0_encoder_checkpoint": str(phase0_path),
                "backbone_name": args.backbone,
                "weights_mode": args.weights,
                "phase0_stage_trainability": {
                    "phase0": "all_backbone_parameters_trainable",
                    "supcon_and_ce": f"frozen_core_{int(getattr(args, 'frozen_core_backbone_modules', 40))}_modules",
                },
            },
        )
    backbone_modules = backbone_leaf_modules(model)
    train_progress = (
        dict(resume_checkpoint.get("train_progress", {}))
        if resume_checkpoint is not None
        else {"global_train_step": 0, "global_source_samples_seen": 0}
    )
    train_progress.setdefault("global_train_step", 0)
    train_progress.setdefault("global_source_samples_seen", 0)

    history: list[dict[str, Any]] = list(resume_checkpoint.get("history", [])) if resume_checkpoint is not None else []
    total_params, _ = parameter_counts(model)
    best_val_loss = float(resume_checkpoint.get("best_val_loss", float("inf"))) if resume_checkpoint is not None else float("inf")
    best_val_acc = float(resume_checkpoint.get("best_val_acc", -1.0)) if resume_checkpoint is not None else -1.0
    best_val_raw_acc = (
        float(resume_checkpoint.get("best_val_raw_acc", resume_checkpoint.get("best_val_acc", -1.0)))
        if resume_checkpoint is not None
        else -1.0
    )
    augmentation_epoch_cursor = int(resume_checkpoint.get("augmentation_epoch_cursor", 0)) if resume_checkpoint is not None else 0
    best_classifier_state = resume_checkpoint.get("best_classifier_state", cpu_state_dict(model)) if resume_checkpoint is not None else cpu_state_dict(model)
    supcon_best_state = (
        resume_checkpoint.get("supcon_best_state", cpu_state_dict(model))
        if resume_checkpoint is not None
        else cpu_state_dict(model)
    )
    resume_state = dict(resume_checkpoint.get("resume", {})) if resume_checkpoint is not None else {}
    resume_from_best_phase = resume_checkpoint is not None and args.resume_mode != "latest"
    if resume_from_best_phase:
        resume_state = {
            "stage": "classifier",
            "epoch": 1,
            "epoch_step_completed": 0,
            "validation_index": 0,
        }
        log_json_event(
            log_path,
            {
                "event": "resume_best_mode",
                "resume_mode": args.resume_mode,
                "resume_path": str(resume_path),
            },
        )

    supcon_loss = SupConLoss(args.supcon_temperature)
    supcon_phases = build_supcon_phase_plan(len(backbone_modules), args)
    supcon_best_loss = float(resume_checkpoint.get("supcon_best_loss", float("inf"))) if resume_checkpoint is not None else float("inf")
    supcon_best_epoch = int(resume_checkpoint.get("supcon_best_epoch", 0)) if resume_checkpoint is not None else 0
    supcon_wait = int(resume_checkpoint.get("supcon_wait", 0)) if resume_checkpoint is not None else 0
    start_supcon_epoch = 1
    start_supcon_epoch_step = 0
    start_supcon_validation_index = 0
    resume_supcon_phase_index = 1
    supcon_completed = bool(args.skip_supcon or not supcon_phases)

    save_training_checkpoint(
        checkpoint_path,
        {
            "model_state_dict": cpu_state_dict(model),
            "class_names": train_dataset.classes,
            "class_to_idx": train_dataset.class_to_idx,
            "args": vars(args),
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_val_raw_acc": best_val_raw_acc,
            "best_classifier_state": best_classifier_state,
            "supcon_best_state": supcon_best_state,
            "supcon_best_loss": supcon_best_loss,
            "supcon_best_epoch": supcon_best_epoch,
            "supcon_wait": supcon_wait,
            "augmentation_epoch_cursor": augmentation_epoch_cursor,
            "train_progress": train_progress,
            "resume": resume_state,
        },
    )
    if resume_checkpoint is None:
        maybe_run_phase_visualizations(
            checkpoint_path=checkpoint_path,
            dataset_root=args.dataset_root,
            output_dir=output_dir,
            phase_label="startup",
            args=args,
            log_path=log_path,
            reason="run_start",
        )

    if args.skip_supcon or resume_from_best_phase or not supcon_phases:
        supcon_completed = True
        log_json_event(
            log_path,
            {
                "event": "supcon_skipped",
                "reason": (
                    "resume_from_best_phase"
                    if resume_from_best_phase
                    else "no_supcon_phases"
                    if not supcon_phases
                    else "flag"
                ),
            },
        )
    elif resume_state.get("stage") == "supcon":
        resume_supcon_phase_index = resolve_phase_start_index(args, supcon_phases, resume_state)
        start_supcon_epoch = int(resume_state.get("epoch", 1))
        start_supcon_epoch_step = int(resume_state.get("epoch_step_completed", 0))
        start_supcon_validation_index = int(resume_state.get("validation_index", 0))
        supcon_completed = False
    elif resume_state.get("stage") == "classifier":
        supcon_completed = True

    if not supcon_completed:
        for supcon_phase_index, supcon_phase in enumerate(supcon_phases, start=1):
            if supcon_phase_index < resume_supcon_phase_index:
                continue
            thawed_supcon = set_trainability_for_supcon(model, backbone_modules, supcon_phase.unfrozen_backbone_modules)
            supcon_optimizer = build_supcon_optimizer(model, args)
            supcon_scheduler = build_scheduler(
                base_optimizer_for_scheduler(supcon_optimizer),
                steps_per_epoch=steps_per_epoch_for_sampler(
                    supcon_sampler, supcon_train_dataset, args.batch_size, args.max_train_batches
                ),
                warmup_epochs=args.warmup_epochs,
                warmup_steps=args.warmup_steps,
            )
            supcon_scaler = build_grad_scaler(device, args)
            _, supcon_trainable_params = parameter_counts(model)
            phase_best_loss = float("inf")
            phase_best_epoch = 0
            phase_best_state = cpu_state_dict(model)
            phase_wait = 0
            same_resume_phase = (
                resume_checkpoint is not None
                and
                resume_state.get("stage") == "supcon"
                and supcon_phase_index == resume_supcon_phase_index
                and resume_state.get("phase_name") == supcon_phase.name
                and "optimizer_state_dict" in resume_state
                and "scheduler_state_dict" in resume_state
            )
            if same_resume_phase:
                try:
                    supcon_optimizer.load_state_dict(resume_state["optimizer_state_dict"])
                    supcon_scheduler.load_state_dict(resume_state["scheduler_state_dict"])
                    set_scheduler_base_lrs(
                        supcon_scheduler,
                        [
                            args.supcon_head_lr,
                            *([args.supcon_backbone_lr] if len(supcon_scheduler.optimizer.param_groups) > 1 else []),
                        ],
                    )
                    if "scaler_state_dict" in resume_state:
                        supcon_scaler.load_state_dict(resume_state["scaler_state_dict"])
                    phase_best_loss = float(resume_state.get("phase_best_loss", float("inf")))
                    phase_best_epoch = int(resume_state.get("phase_best_epoch", 0))
                    phase_best_state = resume_checkpoint.get("phase_best_state", cpu_state_dict(model))
                    phase_wait = int(resume_state.get("phase_wait", 0))
                except ValueError as exc:
                    same_resume_phase = False
                    log_json_event(
                        log_path,
                        {
                            "event": "resume_fallback",
                            "stage": "supcon",
                            "phase_index": supcon_phase_index,
                            "phase_name": supcon_phase.name,
                            "reason": f"incompatible_optimizer_state: {exc}",
                        },
                    )
            elif resume_checkpoint is not None and resume_state.get("stage") == "supcon" and supcon_phase_index == resume_supcon_phase_index:
                log_json_event(
                    log_path,
                    {
                        "event": "resume_fallback",
                        "stage": "supcon",
                        "phase_index": supcon_phase_index,
                        "phase_name": supcon_phase.name,
                        "reason": "missing_optimizer_or_scheduler_state",
                    },
                )

            phase_train_loss_best = float(
                resume_state.get(
                    "phase_train_loss_best",
                    infer_phase_train_loss_baseline(history, "supcon", supcon_phase.name),
                )
            )
            phase_train_loss_wait = int(resume_state.get("phase_train_loss_wait", 0))
            phase_validation_wait = int(resume_state.get("phase_validation_wait", 0))

            supcon_steps_full_epoch = steps_per_epoch_for_sampler(
                supcon_sampler, supcon_train_dataset, args.batch_size, args.max_train_batches
            )
            epoch = start_supcon_epoch if same_resume_phase else 1
            while True:
                train_dataset.set_epoch(augmentation_epoch_cursor)
                val_dataset.set_epoch(0)
                test_dataset.set_epoch(0)
                supcon_train_dataset.set_epoch(augmentation_epoch_cursor)
                supcon_val_dataset.set_epoch(0)
                start_index = start_supcon_epoch_step * args.batch_size if same_resume_phase and epoch == start_supcon_epoch else 0
                supcon_sampler.set_epoch(augmentation_epoch_cursor)
                supcon_sampler.set_start_index(start_index)
                supcon_steps_per_epoch = max(
                    0,
                    supcon_steps_full_epoch - (start_supcon_epoch_step if same_resume_phase and epoch == start_supcon_epoch else 0),
                )
                epoch_steps_done = start_supcon_epoch_step if same_resume_phase and epoch == start_supcon_epoch else 0
                epoch_samples_seen = 0
                epoch_loss_sum = 0.0
                validation_index = start_supcon_validation_index if same_resume_phase and epoch == start_supcon_epoch else 0
                phase_stopped = False
                progress = tqdm(total=supcon_steps_per_epoch, leave=False)

                while epoch_steps_done < supcon_steps_full_epoch:
                    step_start_index = epoch_steps_done * args.batch_size
                    supcon_sampler.set_epoch(augmentation_epoch_cursor)
                    supcon_sampler.set_start_index(step_start_index)
                    supcon_iterator = iter(limited_batches(supcon_train_loader, args.max_train_batches))
                    supcon_step_checkpoint_payload = {
                        "history": history,
                        "best_val_loss": best_val_loss,
                        "best_val_acc": best_val_acc,
                        "best_val_raw_acc": best_val_raw_acc,
                        "best_classifier_state": best_classifier_state,
                        "supcon_best_state": supcon_best_state,
                        "supcon_best_loss": supcon_best_loss,
                        "supcon_best_epoch": supcon_best_epoch,
                        "supcon_wait": supcon_wait,
                        "phase_best_state": phase_best_state,
                        "augmentation_epoch_cursor": augmentation_epoch_cursor,
                        "phase_train_loss_best": phase_train_loss_best,
                        "phase_train_loss_wait": phase_train_loss_wait,
                        "phase_validation_wait": phase_validation_wait,
                    }
                    supcon_step_resume_payload = {
                        "phase_index": supcon_phase_index,
                        "phase_name": supcon_phase.name,
                        "validation_index": validation_index,
                        "phase_best_loss": phase_best_loss,
                        "phase_best_epoch": phase_best_epoch,
                        "phase_wait": phase_wait,
                        "phase_train_loss_best": phase_train_loss_best,
                        "phase_train_loss_wait": phase_train_loss_wait,
                        "phase_validation_wait": phase_validation_wait,
                    }
                    window_train_loss, steps_done, window_samples = train_supcon_steps(
                        model=model,
                        batch_iterator=supcon_iterator,
                        step_limit=supcon_steps_full_epoch - epoch_steps_done,
                        criterion=supcon_loss,
                        optimizer=supcon_optimizer,
                        scaler=supcon_scaler,
                        scheduler=supcon_scheduler,
                        device=device,
                        backbone_modules=backbone_modules,
                        epoch=epoch,
                        epoch_step_offset=epoch_steps_done,
                        log_path=log_path,
                        log_every_steps=args.log_every_steps,
                        train_progress=train_progress,
                        step_checkpoint_path=step_checkpoint_path,
                        class_names=train_dataset.classes,
                        class_to_idx=train_dataset.class_to_idx,
                        args=args,
                        phase_index=supcon_phase_index,
                        phase_name=supcon_phase.name,
                        step_checkpoint_payload=supcon_step_checkpoint_payload,
                        step_resume_payload=supcon_step_resume_payload,
                    )
                    del supcon_iterator
                    if steps_done == 0:
                        break

                    epoch_steps_done += steps_done
                    epoch_samples_seen += window_samples
                    epoch_loss_sum += window_train_loss * window_samples
                    validation_index += 1
                    progress.update(steps_done)
                    last_supcon_metrics = supcon_step_resume_payload.get("last_batch_contrastive_metrics", {})
                    progress.set_postfix(
                        loss=f"{supcon_step_resume_payload.get('last_batch_loss', window_train_loss):.4f}",
                        margin=f"{last_supcon_metrics.get('positive_negative_cosine_margin', 0.0):.4f}",
                    )
                    release_training_memory(device, supcon_train_loader)

                    phase_train_loss_best = float(
                        supcon_step_resume_payload.get("phase_train_loss_best", phase_train_loss_best)
                    )
                    phase_train_loss_wait = int(
                        supcon_step_resume_payload.get("phase_train_loss_wait", phase_train_loss_wait)
                    )
                    phase_validation_wait = int(
                        supcon_step_resume_payload.get("phase_validation_wait", phase_validation_wait)
                    )
                    validation_trigger_reason = supcon_step_resume_payload.get("validation_reason")
                    if validation_trigger_reason is None and epoch_steps_done < supcon_steps_full_epoch:
                        validation_trigger_reason = "epoch_boundary"

                    if args.train_only:
                        if validation_trigger_reason == "train_loss_plateau":
                            log_json_event(
                                log_path,
                                {
                                    "event": "train_only_phase_stopped",
                                    "stage": "supcon",
                                    "phase_index": supcon_phase_index,
                                    "phase_name": supcon_phase.name,
                                    "stopped_at_epoch_step": epoch_steps_done,
                                    "global_train_step": train_progress["global_train_step"],
                                    "reason": validation_trigger_reason,
                                    "phase_train_loss_best": phase_train_loss_best,
                                    "phase_train_loss_wait": phase_train_loss_wait,
                                },
                            )
                            phase_stopped = True
                            break
                        continue

                    log_json_event(
                        log_path,
                        {
                            "event": "validation_started",
                            "stage": "supcon",
                            "phase_index": supcon_phase_index,
                            "phase_name": supcon_phase.name,
                            "epoch_in_phase": epoch,
                            "validation_index": validation_index,
                            "epoch_step": epoch_steps_done,
                            "global_train_step": train_progress["global_train_step"],
                            "eval_batches": val_eval_batches,
                            "validation_reason": validation_trigger_reason,
                        },
                    )
                    val_loss, val_per_class_metrics = evaluate_supcon(
                        model=model,
                        loader=supcon_val_loader,
                        criterion=supcon_loss,
                        device=device,
                        max_batches=args.max_eval_batches,
                        log_path=log_path,
                        log_every_eval_steps=args.log_eval_every_steps,
                        stage="supcon",
                        split="val",
                        eval_context={
                            "phase_index": supcon_phase_index,
                            "phase_name": supcon_phase.name,
                            "epoch_in_phase": epoch,
                            "validation_index": validation_index,
                            "epoch_step": epoch_steps_done,
                            "global_train_step": train_progress["global_train_step"],
                        },
                        args=args,
                    )
                    release_training_memory(device, supcon_val_loader)
                    log_json_event(
                        log_path,
                        {
                            "event": "validation_finished",
                            "stage": "supcon",
                            "phase_index": supcon_phase_index,
                            "phase_name": supcon_phase.name,
                            "epoch_in_phase": epoch,
                            "validation_index": validation_index,
                            "epoch_step": epoch_steps_done,
                            "global_train_step": train_progress["global_train_step"],
                            "eval_batches": val_eval_batches,
                            "val_loss": val_loss,
                            **val_per_class_metrics["contrastive_metrics"],
                            "phase_train_loss_best": phase_train_loss_best,
                            "phase_train_loss_wait": phase_train_loss_wait,
                            "phase_validation_wait": phase_validation_wait,
                            "validation_reason": validation_trigger_reason,
                        },
                    )
                    if val_loss < phase_best_loss - args.early_stopping_min_delta:
                        phase_best_loss = val_loss
                        phase_best_epoch = epoch
                        phase_best_state = cpu_state_dict(model)
                        phase_wait = 0
                        supcon_phase_dir = phase_artifact_dir(output_dir, supcon_phase.name)
                        supcon_phase_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "model_state_dict": phase_best_state,
                                "class_names": train_dataset.classes,
                                "class_to_idx": train_dataset.class_to_idx,
                                "args": vars(args),
                                "phase_index": supcon_phase_index,
                                "phase_name": supcon_phase.name,
                                "val_loss": val_loss,
                            },
                            supcon_phase_dir / "best.pt",
                        )
                    else:
                        phase_wait += 1

                    if val_loss < supcon_best_loss - args.early_stopping_min_delta:
                        supcon_best_loss = val_loss
                        supcon_best_epoch = epoch
                        supcon_best_state = cpu_state_dict(model)
                        supcon_wait = 0
                        save_supcon_best_checkpoint(
                            output_dir / "best.pt",
                            model_state_dict=supcon_best_state,
                            class_names=train_dataset.classes,
                            class_to_idx=train_dataset.class_to_idx,
                            args=args,
                            history=history,
                            train_progress=train_progress,
                            best_val_loss=best_val_loss,
                            best_val_acc=best_val_acc,
                            best_val_raw_acc=best_val_raw_acc,
                            best_classifier_state=best_classifier_state,
                            supcon_best_state=supcon_best_state,
                            supcon_best_loss=supcon_best_loss,
                            supcon_best_epoch=supcon_best_epoch,
                            supcon_wait=supcon_wait,
                            augmentation_epoch_cursor=augmentation_epoch_cursor,
                        )
                        save_supcon_best_checkpoint(
                            output_dir / "supcon_best.pt",
                            model_state_dict=supcon_best_state,
                            class_names=train_dataset.classes,
                            class_to_idx=train_dataset.class_to_idx,
                            args=args,
                            history=history,
                            train_progress=train_progress,
                            best_val_loss=best_val_loss,
                            best_val_acc=best_val_acc,
                            best_val_raw_acc=best_val_raw_acc,
                            best_classifier_state=best_classifier_state,
                            supcon_best_state=supcon_best_state,
                            supcon_best_loss=supcon_best_loss,
                            supcon_best_epoch=supcon_best_epoch,
                            supcon_wait=supcon_wait,
                            augmentation_epoch_cursor=augmentation_epoch_cursor,
                        )
                    else:
                        supcon_wait += 1

                    row = {
                        "stage": "supcon",
                        "phase_index": supcon_phase_index,
                        "phase_name": supcon_phase.name,
                        "epoch_in_phase": epoch,
                        "validation_index": validation_index,
                        "epoch_step": epoch_steps_done,
                        "global_train_step": train_progress["global_train_step"],
                        "unfrozen_backbone_modules": supcon_phase.unfrozen_backbone_modules,
                        "newly_unfrozen_tail_modules": thawed_supcon[-args.unfreeze_chunk_size :],
                        "trainable_params": supcon_trainable_params,
                        "total_params": total_params,
                        "batch_train_loss": supcon_step_resume_payload.get("last_batch_loss", window_train_loss),
                        **{
                            f"batch_{key}": value
                            for key, value in (
                                supcon_step_resume_payload.get("last_batch_contrastive_metrics", {}) or {}
                            ).items()
                        },
                        "val_loss": val_loss,
                        **{
                            f"val_{key}": value
                            for key, value in val_per_class_metrics["contrastive_metrics"].items()
                        },
                        "phase_best_val_loss": phase_best_loss,
                        "supcon_best_val_loss": supcon_best_loss,
                        "phase_train_loss_best": phase_train_loss_best,
                        "phase_train_loss_wait": phase_train_loss_wait,
                        "phase_validation_wait": phase_validation_wait,
                        "validation_reason": validation_trigger_reason,
                        "checks_without_improvement": phase_wait,
                        "patience_limit": args.supcon_early_stopping_patience,
                        "patience_unit": "train_steps",
                        "validation_interval_train_steps": None,
                        "train_loss_validation_patience": args.train_loss_validation_patience,
                        "validation_patience": args.validation_patience,
                        "supcon_head_lr": args.supcon_head_lr,
                        "supcon_backbone_lr": args.supcon_backbone_lr,
                    }
                    history.append(row)
                    log_json_event(log_path, row)
                    phase_train_loss_wait = 0
                    phase_validation_wait = 0
                    supcon_step_checkpoint_payload["phase_train_loss_wait"] = 0
                    supcon_step_checkpoint_payload["phase_validation_wait"] = 0
                    supcon_step_checkpoint_payload["validation_due"] = False
                    supcon_step_checkpoint_payload["validation_reason"] = None
                    supcon_step_resume_payload["phase_train_loss_wait"] = 0
                    supcon_step_resume_payload["phase_validation_wait"] = 0
                    supcon_step_resume_payload["validation_due"] = False
                    supcon_step_resume_payload["validation_reason"] = None
                    save_step_checkpoint(
                        path=step_checkpoint_path,
                        model=model,
                        class_names=train_dataset.classes,
                        class_to_idx=train_dataset.class_to_idx,
                        args=args,
                        train_progress=train_progress,
                        stage="supcon",
                        epoch=epoch,
                        epoch_step=epoch_steps_done,
                        optimizer=supcon_optimizer,
                        scheduler=supcon_scheduler,
                        scaler=supcon_scaler,
                        phase_index=supcon_phase_index,
                        phase_name=supcon_phase.name,
                        extra_payload=supcon_step_checkpoint_payload,
                        extra_resume=supcon_step_resume_payload,
                    )
                    save_training_checkpoint(
                        checkpoint_path,
                        {
                            "model_state_dict": cpu_state_dict(model),
                            "class_names": train_dataset.classes,
                            "class_to_idx": train_dataset.class_to_idx,
                            "args": vars(args),
                            "history": history,
                            "train_progress": train_progress,
                            "best_val_loss": best_val_loss,
                            "best_val_acc": best_val_acc,
                            "best_val_raw_acc": best_val_raw_acc,
                            "best_classifier_state": best_classifier_state,
                            "supcon_best_state": supcon_best_state,
                            "supcon_best_loss": supcon_best_loss,
                            "supcon_best_epoch": supcon_best_epoch,
                            "supcon_wait": supcon_wait,
                            "phase_best_state": phase_best_state,
                            "augmentation_epoch_cursor": augmentation_epoch_cursor,
                            "phase_train_loss_best": phase_train_loss_best,
                            "phase_train_loss_wait": phase_train_loss_wait,
                            "phase_validation_wait": phase_validation_wait,
                            "resume": {
                                "stage": "supcon",
                                "phase_index": supcon_phase_index,
                                "phase_name": supcon_phase.name,
                                "epoch": epoch,
                                "epoch_step_completed": epoch_steps_done,
                                "validation_index": validation_index,
                                "phase_best_loss": phase_best_loss,
                                "phase_best_epoch": phase_best_epoch,
                                "phase_wait": phase_wait,
                                "phase_train_loss_best": phase_train_loss_best,
                                "phase_train_loss_wait": phase_train_loss_wait,
                                "phase_validation_wait": phase_validation_wait,
                                "optimizer_state_dict": supcon_optimizer.state_dict(),
                                "scheduler_state_dict": supcon_scheduler.state_dict(),
                                "scaler_state_dict": supcon_scaler.state_dict(),
                            },
                        },
                    )
                    if phase_wait >= args.supcon_early_stopping_patience:
                        log_json_event(
                            log_path,
                            {
                                "event": "supcon_phase_stopped_early",
                                "phase_index": supcon_phase_index,
                                "phase_name": supcon_phase.name,
                                "best_epoch_in_phase": phase_best_epoch,
                                "phase_best_val_loss": phase_best_loss,
                                "stopped_at_epoch_step": epoch_steps_done,
                                "global_train_step": train_progress["global_train_step"],
                            },
                        )
                        phase_stopped = True
                        break
                    epoch += 1
                progress.close()
                supcon_sampler.set_start_index(0)
                start_supcon_epoch_step = 0
                start_supcon_validation_index = 0
                if phase_stopped:
                    break
                augmentation_epoch_cursor += 1
                release_training_memory(device, supcon_train_loader, supcon_val_loader)

            model.load_state_dict(phase_best_state)
            release_training_memory(device, supcon_train_loader, supcon_val_loader)
            save_training_checkpoint(
                checkpoint_path,
                {
                    "model_state_dict": cpu_state_dict(model),
                    "class_names": train_dataset.classes,
                    "class_to_idx": train_dataset.class_to_idx,
                    "args": vars(args),
                    "history": history,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "best_val_raw_acc": best_val_raw_acc,
                    "best_classifier_state": best_classifier_state,
                    "supcon_best_state": supcon_best_state,
                    "supcon_best_loss": supcon_best_loss,
                    "supcon_best_epoch": supcon_best_epoch,
                    "supcon_wait": supcon_wait,
                    "phase_best_state": phase_best_state,
                    "augmentation_epoch_cursor": augmentation_epoch_cursor,
                    "train_progress": train_progress,
                    "resume": {
                        "stage": "supcon",
                        "phase_index": supcon_phase_index + 1,
                        "phase_name": None,
                        "epoch": 1,
                        "epoch_step_completed": 0,
                        "validation_index": 0,
                    },
                },
            )
            maybe_run_phase_visualizations(
                checkpoint_path=checkpoint_path,
                dataset_root=args.dataset_root,
                output_dir=phase_artifact_dir(output_dir, supcon_phase.name),
                phase_label=f"{supcon_phase.name}_phase",
                args=args,
                log_path=log_path,
                reason="supcon_phase_complete",
            )
            start_supcon_epoch = 1
            start_supcon_epoch_step = 0
            start_supcon_validation_index = 0
            resume_supcon_phase_index = supcon_phase_index + 1
            resume_state = {
                "stage": "supcon",
                "phase_index": supcon_phase_index + 1,
                "phase_name": None,
                "epoch": 1,
                "epoch_step_completed": 0,
                "validation_index": 0,
            }
        model.load_state_dict(supcon_best_state)
        release_training_memory(device, supcon_train_loader, supcon_val_loader)
    else:
        start_supcon_epoch_step = 0
        start_supcon_validation_index = 0

    classifier_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    phases = build_classifier_phase_plan(len(backbone_modules), args)
    resume_phase_index = resolve_phase_start_index(args, phases, resume_state)
    start_phase_epoch = int(resume_state.get("epoch", 1)) if resume_state.get("stage") == "classifier" and not resume_from_best_phase else 1
    start_phase_epoch_step = int(resume_state.get("epoch_step_completed", 0)) if resume_state.get("stage") == "classifier" and not resume_from_best_phase else 0
    start_phase_validation_index = int(resume_state.get("validation_index", 0)) if resume_state.get("stage") == "classifier" and not resume_from_best_phase else 0
    checkpoint_phase_name = phases[resume_phase_index - 1].name if 1 <= resume_phase_index <= len(phases) else None
    save_training_checkpoint(
        checkpoint_path,
        {
            "model_state_dict": cpu_state_dict(model),
            "class_names": train_dataset.classes,
            "class_to_idx": train_dataset.class_to_idx,
            "args": vars(args),
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_val_raw_acc": best_val_raw_acc,
            "best_classifier_state": best_classifier_state,
            "supcon_best_state": supcon_best_state,
            "supcon_best_loss": supcon_best_loss,
            "supcon_best_epoch": supcon_best_epoch,
            "supcon_wait": supcon_wait,
            "augmentation_epoch_cursor": augmentation_epoch_cursor,
            "train_progress": train_progress,
            "resume": {
                "stage": "classifier",
                "phase_index": resume_phase_index,
                "phase_name": checkpoint_phase_name,
                "epoch": start_phase_epoch,
                "epoch_step_completed": start_phase_epoch_step,
                "validation_index": start_phase_validation_index,
            },
        },
    )

    for phase_index, phase in enumerate(phases, start=1):
        if phase_index < resume_phase_index:
            continue
        global_best_loss_before_phase = best_val_loss
        global_best_raw_acc_before_phase = best_val_raw_acc
        thawed = set_trainability_for_classifier(model, backbone_modules, phase.unfrozen_backbone_modules)
        phase_head_lr, phase_backbone_lr = classifier_phase_learning_rates(
            args,
            unfrozen_backbone_modules=phase.unfrozen_backbone_modules,
            total_backbone_modules=len(backbone_modules),
        )
        optimizer = build_classifier_optimizer(model, args, head_lr=phase_head_lr, backbone_lr=phase_backbone_lr)
        scaler = build_grad_scaler(device, args)
        scheduler = build_scheduler(
            base_optimizer_for_scheduler(optimizer),
            steps_per_epoch=steps_per_epoch_for_sampler(train_sampler, train_dataset, args.batch_size, args.max_train_batches),
            warmup_epochs=args.warmup_epochs,
            warmup_steps=args.warmup_steps,
        )
        _, trainable_params = parameter_counts(model)
        same_resume_phase = (
            resume_checkpoint is not None
            and
            resume_state.get("stage") == "classifier"
            and phase_index == resume_phase_index
            and resume_state.get("phase_name") == phase.name
            and "optimizer_state_dict" in resume_state
            and "scheduler_state_dict" in resume_state
        )
        if resume_checkpoint is not None and resume_state.get("stage") == "classifier" and phase_index == resume_phase_index and not same_resume_phase:
            log_json_event(
                log_path,
                {
                    "event": "resume_fallback",
                    "stage": "classifier",
                    "phase_index": phase_index,
                    "reason": "missing_optimizer_or_scheduler_state",
                },
            )
        if same_resume_phase:
            try:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
                set_scheduler_base_lrs(
                    scheduler,
                    [
                        phase_head_lr,
                        *([phase_backbone_lr] if len(scheduler.optimizer.param_groups) > 1 else []),
                    ],
                )
                if "scaler_state_dict" in resume_state:
                    scaler.load_state_dict(resume_state["scaler_state_dict"])
                phase_best_loss = float(resume_state.get("phase_best_loss", float("inf")))
                phase_best_acc = float(resume_state.get("phase_best_acc", -1.0))
                phase_best_raw_acc = float(
                    resume_state.get("phase_best_raw_acc", resume_state.get("phase_best_acc", -1.0))
                )
                phase_best_epoch = int(resume_state.get("phase_best_epoch", 0))
                phase_best_state = resume_checkpoint.get("phase_best_state", cpu_state_dict(model))
                phase_wait = int(resume_state.get("phase_wait", 0))
            except ValueError as exc:
                same_resume_phase = False
                log_json_event(
                    log_path,
                    {
                        "event": "resume_fallback",
                        "stage": "classifier",
                        "phase_index": phase_index,
                        "phase_name": phase.name,
                        "reason": f"incompatible_optimizer_state: {exc}",
                    },
                )
        if not same_resume_phase:
            phase_best_loss = float("inf")
            phase_best_acc = -1.0
            phase_best_raw_acc = -1.0
            phase_best_epoch = 0
            phase_best_state = cpu_state_dict(model)
            phase_wait = 0
        phase_train_loss_best = float(
            resume_state.get(
                "phase_train_loss_best",
                infer_phase_train_loss_baseline(history, "classifier", phase.name),
            )
        )
        phase_train_loss_wait = int(resume_state.get("phase_train_loss_wait", 0))
        phase_validation_wait = int(resume_state.get("phase_validation_wait", 0))
        phase_patience = (
            args.head_early_stopping_patience
            if phase_index == 1 and args.classifier_train_mode == "progressive"
            else args.stage_early_stopping_patience
        )

        epoch = start_phase_epoch if same_resume_phase else 1
        while True:
            train_dataset.set_epoch(augmentation_epoch_cursor)
            val_dataset.set_epoch(0)
            test_dataset.set_epoch(0)
            start_index = start_phase_epoch_step * args.batch_size if same_resume_phase and epoch == start_phase_epoch else 0
            train_sampler.set_epoch(augmentation_epoch_cursor)
            train_sampler.set_start_index(start_index)
            phase_steps_full_epoch = steps_per_epoch_for_sampler(train_sampler, train_dataset, args.batch_size, args.max_train_batches)
            phase_steps_per_epoch = max(
                0,
                phase_steps_full_epoch - (start_phase_epoch_step if same_resume_phase and epoch == start_phase_epoch else 0),
            )
            epoch_steps_done = start_phase_epoch_step if same_resume_phase and epoch == start_phase_epoch else 0
            epoch_samples_seen = 0
            epoch_loss_sum = 0.0
            epoch_correct_sum = 0.0
            validation_index = start_phase_validation_index if same_resume_phase and epoch == start_phase_epoch else 0
            phase_stopped = False
            progress = tqdm(total=phase_steps_per_epoch, leave=False)

            while epoch_steps_done < phase_steps_full_epoch:
                step_start_index = epoch_steps_done * args.batch_size
                train_sampler.set_epoch(augmentation_epoch_cursor)
                train_sampler.set_start_index(step_start_index)
                phase_iterator = iter(limited_batches(train_loader, args.max_train_batches))
                classifier_step_checkpoint_payload = {
                    "history": history,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "best_val_raw_acc": best_val_raw_acc,
                    "best_classifier_state": best_classifier_state,
                    "supcon_best_state": supcon_best_state,
                    "supcon_best_loss": supcon_best_loss,
                    "supcon_best_epoch": supcon_best_epoch,
                    "supcon_wait": supcon_wait,
                    "phase_best_state": phase_best_state,
                    "augmentation_epoch_cursor": augmentation_epoch_cursor,
                    "phase_train_loss_best": phase_train_loss_best,
                    "phase_train_loss_wait": phase_train_loss_wait,
                    "phase_validation_wait": phase_validation_wait,
                }
                classifier_step_resume_payload = {
                    "validation_index": validation_index,
                    "phase_best_loss": phase_best_loss,
                    "phase_best_acc": phase_best_acc,
                    "phase_best_raw_acc": phase_best_raw_acc,
                    "phase_best_epoch": phase_best_epoch,
                    "phase_wait": phase_wait,
                    "phase_train_loss_best": phase_train_loss_best,
                    "phase_train_loss_wait": phase_train_loss_wait,
                    "phase_validation_wait": phase_validation_wait,
                }
                window_train_loss, window_train_acc, steps_done, window_samples = train_classifier_steps(
                    model=model,
                    batch_iterator=phase_iterator,
                    step_limit=phase_steps_full_epoch - epoch_steps_done,
                    criterion=classifier_criterion,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    device=device,
                    backbone_modules=backbone_modules,
                    epoch=epoch,
                    epoch_step_offset=epoch_steps_done,
                    phase_index=phase_index,
                    phase_name=phase.name,
                    log_path=log_path,
                    log_every_steps=args.log_every_steps,
                    train_progress=train_progress,
                    step_checkpoint_path=step_checkpoint_path,
                    class_names=train_dataset.classes,
                    class_to_idx=train_dataset.class_to_idx,
                    args=args,
                    step_checkpoint_payload=classifier_step_checkpoint_payload,
                    step_resume_payload=classifier_step_resume_payload,
                )
                del phase_iterator
                if steps_done == 0:
                    break

                epoch_steps_done += steps_done
                epoch_samples_seen += window_samples
                epoch_loss_sum += window_train_loss * window_samples
                epoch_correct_sum += window_train_acc * window_samples
                validation_index += 1
                progress.update(steps_done)
                progress.set_postfix(
                    loss=f"{classifier_step_resume_payload.get('last_batch_loss', window_train_loss):.4f}",
                    acc=f"{classifier_step_resume_payload.get('last_batch_acc', window_train_acc):.4f}",
                )
                release_training_memory(device, train_loader)

                phase_train_loss_best = float(classifier_step_resume_payload.get("phase_train_loss_best", phase_train_loss_best))
                phase_train_loss_wait = int(classifier_step_resume_payload.get("phase_train_loss_wait", phase_train_loss_wait))
                phase_validation_wait = int(classifier_step_resume_payload.get("phase_validation_wait", phase_validation_wait))
                validation_trigger_reason = classifier_step_resume_payload.get("validation_reason")
                if validation_trigger_reason is None and epoch_steps_done < phase_steps_full_epoch:
                    validation_trigger_reason = "epoch_boundary"

                if args.train_only:
                    if validation_trigger_reason == "train_loss_plateau":
                        log_json_event(
                            log_path,
                            {
                                "event": "train_only_phase_stopped",
                                "stage": "classifier",
                                "phase_index": phase_index,
                                "phase_name": phase.name,
                                "stopped_at_epoch_step": epoch_steps_done,
                                "global_train_step": train_progress["global_train_step"],
                                "reason": validation_trigger_reason,
                                "phase_train_loss_best": phase_train_loss_best,
                                "phase_train_loss_wait": phase_train_loss_wait,
                            },
                        )
                        phase_stopped = True
                        break
                    continue

                log_json_event(
                    log_path,
                    {
                        "event": "validation_started",
                        "stage": "classifier",
                        "phase_index": phase_index,
                        "phase_name": phase.name,
                        "epoch_in_phase": epoch,
                        "validation_index": validation_index,
                        "epoch_step": epoch_steps_done,
                        "global_train_step": train_progress["global_train_step"],
                        "eval_batches": val_eval_batches,
                        "validation_reason": validation_trigger_reason,
                    },
                )
                val_metrics = evaluate_classifier(
                    model=model,
                    loader=val_loader,
                    criterion=classifier_criterion,
                    device=device,
                    max_batches=args.max_eval_batches,
                    log_path=log_path,
                    log_every_eval_steps=args.log_eval_every_steps,
                    stage="classifier",
                    split="val",
                    eval_context={
                        "phase_index": phase_index,
                        "phase_name": phase.name,
                        "epoch_in_phase": epoch,
                        "validation_index": validation_index,
                        "epoch_step": epoch_steps_done,
                        "global_train_step": train_progress["global_train_step"],
                    },
                    args=args,
                )
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["accuracy"]
                val_per_class_accuracy = val_metrics["per_class_accuracy"]
                val_per_class_avg_confidence = val_metrics["per_class_avg_confidence"]
                val_raw_acc = val_metrics["raw_accuracy"]

                release_training_memory(device, val_loader)
                val_improved = improved_classifier_selection_metric(
                    args.classifier_early_stopping_metric,
                    phase_best_loss,
                    phase_best_raw_acc,
                    val_loss,
                    val_raw_acc,
                    args.early_stopping_min_delta,
                )
                log_json_event(
                    log_path,
                    {
                        "event": "validation_finished",
                        "stage": "classifier",
                        "phase_index": phase_index,
                        "phase_name": phase.name,
                        "epoch_in_phase": epoch,
                            "validation_index": validation_index,
                            "epoch_step": epoch_steps_done,
                            "global_train_step": train_progress["global_train_step"],
                            "eval_batches": val_eval_batches,
                            "val_loss": val_loss,
                            "accuracy": val_acc,
                            "per_class_accuracy": val_per_class_accuracy,
                            "per_class_avg_confidence": val_per_class_avg_confidence,
                            "val_macro_precision": val_metrics["macro_precision"],
                            "val_macro_recall": val_metrics["macro_recall"],
                            "val_macro_f1": val_metrics["macro_f1"],
                            "val_weighted_precision": val_metrics["weighted_precision"],
                            "val_weighted_recall": val_metrics["weighted_recall"],
                            "val_weighted_f1": val_metrics["weighted_f1"],
                            "val_ece": val_metrics["calibration"]["expected_calibration_error"],
                            "val_brier_score": val_metrics["calibration"]["brier_score"],
                            "val_nll": val_metrics["calibration"]["negative_log_likelihood"],
                            "phase_train_loss_best": phase_train_loss_best,
                        "phase_train_loss_wait": phase_train_loss_wait,
                        "phase_validation_wait": phase_validation_wait,
                        "validation_reason": validation_trigger_reason,
                    },
                )
                if val_improved:
                    phase_best_loss = val_loss
                    phase_best_acc = val_acc
                    phase_best_raw_acc = val_raw_acc
                    phase_best_epoch = epoch
                    phase_best_state = cpu_state_dict(model)
                    phase_wait = 0
                    classifier_phase_dir = phase_artifact_dir(output_dir, phase.name)
                    classifier_phase_dir.mkdir(parents=True, exist_ok=True)

                    if "confusion_matrix" in val_metrics:
                        save_confusion_matrix_plot(
                            classifier_phase_dir / "best_confusion_matrix.png",
                            np.asarray(val_metrics["confusion_matrix"], dtype=np.int64),
                            train_dataset.classes,
                            f"Phase {phase_index} Best Validation Confusion Matrix",
                        )

                    torch.save(
                        {
                            "model_state_dict": cpu_state_dict(model),
                            "class_names": train_dataset.classes,
                            "class_to_idx": train_dataset.class_to_idx,
                            "args": vars(args),
                            "phase_index": phase_index,
                            "phase_name": phase.name,
                            "val_loss": val_loss,
                            "val_raw_acc": val_raw_acc,
                        },
                        classifier_phase_dir / "best.pt",
                    )
                else:
                    phase_wait += 1

                if improved_classifier_selection_metric(
                    args.classifier_early_stopping_metric,
                    best_val_loss,
                    best_val_raw_acc,
                    val_loss,
                    val_raw_acc,
                    args.early_stopping_min_delta,
                ):
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_val_raw_acc = val_raw_acc
                    best_classifier_state = cpu_state_dict(model)

                row = {
                    "stage": "classifier",
                    "phase_index": phase_index,
                    "phase_name": phase.name,
                    "epoch_in_phase": epoch,
                    "validation_index": validation_index,
                    "epoch_step": epoch_steps_done,
                    "global_train_step": train_progress["global_train_step"],
                    "unfrozen_backbone_modules": phase.unfrozen_backbone_modules,
                    "newly_unfrozen_tail_modules": thawed[-args.unfreeze_chunk_size :],
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "batch_train_loss": classifier_step_resume_payload.get("last_batch_loss", window_train_loss),
                    "batch_train_acc": classifier_step_resume_payload.get("last_batch_acc", window_train_acc),
                    "batch_per_class_accuracy": classifier_step_resume_payload.get(
                        "last_batch_per_class_accuracy"
                    ),
                    "batch_per_class_avg_confidence": classifier_step_resume_payload.get(
                        "last_batch_per_class_avg_confidence"
                    ),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_per_class_accuracy": val_per_class_accuracy,
                    "val_per_class_avg_confidence": val_metrics["per_class_avg_confidence"],
                    "val_macro_precision": val_metrics["macro_precision"],
                    "val_macro_recall": val_metrics["macro_recall"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_weighted_precision": val_metrics["weighted_precision"],
                    "val_weighted_recall": val_metrics["weighted_recall"],
                    "val_weighted_f1": val_metrics["weighted_f1"],
                    "val_ece": val_metrics["calibration"]["expected_calibration_error"],
                    "val_brier_score": val_metrics["calibration"]["brier_score"],
                    "val_nll": val_metrics["calibration"]["negative_log_likelihood"],
                    "phase_best_val_loss": phase_best_loss,
                    "phase_best_val_acc": phase_best_acc,
                    "phase_best_val_accuracy": phase_best_raw_acc,
                    "phase_train_loss_best": phase_train_loss_best,
                    "phase_train_loss_wait": phase_train_loss_wait,
                    "phase_validation_wait": phase_validation_wait,
                    "validation_reason": validation_trigger_reason,
                    "checks_without_improvement": phase_wait,
                    "patience_limit": phase_patience,
                    "patience_unit": "train_steps",
                    "validation_interval_train_steps": None,
                    "train_loss_validation_patience": args.train_loss_validation_patience,
                    "validation_patience": args.validation_patience,
                    "early_stopping_metric": args.classifier_early_stopping_metric,
                    "phase_head_lr": phase_head_lr,
                    "phase_backbone_lr": phase_backbone_lr,
                }
                history.append(row)
                log_json_event(log_path, row)
                phase_train_loss_wait = 0
                phase_validation_wait = 0
                classifier_step_checkpoint_payload["phase_train_loss_wait"] = 0
                classifier_step_checkpoint_payload["phase_validation_wait"] = 0
                classifier_step_checkpoint_payload["validation_due"] = False
                classifier_step_checkpoint_payload["validation_reason"] = None
                classifier_step_resume_payload["phase_train_loss_wait"] = 0
                classifier_step_resume_payload["phase_validation_wait"] = 0
                classifier_step_resume_payload["validation_due"] = False
                classifier_step_resume_payload["validation_reason"] = None
                save_step_checkpoint(
                    path=step_checkpoint_path,
                    model=model,
                    class_names=train_dataset.classes,
                    class_to_idx=train_dataset.class_to_idx,
                    args=args,
                    train_progress=train_progress,
                    stage="classifier",
                    epoch=epoch,
                    epoch_step=epoch_steps_done,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    phase_index=phase_index,
                    phase_name=phase.name,
                    extra_payload=classifier_step_checkpoint_payload,
                    extra_resume=classifier_step_resume_payload,
                )
                save_training_checkpoint(
                    checkpoint_path,
                    {
                        "model_state_dict": cpu_state_dict(model),
                        "class_names": train_dataset.classes,
                        "class_to_idx": train_dataset.class_to_idx,
                        "args": vars(args),
                        "history": history,
                        "best_val_loss": best_val_loss,
                        "best_val_acc": best_val_acc,
                        "best_val_raw_acc": best_val_raw_acc,
                        "best_classifier_state": best_classifier_state,
                        "supcon_best_state": supcon_best_state,
                        "supcon_best_loss": supcon_best_loss,
                        "supcon_best_epoch": supcon_best_epoch,
                        "supcon_wait": supcon_wait,
                        "augmentation_epoch_cursor": augmentation_epoch_cursor,
                        "phase_train_loss_best": phase_train_loss_best,
                        "phase_train_loss_wait": phase_train_loss_wait,
                        "phase_validation_wait": phase_validation_wait,
                        "train_progress": train_progress,
                        "phase_best_state": phase_best_state,
                        "resume": {
                            "stage": "classifier",
                            "phase_index": phase_index,
                            "phase_name": phase.name,
                            "epoch": epoch,
                            "epoch_step_completed": epoch_steps_done,
                            "validation_index": validation_index,
                            "phase_best_loss": phase_best_loss,
                            "phase_best_acc": phase_best_acc,
                            "phase_best_raw_acc": phase_best_raw_acc,
                            "phase_best_epoch": phase_best_epoch,
                            "phase_wait": phase_wait,
                            "phase_train_loss_best": phase_train_loss_best,
                            "phase_train_loss_wait": phase_train_loss_wait,
                            "phase_validation_wait": phase_validation_wait,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                        },
                    },
                )

                if phase_wait >= phase_patience:
                    event = {
                        "stage": "classifier",
                        "phase_index": phase_index,
                        "phase_name": phase.name,
                        "stopped_early": True,
                        "best_epoch_in_phase": phase_best_epoch,
                        "phase_best_val_loss": phase_best_loss,
                        "phase_best_val_acc": phase_best_acc,
                        "phase_best_val_accuracy": phase_best_raw_acc,
                        "stopped_at_epoch_step": epoch_steps_done,
                        "global_train_step": train_progress["global_train_step"],
                    }
                    log_json_event(log_path, event)
                    phase_stopped = True
                    break
                epoch += 1
            progress.close()
            train_sampler.set_start_index(0)
            start_phase_epoch_step = 0
            start_phase_validation_index = 0
            if phase_stopped:
                break
            augmentation_epoch_cursor += 1
            release_training_memory(device, train_loader, val_loader)

        phase_improved_global_best = improved_classifier_selection_metric(
            args.classifier_early_stopping_metric,
            global_best_loss_before_phase,
            global_best_raw_acc_before_phase,
            phase_best_loss,
            phase_best_raw_acc,
            args.early_stopping_min_delta,
        )
        log_json_event(
            log_path,
                {
                    "event": "phase_global_best_comparison",
                    "phase_index": phase_index,
                    "phase_name": phase.name,
                    "classifier_metric": args.classifier_early_stopping_metric,
                    "phase_improved_global_best": phase_improved_global_best,
                    "reject_current_enabled": args.reject_current_phase_on_global_miss,
                    "phase_best_val_loss": phase_best_loss,
                    "phase_best_val_accuracy": phase_best_raw_acc,
                    "global_best_val_loss_before_phase": global_best_loss_before_phase,
                    "global_best_val_accuracy_before_phase": global_best_raw_acc_before_phase,
                    "min_delta": args.early_stopping_min_delta,
                },
            )
        reject_current_phase_for_next = (
            args.classifier_train_mode == "progressive"
            and args.reject_current_phase_on_global_miss
            and not phase_improved_global_best
        )
        if reject_current_phase_for_next:
            log_json_event(
                log_path,
                {
                    "event": "phase_rejected_for_next_initialization",
                    "reason": "phase_failed_to_beat_global_best",
                    "phase_index": phase_index,
                    "phase_name": phase.name,
                    "classifier_metric": args.classifier_early_stopping_metric,
                    "phase_best_val_loss": phase_best_loss,
                    "phase_best_val_accuracy": phase_best_raw_acc,
                    "global_best_val_loss_before_phase": global_best_loss_before_phase,
                    "global_best_val_accuracy_before_phase": global_best_raw_acc_before_phase,
                    "min_delta": args.early_stopping_min_delta,
                },
            )
        next_phase_state = best_classifier_state if reject_current_phase_for_next else phase_best_state
        next_phase_init_source = "global_best" if reject_current_phase_for_next else "phase_best"
        model.load_state_dict(next_phase_state)
        release_training_memory(device, train_loader, val_loader)
        save_training_checkpoint(
            checkpoint_path,
            {
                "model_state_dict": cpu_state_dict(model),
                "class_names": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx,
                "args": vars(args),
                "history": history,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "best_val_raw_acc": best_val_raw_acc,
                "best_classifier_state": best_classifier_state,
                "supcon_best_state": supcon_best_state,
                "supcon_best_loss": supcon_best_loss,
                "supcon_best_epoch": supcon_best_epoch,
                "supcon_wait": supcon_wait,
                "augmentation_epoch_cursor": augmentation_epoch_cursor,
                "train_progress": train_progress,
                "phase_best_state": phase_best_state,
                "resume": {
                    "stage": "classifier",
                    "phase_index": phase_index + 1,
                    "phase_name": None,
                    "epoch": 1,
                    "epoch_step_completed": 0,
                    "validation_index": 0,
                    "next_phase_init_source": next_phase_init_source,
                },
            },
        )
        maybe_run_phase_visualizations(
            checkpoint_path=checkpoint_path,
            dataset_root=args.dataset_root,
            output_dir=phase_artifact_dir(output_dir, phase.name),
            phase_label=f"{phase.name}_phase",
            args=args,
            log_path=log_path,
            reason="classifier_phase_complete",
        )
        log_json_event(
            log_path,
            {
                "event": "next_phase_initialization_selected",
                "phase_index": phase_index,
                "phase_name": phase.name,
                "next_phase_init_source": next_phase_init_source,
                "phase_improved_global_best": phase_improved_global_best,
            },
        )
        start_phase_epoch = 1
        start_phase_epoch_step = 0
        start_phase_validation_index = 0
        resume_phase_index = phase_index + 1
        resume_state = {
            "stage": "classifier",
            "phase_index": phase_index + 1,
            "phase_name": None,
            "epoch": 1,
            "epoch_step_completed": 0,
            "validation_index": 0,
        }

    if args.train_only:
        final_checkpoint = {
            "model_state_dict": cpu_state_dict(model),
            "class_names": train_dataset.classes,
            "class_to_idx": train_dataset.class_to_idx,
            "args": vars(args),
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_val_raw_acc": best_val_raw_acc,
            "best_classifier_state": cpu_state_dict(model),
            "supcon_best_state": supcon_best_state,
            "supcon_best_loss": supcon_best_loss,
            "supcon_best_epoch": supcon_best_epoch,
            "supcon_wait": supcon_wait,
            "augmentation_epoch_cursor": augmentation_epoch_cursor,
            "phase_train_loss_best": phase_train_loss_best,
            "phase_train_loss_wait": phase_train_loss_wait,
            "phase_validation_wait": phase_validation_wait,
            "train_progress": train_progress,
            "phase_best_state": cpu_state_dict(model),
        }
        torch.save(final_checkpoint, output_dir / "best.pt")
        save_json(
            output_dir / "train_only_summary.json",
            {
                "dataset_root": str(args.dataset_root),
                "checkpoint": str(output_dir / "best.pt"),
                "train_loss_validation_patience": args.train_loss_validation_patience,
                "train_loss_best": phase_train_loss_best,
                "train_loss_wait": phase_train_loss_wait,
                "global_train_step": train_progress["global_train_step"],
                "global_source_samples_seen": train_progress["global_source_samples_seen"],
                "train_only": True,
            },
        )
        log_json_event(
            log_path,
            {
                "event": "train_only_complete",
                "checkpoint": str(output_dir / "best.pt"),
                "train_loss_best": phase_train_loss_best,
                "train_loss_wait": phase_train_loss_wait,
                "global_train_step": train_progress["global_train_step"],
            },
        )
        return 0

    model.load_state_dict(best_classifier_state)
    release_training_memory(device, train_loader, val_loader, supcon_train_loader, supcon_val_loader)

    log_json_event(
        log_path,
        {
            "event": "validation_started",
            "stage": "post_progressive_evaluation",
            "phase_name": "best_classifier_validation",
            "split": "val",
            "eval_batches": val_eval_batches,
        },
    )
    val_logits, val_targets = collect_logits_and_labels(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=args.max_eval_batches,
        log_path=log_path,
        log_every_eval_steps=args.log_eval_every_steps,
        criterion=classifier_criterion,
        split="val",
        args=args,
        stage="post_progressive_evaluation",
        phase_name="best_classifier_validation",
    )
    release_training_memory(device, val_loader)
    val_metrics = compute_classification_metrics(val_logits, val_targets, train_dataset.classes, args.confidence_threshold)
    val_probabilities = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
    log_json_event(
        log_path,
        {
            "event": "validation_finished",
            "stage": "post_progressive_evaluation",
            "phase_name": "best_classifier_validation",
            "split": "val",
            "eval_batches": val_eval_batches,
            "accuracy": val_metrics["accuracy"],
            "per_class_accuracy": val_metrics["per_class_accuracy"],
            "per_class_avg_confidence": val_metrics["per_class_avg_confidence"],
            "loss": val_metrics["loss"],
        },
    )

    if args.run_final_test:
        log_json_event(
            log_path,
            {
                "event": "final_evaluation_started",
                "stage": "final_test_evaluation",
                "phase_name": "final_test",
                "split": "test",
                "eval_batches": test_eval_batches,
            },
        )
        test_logits, test_targets = collect_logits_and_labels(
            model=model,
            loader=test_loader,
            device=device,
            max_batches=args.max_eval_batches,
            log_path=log_path,
            log_every_eval_steps=args.log_eval_every_steps,
            criterion=classifier_criterion,
            split="test",
            args=args,
            stage="final_test_evaluation",
            phase_name="final_test",
        )
        release_training_memory(device, test_loader)
        test_metrics = compute_classification_metrics(test_logits, test_targets, train_dataset.classes, args.confidence_threshold)
        test_probabilities = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
        test_confmat = np.asarray(test_metrics["confusion_matrix"], dtype=np.int64)
        test_loss = float(test_metrics.get("cross_entropy_loss", test_metrics.get("loss", 0.0)))
        save_confusion_matrix_csv(output_dir / "test_confmat_counts.csv", test_confmat, train_dataset.classes, percent=False)
        save_confusion_matrix_csv(output_dir / "test_confmat_rate_pct.csv", test_confmat, train_dataset.classes, percent=True)
        save_classification_report_csv(output_dir / "test_classification_report.csv", test_metrics, train_dataset.classes)
        log_json_event(
            log_path,
            {
                "event": "final_evaluation_finished",
                "stage": "final_test_evaluation",
                "phase_name": "final_test",
                "split": "test",
                "eval_batches": test_eval_batches,
                "accuracy": test_metrics["accuracy"],
                "loss": test_loss,
                "raw_accuracy": test_metrics["raw_accuracy"],
                "top1_accuracy": test_metrics["top1_accuracy"],
                "top3_accuracy": test_metrics["top3_accuracy"],
                "top5_accuracy": test_metrics["top5_accuracy"],
                "macro_precision": test_metrics["macro_precision"],
                "macro_recall": test_metrics["macro_recall"],
                "macro_f1": test_metrics["macro_f1"],
                "weighted_precision": test_metrics["weighted_precision"],
                "weighted_recall": test_metrics["weighted_recall"],
                "weighted_f1": test_metrics["weighted_f1"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "macro_roc_auc_ovr": test_metrics["macro_roc_auc_ovr"],
                "weighted_roc_auc_ovr": test_metrics["weighted_roc_auc_ovr"],
                "macro_pr_auc_ovr": test_metrics["macro_pr_auc_ovr"],
                "weighted_pr_auc_ovr": test_metrics["weighted_pr_auc_ovr"],
                "cohen_kappa": test_metrics["cohen_kappa"],
                "mcc": test_metrics["mcc"],
                "per_class_accuracy": test_metrics["per_class_accuracy"],
                "per_class_avg_confidence": test_metrics["per_class_avg_confidence"],
                "ece": test_metrics["calibration"]["expected_calibration_error"],
                "mce": test_metrics["calibration"]["maximum_calibration_error"],
                "brier_score": test_metrics["calibration"]["brier_score"],
                "nll": test_metrics["calibration"]["negative_log_likelihood"],
            },
        )
        test_correct_confidence = compute_correct_confidence_by_class(test_logits, test_targets, train_dataset.classes)  # noqa: F841
    else:
        test_metrics = {}
        test_probabilities = np.empty((0, len(train_dataset.classes)), dtype=np.float32)
        log_json_event(log_path, {"event": "final_test_evaluation_skipped", "reason": "--run-final-test not set (default: protected holdout)"})

    validation_summary = {
        "split": "val",
        "num_samples": val_metrics["num_samples"],
        "accuracy": val_metrics["accuracy"],
        "top1_accuracy": val_metrics["top1_accuracy"],
        "top3_accuracy": val_metrics["top3_accuracy"],
        "macro_precision": val_metrics["macro_precision"],
        "macro_recall": val_metrics["macro_recall"],
        "macro_f1": val_metrics["macro_f1"],
        "weighted_precision": val_metrics["weighted_precision"],
        "weighted_recall": val_metrics["weighted_recall"],
        "weighted_f1": val_metrics["weighted_f1"],
        "balanced_accuracy": val_metrics["balanced_accuracy"],
        "per_class_accuracy": val_metrics["per_class_accuracy"],
        "per_class_avg_confidence": val_metrics["per_class_avg_confidence"],
        "calibration": val_metrics["calibration"],
        "per_class": val_metrics["per_class"],
    }
    save_json(output_dir / "validation_summary.json", validation_summary)
    save_reliability_diagram(
        output_dir / "validation_reliability_diagram.png",
        val_metrics["calibration"],
        "Validation Reliability Diagram",
    )
    save_confidence_histogram(
        output_dir / "validation_confidence_histogram.png",
        val_probabilities,
        "Validation Confidence Histogram",
    )
    save_confusion_matrix_plot(
        output_dir / "validation_confusion_matrix.png",
        np.asarray(val_metrics["confusion_matrix"], dtype=np.int64),
        train_dataset.classes,
        "Validation Confusion Matrix",
    )
    if args.run_final_test:
        test_summary = {
            "split": "test",
            "num_samples": test_metrics["num_samples"],
            "loss": test_loss,
            "cross_entropy_loss": test_loss,
            "accuracy": test_metrics["accuracy"],
            "top1_accuracy": test_metrics["top1_accuracy"],
            "top3_accuracy": test_metrics["top3_accuracy"],
            "top5_accuracy": test_metrics["top5_accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_precision": test_metrics["weighted_precision"],
            "weighted_recall": test_metrics["weighted_recall"],
            "weighted_f1": test_metrics["weighted_f1"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_roc_auc_ovr": test_metrics["macro_roc_auc_ovr"],
            "weighted_roc_auc_ovr": test_metrics["weighted_roc_auc_ovr"],
            "macro_pr_auc_ovr": test_metrics["macro_pr_auc_ovr"],
            "weighted_pr_auc_ovr": test_metrics["weighted_pr_auc_ovr"],
            "cohen_kappa": test_metrics["cohen_kappa"],
            "mcc": test_metrics["mcc"],
            "per_class_accuracy": test_metrics["per_class_accuracy"],
            "per_class_avg_confidence": test_metrics["per_class_avg_confidence"],
            "calibration": test_metrics["calibration"],
            "per_class": test_metrics["per_class"],
        }
        save_json(output_dir / "test_summary.json", test_summary)
        log_json_event(
            log_path,
            {
                "event": "final_test_evaluation_finished",
                "split": "test",
                "eval_batches": test_eval_batches,
                "num_samples": test_metrics["num_samples"],
                "accuracy": test_metrics["accuracy"],
                "loss": test_loss,
                "raw_accuracy": test_metrics["raw_accuracy"],
                "top1_accuracy": test_metrics["top1_accuracy"],
                "top3_accuracy": test_metrics["top3_accuracy"],
                "top5_accuracy": test_metrics["top5_accuracy"],
                "per_class_accuracy": test_metrics["per_class_accuracy"],
                "macro_precision": test_metrics["macro_precision"],
                "macro_recall": test_metrics["macro_recall"],
                "macro_f1": test_metrics["macro_f1"],
                "weighted_precision": test_metrics["weighted_precision"],
                "weighted_recall": test_metrics["weighted_recall"],
                "weighted_f1": test_metrics["weighted_f1"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "macro_roc_auc_ovr": test_metrics["macro_roc_auc_ovr"],
                "weighted_roc_auc_ovr": test_metrics["weighted_roc_auc_ovr"],
                "macro_pr_auc_ovr": test_metrics["macro_pr_auc_ovr"],
                "weighted_pr_auc_ovr": test_metrics["weighted_pr_auc_ovr"],
                "cohen_kappa": test_metrics["cohen_kappa"],
                "mcc": test_metrics["mcc"],
                "per_class_avg_confidence": test_metrics["per_class_avg_confidence"],
                "ece": test_metrics["calibration"]["expected_calibration_error"],
                "mce": test_metrics["calibration"]["maximum_calibration_error"],
                "brier_score": test_metrics["calibration"]["brier_score"],
                "nll": test_metrics["calibration"]["negative_log_likelihood"],
            },
        )
        save_reliability_diagram(
            output_dir / "test_reliability_diagram.png",
            test_metrics["calibration"],
            "Test Reliability Diagram",
        )
        save_confidence_histogram(
            output_dir / "test_confidence_histogram.png",
            test_probabilities,
            "Test Confidence Histogram",
        )
        save_confusion_matrix_plot(
            output_dir / "test_confusion_matrix.png",
            test_confmat,
            train_dataset.classes,
            "Test Confusion Matrix",
        )

    # cpu_state_dict: detach + move to CPU + clone — consistent with all other
    # checkpoint saves in this file and safe for map_location-free loading.
    final_checkpoint = {
        "model_state_dict": cpu_state_dict(model),
        "class_names": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "args": vars(args),
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "validation_summary": validation_summary,
        "test_summary": test_summary if args.run_final_test else {},
        "best_val_loss": best_val_loss,
        "best_val_raw_acc": best_val_raw_acc,
    }
    torch.save(final_checkpoint, output_dir / "best.pt")

    metrics = {
        "model_name": model_name,
        "backbone": args.backbone,
        "device": str(device),
        "output_dir": str(output_dir),
        "log_file": str(log_path),
        "weights": args.weights,
        "class_names": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "source_train_count": train_dataset.source_count(),
        "source_val_count": val_dataset.source_count(),
        "source_test_count": test_dataset.source_count(),
        "train_samples_per_epoch": len(train_sampler),
        "val_samples_per_eval": len(val_dataset),
        "test_samples_per_eval": len(test_dataset),
        "augmentation_bank_train_count": len(train_dataset),
        "augmentation_bank_val_count": len(val_dataset),
        "augmentation_bank_test_count": len(test_dataset),
        "augment_repeats": args.augment_repeats,
        "augment_gaussian_sigmas": args.augment_gaussian_sigmas,
        "camera_color_cast_probability": args.camera_color_cast_probability,
        "camera_color_cast_strength": args.camera_color_cast_strength,
        "camera_color_cast_eval": args.camera_color_cast_eval,
        "sampling_strategy": args.sampling_strategy,
        "train_class_counts": class_counts(train_dataset),
        "val_class_counts": class_counts(val_dataset),
        "test_class_counts": class_counts(test_dataset),
        "effective_train_class_counts": effective_class_counts(train_dataset),
        "effective_val_class_counts": effective_class_counts(val_dataset),
        "effective_test_class_counts": effective_class_counts(test_dataset),
        "supcon": {
            "temperature": args.supcon_temperature,
            "phase_plan": [asdict(phase) for phase in supcon_phases],
            "best_val_loss": supcon_best_loss,
            "best_epoch": supcon_best_epoch,
            "frozen_core_backbone_modules": args.frozen_core_backbone_modules,
            "unfreeze_cap": effective_unfrozen_backbone_cap(len(backbone_modules), args, "supcon"),
        },
        "classifier": {
            "phase_plan": [asdict(phase) for phase in phases],
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_val_raw_acc": best_val_raw_acc,
        },
        "optimization": {
            "optimizer": args.optimizer,
            "scheduler": "Linear warmup + cosine decay",
            "sam_rho": args.sam_rho,
            "weight_decay": args.weight_decay,
            "adam_betas": [args.adam_beta1, args.adam_beta2],
            "warmup_epochs": args.warmup_epochs,
            "warmup_steps": args.warmup_steps,
        },
        "image_augmentation": {
            "policy": "aspect_preserving_random_crop_plus_flips_for_train_only",
            "stochastic_augmentations_enabled": bool(getattr(train_dataset, "stochastic_augmentation", False)),
            "spatial_augmentations_enabled": bool(getattr(train_dataset, "stochastic_augmentation", False)),
            "photometric_augmentations_enabled": False,
            "camera_color_cast_strength": args.camera_color_cast_strength,
            "camera_color_cast_eval": args.camera_color_cast_eval,
        },
        "early_stopping": {
            "supcon_patience": args.supcon_early_stopping_patience,
            "head_patience": args.head_early_stopping_patience,
            "stage_patience": args.stage_early_stopping_patience,
            "train_loss_validation_patience": args.train_loss_validation_patience,
            "validation_patience": args.validation_patience,
            "min_delta": args.early_stopping_min_delta,
            "monitor": "val_loss",
            "check_unit": "train_steps",
        },
        "embedding_dim": args.embedding_dim,
        "projection_dim": args.projection_dim,
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "validation_summary": validation_summary,
        "test_summary": test_summary if args.run_final_test else {},
    }
    return 0
