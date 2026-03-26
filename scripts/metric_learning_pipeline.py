#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import gc
import io
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_OFFSETS = {"train": 0, "val": 10_000_000, "test": 20_000_000}
SHADOW_PROBABILITY = 0.20
GLARE_PROBABILITY = 0.25
MOTION_BLUR_PROBABILITY = 0.18
DEFOCUS_BLUR_PROBABILITY = 0.18
RESOLUTION_DEGRADE_PROBABILITY = 0.20
TRUNCATION_PROBABILITY = 0.20
SMUDGE_PROBABILITY = 0.18


@dataclass
class PhaseSpec:
    name: str
    unfrozen_backbone_modules: int
    max_epochs: int


class DeterministicAugmentedImageFolder(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        root: Path,
        image_size: int,
        augment_repeats: int,
        split_name: str,
        seed: int,
        gaussian_sigmas: float,
        apply_augmentation: bool = True,
    ) -> None:
        self.base_dataset = datasets.ImageFolder(root)
        self.image_size = image_size
        self.augment_repeats = augment_repeats
        self.split_name = split_name
        self.seed = seed
        self.gaussian_sigmas = gaussian_sigmas
        self.apply_augmentation = apply_augmentation
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx
        self.samples = self.base_dataset.samples
        self.targets = self.base_dataset.targets
        self.current_epoch = 0

    def __len__(self) -> int:
        return len(self.samples) * self.augment_repeats

    def source_count(self) -> int:
        return len(self.samples)

    def target_for_index(self, index: int) -> int:
        source_index = index % len(self.samples)
        return self.targets[source_index]

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(0, int(epoch))

    def variant_for_source(self, source_index: int) -> int:
        return (source_index * 17) % self.augment_repeats

    def _seed_for_variant(self, source_index: int, variant_index: int, view_offset: int = 0) -> int:
        return (
            self.seed * 1_000_003
            + SPLIT_OFFSETS[self.split_name]
            + source_index * 9_973
            + variant_index * 99_991
            + self.current_epoch * 104_729
            + view_offset * 1_299_721
        )

    def load_augmented(self, source_index: int, variant_index: int, view_offset: int = 0) -> tuple[torch.Tensor, int]:
        path, target = self.samples[source_index]
        image = self.base_dataset.loader(path).convert("RGB")
        if self.apply_augmentation:
            rng = random.Random(self._seed_for_variant(source_index, variant_index, view_offset))
            tensor = augmented_tensor_from_image(image, self.image_size, rng, self.gaussian_sigmas)
        else:
            tensor = evaluation_tensor_from_image(image, self.image_size)
        return tensor, target

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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        source_index = index % self.base_dataset.source_count()
        variant_index = index // self.base_dataset.source_count()
        view_one, target = self.base_dataset.load_augmented(source_index, variant_index, view_offset=0)
        view_two, _ = self.base_dataset.load_augmented(source_index, variant_index, view_offset=1)
        return view_one, view_two, target


def sample_gaussian_clipped(rng: random.Random, mean: float, std: float, low: float, high: float) -> float:
    return min(max(rng.gauss(mean, max(std, 1e-6)), low), high)


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
    if mean is None:
        mean = (safe_low + safe_high) / 2.0
    std = (safe_high - safe_low) / max(1e-6, 2.0 * gaussian_sigmas)
    return sample_gaussian_clipped(rng, mean, std, hard_low, hard_high)


def sample_symmetric(
    rng: random.Random,
    safe_abs: float,
    hard_abs: float,
    gaussian_sigmas: float,
    *,
    mean: float = 0.0,
) -> float:
    return sample_safe_range(rng, -safe_abs, safe_abs, -hard_abs, hard_abs, gaussian_sigmas, mean=mean)


def sample_log_safe_ratio(
    rng: random.Random,
    safe_low: float,
    safe_high: float,
    hard_low: float,
    hard_high: float,
    gaussian_sigmas: float,
) -> float:
    safe_low_log = math.log(safe_low)
    safe_high_log = math.log(safe_high)
    hard_low_log = math.log(hard_low)
    hard_high_log = math.log(hard_high)
    value_log = sample_safe_range(
        rng,
        safe_low_log,
        safe_high_log,
        hard_low_log,
        hard_high_log,
        gaussian_sigmas,
        mean=0.0,
    )
    return math.exp(value_log)


def random_resized_crop(image: Image.Image, image_size: int, rng: random.Random, gaussian_sigmas: float) -> Image.Image:
    width, height = image.size
    area = width * height

    for _ in range(10):
        target_area = area * sample_safe_range(rng, 0.70, 1.0, 0.55, 1.0, gaussian_sigmas, mean=0.85)
        aspect_ratio = sample_log_safe_ratio(rng, 0.8, 1.25, 0.7, 1.35, gaussian_sigmas)
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = rng.randint(0, height - crop_height)
            left = rng.randint(0, width - crop_width)
            return TF.resized_crop(
                image,
                top=top,
                left=left,
                height=crop_height,
                width=crop_width,
                size=[image_size, image_size],
                interpolation=InterpolationMode.BILINEAR,
            )

    side = min(width, height)
    top = (height - side) // 2
    left = (width - side) // 2
    return TF.resized_crop(
        image,
        top=top,
        left=left,
        height=side,
        width=side,
        size=[image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
    )


def random_perspective(image: Image.Image, rng: random.Random, gaussian_sigmas: float) -> Image.Image:
    distortion = sample_safe_range(rng, 0.05, 0.15, 0.0, 0.2, gaussian_sigmas, mean=0.10)
    if distortion < 1e-4:
        return image

    width, height = image.size
    dx = distortion * width
    dy = distortion * height

    def point(x: float, y: float) -> list[int]:
        px = int(round(min(max(x, 0.0), width - 1.0)))
        py = int(round(min(max(y, 0.0), height - 1.0)))
        return [px, py]

    startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    endpoints = [
        point(sample_gaussian_clipped(rng, dx / 2.0, dx / max(1e-6, 2.0 * gaussian_sigmas), 0.0, dx), sample_gaussian_clipped(rng, dy / 2.0, dy / max(1e-6, 2.0 * gaussian_sigmas), 0.0, dy)),
        point(sample_gaussian_clipped(rng, width - 1 - dx / 2.0, dx / max(1e-6, 2.0 * gaussian_sigmas), width - 1 - dx, width - 1), sample_gaussian_clipped(rng, dy / 2.0, dy / max(1e-6, 2.0 * gaussian_sigmas), 0.0, dy)),
        point(sample_gaussian_clipped(rng, width - 1 - dx / 2.0, dx / max(1e-6, 2.0 * gaussian_sigmas), width - 1 - dx, width - 1), sample_gaussian_clipped(rng, height - 1 - dy / 2.0, dy / max(1e-6, 2.0 * gaussian_sigmas), height - 1 - dy, height - 1)),
        point(sample_gaussian_clipped(rng, dx / 2.0, dx / max(1e-6, 2.0 * gaussian_sigmas), 0.0, dx), sample_gaussian_clipped(rng, height - 1 - dy / 2.0, dy / max(1e-6, 2.0 * gaussian_sigmas), height - 1 - dy, height - 1)),
    ]
    return TF.perspective(
        image,
        startpoints=startpoints,
        endpoints=endpoints,
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )


def apply_resolution_degradation(image: Image.Image, rng: random.Random, gaussian_sigmas: float) -> Image.Image:
    if rng.random() >= RESOLUTION_DEGRADE_PROBABILITY:
        return image
    width, height = image.size
    scale = sample_safe_range(rng, 0.45, 0.80, 0.25, 1.0, gaussian_sigmas, mean=0.60)
    down_w = max(8, int(round(width * scale)))
    down_h = max(8, int(round(height * scale)))
    down_mode = InterpolationMode.BILINEAR if rng.random() < 0.5 else InterpolationMode.BICUBIC
    up_mode = InterpolationMode.BILINEAR if rng.random() < 0.7 else InterpolationMode.BICUBIC
    image = TF.resize(image, [down_h, down_w], interpolation=down_mode)
    return TF.resize(image, [height, width], interpolation=up_mode)


def apply_border_truncation(image: Image.Image, rng: random.Random, gaussian_sigmas: float) -> Image.Image:
    if rng.random() >= TRUNCATION_PROBABILITY:
        return image
    width, height = image.size
    keep_fraction = sample_safe_range(rng, 0.85, 1.0, 0.70, 1.0, gaussian_sigmas, mean=0.93)
    keep_ratio = sample_log_safe_ratio(rng, 0.85, 1.15, 0.70, 1.30, gaussian_sigmas)
    crop_w = min(width, max(2, int(round(width * keep_fraction * math.sqrt(keep_ratio)))))
    crop_h = min(height, max(2, int(round(height * keep_fraction / math.sqrt(keep_ratio)))))
    max_left = max(0, width - crop_w)
    max_top = max(0, height - crop_h)

    anchor = rng.choice(("left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right"))
    if anchor in {"left", "top_left", "bottom_left"}:
        left = 0
    elif anchor in {"right", "top_right", "bottom_right"}:
        left = max_left
    else:
        left = rng.randint(0, max_left)

    if anchor in {"top", "top_left", "top_right"}:
        top = 0
    elif anchor in {"bottom", "bottom_left", "bottom_right"}:
        top = max_top
    else:
        top = rng.randint(0, max_top)

    image = TF.crop(image, top=top, left=left, height=crop_h, width=crop_w)
    return TF.resize(image, [height, width], interpolation=InterpolationMode.BILINEAR)


def jpeg_compress(image: Image.Image, rng: random.Random, gaussian_sigmas: float) -> Image.Image:
    quality = int(round(sample_safe_range(rng, 50.0, 100.0, 35.0, 100.0, gaussian_sigmas, mean=75.0)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    compressed = Image.open(buffer)
    return compressed.convert("RGB")


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
    if rng.random() >= MOTION_BLUR_PROBABILITY:
        return tensor
    length = sample_safe_range(rng, 3.0, 9.0, 1.0, 15.0, gaussian_sigmas, mean=5.5)
    kernel_size = max(3, int(round(length)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    angle = rng.uniform(0.0, 180.0)
    kernel = motion_blur_kernel(kernel_size, angle).to(dtype=tensor.dtype, device=tensor.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
    blurred = F.conv2d(tensor.unsqueeze(0), kernel, padding=kernel_size // 2, groups=3)
    return blurred.squeeze(0)


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
    if rng.random() >= DEFOCUS_BLUR_PROBABILITY:
        return tensor
    radius = sample_safe_range(rng, 0.8, 1.8, 0.0, 3.0, gaussian_sigmas, mean=1.2)
    if radius < 0.25:
        return tensor
    kernel = defocus_blur_kernel(radius).to(dtype=tensor.dtype, device=tensor.device)
    kernel_size = kernel.shape[0]
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
    blurred = F.conv2d(tensor.unsqueeze(0), kernel, padding=kernel_size // 2, groups=3)
    return blurred.squeeze(0)


def apply_gaussian_noise(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    noise_std = sample_safe_range(rng, 0.0, 0.02, 0.0, 0.06, gaussian_sigmas, mean=0.01)
    if noise_std < 1e-5:
        return tensor
    generator = torch.Generator()
    generator.manual_seed(rng.randrange(0, 2**31 - 1))
    noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
    return tensor + (noise * noise_std)


def apply_channel_shift(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    shifts = torch.tensor(
        [
            sample_symmetric(rng, 0.02, 0.05, gaussian_sigmas),
            sample_symmetric(rng, 0.02, 0.05, gaussian_sigmas),
            sample_symmetric(rng, 0.02, 0.05, gaussian_sigmas),
        ],
        dtype=tensor.dtype,
    ).view(3, 1, 1)
    return tensor + shifts


def apply_grayscale_mix(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    mix = sample_safe_range(rng, 0.0, 0.08, 0.0, 0.15, gaussian_sigmas, mean=0.04)
    gray = tensor.mean(dim=0, keepdim=True).repeat(3, 1, 1)
    return tensor.mul(1.0 - mix).add(gray.mul(mix))


def apply_illumination_gradient(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    _, height, width = tensor.shape
    horizontal = rng.random() < 0.5
    start = sample_safe_range(rng, 0.8, 1.05, 0.7, 1.1, gaussian_sigmas, mean=0.95)
    end = sample_safe_range(rng, 0.8, 1.05, 0.7, 1.1, gaussian_sigmas, mean=0.95)
    if horizontal:
        ramp = torch.linspace(start, end, steps=width, dtype=tensor.dtype).view(1, 1, width)
        mask = ramp.expand(1, height, width)
    else:
        ramp = torch.linspace(start, end, steps=height, dtype=tensor.dtype).view(1, height, 1)
        mask = ramp.expand(1, height, width)
    return tensor * mask


def apply_smudge_overlay(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    if rng.random() >= SMUDGE_PROBABILITY:
        return tensor
    _, height, width = tensor.shape
    ys = torch.linspace(-1.0, 1.0, steps=height, dtype=tensor.dtype, device=tensor.device)
    xs = torch.linspace(-1.0, 1.0, steps=width, dtype=tensor.dtype, device=tensor.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    overlay = torch.zeros((height, width), dtype=tensor.dtype, device=tensor.device)
    blob_count = rng.randint(1, 3)
    for _ in range(blob_count):
        cx = sample_safe_range(rng, -0.45, 0.45, -0.85, 0.85, gaussian_sigmas, mean=0.0)
        cy = sample_safe_range(rng, -0.45, 0.45, -0.85, 0.85, gaussian_sigmas, mean=0.0)
        sigma_x = sample_safe_range(rng, 0.10, 0.22, 0.04, 0.35, gaussian_sigmas, mean=0.15)
        sigma_y = sample_safe_range(rng, 0.07, 0.18, 0.03, 0.30, gaussian_sigmas, mean=0.12)
        opacity = sample_safe_range(rng, 0.05, 0.14, 0.0, 0.25, gaussian_sigmas, mean=0.09)
        blob = torch.exp(-(((xx - cx) ** 2) / (2 * sigma_x * sigma_x) + ((yy - cy) ** 2) / (2 * sigma_y * sigma_y)))
        overlay = torch.maximum(overlay, blob * opacity)
    stain_color = torch.tensor(
        [
            sample_safe_range(rng, 0.32, 0.55, 0.20, 0.65, gaussian_sigmas, mean=0.43),
            sample_safe_range(rng, 0.28, 0.46, 0.18, 0.56, gaussian_sigmas, mean=0.36),
            sample_safe_range(rng, 0.18, 0.34, 0.10, 0.44, gaussian_sigmas, mean=0.24),
        ],
        dtype=tensor.dtype,
        device=tensor.device,
    ).view(3, 1, 1)
    overlay = overlay.unsqueeze(0)
    return tensor * (1.0 - overlay) + stain_color * overlay


def apply_shadow_overlay(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float, shadow_prob: float) -> torch.Tensor:
    if rng.random() >= shadow_prob:
        return tensor
    _, height, width = tensor.shape
    horizontal = rng.random() < 0.5
    darkness = sample_safe_range(rng, 0.82, 0.95, 0.7, 1.0, gaussian_sigmas, mean=0.90)
    midpoint = sample_safe_range(rng, 0.35, 0.65, 0.15, 0.85, gaussian_sigmas, mean=0.5)
    softness = sample_safe_range(rng, 0.08, 0.18, 0.03, 0.30, gaussian_sigmas, mean=0.12)
    if horizontal:
        coords = torch.linspace(0.0, 1.0, steps=width, dtype=tensor.dtype).view(1, 1, width)
    else:
        coords = torch.linspace(0.0, 1.0, steps=height, dtype=tensor.dtype).view(1, height, 1)
    transition = torch.sigmoid((coords - midpoint) / max(softness, 1e-4))
    shadow_mask = darkness + (1.0 - darkness) * transition
    shadow_mask = shadow_mask.expand(1, height, width)
    return tensor * shadow_mask


def apply_specular_glare(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float, glare_prob: float) -> torch.Tensor:
    if rng.random() >= glare_prob:
        return tensor
    _, height, width = tensor.shape
    ys = torch.linspace(-1.0, 1.0, steps=height, dtype=tensor.dtype)
    xs = torch.linspace(-1.0, 1.0, steps=width, dtype=tensor.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    glare = torch.zeros((height, width), dtype=tensor.dtype)
    blob_count = rng.randint(1, 3)
    for _ in range(blob_count):
        cx = sample_safe_range(rng, -0.4, 0.4, -0.8, 0.8, gaussian_sigmas, mean=0.0)
        cy = sample_safe_range(rng, -0.4, 0.4, -0.8, 0.8, gaussian_sigmas, mean=0.0)
        sigma_x = sample_safe_range(rng, 0.08, 0.18, 0.03, 0.30, gaussian_sigmas, mean=0.12)
        sigma_y = sample_safe_range(rng, 0.05, 0.14, 0.02, 0.22, gaussian_sigmas, mean=0.09)
        intensity = sample_safe_range(rng, 0.10, 0.22, 0.0, 0.35, gaussian_sigmas, mean=0.16)
        blob = torch.exp(-(((xx - cx) ** 2) / (2 * sigma_x * sigma_x) + ((yy - cy) ** 2) / (2 * sigma_y * sigma_y)))
        glare = torch.maximum(glare, blob * intensity)
    color_scale = torch.tensor(
        [
            sample_safe_range(rng, 0.95, 1.05, 0.9, 1.1, gaussian_sigmas, mean=1.0),
            sample_safe_range(rng, 0.97, 1.08, 0.9, 1.12, gaussian_sigmas, mean=1.02),
            sample_safe_range(rng, 1.00, 1.12, 0.92, 1.18, gaussian_sigmas, mean=1.06),
        ],
        dtype=tensor.dtype,
    ).view(3, 1, 1)
    glare = glare.unsqueeze(0)
    return tensor + (1.0 - tensor) * glare * color_scale


def apply_cutout(tensor: torch.Tensor, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    _, height, width = tensor.shape
    cutout_count = rng.randint(1, 3)
    total_fraction = sample_safe_range(rng, 0.08, 0.15, 0.02, 0.25, gaussian_sigmas, mean=0.11)
    remaining_area = total_fraction * height * width
    for _ in range(cutout_count):
        if remaining_area <= 1.0:
            break
        share = remaining_area if cutout_count == 1 else remaining_area * rng.uniform(0.25, 0.75)
        aspect = sample_log_safe_ratio(rng, 0.8, 1.25, 0.7, 1.35, gaussian_sigmas)
        cutout_h = max(1, int(round(math.sqrt(share / max(aspect, 1e-6)))))
        cutout_w = max(1, int(round(cutout_h * aspect)))
        cutout_h = min(cutout_h, height)
        cutout_w = min(cutout_w, width)
        top = rng.randint(0, max(0, height - cutout_h))
        left = rng.randint(0, max(0, width - cutout_w))
        fill = torch.tensor(
            [
                sample_gaussian_clipped(rng, 0.5, 0.25, 0.0, 1.0),
                sample_gaussian_clipped(rng, 0.5, 0.25, 0.0, 1.0),
                sample_gaussian_clipped(rng, 0.5, 0.25, 0.0, 1.0),
            ],
            dtype=tensor.dtype,
        ).view(3, 1, 1)
        tensor[:, top : top + cutout_h, left : left + cutout_w] = fill
        remaining_area -= cutout_h * cutout_w
        cutout_count -= 1
    return tensor


def augmented_tensor_from_image(image: Image.Image, image_size: int, rng: random.Random, gaussian_sigmas: float) -> torch.Tensor:
    image = random_resized_crop(image, image_size, rng, gaussian_sigmas)
    image = apply_border_truncation(image, rng, gaussian_sigmas)

    if rng.random() < 0.5:
        image = TF.hflip(image)
    if rng.random() < 0.05:
        image = TF.vflip(image)

    translate = (
        int(round(sample_symmetric(rng, 0.10, 0.15, gaussian_sigmas) * image_size)),
        int(round(sample_symmetric(rng, 0.10, 0.15, gaussian_sigmas) * image_size)),
    )
    image = TF.affine(
        image,
        angle=sample_symmetric(rng, 15.0, 25.0, gaussian_sigmas),
        translate=translate,
        scale=sample_safe_range(rng, 0.85, 1.15, 0.75, 1.25, gaussian_sigmas, mean=1.0),
        shear=[
            sample_symmetric(rng, 8.0, 12.0, gaussian_sigmas),
            sample_symmetric(rng, 8.0, 12.0, gaussian_sigmas),
        ],
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )
    image = random_perspective(image, rng, gaussian_sigmas)
    image = TF.adjust_brightness(image, sample_safe_range(rng, 0.8, 1.2, 0.65, 1.35, gaussian_sigmas, mean=1.0))
    image = TF.adjust_contrast(image, sample_safe_range(rng, 0.8, 1.25, 0.7, 1.4, gaussian_sigmas, mean=1.0))
    image = TF.adjust_saturation(image, sample_safe_range(rng, 0.8, 1.25, 0.7, 1.35, gaussian_sigmas, mean=1.0))
    image = TF.adjust_hue(image, sample_symmetric(rng, 0.04, 0.08, gaussian_sigmas))
    image = TF.adjust_gamma(image, sample_safe_range(rng, 0.85, 1.2, 0.75, 1.3, gaussian_sigmas, mean=1.0), gain=1.0)
    image = TF.adjust_sharpness(image, sample_safe_range(rng, 0.7, 1.4, 0.5, 1.8, gaussian_sigmas, mean=1.0))
    image = jpeg_compress(image, rng, gaussian_sigmas)
    image = apply_resolution_degradation(image, rng, gaussian_sigmas)

    tensor = TF.to_tensor(image)
    blur_sigma = sample_safe_range(rng, 0.0, 1.0, 0.0, 1.8, gaussian_sigmas, mean=0.5)
    if blur_sigma > 0.05:
        kernel_size = 5 if blur_sigma < 1.0 else 7
        tensor = TF.gaussian_blur(
            tensor,
            kernel_size=[kernel_size, kernel_size],
            sigma=[blur_sigma, blur_sigma],
        )
    tensor = apply_motion_blur(tensor, rng, gaussian_sigmas)
    tensor = apply_defocus_blur(tensor, rng, gaussian_sigmas)
    tensor = apply_gaussian_noise(tensor, rng, gaussian_sigmas)
    tensor = apply_channel_shift(tensor, rng, gaussian_sigmas)
    tensor = apply_grayscale_mix(tensor, rng, gaussian_sigmas)
    tensor = apply_illumination_gradient(tensor, rng, gaussian_sigmas)
    tensor = apply_shadow_overlay(tensor, rng, gaussian_sigmas, shadow_prob=SHADOW_PROBABILITY)
    tensor = apply_specular_glare(tensor, rng, gaussian_sigmas, glare_prob=GLARE_PROBABILITY)
    tensor = apply_smudge_overlay(tensor, rng, gaussian_sigmas)
    tensor = apply_cutout(tensor, rng, gaussian_sigmas)
    tensor = tensor.clamp(0.0, 1.0)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def evaluation_tensor_from_image(image: Image.Image, image_size: int) -> torch.Tensor:
    image = TF.resize(image, image_size, interpolation=InterpolationMode.BILINEAR)
    image = TF.center_crop(image, [image_size, image_size])
    tensor = TF.to_tensor(image).clamp(0.0, 1.0)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class IdentityFrontEnd(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_gabor_kernel(
    kernel_size: int,
    sigma: float,
    theta: float,
    wavelength: float,
    gamma: float,
    psi: float = 0.0,
) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    x_theta = xx * math.cos(theta) + yy * math.sin(theta)
    y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
    gaussian = torch.exp(-(x_theta.square() + (gamma**2) * y_theta.square()) / (2 * sigma * sigma))
    wave = torch.cos((2 * math.pi * x_theta / wavelength) + psi)
    kernel = gaussian * wave
    kernel = kernel - kernel.mean()
    kernel = kernel / (kernel.abs().sum() + 1e-6)
    return kernel


class GaborAdapter(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        orientations: int,
        wavelengths: list[float],
        sigma: float,
        gamma: float,
    ) -> None:
        super().__init__()
        kernels = []
        for wavelength in wavelengths:
            for orientation_index in range(orientations):
                theta = orientation_index * math.pi / orientations
                kernels.append(build_gabor_kernel(kernel_size, sigma, theta, wavelength, gamma))
        bank = torch.stack(kernels).unsqueeze(1)
        self.register_buffer("gabor_bank", bank)
        self.num_filters = bank.shape[0]
        self.adapter = nn.Sequential(
            nn.Conv2d(3 + self.num_filters, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.SiLU(inplace=True),
        )
        with torch.no_grad():
            conv = self.adapter[0]
            conv.weight.zero_()
            conv.weight[0, 0, 0, 0] = 1.0
            conv.weight[1, 1, 0, 0] = 1.0
            conv.weight[2, 2, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = (0.2989 * x[:, 0:1]) + (0.5870 * x[:, 1:2]) + (0.1140 * x[:, 2:3])
        responses = F.conv2d(gray, self.gabor_bank, padding=self.gabor_bank.shape[-1] // 2)
        responses = responses / (responses.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)
        fused = torch.cat([x, responses], dim=1)
        return self.adapter(fused)


class MetricLearningEfficientNetB0(nn.Module):
    def __init__(
        self,
        num_classes: int,
        weights_mode: str,
        embedding_dim: int,
        projection_dim: int,
        use_gabor: bool,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if weights_mode == "default" else None
        self.front_end: nn.Module
        if use_gabor:
            self.front_end = GaborAdapter(
                kernel_size=args.gabor_kernel_size,
                orientations=args.gabor_orientations,
                wavelengths=parse_wavelengths(args.gabor_wavelengths),
                sigma=args.gabor_sigma,
                gamma=args.gabor_gamma,
            )
        else:
            self.front_end = IdentityFrontEnd()

        backbone = efficientnet_b0(weights=weights)
        self.backbone = backbone
        self.in_features = backbone.classifier[1].in_features
        self.dropout_p = backbone.classifier[0].p if isinstance(backbone.classifier[0], nn.Dropout) else 0.0
        self.embedding = nn.Linear(self.in_features, embedding_dim, bias=False)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.arcface_head = ArcMarginProduct(embedding_dim, num_classes, s=args.arcface_scale, m=args.arcface_margin)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.front_end(x)
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
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
        return self.arcface_head(embeddings, labels)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.s

        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s


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


def make_weighted_sampler(dataset: Dataset, classes: list[str], target_fn, seed: int) -> DeterministicEpochSampler:
    counts = {name: 0 for name in classes}
    for index in range(len(dataset)):
        counts[classes[target_fn(index)]] += 1
    weights = [1.0 / counts[classes[target_fn(index)]] for index in range(len(dataset))]
    return DeterministicEpochSampler(
        len(weights),
        seed,
        shuffle=False,
        weights=torch.as_tensor(weights, dtype=torch.double),
        replacement=True,
    )


def make_epoch_sampler(dataset: Dataset, seed: int, shuffle: bool) -> DeterministicEpochSampler:
    return DeterministicEpochSampler(len(dataset), seed, shuffle=shuffle)


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
    root = Path(args.dataset_root)
    train_dataset = DeterministicAugmentedImageFolder(
        root / "train", args.image_size, args.augment_repeats, "train", args.seed, args.augment_gaussian_sigmas, apply_augmentation=True
    )
    val_dataset = DeterministicAugmentedImageFolder(
        root / "val", args.image_size, 1, "val", args.seed, args.augment_gaussian_sigmas, apply_augmentation=False
    )
    test_dataset = DeterministicAugmentedImageFolder(
        root / "test", args.image_size, args.augment_repeats, "test", args.seed, args.augment_gaussian_sigmas, apply_augmentation=True
    )
    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError("Class folders differ across train/val/test")
    return (
        train_dataset,
        val_dataset,
        test_dataset,
        DeterministicSupConDataset(train_dataset),
        DeterministicSupConDataset(val_dataset),
    )


def build_phase_plan(total_modules: int, args: argparse.Namespace) -> list[PhaseSpec]:
    phases: list[PhaseSpec] = []
    if args.head_epochs > 0:
        phases.append(PhaseSpec(name="arcface_head_only", unfrozen_backbone_modules=0, max_epochs=args.head_epochs))
    count = 0
    phase_index = 0
    while count < total_modules and args.stage_epochs > 0:
        count = min(total_modules, count + args.unfreeze_chunk_size)
        phase_index += 1
        if args.max_progressive_phases and phase_index > args.max_progressive_phases:
            break
        phases.append(PhaseSpec(name=f"arcface_last_{count}_modules", unfrozen_backbone_modules=count, max_epochs=args.stage_epochs))
    return phases


def backbone_leaf_modules(model: MetricLearningEfficientNetB0) -> list[tuple[str, nn.Module]]:
    modules: list[tuple[str, nn.Module]] = []
    for name, module in model.backbone.features.named_modules():
        if not name:
            continue
        if any(True for _ in module.parameters(recurse=False)):
            modules.append((f"backbone.features.{name}", module))
    return modules


def set_trainability_for_supcon(
    model: MetricLearningEfficientNetB0,
    backbone_modules: list[tuple[str, nn.Module]],
    unfrozen_backbone_modules: int,
) -> list[str]:
    for parameter in model.backbone.features.parameters():
        parameter.requires_grad = False
    for parameter in model.embedding.parameters():
        parameter.requires_grad = True
    for parameter in model.projection_head.parameters():
        parameter.requires_grad = True
    for parameter in model.embedding_norm.parameters():
        parameter.requires_grad = True
    for parameter in model.arcface_head.parameters():
        parameter.requires_grad = False
    if not isinstance(model.front_end, IdentityFrontEnd):
        for parameter in model.front_end.parameters():
            parameter.requires_grad = True

    if unfrozen_backbone_modules <= 0:
        return []
    thawed = backbone_modules[-unfrozen_backbone_modules:]
    for _, module in thawed:
        for parameter in module.parameters(recurse=False):
            parameter.requires_grad = True
    return [name for name, _ in thawed]


def set_trainability_for_arcface(
    model: MetricLearningEfficientNetB0,
    backbone_modules: list[tuple[str, nn.Module]],
    unfrozen_backbone_modules: int,
) -> list[str]:
    for parameter in model.backbone.features.parameters():
        parameter.requires_grad = False
    for parameter in model.embedding.parameters():
        parameter.requires_grad = True
    for parameter in model.embedding_norm.parameters():
        parameter.requires_grad = True
    for parameter in model.arcface_head.parameters():
        parameter.requires_grad = True
    for parameter in model.projection_head.parameters():
        parameter.requires_grad = False
    if not isinstance(model.front_end, IdentityFrontEnd):
        for parameter in model.front_end.parameters():
            parameter.requires_grad = True

    if unfrozen_backbone_modules <= 0:
        return []
    thawed = backbone_modules[-unfrozen_backbone_modules:]
    for _, module in thawed:
        for parameter in module.parameters(recurse=False):
            parameter.requires_grad = True
    return [name for name, _ in thawed]


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


def build_supcon_optimizer(model: MetricLearningEfficientNetB0, args: argparse.Namespace) -> Optimizer:
    head_params = []
    if not isinstance(model.front_end, IdentityFrontEnd):
        head_params.extend(parameter for parameter in model.front_end.parameters() if parameter.requires_grad)
    head_params.extend(parameter for parameter in model.embedding.parameters() if parameter.requires_grad)
    head_params.extend(parameter for parameter in model.embedding_norm.parameters() if parameter.requires_grad)
    head_params.extend(parameter for parameter in model.projection_head.parameters() if parameter.requires_grad)
    head_ids = {id(parameter) for parameter in head_params}
    backbone_params = [
        parameter
        for parameter in model.backbone.features.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.supcon_head_lr, "rho": args.sam_rho})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.supcon_backbone_lr, "rho": args.sam_rho})
    return build_optimizer_from_groups(param_groups, args)


def build_arcface_optimizer(model: MetricLearningEfficientNetB0, args: argparse.Namespace) -> Optimizer:
    head_params = []
    if not isinstance(model.front_end, IdentityFrontEnd):
        head_params.extend(parameter for parameter in model.front_end.parameters() if parameter.requires_grad)
    head_params.extend(parameter for parameter in model.embedding.parameters() if parameter.requires_grad)
    head_params.extend(parameter for parameter in model.embedding_norm.parameters() if parameter.requires_grad)
    head_params.extend(parameter for parameter in model.arcface_head.parameters() if parameter.requires_grad)
    head_ids = {id(parameter) for parameter in head_params}
    backbone_params = [
        parameter
        for parameter in model.backbone.features.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_lr, "rho": args.sam_rho})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr, "rho": args.sam_rho})
    return build_optimizer_from_groups(param_groups, args)


def base_optimizer_for_scheduler(optimizer: Optimizer) -> Optimizer:
    return optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer


def optimizer_learning_rates(optimizer: Optimizer) -> list[float]:
    return [group["lr"] for group in base_optimizer_for_scheduler(optimizer).param_groups]


def model_dtype_for_args(args: argparse.Namespace) -> torch.dtype:
    return torch.float64 if str(args.precision) == "64" else torch.float32


def autocast_enabled(device: torch.device, args: argparse.Namespace) -> bool:
    return device.type == "cuda" and str(args.precision) == "mixed"


def build_grad_scaler(device: torch.device, args: argparse.Namespace) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=autocast_enabled(device, args))


def move_images_to_device(images: torch.Tensor, device: torch.device, args: argparse.Namespace) -> torch.Tensor:
    return images.to(device=device, dtype=model_dtype_for_args(args), non_blocking=True)


class WarmupCosineScheduler:
    def __init__(self, optimizer: Optimizer, max_epochs: int, steps_per_epoch: int, warmup_epochs: int, warmup_steps: int = 0) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, max_epochs * max(1, steps_per_epoch))
        requested_warmup_steps = max(0, int(warmup_steps)) if warmup_steps > 0 else max(0, warmup_epochs * max(1, steps_per_epoch))
        self.warmup_steps = min(self.total_steps - 1, requested_warmup_steps) if self.total_steps > 1 else 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_index = 0

    def _factor(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        if self.total_steps <= self.warmup_steps + 1:
            return 1.0
        progress = float(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self) -> None:
        factor = self._factor(self.step_index)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor
        self.step_index += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "base_lrs": self.base_lrs,
            "step_index": self.step_index,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.total_steps = int(state_dict["total_steps"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.base_lrs = [float(value) for value in state_dict["base_lrs"]]
        self.step_index = int(state_dict["step_index"])


def build_scheduler(
    optimizer: Optimizer,
    max_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
    warmup_steps: int = 0,
) -> WarmupCosineScheduler:
    return WarmupCosineScheduler(
        optimizer,
        max_epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=warmup_epochs,
        warmup_steps=warmup_steps,
    )


def steps_per_epoch_for_dataset(dataset: Dataset, batch_size: int, max_batches: int) -> int:
    steps = math.ceil(len(dataset) / max(1, batch_size))
    return min(steps, max_batches) if max_batches > 0 else steps


def limited_batches(loader: DataLoader, max_batches: int):
    if max_batches <= 0:
        yield from loader
        return
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        yield batch


def improved_metric(best_loss: float, val_loss: float, min_delta: float) -> bool:
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
        total_loss += second_loss.item() * batch_size
        total_seen += batch_size
        train_progress["global_train_step"] += 1
        train_progress["global_source_samples_seen"] += batch_size
        progress.set_postfix(loss=f"{total_loss / max(1, total_seen):.4f}")

        if train_progress["global_train_step"] % log_every_steps == 0:
            event = {
                "event": "train_step",
                "stage": "supcon",
                "epoch": epoch,
                "epoch_step": step_in_epoch,
                "global_train_step": train_progress["global_train_step"],
                "batch_size": batch_size,
                "epoch_source_samples_seen": total_seen,
                "global_source_samples_seen": train_progress["global_source_samples_seen"],
                "running_loss": total_loss / max(1, total_seen),
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
) -> tuple[float, int, int]:
    model.train()
    freeze_frozen_batchnorms(backbone_modules)
    total_loss = 0.0
    total_seen = 0
    steps_done = 0

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

        batch_size = labels.size(0)
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
        )

        if train_progress["global_train_step"] % log_every_steps == 0:
            event = {
                "event": "train_step",
                "stage": "supcon",
                "epoch": epoch,
                "epoch_step": step_in_epoch,
                "global_train_step": train_progress["global_train_step"],
                "batch_size": batch_size,
                "epoch_source_samples_seen": total_seen,
                "global_source_samples_seen": train_progress["global_source_samples_seen"],
                "running_loss": total_loss / max(1, total_seen),
                "learning_rates": optimizer_learning_rates(optimizer),
            }
            log_json_event(log_path, event)

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
) -> float:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    eval_context = dict(eval_context or {})
    with torch.no_grad():
        for eval_step, (view_one, view_two, labels) in enumerate(limited_batches(loader, max_batches), start=1):
            view_one = move_images_to_device(view_one, device, args) if args is not None else view_one.to(device, non_blocking=True)
            view_two = move_images_to_device(view_two, device, args) if args is not None else view_two.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=args is not None and autocast_enabled(device, args)):
                emb_one = model.encode(view_one)
                emb_two = model.encode(view_two)
                proj_one = model.supcon_projection(emb_one)
                proj_two = model.supcon_projection(emb_two)
                loss = criterion(torch.stack([proj_one, proj_two], dim=1), labels)
            total_loss += loss.item() * labels.size(0)
            total_seen += labels.size(0)
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
                        "running_loss": total_loss / max(1, total_seen),
                        **eval_context,
                    },
                )
    return total_loss / max(1, total_seen)


def arcface_loss(
    model: MetricLearningEfficientNetB0,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model.classify(embeddings, labels)
    base_loss = criterion(logits, labels)
    plain_logits = model.classify(embeddings, labels=None)
    probabilities = torch.softmax(plain_logits, dim=1)
    true_class_confidence = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    predictions = plain_logits.argmax(dim=1)
    correct_mask = predictions == labels
    if correct_mask.any():
        confidence_gap = (args.confidence_threshold - true_class_confidence).clamp(min=0.0)
        penalty = confidence_gap[correct_mask].square().mean()
    else:
        penalty = embeddings.new_zeros(())
    return base_loss, penalty


def confidence_qualified_predictions(logits: torch.Tensor, confidence_threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    return predictions, confidence >= confidence_threshold


def confidence_qualified_correct_count(
    logits: torch.Tensor,
    labels: torch.Tensor,
    confidence_threshold: float,
) -> tuple[int, int]:
    predictions, confident_mask = confidence_qualified_predictions(logits, confidence_threshold)
    raw_correct = (predictions == labels)
    qualified_correct = raw_correct & confident_mask
    return int(qualified_correct.sum().item()), int(raw_correct.sum().item())


def train_arcface_epoch(
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
    total_raw_correct = 0.0
    total_confidence_gap = 0.0
    total_seen = 0
    total_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    progress = tqdm(enumerate(limited_batches(loader, max_batches), start=1), total=total_batches, leave=False)

    for step_in_epoch, (images, labels) in progress:
        images = move_images_to_device(images, device, args)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
            embeddings = model.encode(images)
            loss, penalty = arcface_loss(model, embeddings, labels, criterion, args)
        if isinstance(optimizer, SAM):
            scaler.scale(loss).backward()
            scaler.unscale_(base_optimizer_for_scheduler(optimizer))
            optimizer.first_step(zero_grad=True)

            with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
                embeddings = model.encode(images)
                second_loss, second_penalty = arcface_loss(model, embeddings, labels, criterion, args)
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
            second_penalty = penalty
        scheduler.step()

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
                eval_logits = model.classify(model.encode(images), labels=None)
            qualified_correct, raw_correct = confidence_qualified_correct_count(eval_logits, labels, args.confidence_threshold)

        batch_size = labels.size(0)
        total_loss += second_loss.item() * batch_size
        total_confidence_gap += second_penalty.item() * batch_size
        total_correct += qualified_correct
        total_raw_correct += raw_correct
        total_seen += batch_size
        train_progress["global_train_step"] += 1
        train_progress["global_source_samples_seen"] += batch_size
        progress.set_postfix(loss=f"{total_loss / max(1, total_seen):.4f}", acc=f"{total_correct / max(1, total_seen):.4f}")

        if train_progress["global_train_step"] % log_every_steps == 0:
            event = {
                "event": "train_step",
                "stage": "arcface",
                "phase_index": phase_index,
                "phase_name": phase_name,
                "epoch": epoch,
                "epoch_step": step_in_epoch,
                "global_train_step": train_progress["global_train_step"],
                "batch_size": batch_size,
                "epoch_source_samples_seen": total_seen,
                "global_source_samples_seen": train_progress["global_source_samples_seen"],
                "running_loss": total_loss / max(1, total_seen),
                "running_acc": total_correct / max(1, total_seen),
                "running_raw_acc": total_raw_correct / max(1, total_seen),
                "running_confidence_gap": total_confidence_gap / max(1, total_seen),
                "confidence_threshold": args.confidence_threshold,
                "learning_rates": optimizer_learning_rates(optimizer),
            }
            log_json_event(log_path, event)

    return total_loss / max(1, total_seen), total_correct / max(1, total_seen)


def train_arcface_steps(
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
) -> tuple[float, float, float, int, int]:
    model.train()
    freeze_frozen_batchnorms(backbone_modules)
    total_loss = 0.0
    total_correct = 0.0
    total_raw_correct = 0.0
    total_confidence_gap = 0.0
    total_seen = 0
    steps_done = 0

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
            loss, penalty = arcface_loss(model, embeddings, labels, criterion, args)
        if isinstance(optimizer, SAM):
            scaler.scale(loss).backward()
            scaler.unscale_(base_optimizer_for_scheduler(optimizer))
            optimizer.first_step(zero_grad=True)

            with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
                embeddings = model.encode(images)
                second_loss, second_penalty = arcface_loss(model, embeddings, labels, criterion, args)
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
            second_penalty = penalty
        scheduler.step()

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=autocast_enabled(device, args)):
                eval_logits = model.classify(model.encode(images), labels=None)
            qualified_correct, raw_correct = confidence_qualified_correct_count(eval_logits, labels, args.confidence_threshold)

        batch_size = labels.size(0)
        total_loss += second_loss.item() * batch_size
        total_confidence_gap += second_penalty.item() * batch_size
        total_correct += qualified_correct
        total_raw_correct += raw_correct
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
            stage="arcface",
            epoch=epoch,
            epoch_step=step_in_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            phase_index=phase_index,
            phase_name=phase_name,
        )

        if train_progress["global_train_step"] % log_every_steps == 0:
            event = {
                "event": "train_step",
                "stage": "arcface",
                "phase_index": phase_index,
                "phase_name": phase_name,
                "epoch": epoch,
                "epoch_step": step_in_epoch,
                "global_train_step": train_progress["global_train_step"],
                "batch_size": batch_size,
                "epoch_source_samples_seen": total_seen,
                "global_source_samples_seen": train_progress["global_source_samples_seen"],
                "running_loss": total_loss / max(1, total_seen),
                "running_acc": total_correct / max(1, total_seen),
                "running_raw_acc": total_raw_correct / max(1, total_seen),
                "running_confidence_gap": total_confidence_gap / max(1, total_seen),
                "confidence_threshold": args.confidence_threshold,
                "learning_rates": optimizer_learning_rates(optimizer),
            }
            log_json_event(log_path, event)

    return (
        total_loss / max(1, total_seen),
        total_correct / max(1, total_seen),
        total_raw_correct / max(1, total_seen),
        steps_done,
        total_seen,
    )


def evaluate_arcface(
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
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_raw_correct = 0
    total_seen = 0
    eval_context = dict(eval_context or {})
    with torch.no_grad():
        for eval_step, (images, labels) in enumerate(limited_batches(loader, max_batches), start=1):
            images = move_images_to_device(images, device, args) if args is not None else images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=args is not None and autocast_enabled(device, args)):
                embeddings = model.encode(images)
                margin_logits = model.classify(embeddings, labels)
                logits = model.classify(embeddings, labels=None)
                loss = criterion(margin_logits, labels)
            total_loss += loss.item() * labels.size(0)
            qualified_correct, raw_correct = confidence_qualified_correct_count(logits, labels, args.confidence_threshold)
            total_correct += qualified_correct
            total_raw_correct += raw_correct
            total_seen += labels.size(0)
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
                        "running_loss": total_loss / max(1, total_seen),
                        "running_acc": total_correct / max(1, total_seen),
                        "running_raw_acc": total_raw_correct / max(1, total_seen),
                        "confidence_threshold": args.confidence_threshold,
                        **eval_context,
                    },
                )
    return (
        total_loss / max(1, total_seen),
        total_correct / max(1, total_seen),
        total_raw_correct / max(1, total_seen),
    )


def collect_logits_and_labels(
    model: MetricLearningEfficientNetB0,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    log_path: Path,
    log_every_eval_steps: int,
    split: str,
    args: argparse.Namespace | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    with torch.no_grad():
        total_seen = 0
        for eval_step, (images, labels) in enumerate(limited_batches(loader, max_batches), start=1):
            images = move_images_to_device(images, device, args) if args is not None else images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=args is not None and autocast_enabled(device, args)):
                embeddings = model.encode(images)
                logits = model.classify(embeddings, labels=None)
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())
            total_seen += labels.size(0)
            if eval_step % log_every_eval_steps == 0:
                log_json_event(
                    log_path,
                    {
                        "event": "eval_step",
                        "stage": "final_evaluation",
                        "split": split,
                        "eval_step": eval_step,
                        "batch_size": labels.size(0),
                        "samples_seen": total_seen,
                    },
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
    qualified_matches = (predictions == targets) & (confidence >= confidence_threshold)
    num_classes = len(class_names)
    support = np.bincount(targets, minlength=num_classes)
    cm = confusion_matrix_from_predictions(targets, predictions, num_classes)

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
    accuracy = float(qualified_matches.mean())
    balanced_accuracy = float(sum(recalls) / num_classes)
    macro_roc_auc, weighted_roc_auc = macro_weighted(roc_aucs, support.tolist())
    macro_pr_auc, weighted_pr_auc = macro_weighted(pr_aucs, support.tolist())
    weighted_precision = float(np.average(precisions, weights=support)) if support.sum() > 0 else 0.0
    weighted_recall = float(np.average(recalls, weights=support)) if support.sum() > 0 else 0.0
    weighted_f1 = float(np.average(f1_scores, weights=support)) if support.sum() > 0 else 0.0
    top1 = accuracy
    top3 = top_k_accuracy(logits, targets, 3)

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

    return {
        "num_classes": num_classes,
        "num_samples": int(total_samples),
        "confidence_threshold": confidence_threshold,
        "rejection_rate": float((confidence < confidence_threshold).mean()),
        "raw_accuracy": raw_accuracy,
        "accuracy": accuracy,
        "top1_accuracy": top1,
        "top3_accuracy": top3,
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
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def save_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")


def log_json_event(path: Path, payload: object) -> None:
    if isinstance(payload, dict) and "timestamp" not in payload:
        payload = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), **payload}
    print(payload)
    append_jsonl(path, payload)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_training_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    torch.save(payload, path)


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
    torch.save(
        {
            "model_state_dict": cpu_state_dict(model),
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "args": vars(args),
            "train_progress": dict(train_progress),
            "resume": resume_payload,
        },
        path,
    )


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


def checkpoint_state_for_mode(resume_checkpoint: dict[str, Any], resume_mode: str) -> dict[str, torch.Tensor]:
    if resume_mode == "global_best":
        return resume_checkpoint.get("best_arcface_state", resume_checkpoint["model_state_dict"])
    if resume_mode == "phase_best":
        return resume_checkpoint.get(
            "phase_best_state",
            resume_checkpoint.get("best_arcface_state", resume_checkpoint["model_state_dict"]),
        )
    return resume_checkpoint["model_state_dict"]


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
    if resume_state.get("stage") == "arcface":
        return int(resume_state.get("phase_index", 1))
    return 1


def build_parser(use_gabor: bool) -> argparse.ArgumentParser:
    description = (
        "Metric-learning EfficientNet B0 training with deterministic 16x split-safe augmentation, "
        "SupCon warmup, SAM optimization, ArcFace classification, progressive unfreezing, and paper-style metrics"
    )
    if use_gabor:
        description += " using a Gabor front-end."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--augment-repeats", type=int, default=16)
    parser.add_argument("--augment-gaussian-sigmas", type=float, default=2.0)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--supcon-epochs", type=int, default=200)
    parser.add_argument("--supcon-unfreeze-backbone-modules", type=int, default=0)
    parser.add_argument("--skip-supcon", action="store_true")
    parser.add_argument("--head-epochs", type=int, default=200)
    parser.add_argument("--stage-epochs", type=int, default=200)
    parser.add_argument("--unfreeze-chunk-size", type=int, default=20)
    parser.add_argument("--max-progressive-phases", type=int, default=0)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--arcface-margin", type=float, default=0.35)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--supcon-head-lr", type=float, default=3e-4)
    parser.add_argument("--supcon-backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["sam", "adamw"], default="sam")
    parser.add_argument("--precision", choices=("mixed", "32", "64"), default="mixed")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weighted-sampling", action="store_true")
    parser.add_argument("--weights", choices=("default", "none"), default="default")
    parser.add_argument("--output-dir", default="Results/metric_learning_experiment")
    parser.add_argument("--log-file", default="logs/metric_learning_experiment.log.jsonl")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-mode", choices=("latest", "global_best", "phase_best"), default="latest")
    parser.add_argument("--resume-phase-index", type=int, default=0)
    parser.add_argument("--resume-phase-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--log-eval-every-steps", type=int, default=1000)
    parser.add_argument("--eval-every-train-steps", type=int, default=1024)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--supcon-early-stopping-patience", type=int, default=10)
    parser.add_argument("--head-early-stopping-patience", type=int, default=10)
    parser.add_argument("--stage-early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=1024)
    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    if use_gabor:
        parser.add_argument("--gabor-kernel-size", type=int, default=15)
        parser.add_argument("--gabor-orientations", type=int, default=8)
        parser.add_argument("--gabor-wavelengths", type=str, default="4.0,8.0")
        parser.add_argument("--gabor-sigma", type=float, default=4.5)
        parser.add_argument("--gabor-gamma", type=float, default=0.6)
    return parser


def run_experiment(args: argparse.Namespace, use_gabor: bool) -> int:
    if args.optimizer == "sam" and args.grad_accum_steps != 1:
        raise ValueError("SAM support in this trainer requires --grad-accum-steps 1")
    if args.unfreeze_chunk_size < 1:
        raise ValueError("--unfreeze-chunk-size must be >= 1")
    if args.augment_repeats < 1:
        raise ValueError("--augment-repeats must be >= 1")
    if args.augment_gaussian_sigmas <= 0:
        raise ValueError("--augment-gaussian-sigmas must be > 0")
    if args.log_every_steps < 1:
        raise ValueError("--log-every-steps must be >= 1")
    if args.log_eval_every_steps < 1:
        raise ValueError("--log-eval-every-steps must be >= 1")
    if args.eval_every_train_steps < 1:
        raise ValueError("--eval-every-train-steps must be >= 1")
    if args.confidence_threshold < 0.0 or args.confidence_threshold > 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1")
    if args.supcon_early_stopping_patience < 1:
        raise ValueError("--supcon-early-stopping-patience must be >= 1")
    if args.head_early_stopping_patience < 1:
        raise ValueError("--head-early-stopping-patience must be >= 1")
    if args.stage_early_stopping_patience < 1:
        raise ValueError("--stage-early-stopping-patience must be >= 1")

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    if str(args.precision) in {"32", "mixed"}:
        torch.set_float32_matmul_precision("high")
    model_name = "efficientnet_b0_metric_learning_gabor" if use_gabor else "efficientnet_b0_metric_learning"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "last.pt"
    step_checkpoint_path = output_dir / "step_last.pt"
    resume_path = Path(args.resume_checkpoint) if args.resume_checkpoint else checkpoint_path
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resume_checkpoint, resume_warning = load_resume_checkpoint(resume_path)
    if resume_checkpoint is None:
        log_path.write_text("", encoding="utf-8")
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
    if resume_warning is not None:
        log_json_event(log_path, {"event": "resume_checkpoint_ignored", "message": resume_warning})

    train_dataset, val_dataset, test_dataset, supcon_train_dataset, supcon_val_dataset = build_datasets(args)
    if use_gabor and "plastic" not in train_dataset.class_to_idx:
        note = "note: current dataset has no `plastic` class; the Gabor front-end remains an ablation for texture-sensitive separation."
        print(note)
        append_jsonl(log_path, {"event": "note", "message": note})

    train_sampler = (
        make_weighted_sampler(train_dataset, train_dataset.classes, train_dataset.target_for_index, args.seed + 101)
        if args.weighted_sampling
        else make_epoch_sampler(train_dataset, args.seed + 101, shuffle=True)
    )
    supcon_sampler = (
        make_weighted_sampler(supcon_train_dataset, train_dataset.classes, supcon_train_dataset.target_for_index, args.seed + 202)
        if args.weighted_sampling
        else make_epoch_sampler(supcon_train_dataset, args.seed + 202, shuffle=True)
    )

    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, args.prefetch_factor, shuffle=False, sampler=train_sampler)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, args.prefetch_factor, shuffle=False)
    test_loader = make_loader(test_dataset, args.batch_size, args.num_workers, args.prefetch_factor, shuffle=False)
    supcon_train_loader = make_loader(
        supcon_train_dataset, args.batch_size, args.num_workers, args.prefetch_factor, shuffle=False, sampler=supcon_sampler
    )
    supcon_val_loader = make_loader(supcon_val_dataset, args.batch_size, args.num_workers, args.prefetch_factor, shuffle=False)
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
            "train_samples_per_epoch": len(train_dataset),
            "val_samples_per_eval": len(val_dataset),
            "test_samples_per_eval": len(test_dataset),
            "train_steps_per_epoch": len(train_loader),
            "supcon_steps_per_epoch": len(supcon_train_loader),
            "val_steps_per_eval": len(val_loader),
            "test_steps_per_eval": len(test_loader),
            "eval_every_train_steps": args.eval_every_train_steps,
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
    model = MetricLearningEfficientNetB0(
        num_classes=len(train_dataset.classes),
        weights_mode=args.weights,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        use_gabor=use_gabor,
        args=args,
    ).to(device=device, dtype=model_dtype)
    if resume_checkpoint is not None:
        model.load_state_dict(checkpoint_state_for_mode(resume_checkpoint, args.resume_mode))
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
    best_arcface_state = (
        resume_checkpoint.get("best_arcface_state", cpu_state_dict(model))
        if resume_checkpoint is not None
        else cpu_state_dict(model)
    )
    supcon_best_state = (
        resume_checkpoint.get("supcon_best_state", cpu_state_dict(model))
        if resume_checkpoint is not None
        else cpu_state_dict(model)
    )
    resume_state = dict(resume_checkpoint.get("resume", {})) if resume_checkpoint is not None else {}
    resume_from_best_phase = resume_checkpoint is not None and args.resume_mode != "latest"
    if resume_from_best_phase:
        resume_state = {
            "stage": "arcface",
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
    thawed_supcon = set_trainability_for_supcon(model, backbone_modules, args.supcon_unfreeze_backbone_modules)
    supcon_optimizer = build_supcon_optimizer(model, args)
    supcon_scheduler = build_scheduler(
        base_optimizer_for_scheduler(supcon_optimizer),
        max_epochs=args.supcon_epochs,
        steps_per_epoch=steps_per_epoch_for_dataset(supcon_train_dataset, args.batch_size, args.max_train_batches),
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    _, supcon_trainable_params = parameter_counts(model)
    supcon_best_loss = float(resume_checkpoint.get("supcon_best_loss", float("inf"))) if resume_checkpoint is not None else float("inf")
    supcon_best_epoch = int(resume_checkpoint.get("supcon_best_epoch", 0)) if resume_checkpoint is not None else 0
    supcon_wait = int(resume_checkpoint.get("supcon_wait", 0)) if resume_checkpoint is not None else 0
    stop_supcon = False
    start_supcon_epoch = 1
    start_supcon_epoch_step = 0
    start_supcon_validation_index = 0
    supcon_completed = bool(args.skip_supcon)

    if resume_state.get("stage") == "supcon" and "optimizer_state_dict" in resume_state and "scheduler_state_dict" in resume_state:
        supcon_optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        supcon_scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_supcon_epoch = int(resume_state.get("epoch", 1))
        start_supcon_epoch_step = int(resume_state.get("epoch_step_completed", 0))
        start_supcon_validation_index = int(resume_state.get("validation_index", 0))
        supcon_completed = False
    elif resume_state.get("stage") == "supcon":
        log_json_event(log_path, {"event": "resume_fallback", "stage": "supcon", "reason": "missing_optimizer_or_scheduler_state"})
        supcon_completed = False
    elif resume_state.get("stage") == "arcface":
        supcon_completed = True
    elif args.skip_supcon or resume_from_best_phase:
        supcon_completed = True
        log_json_event(
            log_path,
            {
                "event": "supcon_skipped",
                "reason": "resume_from_best_phase" if resume_from_best_phase else "flag",
            },
        )

    if not supcon_completed:
        supcon_steps_full_epoch = steps_per_epoch_for_dataset(supcon_train_dataset, args.batch_size, args.max_train_batches)
        supcon_scaler = build_grad_scaler(device, args)
        if resume_state.get("stage") == "supcon" and "scaler_state_dict" in resume_state:
            supcon_scaler.load_state_dict(resume_state["scaler_state_dict"])
        for epoch in range(start_supcon_epoch, args.supcon_epochs + 1):
            train_dataset.set_epoch(augmentation_epoch_cursor)
            val_dataset.set_epoch(0)
            test_dataset.set_epoch(0)
            supcon_train_dataset.set_epoch(augmentation_epoch_cursor)
            supcon_val_dataset.set_epoch(0)
            start_index = start_supcon_epoch_step * args.batch_size if epoch == start_supcon_epoch else 0
            supcon_sampler.set_epoch(augmentation_epoch_cursor)
            supcon_sampler.set_start_index(start_index)
            supcon_steps_per_epoch = max(0, supcon_steps_full_epoch - (start_supcon_epoch_step if epoch == start_supcon_epoch else 0))
            epoch_steps_done = start_supcon_epoch_step if epoch == start_supcon_epoch else 0
            epoch_samples_seen = 0
            epoch_loss_sum = 0.0
            validation_index = start_supcon_validation_index if epoch == start_supcon_epoch else 0
            progress = tqdm(total=supcon_steps_per_epoch, leave=False)

            while epoch_steps_done < supcon_steps_full_epoch:
                step_window = min(args.eval_every_train_steps, supcon_steps_full_epoch - epoch_steps_done)
                step_start_index = epoch_steps_done * args.batch_size
                supcon_sampler.set_epoch(augmentation_epoch_cursor)
                supcon_sampler.set_start_index(step_start_index)
                supcon_iterator = iter(limited_batches(supcon_train_loader, args.max_train_batches))
                window_train_loss, steps_done, window_samples = train_supcon_steps(
                    model=model,
                    batch_iterator=supcon_iterator,
                    step_limit=step_window,
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
                )
                del supcon_iterator
                if steps_done == 0:
                    break

                epoch_steps_done += steps_done
                epoch_samples_seen += window_samples
                epoch_loss_sum += window_train_loss * window_samples
                validation_index += 1
                progress.update(steps_done)
                progress.set_postfix(loss=f"{epoch_loss_sum / max(1, epoch_samples_seen):.4f}")
                release_training_memory(device, supcon_train_loader)

                log_json_event(
                    log_path,
                    {
                        "event": "validation_started",
                        "stage": "supcon",
                        "epoch": epoch,
                        "validation_index": validation_index,
                        "epoch_step": epoch_steps_done,
                        "global_train_step": train_progress["global_train_step"],
                        "eval_batches": val_eval_batches,
                    },
                )
                val_loss = evaluate_supcon(
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
                        "epoch": epoch,
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
                        "epoch": epoch,
                        "validation_index": validation_index,
                        "epoch_step": epoch_steps_done,
                        "global_train_step": train_progress["global_train_step"],
                        "eval_batches": val_eval_batches,
                        "val_loss": val_loss,
                    },
                )
                if val_loss < supcon_best_loss - args.early_stopping_min_delta:
                    supcon_best_loss = val_loss
                    supcon_best_epoch = epoch
                    supcon_best_state = cpu_state_dict(model)
                    supcon_wait = 0
                else:
                    supcon_wait += 1

                row = {
                    "stage": "supcon",
                    "epoch": epoch,
                    "validation_index": validation_index,
                    "epoch_step": epoch_steps_done,
                    "global_train_step": train_progress["global_train_step"],
                    "window_train_loss": window_train_loss,
                    "epoch_running_train_loss": epoch_loss_sum / max(1, epoch_samples_seen),
                    "val_loss": val_loss,
                    "best_val_loss": supcon_best_loss,
                    "checks_without_improvement": supcon_wait,
                    "patience_unit": "validation_window",
                    "validation_window_steps": args.eval_every_train_steps,
                    "trainable_params": supcon_trainable_params,
                    "total_params": total_params,
                    "unfrozen_backbone_modules": args.supcon_unfreeze_backbone_modules,
                    "newly_unfrozen_tail_modules": thawed_supcon[-args.unfreeze_chunk_size :],
                }
                history.append(row)
                log_json_event(log_path, row)
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
                        "best_arcface_state": best_arcface_state,
                        "supcon_best_state": supcon_best_state,
                        "supcon_best_loss": supcon_best_loss,
                        "supcon_best_epoch": supcon_best_epoch,
                        "supcon_wait": supcon_wait,
                        "augmentation_epoch_cursor": augmentation_epoch_cursor,
                        "resume": {
                            "stage": "supcon",
                            "epoch": epoch,
                            "epoch_step_completed": epoch_steps_done,
                            "validation_index": validation_index,
                            "optimizer_state_dict": supcon_optimizer.state_dict(),
                            "scheduler_state_dict": supcon_scheduler.state_dict(),
                            "scaler_state_dict": supcon_scaler.state_dict(),
                        },
                    },
                )
                if supcon_wait >= args.supcon_early_stopping_patience:
                    event = {
                        "stage": "supcon",
                        "stopped_early": True,
                        "best_epoch": supcon_best_epoch,
                        "best_val_loss": supcon_best_loss,
                        "stopped_at_epoch_step": epoch_steps_done,
                        "global_train_step": train_progress["global_train_step"],
                    }
                    log_json_event(log_path, event)
                    stop_supcon = True
                    break
            progress.close()
            supcon_sampler.set_start_index(0)
            start_supcon_epoch_step = 0
            start_supcon_validation_index = 0
            if stop_supcon:
                break
            augmentation_epoch_cursor += 1
            release_training_memory(device, supcon_train_loader, supcon_val_loader)
        model.load_state_dict(supcon_best_state)
        release_training_memory(device, supcon_train_loader, supcon_val_loader)
    else:
        start_supcon_epoch_step = 0
        start_supcon_validation_index = 0

    arcface_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    phases = build_phase_plan(len(backbone_modules), args)
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
            "best_arcface_state": best_arcface_state,
            "supcon_best_state": supcon_best_state,
            "supcon_best_loss": supcon_best_loss,
            "supcon_best_epoch": supcon_best_epoch,
            "supcon_wait": supcon_wait,
            "augmentation_epoch_cursor": augmentation_epoch_cursor,
            "train_progress": train_progress,
            "resume": {
                "stage": "arcface",
                "phase_index": 1,
                "epoch": 1,
                "epoch_step_completed": 0,
                "validation_index": 0,
            },
        },
    )
    resume_phase_index = resolve_phase_start_index(args, phases, resume_state)
    start_phase_epoch = int(resume_state.get("epoch", 1)) if resume_state.get("stage") == "arcface" and not resume_from_best_phase else 1
    start_phase_epoch_step = int(resume_state.get("epoch_step_completed", 0)) if resume_state.get("stage") == "arcface" and not resume_from_best_phase else 0
    start_phase_validation_index = int(resume_state.get("validation_index", 0)) if resume_state.get("stage") == "arcface" and not resume_from_best_phase else 0

    for phase_index, phase in enumerate(phases, start=1):
        if phase_index < resume_phase_index:
            continue
        thawed = set_trainability_for_arcface(model, backbone_modules, phase.unfrozen_backbone_modules)
        optimizer = build_arcface_optimizer(model, args)
        scaler = build_grad_scaler(device, args)
        scheduler = build_scheduler(
            base_optimizer_for_scheduler(optimizer),
            max_epochs=phase.max_epochs,
            steps_per_epoch=steps_per_epoch_for_dataset(train_dataset, args.batch_size, args.max_train_batches),
            warmup_epochs=args.warmup_epochs,
            warmup_steps=args.warmup_steps,
        )
        _, trainable_params = parameter_counts(model)
        same_resume_phase = (
            resume_state.get("stage") == "arcface"
            and phase_index == resume_phase_index
            and "optimizer_state_dict" in resume_state
            and "scheduler_state_dict" in resume_state
        )
        if resume_state.get("stage") == "arcface" and phase_index == resume_phase_index and not same_resume_phase:
            log_json_event(
                log_path,
                {
                    "event": "resume_fallback",
                    "stage": "arcface",
                    "phase_index": phase_index,
                    "reason": "missing_optimizer_or_scheduler_state",
                },
            )
        if same_resume_phase:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            if "scaler_state_dict" in resume_state:
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            phase_best_loss = float(resume_state.get("phase_best_loss", float("inf")))
            phase_best_acc = float(resume_state.get("phase_best_acc", -1.0))
            phase_best_raw_acc = float(resume_state.get("phase_best_raw_acc", resume_state.get("phase_best_acc", -1.0)))
            phase_best_epoch = int(resume_state.get("phase_best_epoch", 0))
            phase_best_state = resume_checkpoint.get("phase_best_state", cpu_state_dict(model))
            phase_wait = int(resume_state.get("phase_wait", 0))
        else:
            phase_best_loss = float("inf")
            phase_best_acc = -1.0
            phase_best_raw_acc = -1.0
            phase_best_epoch = 0
            phase_best_state = cpu_state_dict(model)
            phase_wait = 0
        phase_patience = args.head_early_stopping_patience if phase_index == 1 else args.stage_early_stopping_patience

        for epoch in range(start_phase_epoch if same_resume_phase else 1, phase.max_epochs + 1):
            train_dataset.set_epoch(augmentation_epoch_cursor)
            val_dataset.set_epoch(0)
            test_dataset.set_epoch(0)
            start_index = start_phase_epoch_step * args.batch_size if same_resume_phase and epoch == start_phase_epoch else 0
            train_sampler.set_epoch(augmentation_epoch_cursor)
            train_sampler.set_start_index(start_index)
            phase_steps_full_epoch = steps_per_epoch_for_dataset(train_dataset, args.batch_size, args.max_train_batches)
            phase_steps_per_epoch = max(0, phase_steps_full_epoch - (start_phase_epoch_step if same_resume_phase and epoch == start_phase_epoch else 0))
            epoch_steps_done = start_phase_epoch_step if same_resume_phase and epoch == start_phase_epoch else 0
            epoch_samples_seen = 0
            epoch_loss_sum = 0.0
            epoch_correct_sum = 0.0
            epoch_raw_correct_sum = 0.0
            validation_index = start_phase_validation_index if same_resume_phase and epoch == start_phase_epoch else 0
            phase_stopped = False
            progress = tqdm(total=phase_steps_per_epoch, leave=False)

            while epoch_steps_done < phase_steps_full_epoch:
                step_window = min(args.eval_every_train_steps, phase_steps_full_epoch - epoch_steps_done)
                step_start_index = epoch_steps_done * args.batch_size
                train_sampler.set_epoch(augmentation_epoch_cursor)
                train_sampler.set_start_index(step_start_index)
                phase_iterator = iter(limited_batches(train_loader, args.max_train_batches))
                window_train_loss, window_train_acc, window_train_raw_acc, steps_done, window_samples = train_arcface_steps(
                    model=model,
                    batch_iterator=phase_iterator,
                    step_limit=step_window,
                    criterion=arcface_criterion,
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
                )
                del phase_iterator
                if steps_done == 0:
                    break

                epoch_steps_done += steps_done
                epoch_samples_seen += window_samples
                epoch_loss_sum += window_train_loss * window_samples
                epoch_correct_sum += window_train_acc * window_samples
                epoch_raw_correct_sum += window_train_raw_acc * window_samples
                validation_index += 1
                progress.update(steps_done)
                progress.set_postfix(
                    loss=f"{epoch_loss_sum / max(1, epoch_samples_seen):.4f}",
                    acc=f"{epoch_correct_sum / max(1, epoch_samples_seen):.4f}",
                )
                release_training_memory(device, train_loader)

                log_json_event(
                    log_path,
                    {
                        "event": "validation_started",
                        "stage": "arcface",
                        "phase_index": phase_index,
                        "phase_name": phase.name,
                        "epoch_in_phase": epoch,
                        "validation_index": validation_index,
                        "epoch_step": epoch_steps_done,
                        "global_train_step": train_progress["global_train_step"],
                        "eval_batches": val_eval_batches,
                    },
                )
                val_loss, val_acc, val_raw_acc = evaluate_arcface(
                    model=model,
                    loader=val_loader,
                    criterion=arcface_criterion,
                    device=device,
                    max_batches=args.max_eval_batches,
                    log_path=log_path,
                    log_every_eval_steps=args.log_eval_every_steps,
                    stage="arcface",
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
                release_training_memory(device, val_loader)
                log_json_event(
                    log_path,
                    {
                        "event": "validation_finished",
                        "stage": "arcface",
                        "phase_index": phase_index,
                        "phase_name": phase.name,
                        "epoch_in_phase": epoch,
                        "validation_index": validation_index,
                        "epoch_step": epoch_steps_done,
                        "global_train_step": train_progress["global_train_step"],
                        "eval_batches": val_eval_batches,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_raw_acc": val_raw_acc,
                    },
                )
                if improved_metric(phase_best_loss, val_loss, args.early_stopping_min_delta):
                    phase_best_loss = val_loss
                    phase_best_acc = val_acc
                    phase_best_raw_acc = val_raw_acc
                    phase_best_epoch = epoch
                    phase_best_state = cpu_state_dict(model)
                    phase_wait = 0
                else:
                    phase_wait += 1

                if improved_metric(best_val_loss, val_loss, args.early_stopping_min_delta):
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_val_raw_acc = val_raw_acc
                    best_arcface_state = cpu_state_dict(model)

                row = {
                    "stage": "arcface",
                    "phase_index": phase_index,
                    "phase_name": phase.name,
                    "epoch_in_phase": epoch,
                    "validation_index": validation_index,
                    "epoch_step": epoch_steps_done,
                    "global_train_step": train_progress["global_train_step"],
                    "phase_max_epochs": phase.max_epochs,
                    "unfrozen_backbone_modules": phase.unfrozen_backbone_modules,
                    "newly_unfrozen_tail_modules": thawed[-args.unfreeze_chunk_size :],
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "window_train_loss": window_train_loss,
                    "window_train_acc": window_train_acc,
                    "window_train_raw_acc": window_train_raw_acc,
                    "epoch_running_train_loss": epoch_loss_sum / max(1, epoch_samples_seen),
                    "epoch_running_train_acc": epoch_correct_sum / max(1, epoch_samples_seen),
                    "epoch_running_train_raw_acc": epoch_raw_correct_sum / max(1, epoch_samples_seen),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_raw_acc": val_raw_acc,
                    "phase_best_val_loss": phase_best_loss,
                    "phase_best_val_acc": phase_best_acc,
                    "phase_best_val_raw_acc": phase_best_raw_acc,
                    "checks_without_improvement": phase_wait,
                    "patience_limit": phase_patience,
                    "patience_unit": "validation_window",
                    "validation_window_steps": args.eval_every_train_steps,
                }
                history.append(row)
                log_json_event(log_path, row)
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
                        "best_arcface_state": best_arcface_state,
                        "supcon_best_state": supcon_best_state,
                        "supcon_best_loss": supcon_best_loss,
                        "supcon_best_epoch": supcon_best_epoch,
                        "supcon_wait": supcon_wait,
                        "augmentation_epoch_cursor": augmentation_epoch_cursor,
                        "train_progress": train_progress,
                        "phase_best_state": phase_best_state,
                        "resume": {
                            "stage": "arcface",
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
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                        },
                    },
                )

                if phase_wait >= phase_patience:
                    event = {
                        "stage": "arcface",
                        "phase_index": phase_index,
                        "phase_name": phase.name,
                        "stopped_early": True,
                        "best_epoch_in_phase": phase_best_epoch,
                        "phase_best_val_loss": phase_best_loss,
                        "phase_best_val_acc": phase_best_acc,
                        "phase_best_val_raw_acc": phase_best_raw_acc,
                        "stopped_at_epoch_step": epoch_steps_done,
                        "global_train_step": train_progress["global_train_step"],
                    }
                    log_json_event(log_path, event)
                    phase_stopped = True
                    break
            progress.close()
            train_sampler.set_start_index(0)
            start_phase_epoch_step = 0
            start_phase_validation_index = 0
            if phase_stopped:
                break
            augmentation_epoch_cursor += 1
            release_training_memory(device, train_loader, val_loader)

        model.load_state_dict(phase_best_state)
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
                "best_arcface_state": best_arcface_state,
                "supcon_best_state": supcon_best_state,
                "supcon_best_loss": supcon_best_loss,
                "supcon_best_epoch": supcon_best_epoch,
                "supcon_wait": supcon_wait,
                "augmentation_epoch_cursor": augmentation_epoch_cursor,
                "train_progress": train_progress,
                "phase_best_state": phase_best_state,
                "resume": {
                    "stage": "arcface",
                    "phase_index": phase_index + 1,
                    "epoch": 1,
                    "epoch_step_completed": 0,
                    "validation_index": 0,
                },
            },
        )
        start_phase_epoch = 1
        resume_phase_index = phase_index + 1

    model.load_state_dict(best_arcface_state)
    release_training_memory(device, train_loader, val_loader, supcon_train_loader, supcon_val_loader)

    log_json_event(log_path, {"event": "final_evaluation_started", "split": "val", "eval_batches": val_eval_batches})
    val_logits, val_targets = collect_logits_and_labels(
        model,
        val_loader,
        device,
        args.max_eval_batches,
        log_path,
        args.log_eval_every_steps,
        "val",
        args,
    )
    release_training_memory(device, val_loader)
    log_json_event(log_path, {"event": "final_evaluation_finished", "split": "val", "eval_batches": val_eval_batches})
    log_json_event(log_path, {"event": "final_evaluation_started", "split": "test", "eval_batches": test_eval_batches})
    test_logits, test_targets = collect_logits_and_labels(
        model,
        test_loader,
        device,
        args.max_eval_batches,
        log_path,
        args.log_eval_every_steps,
        "test",
        args,
    )
    release_training_memory(device, test_loader)
    log_json_event(log_path, {"event": "final_evaluation_finished", "split": "test", "eval_batches": test_eval_batches})

    val_metrics = compute_classification_metrics(val_logits, val_targets, train_dataset.classes, args.confidence_threshold)
    test_metrics = compute_classification_metrics(test_logits, test_targets, train_dataset.classes, args.confidence_threshold)

    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "args": vars(args),
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    torch.save(final_checkpoint, output_dir / "best.pt")

    metrics = {
        "model_name": model_name,
        "device": str(device),
        "output_dir": str(output_dir),
        "log_file": str(log_path),
        "weights": args.weights,
        "class_names": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "source_train_count": train_dataset.source_count(),
        "source_val_count": val_dataset.source_count(),
        "source_test_count": test_dataset.source_count(),
        "train_samples_per_epoch": len(train_dataset),
        "val_samples_per_eval": len(val_dataset),
        "test_samples_per_eval": len(test_dataset),
        "augmentation_bank_train_count": len(train_dataset),
        "augmentation_bank_val_count": len(val_dataset),
        "augmentation_bank_test_count": len(test_dataset),
        "augment_repeats": args.augment_repeats,
        "augment_gaussian_sigmas": args.augment_gaussian_sigmas,
        "train_class_counts": class_counts(train_dataset),
        "val_class_counts": class_counts(val_dataset),
        "test_class_counts": class_counts(test_dataset),
        "effective_train_class_counts": effective_class_counts(train_dataset),
        "effective_val_class_counts": effective_class_counts(val_dataset),
        "effective_test_class_counts": effective_class_counts(test_dataset),
        "supcon": {
            "temperature": args.supcon_temperature,
            "max_epochs": args.supcon_epochs,
            "best_val_loss": supcon_best_loss,
            "best_epoch": supcon_best_epoch,
            "unfrozen_backbone_modules": args.supcon_unfreeze_backbone_modules,
        },
        "arcface": {
            "margin": args.arcface_margin,
            "scale": args.arcface_scale,
            "phase_plan": [asdict(phase) for phase in phases],
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_val_raw_acc": best_val_raw_acc,
        },
        "optimization": {
            "optimizer": "AdamW + SAM",
            "scheduler": "Linear warmup + cosine decay",
            "sam_rho": args.sam_rho,
            "weight_decay": args.weight_decay,
            "adam_betas": [args.adam_beta1, args.adam_beta2],
            "warmup_epochs": args.warmup_epochs,
            "warmup_steps": args.warmup_steps,
        },
        "image_augmentation": {
            "gaussian_sigmas": args.augment_gaussian_sigmas,
            "shadow_probability": SHADOW_PROBABILITY,
            "glare_probability": GLARE_PROBABILITY,
        },
        "early_stopping": {
            "supcon_patience": args.supcon_early_stopping_patience,
            "head_patience": args.head_early_stopping_patience,
            "stage_patience": args.stage_early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "monitor": "val_loss",
            "check_unit": "validation_window",
            "eval_every_train_steps": args.eval_every_train_steps,
        },
        "embedding_dim": args.embedding_dim,
        "projection_dim": args.projection_dim,
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    if use_gabor:
        metrics["gabor"] = {
            "kernel_size": args.gabor_kernel_size,
            "orientations": args.gabor_orientations,
            "wavelengths": parse_wavelengths(args.gabor_wavelengths),
            "sigma": args.gabor_sigma,
            "gamma": args.gabor_gamma,
        }

    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "validation_metrics.json", val_metrics)
    save_json(output_dir / "test_metrics.json", test_metrics)
    final_event = {
        "event": "run_finished",
        "validation_accuracy": val_metrics["accuracy"],
        "validation_raw_accuracy": val_metrics["raw_accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_raw_accuracy": test_metrics["raw_accuracy"],
    }
    log_json_event(log_path, final_event)
    print(
        {
            "validation_accuracy": val_metrics["accuracy"],
            "validation_raw_accuracy": val_metrics["raw_accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_raw_accuracy": test_metrics["raw_accuracy"],
        }
    )
    return 0
