#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets
from torchvision.models import EfficientNet_V2_L_Weights, efficientnet_v2_l
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_OFFSETS = {"train": 0, "val": 10_000_000, "test": 20_000_000}


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
    ) -> None:
        self.base_dataset = datasets.ImageFolder(root)
        self.image_size = image_size
        self.augment_repeats = augment_repeats
        self.split_name = split_name
        self.seed = seed
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx
        self.samples = self.base_dataset.samples
        self.targets = self.base_dataset.targets

    def __len__(self) -> int:
        return len(self.samples) * self.augment_repeats

    def source_count(self) -> int:
        return len(self.samples)

    def target_for_index(self, index: int) -> int:
        source_index = index % len(self.samples)
        return self.targets[source_index]

    def _seed_for_variant(self, source_index: int, variant_index: int, view_offset: int = 0) -> int:
        return (
            self.seed * 1_000_003
            + SPLIT_OFFSETS[self.split_name]
            + source_index * 9_973
            + variant_index * 99_991
            + view_offset * 1_299_721
        )

    def load_augmented(self, source_index: int, variant_index: int, view_offset: int = 0) -> tuple[torch.Tensor, int]:
        path, target = self.samples[source_index]
        image = self.base_dataset.loader(path).convert("RGB")
        rng = random.Random(self._seed_for_variant(source_index, variant_index, view_offset))
        tensor = augmented_tensor_from_image(image, self.image_size, rng)
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

    def target_for_index(self, index: int) -> int:
        return self.base_dataset.target_for_index(index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        source_index = index % self.base_dataset.source_count()
        variant_index = index // self.base_dataset.source_count()
        view_one, target = self.base_dataset.load_augmented(source_index, variant_index, view_offset=0)
        view_two, _ = self.base_dataset.load_augmented(source_index, variant_index, view_offset=1)
        return view_one, view_two, target


def uniform(rng: random.Random, low: float, high: float) -> float:
    return low + (high - low) * rng.random()


def random_resized_crop(image: Image.Image, image_size: int, rng: random.Random) -> Image.Image:
    width, height = image.size
    area = width * height
    log_ratio = (math.log(0.7), math.log(1.35))

    for _ in range(10):
        target_area = area * uniform(rng, 0.55, 1.0)
        aspect_ratio = math.exp(uniform(rng, *log_ratio))
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


def random_perspective(image: Image.Image, rng: random.Random) -> Image.Image:
    distortion = uniform(rng, 0.0, 0.22)
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
        point(uniform(rng, 0.0, dx), uniform(rng, 0.0, dy)),
        point(uniform(rng, width - 1 - dx, width - 1), uniform(rng, 0.0, dy)),
        point(uniform(rng, width - 1 - dx, width - 1), uniform(rng, height - 1 - dy, height - 1)),
        point(uniform(rng, 0.0, dx), uniform(rng, height - 1 - dy, height - 1)),
    ]
    return TF.perspective(
        image,
        startpoints=startpoints,
        endpoints=endpoints,
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )


def jpeg_compress(image: Image.Image, rng: random.Random) -> Image.Image:
    quality = int(round(uniform(rng, 35.0, 100.0)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    compressed = Image.open(buffer)
    return compressed.convert("RGB")


def apply_gaussian_noise(tensor: torch.Tensor, rng: random.Random) -> torch.Tensor:
    noise_std = uniform(rng, 0.0, 0.06)
    if noise_std < 1e-5:
        return tensor
    generator = torch.Generator()
    generator.manual_seed(rng.randrange(0, 2**31 - 1))
    noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
    return tensor + (noise * noise_std)


def apply_channel_shift(tensor: torch.Tensor, rng: random.Random) -> torch.Tensor:
    shifts = torch.tensor(
        [uniform(rng, -0.05, 0.05), uniform(rng, -0.05, 0.05), uniform(rng, -0.05, 0.05)],
        dtype=tensor.dtype,
    ).view(3, 1, 1)
    return tensor + shifts


def apply_grayscale_mix(tensor: torch.Tensor, rng: random.Random) -> torch.Tensor:
    mix = uniform(rng, 0.0, 0.15)
    gray = tensor.mean(dim=0, keepdim=True).repeat(3, 1, 1)
    return tensor.mul(1.0 - mix).add(gray.mul(mix))


def apply_illumination_gradient(tensor: torch.Tensor, rng: random.Random) -> torch.Tensor:
    _, height, width = tensor.shape
    horizontal = rng.random() < 0.5
    start = uniform(rng, 0.7, 1.05)
    end = uniform(rng, 0.7, 1.05)
    if horizontal:
        ramp = torch.linspace(start, end, steps=width, dtype=tensor.dtype).view(1, 1, width)
        mask = ramp.expand(1, height, width)
    else:
        ramp = torch.linspace(start, end, steps=height, dtype=tensor.dtype).view(1, height, 1)
        mask = ramp.expand(1, height, width)
    return tensor * mask


def apply_cutout(tensor: torch.Tensor, rng: random.Random) -> torch.Tensor:
    _, height, width = tensor.shape
    cutout_count = rng.randint(1, 3)
    for _ in range(cutout_count):
        cutout_h = max(1, int(round(uniform(rng, 0.04, 0.18) * height)))
        cutout_w = max(1, int(round(uniform(rng, 0.04, 0.18) * width)))
        top = rng.randint(0, max(0, height - cutout_h))
        left = rng.randint(0, max(0, width - cutout_w))
        fill = torch.tensor(
            [uniform(rng, 0.0, 1.0), uniform(rng, 0.0, 1.0), uniform(rng, 0.0, 1.0)],
            dtype=tensor.dtype,
        ).view(3, 1, 1)
        tensor[:, top : top + cutout_h, left : left + cutout_w] = fill
    return tensor


def augmented_tensor_from_image(image: Image.Image, image_size: int, rng: random.Random) -> torch.Tensor:
    image = random_resized_crop(image, image_size, rng)

    if rng.random() < 0.5:
        image = TF.hflip(image)
    if rng.random() < 0.2:
        image = TF.vflip(image)

    translate = (
        int(round(uniform(rng, -0.14, 0.14) * image_size)),
        int(round(uniform(rng, -0.14, 0.14) * image_size)),
    )
    image = TF.affine(
        image,
        angle=uniform(rng, -35.0, 35.0),
        translate=translate,
        scale=uniform(rng, 0.82, 1.18),
        shear=[uniform(rng, -14.0, 14.0), uniform(rng, -10.0, 10.0)],
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )
    image = random_perspective(image, rng)
    image = TF.adjust_brightness(image, uniform(rng, 0.65, 1.35))
    image = TF.adjust_contrast(image, uniform(rng, 0.7, 1.4))
    image = TF.adjust_saturation(image, uniform(rng, 0.7, 1.4))
    image = TF.adjust_hue(image, uniform(rng, -0.08, 0.08))
    image = TF.adjust_gamma(image, uniform(rng, 0.75, 1.35), gain=1.0)
    image = TF.adjust_sharpness(image, uniform(rng, 0.5, 1.8))
    image = jpeg_compress(image, rng)

    tensor = TF.to_tensor(image)
    blur_sigma = uniform(rng, 0.0, 1.8)
    if blur_sigma > 0.05:
        kernel_size = 5 if blur_sigma < 1.0 else 7
        tensor = TF.gaussian_blur(
            tensor,
            kernel_size=[kernel_size, kernel_size],
            sigma=[blur_sigma, blur_sigma],
        )
    tensor = apply_gaussian_noise(tensor, rng)
    tensor = apply_channel_shift(tensor, rng)
    tensor = apply_grayscale_mix(tensor, rng)
    tensor = apply_illumination_gradient(tensor, rng)
    tensor = apply_cutout(tensor, rng)
    tensor = tensor.clamp(0.0, 1.0)
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


class MetricLearningEfficientNetV2L(nn.Module):
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
        weights = EfficientNet_V2_L_Weights.DEFAULT if weights_mode == "default" else None
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

        backbone = efficientnet_v2_l(weights=weights)
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


def make_weighted_sampler(dataset: Dataset, classes: list[str], target_fn) -> WeightedRandomSampler:
    counts = {name: 0 for name in classes}
    for index in range(len(dataset)):
        counts[classes[target_fn(index)]] += 1
    weights = [1.0 / counts[classes[target_fn(index)]] for index in range(len(dataset))]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


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
    train_dataset = DeterministicAugmentedImageFolder(root / "train", args.image_size, args.augment_repeats, "train", args.seed)
    val_dataset = DeterministicAugmentedImageFolder(root / "val", args.image_size, args.augment_repeats, "val", args.seed)
    test_dataset = DeterministicAugmentedImageFolder(root / "test", args.image_size, args.augment_repeats, "test", args.seed)
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


def backbone_leaf_modules(model: MetricLearningEfficientNetV2L) -> list[tuple[str, nn.Module]]:
    modules: list[tuple[str, nn.Module]] = []
    for name, module in model.backbone.features.named_modules():
        if not name:
            continue
        if any(True for _ in module.parameters(recurse=False)):
            modules.append((f"backbone.features.{name}", module))
    return modules


def set_trainability_for_supcon(
    model: MetricLearningEfficientNetV2L,
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
    model: MetricLearningEfficientNetV2L,
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


def build_supcon_optimizer(model: MetricLearningEfficientNetV2L, args: argparse.Namespace) -> SAM:
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
    return build_sam_optimizer(param_groups, args.sam_rho, args.weight_decay, (args.adam_beta1, args.adam_beta2))


def build_arcface_optimizer(model: MetricLearningEfficientNetV2L, args: argparse.Namespace) -> SAM:
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
    return build_sam_optimizer(param_groups, args.sam_rho, args.weight_decay, (args.adam_beta1, args.adam_beta2))


class WarmupCosineScheduler:
    def __init__(self, optimizer: Optimizer, max_epochs: int, steps_per_epoch: int, warmup_epochs: int) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, max_epochs * max(1, steps_per_epoch))
        self.warmup_steps = min(
            self.total_steps - 1,
            max(0, warmup_epochs * max(1, steps_per_epoch)),
        ) if self.total_steps > 1 else 0
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


def build_scheduler(optimizer: Optimizer, max_epochs: int, steps_per_epoch: int, warmup_epochs: int) -> WarmupCosineScheduler:
    total_steps = max(1, max_epochs * max(1, steps_per_epoch))
    warmup_steps = min(total_steps - 1, max(0, warmup_epochs * max(1, steps_per_epoch))) if total_steps > 1 else 0
    del total_steps, warmup_steps
    return WarmupCosineScheduler(optimizer, max_epochs=max_epochs, steps_per_epoch=steps_per_epoch, warmup_epochs=warmup_epochs)


def limited_batches(loader: DataLoader, max_batches: int):
    if max_batches <= 0:
        yield from loader
        return
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        yield batch


def improved_metric(best_loss: float, best_acc: float, val_loss: float, val_acc: float, min_delta: float) -> bool:
    if val_loss < best_loss - min_delta:
        return True
    if abs(val_loss - best_loss) <= min_delta and val_acc > best_acc:
        return True
    return False


def train_supcon_epoch(
    model: MetricLearningEfficientNetV2L,
    loader: DataLoader,
    criterion: SupConLoss,
    optimizer: SAM,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    backbone_modules: list[tuple[str, nn.Module]],
    max_batches: int,
) -> float:
    model.train()
    freeze_frozen_batchnorms(backbone_modules)
    total_loss = 0.0
    total_seen = 0
    total_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    progress = tqdm(enumerate(limited_batches(loader, max_batches), start=1), total=total_batches, leave=False)

    for _, (view_one, view_two, labels) in progress:
        view_one = view_one.to(device, non_blocking=True)
        view_two = view_two.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            emb_one = model.encode(view_one)
            emb_two = model.encode(view_two)
            proj_one = model.supcon_projection(emb_one)
            proj_two = model.supcon_projection(emb_two)
            stacked = torch.stack([proj_one, proj_two], dim=1)
            loss = criterion(stacked, labels)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            emb_one = model.encode(view_one)
            emb_two = model.encode(view_two)
            proj_one = model.supcon_projection(emb_one)
            proj_two = model.supcon_projection(emb_two)
            stacked = torch.stack([proj_one, proj_two], dim=1)
            second_loss = criterion(stacked, labels)
        second_loss.backward()
        optimizer.second_step(zero_grad=True)
        scheduler.step()

        batch_size = labels.size(0)
        total_loss += second_loss.item() * batch_size
        total_seen += batch_size
        progress.set_postfix(loss=f"{total_loss / max(1, total_seen):.4f}")

    return total_loss / max(1, total_seen)


def evaluate_supcon(
    model: MetricLearningEfficientNetV2L,
    loader: DataLoader,
    criterion: SupConLoss,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    with torch.no_grad():
        for view_one, view_two, labels in limited_batches(loader, max_batches):
            view_one = view_one.to(device, non_blocking=True)
            view_two = view_two.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            emb_one = model.encode(view_one)
            emb_two = model.encode(view_two)
            proj_one = model.supcon_projection(emb_one)
            proj_two = model.supcon_projection(emb_two)
            loss = criterion(torch.stack([proj_one, proj_two], dim=1), labels)
            total_loss += loss.item() * labels.size(0)
            total_seen += labels.size(0)
    return total_loss / max(1, total_seen)


def train_arcface_epoch(
    model: MetricLearningEfficientNetV2L,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: SAM,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    backbone_modules: list[tuple[str, nn.Module]],
    max_batches: int,
) -> tuple[float, float]:
    model.train()
    freeze_frozen_batchnorms(backbone_modules)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    total_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    progress = tqdm(enumerate(limited_batches(loader, max_batches), start=1), total=total_batches, leave=False)

    for _, (images, labels) in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            embeddings = model.encode(images)
            logits = model.classify(embeddings, labels)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            embeddings = model.encode(images)
            logits = model.classify(embeddings, labels)
            second_loss = criterion(logits, labels)
        second_loss.backward()
        optimizer.second_step(zero_grad=True)
        scheduler.step()

        with torch.no_grad():
            eval_logits = model.classify(model.encode(images), labels=None)
            predictions = eval_logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += second_loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_seen += batch_size
        progress.set_postfix(loss=f"{total_loss / max(1, total_seen):.4f}", acc=f"{total_correct / max(1, total_seen):.4f}")

    return total_loss / max(1, total_seen), total_correct / max(1, total_seen)


def evaluate_arcface(
    model: MetricLearningEfficientNetV2L,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for images, labels in limited_batches(loader, max_batches):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            embeddings = model.encode(images)
            margin_logits = model.classify(embeddings, labels)
            logits = model.classify(embeddings, labels=None)
            loss = criterion(margin_logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += labels.size(0)
    return total_loss / max(1, total_seen), total_correct / max(1, total_seen)


def collect_logits_and_labels(
    model: MetricLearningEfficientNetV2L,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in limited_batches(loader, max_batches):
            images = images.to(device, non_blocking=True)
            embeddings = model.encode(images)
            logits = model.classify(embeddings, labels=None)
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())
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


def compute_classification_metrics(logits: np.ndarray, targets: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = logits.argmax(axis=1)
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

    accuracy = float((predictions == targets).mean())
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


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def build_parser(use_gabor: bool) -> argparse.ArgumentParser:
    description = (
        "Metric-learning EfficientNet V2-L training with deterministic 20x split-safe augmentation, "
        "SupCon warmup, SAM optimization, ArcFace classification, progressive unfreezing, and paper-style metrics"
    )
    if use_gabor:
        description += " using a Gabor front-end."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--augment-repeats", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--supcon-epochs", type=int, default=200)
    parser.add_argument("--supcon-unfreeze-backbone-modules", type=int, default=0)
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
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weighted-sampling", action="store_true")
    parser.add_argument("--weights", choices=("default", "none"), default="default")
    parser.add_argument("--output-dir", default="Results/metric_learning_experiment")
    parser.add_argument("--log-file", default="logs/metric_learning_experiment.log.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
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
    if args.grad_accum_steps != 1:
        raise ValueError("SAM support in this trainer requires --grad-accum-steps 1")
    if args.unfreeze_chunk_size < 1:
        raise ValueError("--unfreeze-chunk-size must be >= 1")
    if args.augment_repeats < 1:
        raise ValueError("--augment-repeats must be >= 1")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be >= 1")

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    append_jsonl(
        log_path,
        {
            "event": "run_started",
            "model_name": "efficientnet_v2_l_metric_learning_gabor" if use_gabor else "efficientnet_v2_l_metric_learning",
            "output_dir": str(output_dir),
            "log_file": str(log_path),
            "args": vars(args),
        },
    )

    train_dataset, val_dataset, test_dataset, supcon_train_dataset, supcon_val_dataset = build_datasets(args)
    if use_gabor and "plastic" not in train_dataset.class_to_idx:
        note = "note: current dataset has no `plastic` class; the Gabor front-end remains an ablation for texture-sensitive separation."
        print(note)
        append_jsonl(log_path, {"event": "note", "message": note})

    train_sampler = make_weighted_sampler(train_dataset, train_dataset.classes, train_dataset.target_for_index) if args.weighted_sampling else None
    supcon_sampler = make_weighted_sampler(supcon_train_dataset, train_dataset.classes, supcon_train_dataset.target_for_index) if args.weighted_sampling else None

    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=not args.weighted_sampling, sampler=train_sampler)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, shuffle=False)
    test_loader = make_loader(test_dataset, args.batch_size, args.num_workers, shuffle=False)
    supcon_train_loader = make_loader(supcon_train_dataset, args.batch_size, args.num_workers, shuffle=not args.weighted_sampling, sampler=supcon_sampler)
    supcon_val_loader = make_loader(supcon_val_dataset, args.batch_size, args.num_workers, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricLearningEfficientNetV2L(
        num_classes=len(train_dataset.classes),
        weights_mode=args.weights,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        use_gabor=use_gabor,
        args=args,
    ).to(device)
    backbone_modules = backbone_leaf_modules(model)

    history: list[dict[str, Any]] = []
    total_params, _ = parameter_counts(model)
    best_val_loss = float("inf")
    best_val_acc = -1.0

    supcon_loss = SupConLoss(args.supcon_temperature)
    thawed_supcon = set_trainability_for_supcon(model, backbone_modules, args.supcon_unfreeze_backbone_modules)
    supcon_optimizer = build_supcon_optimizer(model, args)
    supcon_scheduler = build_scheduler(
        supcon_optimizer.base_optimizer,
        max_epochs=args.supcon_epochs,
        steps_per_epoch=min(len(supcon_train_loader), args.max_train_batches) if args.max_train_batches > 0 else len(supcon_train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    _, supcon_trainable_params = parameter_counts(model)
    supcon_best_loss = float("inf")
    supcon_best_epoch = 0
    supcon_best_state = cpu_state_dict(model)
    supcon_wait = 0

    for epoch in range(1, args.supcon_epochs + 1):
        train_loss = train_supcon_epoch(
            model=model,
            loader=supcon_train_loader,
            criterion=supcon_loss,
            optimizer=supcon_optimizer,
            scheduler=supcon_scheduler,
            device=device,
            backbone_modules=backbone_modules,
            max_batches=args.max_train_batches,
        )
        val_loss = evaluate_supcon(
            model=model,
            loader=supcon_val_loader,
            criterion=supcon_loss,
            device=device,
            max_batches=args.max_eval_batches,
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
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": supcon_best_loss,
            "epochs_without_improvement": supcon_wait,
            "trainable_params": supcon_trainable_params,
            "total_params": total_params,
            "unfrozen_backbone_modules": args.supcon_unfreeze_backbone_modules,
            "newly_unfrozen_tail_modules": thawed_supcon[-args.unfreeze_chunk_size :],
        }
        history.append(row)
        print(row)
        append_jsonl(log_path, row)
        if supcon_wait >= args.early_stopping_patience:
            event = {
                "stage": "supcon",
                "stopped_early": True,
                "best_epoch": supcon_best_epoch,
                "best_val_loss": supcon_best_loss,
            }
            print(event)
            append_jsonl(log_path, event)
            break

    model.load_state_dict(supcon_best_state)

    arcface_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    phases = build_phase_plan(len(backbone_modules), args)
    best_arcface_state = cpu_state_dict(model)

    for phase_index, phase in enumerate(phases, start=1):
        thawed = set_trainability_for_arcface(model, backbone_modules, phase.unfrozen_backbone_modules)
        optimizer = build_arcface_optimizer(model, args)
        scheduler = build_scheduler(
            optimizer.base_optimizer,
            max_epochs=phase.max_epochs,
            steps_per_epoch=min(len(train_loader), args.max_train_batches) if args.max_train_batches > 0 else len(train_loader),
            warmup_epochs=args.warmup_epochs,
        )
        _, trainable_params = parameter_counts(model)
        phase_best_loss = float("inf")
        phase_best_acc = -1.0
        phase_best_epoch = 0
        phase_best_state = cpu_state_dict(model)
        phase_wait = 0

        for epoch in range(1, phase.max_epochs + 1):
            train_loss, train_acc = train_arcface_epoch(
                model=model,
                loader=train_loader,
                criterion=arcface_criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                backbone_modules=backbone_modules,
                max_batches=args.max_train_batches,
            )
            val_loss, val_acc = evaluate_arcface(
                model=model,
                loader=val_loader,
                criterion=arcface_criterion,
                device=device,
                max_batches=args.max_eval_batches,
            )
            if improved_metric(phase_best_loss, phase_best_acc, val_loss, val_acc, args.early_stopping_min_delta):
                phase_best_loss = val_loss
                phase_best_acc = val_acc
                phase_best_epoch = epoch
                phase_best_state = cpu_state_dict(model)
                phase_wait = 0
            else:
                phase_wait += 1

            if improved_metric(best_val_loss, best_val_acc, val_loss, val_acc, args.early_stopping_min_delta):
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_arcface_state = cpu_state_dict(model)

            row = {
                "stage": "arcface",
                "phase_index": phase_index,
                "phase_name": phase.name,
                "epoch_in_phase": epoch,
                "phase_max_epochs": phase.max_epochs,
                "unfrozen_backbone_modules": phase.unfrozen_backbone_modules,
                "newly_unfrozen_tail_modules": thawed[-args.unfreeze_chunk_size :],
                "trainable_params": trainable_params,
                "total_params": total_params,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "phase_best_val_loss": phase_best_loss,
                "phase_best_val_acc": phase_best_acc,
                "epochs_without_improvement": phase_wait,
            }
            history.append(row)
            print(row)
            append_jsonl(log_path, row)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx,
                "args": vars(args),
                "history": history,
                "current_phase": asdict(phase),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
            }
            torch.save(checkpoint, output_dir / "last.pt")

            if phase_wait >= args.early_stopping_patience:
                event = {
                    "stage": "arcface",
                    "phase_index": phase_index,
                    "phase_name": phase.name,
                    "stopped_early": True,
                    "best_epoch_in_phase": phase_best_epoch,
                    "phase_best_val_loss": phase_best_loss,
                    "phase_best_val_acc": phase_best_acc,
                }
                print(event)
                append_jsonl(log_path, event)
                break

        model.load_state_dict(phase_best_state)

    model.load_state_dict(best_arcface_state)

    val_logits, val_targets = collect_logits_and_labels(model, val_loader, device, args.max_eval_batches)
    test_logits, test_targets = collect_logits_and_labels(model, test_loader, device, args.max_eval_batches)

    val_metrics = compute_classification_metrics(val_logits, val_targets, train_dataset.classes)
    test_metrics = compute_classification_metrics(test_logits, test_targets, train_dataset.classes)

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
        "model_name": "efficientnet_v2_l_metric_learning_gabor" if use_gabor else "efficientnet_v2_l_metric_learning",
        "device": str(device),
        "output_dir": str(output_dir),
        "log_file": str(log_path),
        "weights": args.weights,
        "class_names": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "source_train_count": train_dataset.source_count(),
        "source_val_count": val_dataset.source_count(),
        "source_test_count": test_dataset.source_count(),
        "effective_train_count": len(train_dataset),
        "effective_val_count": len(val_dataset),
        "effective_test_count": len(test_dataset),
        "augment_repeats": args.augment_repeats,
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
        },
        "optimization": {
            "optimizer": "AdamW + SAM",
            "scheduler": "Linear warmup + cosine decay",
            "sam_rho": args.sam_rho,
            "weight_decay": args.weight_decay,
            "adam_betas": [args.adam_beta1, args.adam_beta2],
        },
        "early_stopping": {
            "patience": args.early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "monitor": "val_loss",
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
    final_event = {"event": "run_finished", "validation_accuracy": val_metrics["accuracy"], "test_accuracy": test_metrics["accuracy"]}
    append_jsonl(log_path, final_event)
    print({"validation_accuracy": val_metrics["accuracy"], "test_accuracy": test_metrics["accuracy"]})
    return 0
