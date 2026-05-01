#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

PHASE0_EARLY_STOPPING_MODE = "effective_batch_window_best_v3"
PHASE0_PATCH_NORMALIZATION_EPS = 1e-2
PHASE0_GRAD_CLIP_NORM = 1.0
PHASE0_LOSS_MODE_RAW_MSE = "raw_mse"
PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE = "patch_normalized_mse"
PHASE0_LR_SCALE_BASE_BATCH_SIZE = 256
PHASE0_WARMUP_EPOCHS = 10
PHASE0_SCHEDULER_MODE_WARMUP_CONSTANT = "warmup_constant"
PHASE0_SCHEDULER_MODE_WARMUP_COSINE = "warmup_cosine"
PHASE0_ADAMW_BETA1 = 0.9
PHASE0_ADAMW_BETA2 = 0.95

try:
    from metric_learning_pipeline import (
        DEFAULT_BACKBONE_NAME,
        BACKBONE_REGISTRY,
        CAMERA_COLOR_CAST_PROBABILITY,
        CAMERA_COLOR_CAST_STRENGTH,
        CAMERA_COLOR_CAST_EVAL,
        build_progress_postfix,
        build_datasets,
        make_balanced_sampler,
        evaluation_tensor_from_image,
        training_tensor_from_image,
        load_resume_checkpoint,
        log_json_event,
        seed_everything,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from metric_learning_pipeline import (
        DEFAULT_BACKBONE_NAME,
        BACKBONE_REGISTRY,
        CAMERA_COLOR_CAST_PROBABILITY,
        CAMERA_COLOR_CAST_STRENGTH,
        CAMERA_COLOR_CAST_EVAL,
        build_progress_postfix,
        build_datasets,
        make_balanced_sampler,
        evaluation_tensor_from_image,
        training_tensor_from_image,
        load_resume_checkpoint,
        log_json_event,
        seed_everything,
    )


class Phase0WasteDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, samples: list[tuple[str, int]], classes: list[str], image_size: int, seed: int) -> None:
        self.samples = list(samples)
        self.classes = list(classes)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.current_epoch = 0

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(0, int(epoch))

    def source_count(self) -> int:
        return len(self.samples)

    def source_target_for_index(self, index: int) -> int:
        return int(self.samples[index][1])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            rng = random.Random(
                self.seed * 1_000_003
                + self.current_epoch * 104_729
                + index * 9_973
            )
            tensor = training_tensor_from_image(
                image,
                self.image_size,
                rng,
                gaussian_sigmas=1.0,
                camera_color_cast_probability=CAMERA_COLOR_CAST_PROBABILITY,
                camera_color_cast_strength=CAMERA_COLOR_CAST_STRENGTH,
            )
        return tensor, int(target), str(path)


class SpatialMaskGenerator:
    def __init__(self, input_size: int = 224, patch_size: int = 32, mask_ratio: float = 0.6) -> None:
        if input_size % patch_size != 0:
            raise ValueError("input_size must be divisible by patch_size for the spatial mask generator.")
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.mask_ratio = float(mask_ratio)
        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.num_mask = int(round(self.mask_ratio * self.num_patches))

    def __call__(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones((batch_size, self.num_patches), device=device)
        mask[:, : self.num_patches - self.num_mask] = 0.0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        mask_2d = mask.view(batch_size, 1, self.grid_size, self.grid_size)
        pixel_mask = mask_2d.repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3)
        return pixel_mask, mask_2d


def compute_raw_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pixel_mask: torch.Tensor,
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError(f"Prediction/target shape mismatch: {tuple(predictions.shape)} vs {tuple(targets.shape)}")
    if predictions.ndim != 4:
        raise ValueError(f"Expected 4D image tensors, got shape {tuple(predictions.shape)}")
    if pixel_mask.ndim != 4:
        raise ValueError(f"Expected 4D mask tensor, got shape {tuple(pixel_mask.shape)}")
    if pixel_mask.shape[0] != predictions.shape[0] or pixel_mask.shape[2:] != predictions.shape[2:]:
        raise ValueError(f"Mask shape mismatch: {tuple(pixel_mask.shape)} vs {tuple(predictions.shape)}")

    if pixel_mask.shape[1] == 1 and predictions.shape[1] > 1:
        pixel_mask = pixel_mask.expand(-1, predictions.shape[1], -1, -1)
    elif pixel_mask.shape != predictions.shape:
        raise ValueError(f"Mask channel mismatch: {tuple(pixel_mask.shape)} vs {tuple(predictions.shape)}")

    loss = (predictions - targets).pow(2)
    masked_loss = (loss * pixel_mask).sum()
    normalizer = pixel_mask.sum() + 1e-8
    return masked_loss / normalizer


def compute_patch_normalized_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pixel_mask: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError(f"Prediction/target shape mismatch: {tuple(predictions.shape)} vs {tuple(targets.shape)}")
    if predictions.ndim != 4:
        raise ValueError(f"Expected 4D image tensors, got shape {tuple(predictions.shape)}")
    if predictions.shape[2] % patch_size != 0 or predictions.shape[3] % patch_size != 0:
        raise ValueError("Image size must be divisible by patch_size for patch-normalized loss.")
    if pixel_mask.ndim != 4:
        raise ValueError(f"Expected 4D mask tensor, got shape {tuple(pixel_mask.shape)}")
    if pixel_mask.shape[0] != predictions.shape[0] or pixel_mask.shape[2:] != predictions.shape[2:]:
        raise ValueError(f"Mask shape mismatch: {tuple(pixel_mask.shape)} vs {tuple(predictions.shape)}")
    preds = patchify_phase0_images(predictions, patch_size)
    targs = patchify_phase0_images(targets, patch_size)
    mask = patchify_phase0_masks(pixel_mask, patch_size)

    patch_mean = targs.mean(dim=-1, keepdim=True)
    patch_var = targs.var(dim=-1, unbiased=False, keepdim=True)
    targs = (targs - patch_mean) / torch.sqrt(patch_var + PHASE0_PATCH_NORMALIZATION_EPS)

    loss = (preds - targs).pow(2).mean(dim=-1, keepdim=True)
    masked_loss = (loss * mask).sum()
    normalizer = mask.sum() + 1e-8
    return masked_loss / normalizer


def compute_phase0_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pixel_mask: torch.Tensor,
    patch_size: int,
    loss_mode: str,
) -> torch.Tensor:
    if loss_mode == PHASE0_LOSS_MODE_RAW_MSE:
        return compute_raw_mse_loss(predictions, targets, pixel_mask)
    if loss_mode == PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE:
        return compute_patch_normalized_mse_loss(predictions, targets, pixel_mask, patch_size)
    raise ValueError(
        f"Unknown Phase 0 loss mode: {loss_mode!r}. "
        f"Expected {PHASE0_LOSS_MODE_RAW_MSE!r} or {PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE!r}."
    )


def phase0_tensor_is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def phase0_scalar_is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def get_distributed_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def resolve_phase0_loss_mode_from_checkpoint(
    resume_checkpoint: dict[str, Any] | None,
    fallback_loss_mode: str,
) -> str:
    if not resume_checkpoint:
        return fallback_loss_mode
    checkpoint_args = resume_checkpoint.get("args", {})
    if isinstance(checkpoint_args, dict):
        checkpoint_loss_mode = checkpoint_args.get("loss_mode")
        if isinstance(checkpoint_loss_mode, str) and checkpoint_loss_mode in {
            PHASE0_LOSS_MODE_RAW_MSE,
            PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE,
        }:
            return checkpoint_loss_mode
    checkpoint_loss_mode = resume_checkpoint.get("phase0_loss_mode")
    if isinstance(checkpoint_loss_mode, str) and checkpoint_loss_mode in {
        PHASE0_LOSS_MODE_RAW_MSE,
        PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE,
    }:
        return checkpoint_loss_mode
    return fallback_loss_mode


class RepoSafeConvNeXtMIM(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        weights_mode: str,
        input_res: int = 224,
        decoder_dim: int = 512,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.weights_mode = str(weights_mode)
        self.input_res = int(input_res)
        model_name, pretrained, phase0_source = resolve_phase0_backbone_model_name(self.backbone_name, self.weights_mode)
        self.phase0_source = phase0_source
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.input_res, self.input_res)
            features = self.encoder.forward_features(dummy)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            if features.ndim == 2:
                features = features[:, :, None, None]
            if features.ndim != 4:
                raise RuntimeError(
                    f"Phase 0 MIM requires a spatial feature map, got shape {tuple(features.shape)} from {self.backbone_name}."
                )
            enc_channels = int(features.shape[1])
            feat_res = int(features.shape[2])
            if feat_res < 1:
                raise RuntimeError(f"Invalid feature resolution inferred from {self.backbone_name}: {feat_res}")

        self.feat_res = feat_res
        self.decoder_proj = nn.Conv2d(enc_channels, decoder_dim, kernel_size=1)
        self.decoder_block = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=7, padding=3, groups=decoder_dim),
            nn.GroupNorm(1, decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1),
            nn.GroupNorm(1, decoder_dim),
            nn.GELU(),
        )
        self.decoder_pred = nn.Conv2d(decoder_dim, 3, kernel_size=1)

    def forward(self, images: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
        masked = images * (1.0 - pixel_mask)
        features = self.encoder.forward_features(masked)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.ndim == 2:
            features = features[:, :, None, None]
        if features.ndim != 4:
            raise RuntimeError(f"Unexpected backbone feature shape for Phase 0 MIM: {tuple(features.shape)}")
        decoded = self.decoder_proj(features)
        decoded = self.decoder_block(decoded)
        decoded = self.decoder_pred(decoded)
        if decoded.shape[-2:] != (self.input_res, self.input_res):
            decoded = torch.nn.functional.interpolate(
                decoded,
                size=(self.input_res, self.input_res),
                mode="bilinear",
                align_corners=False,
            )
        return decoded


class Phase0WarmupScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        schedule_mode: str,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(0, int(total_steps))
        self.schedule_mode = str(schedule_mode)
        if self.schedule_mode not in {PHASE0_SCHEDULER_MODE_WARMUP_CONSTANT, PHASE0_SCHEDULER_MODE_WARMUP_COSINE}:
            raise ValueError(
                f"Unknown Phase 0 scheduler mode: {self.schedule_mode!r}. "
                f"Expected {PHASE0_SCHEDULER_MODE_WARMUP_CONSTANT!r} or {PHASE0_SCHEDULER_MODE_WARMUP_COSINE!r}."
            )
        if self.schedule_mode == PHASE0_SCHEDULER_MODE_WARMUP_COSINE and self.total_steps <= 0:
            raise ValueError("--phase0-scheduler-mode warmup_cosine requires --phase0-total-steps > 0")
        if self.schedule_mode == PHASE0_SCHEDULER_MODE_WARMUP_COSINE and self.total_steps <= self.warmup_steps:
            raise ValueError("--phase0-total-steps must be greater than warmup steps for cosine scheduling")
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.step_index = 0

    def _factor(self, step: int) -> float:
        if self.warmup_steps <= 0:
            warmup_factor = 1.0
        elif step < self.warmup_steps:
            warmup_factor = float(step + 1) / float(self.warmup_steps)
        else:
            warmup_factor = 1.0
        if step < self.warmup_steps:
            return warmup_factor
        if self.schedule_mode == PHASE0_SCHEDULER_MODE_WARMUP_CONSTANT:
            return 1.0
        if self.schedule_mode == PHASE0_SCHEDULER_MODE_WARMUP_COSINE:
            if self.total_steps <= self.warmup_steps:
                return 1.0
            decay_steps = max(1, self.total_steps - self.warmup_steps)
            progress = min(max(step - self.warmup_steps, 0), decay_steps) / float(decay_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    def apply_current_lrs(self) -> None:
        factor = self._factor(self.step_index)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def step(self) -> None:
        self.step_index += 1
        self.apply_current_lrs()

    def state_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "schedule_mode": self.schedule_mode,
            "base_lrs": self.base_lrs,
            "step_index": self.step_index,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.schedule_mode = str(state_dict.get("schedule_mode", self.schedule_mode))
        self.base_lrs = [float(value) for value in state_dict.get("base_lrs", self.base_lrs)]
        self.step_index = int(state_dict.get("step_index", self.step_index))


def set_phase0_scheduler_base_lrs(scheduler: Phase0WarmupScheduler, base_lrs: list[float]) -> None:
    scheduler.base_lrs = [float(value) for value in base_lrs]
    scheduler.apply_current_lrs()


def compute_phase0_scaled_learning_rate(args: argparse.Namespace) -> tuple[float, int, int]:
    world_size = get_distributed_world_size()
    effective_batch_size = int(args.batch_size * args.grad_accum_steps * world_size)
    reference_batch_size = int(args.lr_scale_base_batch_size)
    if reference_batch_size < 1:
        raise ValueError("--lr-scale-base-batch-size must be >= 1")
    scaled_learning_rate = float(args.learning_rate) * float(effective_batch_size) / float(reference_batch_size)
    return scaled_learning_rate, effective_batch_size, world_size


def build_phase0_recipe_payload(
    args: argparse.Namespace,
    *,
    scaled_learning_rate: float,
    effective_batch_size_world: int,
    world_size: int,
    warmup_steps: int,
) -> dict[str, Any]:
    return {
        "base_lr": float(args.learning_rate),
        "scaled_lr": float(scaled_learning_rate),
        "effective_batch_size": int(effective_batch_size_world),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "world_size": int(world_size),
        "warmup_epochs": int(args.warmup_epochs),
        "warmup_steps": int(warmup_steps),
        "scheduler_mode": str(args.scheduler_mode),
        "total_steps": int(args.total_steps),
        "adamw_betas": [float(args.adamw_beta1), float(args.adamw_beta2)],
        "weight_decay": float(args.weight_decay),
        "loss_mode": str(args.loss_mode),
        "mask_ratio": float(args.mask_ratio),
        "mask_patch_size": int(args.patch_size),
        "grad_clip_norm": float(args.grad_clip_norm),
    }


def resolve_phase0_backbone_model_name(backbone_name: str, weights_mode: str) -> tuple[str, bool, str]:
    spec = BACKBONE_REGISTRY.get(backbone_name)
    if weights_mode == "default":
        pure_candidate = f"{backbone_name}.fcmae"
        if pure_candidate in timm.list_models(f"{backbone_name}*", pretrained=True):
            return pure_candidate, True, "pure_fcmae"
        if spec is not None:
            return spec.pretrained_name, True, "registry_pretrained_fallback"
        return backbone_name, True, "direct_pretrained_fallback"
    if spec is not None:
        return spec.scratch_name, False, "scratch"
    return backbone_name, False, "direct_scratch"


def build_llrd_optimizer(
    model: RepoSafeConvNeXtMIM,
    base_lr: float,
    weight_decay: float,
    *,
    betas: tuple[float, float],
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = []

    def add_group(parameters, lr: float) -> None:
        filtered = [param for param in parameters if param.requires_grad]
        if filtered:
            groups.append({"params": filtered, "lr": lr, "weight_decay": weight_decay})

    add_group(model.decoder_proj.parameters(), base_lr)
    add_group(model.decoder_block.parameters(), base_lr)
    add_group(model.decoder_pred.parameters(), base_lr)

    backbone_leaf_modules = [
        module
        for name, module in model.encoder.named_modules()
        if name and not list(module.children()) and any(True for _ in module.parameters(recurse=False))
    ]
    if not backbone_leaf_modules:
        backbone_leaf_modules = [module for module in model.encoder.modules() if not list(module.children())]

    decay_rate = 0.8
    for offset, module in enumerate(reversed(backbone_leaf_modules), start=1):
        stage_lr = base_lr * (decay_rate ** offset)
        add_group(module.parameters(recurse=False), stage_lr)

    return torch.optim.AdamW(groups, betas=betas, eps=1e-8, foreach=False)


def save_phase0_checkpoint(
    path: Path,
    *,
    model: RepoSafeConvNeXtMIM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: Phase0WarmupScheduler | None,
    epoch: int,
    epoch_batch_index: int,
    epoch_complete: bool,
    step: int,
    best_loss: float,
    train_loss_window_best_loss: float,
    train_loss_window_batch_count: int,
    loss_plateau_windows_without_improvement: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "epoch_batch_index": int(epoch_batch_index),
        "epoch_complete": bool(epoch_complete),
        "step": int(step),
        "optimizer_step": int(step),
        "best_loss": float(best_loss),
        "best_train_effective_batch_loss": float(best_loss),
        "phase0_early_stopping_mode": PHASE0_EARLY_STOPPING_MODE,
        "train_loss_window_best_loss": float(train_loss_window_best_loss),
        "train_loss_window_batch_count": int(train_loss_window_batch_count),
        "train_loss_window_steps_without_improvement": int(train_loss_window_batch_count),
        "loss_plateau_windows_without_improvement": int(loss_plateau_windows_without_improvement),
        "args": vars(args),
    }
    torch.save(payload, path)


def _denormalize_phase0_images(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return torch.clamp(images * std + mean, 0.0, 1.0)


def _phase0_psnr_from_mse(mse_value: torch.Tensor | float) -> float:
    mse_scalar = float(mse_value if isinstance(mse_value, float) else mse_value.detach().float().item())
    if mse_scalar <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse_scalar))


def _phase0_monitoring_metrics(
    images: torch.Tensor,
    reconstructed: torch.Tensor,
    pixel_mask: torch.Tensor,
    patch_size: int,
    loss_mode: str,
) -> dict[str, float]:
    target_vis = _denormalize_phase0_images(images.detach().float())
    if loss_mode == PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE:
        pred_patches = patchify_phase0_images(reconstructed.detach().float(), patch_size)
        target_patches = patchify_phase0_images(images.detach().float(), patch_size)
        patch_mean = target_patches.mean(dim=-1, keepdim=True)
        patch_var = target_patches.var(dim=-1, unbiased=False, keepdim=True)
        patch_std = torch.sqrt(patch_var + PHASE0_PATCH_NORMALIZATION_EPS)

        pred_norm = (pred_patches - patch_mean) / patch_std
        target_norm = (target_patches - patch_mean) / patch_std
        pred_rgb_after_unnorm = unpatchify_phase0_images(
            pred_patches * patch_std + patch_mean,
            patch_size,
            channels=images.shape[1],
        )
        pred_vis = _denormalize_phase0_images(pred_rgb_after_unnorm)

        return {
            "pred_norm_mean": float(pred_norm.mean().item()),
            "pred_norm_std": float(pred_norm.std(unbiased=False).item()),
            "target_norm_mean": float(target_norm.mean().item()),
            "target_norm_std": float(target_norm.std(unbiased=False).item()),
            "pred_rgb_after_unnorm_std": float(pred_vis.std(unbiased=False).item()),
            "masked_psnr_after_unnorm": float(
                _phase0_masked_psnr(pred_vis, target_vis, pixel_mask)
            ),
            "masked_mse": float(_phase0_masked_mse(pred_vis, target_vis, pixel_mask)),
            "masked_mae": float(_phase0_masked_mae(pred_vis, target_vis, pixel_mask)),
            "masked_psnr": float(_phase0_masked_psnr(pred_vis, target_vis, pixel_mask)),
            "full_mse": float(_phase0_full_mse(pred_vis, target_vis)),
            "full_psnr": float(_phase0_full_psnr(pred_vis, target_vis)),
        }

    pred_vis = _denormalize_phase0_images(reconstructed.detach().float())
    return {
        "masked_mse": float(_phase0_masked_mse(pred_vis, target_vis, pixel_mask)),
        "masked_mae": float(_phase0_masked_mae(pred_vis, target_vis, pixel_mask)),
        "masked_psnr": float(_phase0_masked_psnr(pred_vis, target_vis, pixel_mask)),
        "full_mse": float(_phase0_full_mse(pred_vis, target_vis)),
        "full_psnr": float(_phase0_full_psnr(pred_vis, target_vis)),
    }


def _phase0_masked_mse(predictions: torch.Tensor, targets: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
    mask = pixel_mask.float()
    if mask.shape[1] == 1 and predictions.shape[1] > 1:
        mask = mask.expand(-1, predictions.shape[1], -1, -1)
    return ((predictions - targets).pow(2) * mask).sum() / (mask.sum() + 1e-8)


def _phase0_masked_mae(predictions: torch.Tensor, targets: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
    mask = pixel_mask.float()
    if mask.shape[1] == 1 and predictions.shape[1] > 1:
        mask = mask.expand(-1, predictions.shape[1], -1, -1)
    return ((predictions - targets).abs() * mask).sum() / (mask.sum() + 1e-8)


def _phase0_full_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (predictions - targets).pow(2).mean()


def _phase0_full_psnr(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.tensor(_phase0_psnr_from_mse(_phase0_full_mse(predictions, targets)), device=predictions.device)


def _phase0_masked_psnr(predictions: torch.Tensor, targets: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
    return torch.tensor(_phase0_psnr_from_mse(_phase0_masked_mse(predictions, targets, pixel_mask)), device=predictions.device)


def patchify_phase0_images(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected 4D image tensors, got shape {tuple(images.shape)}")
    if images.shape[2] % patch_size != 0 or images.shape[3] % patch_size != 0:
        raise ValueError("Image size must be divisible by patch_size for patchifying.")
    batch_size, channels, height, width = images.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = images.reshape(batch_size, channels, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    return patches.view(batch_size, grid_h * grid_w, channels * patch_size * patch_size)


def unpatchify_phase0_images(patches: torch.Tensor, patch_size: int, channels: int = 3) -> torch.Tensor:
    if patches.ndim != 3:
        raise ValueError(f"Expected 3D patch tensors, got shape {tuple(patches.shape)}")
    batch_size, num_patches, patch_dim = patches.shape
    if channels < 1:
        raise ValueError("channels must be >= 1")
    if patch_dim != channels * patch_size * patch_size:
        raise ValueError(
            f"Patch dimension mismatch: expected {channels * patch_size * patch_size}, got {patch_dim}"
        )
    grid_size = int(math.isqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Number of patches must be a square, got {num_patches}")
    images = patches.view(batch_size, grid_size, grid_size, channels, patch_size, patch_size)
    images = images.permute(0, 3, 1, 4, 2, 5).contiguous()
    return images.view(batch_size, channels, grid_size * patch_size, grid_size * patch_size)


def patchify_phase0_masks(pixel_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    if pixel_mask.ndim != 4:
        raise ValueError(f"Expected 4D mask tensor, got shape {tuple(pixel_mask.shape)}")
    if pixel_mask.shape[2] % patch_size != 0 or pixel_mask.shape[3] % patch_size != 0:
        raise ValueError("Mask size must be divisible by patch_size for patchifying.")
    mask = pixel_mask.float()
    if mask.shape[1] != 1:
        mask = mask.mean(dim=1, keepdim=True)
    batch_size, _, height, width = mask.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = mask.reshape(batch_size, 1, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.view(batch_size, grid_h * grid_w, patch_size * patch_size)
    return patches.mean(dim=-1, keepdim=True)


def _build_phase0_preview_canvas(
    *,
    title: str,
    originals: torch.Tensor,
    masked: torch.Tensor,
    reconstructed: torch.Tensor,
    epoch: int,
    global_step: int,
    sample_count: int,
) -> Image.Image:
    grid_original = _tensor_batch_to_grid_image(originals, nrow=sample_count)
    grid_masked = _tensor_batch_to_grid_image(masked, nrow=sample_count)
    grid_reconstructed = _tensor_batch_to_grid_image(reconstructed, nrow=sample_count)

    label_height = 28
    gap = 18
    margin = 16
    canvas_width = max(grid_original.width, grid_masked.width, grid_reconstructed.width) + margin * 2
    canvas_height = margin * 2 + label_height * 3 + grid_original.height + grid_masked.height + grid_reconstructed.height + gap * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    draw.text(
        (margin, margin),
        f"Phase 0 MIM Reconstruction Preview | {title} | epoch {epoch} | step {global_step}",
        fill=(245, 245, 245),
        font=title_font,
    )
    sections = [
        ("Originals", grid_original),
        ("Masked inputs", grid_masked),
        (title, grid_reconstructed),
    ]
    y = margin + label_height
    for label, image in sections:
        draw.text((margin, y), label, fill=(220, 220, 220), font=body_font)
        y += 18
        canvas.paste(image, (margin, y))
        y += image.height + gap
    return canvas


def _render_phase0_preview_pair(
    originals: torch.Tensor,
    pixel_mask: torch.Tensor,
    reconstructed: torch.Tensor,
    patch_size: int,
    loss_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if originals.shape != reconstructed.shape:
        raise ValueError(f"Preview shape mismatch: {tuple(originals.shape)} vs {tuple(reconstructed.shape)}")
    if originals.ndim != 4:
        raise ValueError(f"Expected 4D image tensors, got shape {tuple(originals.shape)}")
    if originals.shape[2] % patch_size != 0 or originals.shape[3] % patch_size != 0:
        raise ValueError("Image size must be divisible by patch_size for preview rendering.")

    batch_size, channels, _, _ = originals.shape
    mask = pixel_mask

    if loss_mode == PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE:
        orig_patches = patchify_phase0_images(originals, patch_size)
        pred_patches = patchify_phase0_images(reconstructed, patch_size)
        mask_patches = patchify_phase0_masks(mask, patch_size)

        patch_mean = orig_patches.mean(dim=-1, keepdim=True)
        patch_var = orig_patches.var(dim=-1, unbiased=False, keepdim=True)
        patch_std = torch.sqrt(patch_var + PHASE0_PATCH_NORMALIZATION_EPS)

        full_output_patches = pred_patches * patch_std + patch_mean
        masked_fill_patches = orig_patches * (1.0 - mask_patches) + full_output_patches * mask_patches
        masked_fill = unpatchify_phase0_images(masked_fill_patches, patch_size, channels=channels)
        full_output = unpatchify_phase0_images(full_output_patches, patch_size, channels=channels)
        return _denormalize_phase0_images(masked_fill), _denormalize_phase0_images(full_output)

    if loss_mode == PHASE0_LOSS_MODE_RAW_MSE:
        if mask.shape[1] == 1 and reconstructed.shape[1] > 1:
            mask = mask.expand(-1, reconstructed.shape[1], -1, -1)
        masked_fill = originals * (1.0 - mask) + reconstructed * mask
        return _denormalize_phase0_images(masked_fill), _denormalize_phase0_images(reconstructed)

    raise ValueError(
        f"Unknown Phase 0 loss mode for preview rendering: {loss_mode!r}. "
        f"Expected {PHASE0_LOSS_MODE_RAW_MSE!r} or {PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE!r}."
    )


def _render_phase0_preview(
    originals: torch.Tensor,
    pixel_mask: torch.Tensor,
    reconstructed: torch.Tensor,
    patch_size: int,
    loss_mode: str,
) -> torch.Tensor:
    """Map Phase 0 predictions back to a human-readable RGB preview."""
    masked_fill_preview, _ = _render_phase0_preview_pair(
        originals,
        pixel_mask,
        reconstructed,
        patch_size,
        loss_mode,
    )
    return masked_fill_preview


def _tensor_batch_to_grid_image(images: torch.Tensor, nrow: int) -> Image.Image:
    grid = make_grid(images, nrow=max(1, nrow), padding=2, pad_value=0.06)
    grid_np = grid.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((grid_np * 255.0).round().astype(np.uint8))


def save_phase0_reconstruction_preview(
    output_path: Path,
    *,
    originals: torch.Tensor,
    pixel_mask: torch.Tensor,
    reconstructed: torch.Tensor,
    patch_size: int,
    loss_mode: str,
    epoch: int,
    global_step: int,
    sample_count: int,
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = max(1, min(int(sample_count), int(originals.shape[0])))
    originals_vis = _denormalize_phase0_images(originals[:sample_count].detach())
    masked_vis = _denormalize_phase0_images((originals[:sample_count] * (1.0 - pixel_mask[:sample_count])).detach())
    sample_images = originals[:sample_count].detach()
    sample_mask = pixel_mask[:sample_count].detach()
    sample_reconstructed = reconstructed[:sample_count].detach()

    masked_fill_vis, full_output_vis = _render_phase0_preview_pair(
        sample_images,
        sample_mask,
        sample_reconstructed,
        patch_size=patch_size,
        loss_mode=loss_mode,
    )

    masked_fill_canvas = _build_phase0_preview_canvas(
        title="Masked-fill reconstruction",
        originals=originals_vis,
        masked=masked_vis,
        reconstructed=masked_fill_vis,
        epoch=epoch,
        global_step=global_step,
        sample_count=sample_count,
    )
    full_output_canvas = _build_phase0_preview_canvas(
        title="Full-output reconstruction",
        originals=originals_vis,
        masked=masked_vis,
        reconstructed=full_output_vis,
        epoch=epoch,
        global_step=global_step,
        sample_count=sample_count,
    )

    masked_fill_path = output_path.with_name(f"{output_path.stem}_masked_fill{output_path.suffix}")
    full_output_path = output_path.with_name(f"{output_path.stem}_full_output{output_path.suffix}")
    masked_fill_canvas.save(masked_fill_path)
    full_output_canvas.save(full_output_path)
    return full_output_path, masked_fill_path


def log_phase0_state(
    log_path: Path,
    *,
    event: str,
    epoch: int,
    global_step: int,
    microbatch_index: int | None = None,
    microbatches_in_effective_batch: int | None = None,
    samples_seen: int | None = None,
    epoch_loss_sum: float | None = None,
    epoch_sample_count: int | None = None,
    effective_batch_loss: float | None = None,
    microbatch_loss: float | None = None,
    train_loss_window_best_loss: float | None = None,
    best_loss: float | None = None,
    loss_plateau_windows_without_improvement: int | None = None,
    optimizer_lr: float | None = None,
    grad_norm: float | None = None,
    amp_scale: float | None = None,
    skipped_optimizer_step: bool | None = None,
    step_time_sec: float | None = None,
    gpu_memory_allocated: float | None = None,
    prediction_mean: float | None = None,
    prediction_std: float | None = None,
    target_mean: float | None = None,
    target_std: float | None = None,
    masked_ratio_actual: float | None = None,
    train_loss_mean: float | None = None,
    val_loss_mean: float | None = None,
    masked_mse: float | None = None,
    masked_mae: float | None = None,
    masked_psnr: float | None = None,
    full_mse: float | None = None,
    full_psnr: float | None = None,
    pred_norm_mean: float | None = None,
    pred_norm_std: float | None = None,
    target_norm_mean: float | None = None,
    target_norm_std: float | None = None,
    pred_rgb_after_unnorm_std: float | None = None,
    masked_psnr_after_unnorm: float | None = None,
    args: argparse.Namespace | None = None,
) -> None:
    payload: dict[str, Any] = {
        "event": event,
        "epoch": int(epoch),
        "global_step": int(global_step),
    }
    if microbatch_index is not None:
        payload["microbatch_index"] = int(microbatch_index)
    if microbatches_in_effective_batch is not None:
        payload["microbatches_in_effective_batch"] = int(microbatches_in_effective_batch)
    if samples_seen is not None:
        payload["samples_seen"] = int(samples_seen)
    if epoch_loss_sum is not None:
        payload["epoch_loss_sum"] = float(epoch_loss_sum)
    if epoch_sample_count is not None:
        payload["epoch_sample_count"] = int(epoch_sample_count)
    if effective_batch_loss is not None:
        payload["effective_batch_loss"] = float(effective_batch_loss)
    if microbatch_loss is not None:
        payload["microbatch_loss"] = float(microbatch_loss)
    if train_loss_window_best_loss is not None:
        payload["train_loss_window_best_loss"] = float(train_loss_window_best_loss)
    if best_loss is not None:
        payload["best_loss"] = float(best_loss)
    if loss_plateau_windows_without_improvement is not None:
        payload["loss_plateau_windows_without_improvement"] = int(loss_plateau_windows_without_improvement)
    if optimizer_lr is not None:
        payload["optimizer_lr"] = float(optimizer_lr)
    if grad_norm is not None:
        payload["grad_norm"] = float(grad_norm)
    if amp_scale is not None:
        payload["amp_scale"] = float(amp_scale)
    if skipped_optimizer_step is not None:
        payload["skipped_optimizer_step"] = bool(skipped_optimizer_step)
    if step_time_sec is not None:
        payload["step_time_sec"] = float(step_time_sec)
    if gpu_memory_allocated is not None:
        payload["gpu_memory_allocated"] = float(gpu_memory_allocated)
    if prediction_mean is not None:
        payload["prediction_mean"] = float(prediction_mean)
    if prediction_std is not None:
        payload["prediction_std"] = float(prediction_std)
    if target_mean is not None:
        payload["target_mean"] = float(target_mean)
    if target_std is not None:
        payload["target_std"] = float(target_std)
    if masked_ratio_actual is not None:
        payload["masked_ratio_actual"] = float(masked_ratio_actual)
    if train_loss_mean is not None:
        payload["train_loss_mean"] = float(train_loss_mean)
    if val_loss_mean is not None:
        payload["val_loss_mean"] = float(val_loss_mean)
    if masked_mse is not None:
        payload["masked_mse"] = float(masked_mse)
    if masked_mae is not None:
        payload["masked_mae"] = float(masked_mae)
    if masked_psnr is not None:
        payload["masked_psnr"] = float(masked_psnr)
    if full_mse is not None:
        payload["full_mse"] = float(full_mse)
    if full_psnr is not None:
        payload["full_psnr"] = float(full_psnr)
    if pred_norm_mean is not None:
        payload["pred_norm_mean"] = float(pred_norm_mean)
    if pred_norm_std is not None:
        payload["pred_norm_std"] = float(pred_norm_std)
    if target_norm_mean is not None:
        payload["target_norm_mean"] = float(target_norm_mean)
    if target_norm_std is not None:
        payload["target_norm_std"] = float(target_norm_std)
    if pred_rgb_after_unnorm_std is not None:
        payload["pred_rgb_after_unnorm_std"] = float(pred_rgb_after_unnorm_std)
    if masked_psnr_after_unnorm is not None:
        payload["masked_psnr_after_unnorm"] = float(masked_psnr_after_unnorm)
    if args is not None:
        payload["backbone"] = str(args.backbone)
        payload["weights"] = str(args.weights)
        payload["batch_size"] = int(args.batch_size)
        payload["grad_accum_steps"] = int(args.grad_accum_steps)
        payload["effective_batch_size"] = int(args.batch_size * args.grad_accum_steps)
        payload["mask_ratio"] = float(args.mask_ratio)
        payload["patch_size"] = int(args.patch_size)
        payload["grad_clip_norm"] = float(args.grad_clip_norm)
        payload["train_loss_window"] = int(args.train_loss_window)
        payload["early_stopping_patience"] = int(args.early_stopping_patience)
        payload["early_stopping_min_delta"] = float(args.early_stopping_min_delta)
    log_json_event(log_path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 0 masked image modeling pretraining for ConvNeXt backbones.")
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--output-dir", default="Results/phase0_mim")
    parser.add_argument("--log-file", default="logs/phase0_mim.log.jsonl")
    parser.add_argument(
        "--backbone",
        default=DEFAULT_BACKBONE_NAME,
    )
    parser.add_argument("--weights", choices=("default", "none"), default="default")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--augment-repeats",
        type=int,
        default=1,
        help="Legacy compatibility knob retained for parser stability. Phase 0 now uses seeded crop/flip augmentation with no tint.",
    )
    parser.add_argument(
        "--augment-gaussian-sigmas",
        type=float,
        default=1.0,
        help="Legacy compatibility knob retained for parser stability. Phase 0 now uses seeded crop/flip augmentation with no tint.",
    )
    parser.add_argument(
        "--camera-color-cast-probability",
        type=float,
        default=CAMERA_COLOR_CAST_PROBABILITY,
        help="Legacy compatibility knob retained for parser stability. Pink tint is disabled.",
    )
    parser.add_argument(
        "--camera-color-cast-strength",
        type=float,
        default=CAMERA_COLOR_CAST_STRENGTH,
        help="Legacy compatibility knob retained for parser stability. Pink tint is disabled.",
    )
    parser.add_argument(
        "--camera-color-cast-eval",
        action=argparse.BooleanOptionalAction,
        default=CAMERA_COLOR_CAST_EVAL,
        help="Legacy compatibility knob retained for parser stability. Pink tint is disabled.",
    )
    parser.add_argument("--class-mapping", type=str, default="")
    parser.add_argument("--auto-split-ratios", default="0.9,0.05,0.05")
    parser.add_argument(
        "--runtime-bad-sample-cleanup",
        action="store_true",
        help="Mirror the main trainer's runtime bad-sample cleanup flag for dataset construction.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Phase 0 epoch cap. Use 0 to run until early stopping or max-steps termination.",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--decoder-dim", type=int, default=512)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=PHASE0_GRAD_CLIP_NORM,
        help="Clip Phase 0 gradients to this global norm before the optimizer step. Default is 1.0.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=(PHASE0_LOSS_MODE_RAW_MSE, PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE),
        default=PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE,
        help=(
            "Phase 0 reconstruction objective. "
            f"{PHASE0_LOSS_MODE_RAW_MSE} uses masked raw pixel MSE; "
            f"{PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE} uses masked patch-normalized MSE."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument(
        "--lr-scale-base-batch-size",
        type=int,
        default=PHASE0_LR_SCALE_BASE_BATCH_SIZE,
        help="Reference effective batch size used to linearly scale Phase 0 learning rate.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=PHASE0_WARMUP_EPOCHS,
        help="Linear warmup epochs for Phase 0 learning rate.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Optional explicit linear warmup steps for Phase 0. Overrides --warmup-epochs if > 0.",
    )
    parser.add_argument(
        "--scheduler-mode",
        choices=(PHASE0_SCHEDULER_MODE_WARMUP_CONSTANT, PHASE0_SCHEDULER_MODE_WARMUP_COSINE),
        default=PHASE0_SCHEDULER_MODE_WARMUP_CONSTANT,
        help=(
            "Phase 0 learning-rate schedule after warmup. Use warmup_constant unless you provide "
            "--total-steps for a real cosine horizon."
        ),
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=0,
        help="Required when --scheduler-mode warmup_cosine is selected. Total optimizer steps for the cosine horizon.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--adamw-beta1", type=float, default=PHASE0_ADAMW_BETA1)
    parser.add_argument("--adamw-beta2", type=float, default=PHASE0_ADAMW_BETA2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--train-loss-window",
        type=int,
        default=5000,
        help=(
            "Phase 0 plateau window in optimizer-step batches. The effective image count per window is "
            "train_loss_window x batch_size x grad_accum_steps (and x world_size if distributed)."
        ),
    )
    parser.add_argument(
        "--reconstruction-preview-interval",
        type=int,
        default=1,
        help="Save a reconstruction preview every N completed Phase 0 epochs. Set to 0 to disable previews.",
    )
    parser.add_argument(
        "--reconstruction-preview-count",
        type=int,
        default=6,
        help="Number of images to include in each Phase 0 reconstruction preview grid.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop Phase 0 after this many plateau windows without a new best train loss.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum effective-batch loss decrease required to reset Phase 0 patience.",
    )
    parser.add_argument("--resume-checkpoint", default="")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.epochs < 0:
        raise ValueError("--epochs must be >= 0")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0")
    if args.train_loss_window < 1:
        raise ValueError("--train-loss-window must be >= 1")
    if args.reconstruction_preview_interval < 0:
        raise ValueError("--reconstruction-preview-interval must be >= 0")
    if args.reconstruction_preview_count < 1:
        raise ValueError("--reconstruction-preview-count must be >= 1")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be >= 1")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be >= 0")
    if not (0.0 <= args.camera_color_cast_probability <= 1.0):
        raise ValueError("--camera-color-cast-probability must be between 0 and 1")
    if args.camera_color_cast_strength < 0:
        raise ValueError("--camera-color-cast-strength must be >= 0")
    if not (0.0 < args.mask_ratio < 1.0):
        raise ValueError("--mask-ratio must be between 0 and 1")
    if args.patch_size < 1:
        raise ValueError("--patch-size must be >= 1")
    if args.grad_clip_norm <= 0:
        raise ValueError("--grad-clip-norm must be > 0")
    if args.lr_scale_base_batch_size < 1:
        raise ValueError("--lr-scale-base-batch-size must be >= 1")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be >= 0")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.total_steps < 0:
        raise ValueError("--total-steps must be >= 0")
    if args.scheduler_mode == PHASE0_SCHEDULER_MODE_WARMUP_COSINE and args.total_steps <= 0:
        raise ValueError("--scheduler-mode warmup_cosine requires --total-steps > 0")
    if not (0.0 < args.adamw_beta1 < 1.0) or not (0.0 < args.adamw_beta2 < 1.0):
        raise ValueError("--adamw-beta1 and --adamw-beta2 must be in (0, 1)")
    if args.image_size % args.patch_size != 0:
        raise ValueError("--image-size must be divisible by --patch-size")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    log_path = Path(args.log_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    resume_checkpoint: dict[str, Any] | None = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
    else:
        step_last = output_dir / "step_last.pt"
        last = output_dir / "last.pt"
        resume_path = step_last if step_last.exists() else last
    if resume_path.exists():
        resume_checkpoint, resume_warning = load_resume_checkpoint(resume_path)
        if resume_warning is not None:
            log_json_event(log_path, {"event": "resume_checkpoint_ignored", "message": resume_warning})
    else:
        resume_warning = None

    if resume_checkpoint is not None:
        resolved_loss_mode = resolve_phase0_loss_mode_from_checkpoint(resume_checkpoint, args.loss_mode)
        if resolved_loss_mode != args.loss_mode:
            log_json_event(
                log_path,
                {
                    "event": "phase0_loss_mode_resolved_from_checkpoint",
                    "requested_loss_mode": args.loss_mode,
                    "resolved_loss_mode": resolved_loss_mode,
                    "resume_checkpoint": str(resume_path),
                },
            )
        args.loss_mode = resolved_loss_mode

    train_dataset, _, _, _, _ = build_datasets(args)
    phase0_dataset = Phase0WasteDataset(list(train_dataset.samples), list(train_dataset.classes), args.image_size, args.seed)
    phase0_sampler = make_balanced_sampler(phase0_dataset, phase0_dataset.classes, args.batch_size, args.seed + 202)
    loader = DataLoader(
        phase0_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=phase0_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RepoSafeConvNeXtMIM(args.backbone, args.weights, input_res=args.image_size, decoder_dim=args.decoder_dim).to(device)
    scaled_learning_rate, effective_batch_size_world, world_size = compute_phase0_scaled_learning_rate(args)
    phase0_recipe = build_phase0_recipe_payload(
        args,
        scaled_learning_rate=scaled_learning_rate,
        effective_batch_size_world=effective_batch_size_world,
        world_size=world_size,
        warmup_steps=int(args.warmup_steps) if args.warmup_steps > 0 else int(args.warmup_epochs * max(1, math.ceil(len(loader) / max(1, args.grad_accum_steps)))),
    )
    log_json_event(
        log_path,
        {
            "event": "phase0_backbone_initialization",
            "backbone": args.backbone,
            "resolved_model_name": getattr(model.encoder, "default_cfg", {}).get("architecture", args.backbone),
            "phase0_source": getattr(model, "phase0_source", None),
            "encoder_all_parameters_trainable": all(parameter.requires_grad for parameter in model.encoder.parameters()),
            "frozen_backbone_modules": 0,
            "sampler": "balanced_class_epoch_sampler",
            "batch_size": int(args.batch_size),
            "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
            "effective_batch_size_world": int(effective_batch_size_world),
            "world_size": int(world_size),
            "train_loss_window_effective_batches": int(args.train_loss_window),
            "early_stopping_patience": int(args.early_stopping_patience),
            "mask_ratio": float(args.mask_ratio),
            "patch_size": int(args.patch_size),
            "image_size": int(args.image_size),
            "decoder_dim": int(args.decoder_dim),
            "phase0_loss_mode": str(args.loss_mode),
            "phase0_base_learning_rate": float(args.learning_rate),
            "phase0_scaled_learning_rate": float(scaled_learning_rate),
            "phase0_lr_scale_base_batch_size": int(args.lr_scale_base_batch_size),
            "phase0_warmup_epochs": int(args.warmup_epochs),
            "phase0_warmup_steps": int(args.warmup_steps),
            "phase0_scheduler_mode": str(args.scheduler_mode),
            "phase0_total_steps": int(args.total_steps),
            "phase0_adamw_betas": [float(args.adamw_beta1), float(args.adamw_beta2)],
            "phase0_recipe": phase0_recipe,
            "camera_color_cast_probability": float(args.camera_color_cast_probability),
            "camera_color_cast_strength": float(args.camera_color_cast_strength),
            "camera_color_cast_eval": bool(args.camera_color_cast_eval),
        },
    )
    model.train()
    if device.type == "cuda" and hasattr(model.encoder, "set_grad_checkpointing"):
        try:
            model.encoder.set_grad_checkpointing(True)
        except TypeError:
            model.encoder.set_grad_checkpointing(enable=True)

    optimizer = build_llrd_optimizer(
        model,
        scaled_learning_rate,
        args.weight_decay,
        betas=(float(args.adamw_beta1), float(args.adamw_beta2)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    optimizer_steps_per_epoch = max(1, math.ceil(len(loader) / max(1, args.grad_accum_steps)))
    warmup_steps = int(args.warmup_steps) if args.warmup_steps > 0 else int(args.warmup_epochs * optimizer_steps_per_epoch)
    scheduler = Phase0WarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=int(args.total_steps),
        schedule_mode=str(args.scheduler_mode),
    )
    log_json_event(
        log_path,
        {
            "event": "phase0_run_started",
            "output_dir": str(output_dir),
            "log_file": str(log_path),
            "resume_path": str(resume_path),
            "phase0_source": None,
            "phase0_loss_mode": args.loss_mode,
            "phase0_recipe": build_phase0_recipe_payload(
                args,
                scaled_learning_rate=scaled_learning_rate,
                effective_batch_size_world=effective_batch_size_world,
                world_size=world_size,
                warmup_steps=warmup_steps,
            ),
            "args": vars(args),
        },
    )

    start_epoch = 0
    global_step = 0
    best_loss = math.inf
    train_loss_window_best_loss = math.inf
    train_loss_window_batch_count = 0
    loss_plateau_windows_without_improvement = 0
    resume_epoch = 0
    resume_batch_index = 0
    resume_epoch_complete = True
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        resume_epoch = int(resume_checkpoint.get("epoch", -1))
        resume_batch_index = int(resume_checkpoint.get("epoch_batch_index", 0))
        resume_epoch_complete = bool(resume_checkpoint.get("epoch_complete", False))
        if resume_epoch_complete or resume_batch_index <= 0:
            start_epoch = resume_epoch + 1
            resume_batch_index = 0
        else:
            start_epoch = resume_epoch
        global_step = int(resume_checkpoint.get("step", 0))
        resumed_mode = str(resume_checkpoint.get("phase0_early_stopping_mode", ""))
        if resumed_mode == PHASE0_EARLY_STOPPING_MODE:
            best_loss = float(
                resume_checkpoint.get(
                    "best_train_effective_batch_loss",
                    resume_checkpoint.get("best_loss", math.inf),
                )
            )
            train_loss_window_best_loss = float(resume_checkpoint.get("train_loss_window_best_loss", math.inf))
            train_loss_window_batch_count = int(resume_checkpoint.get("train_loss_window_batch_count", 0))
            loss_plateau_windows_without_improvement = int(
                resume_checkpoint.get(
                    "loss_plateau_windows_without_improvement",
                    resume_checkpoint.get("epochs_without_improvement", 0),
                )
            )
        else:
            best_loss = math.inf
            train_loss_window_best_loss = math.inf
            train_loss_window_batch_count = 0
            loss_plateau_windows_without_improvement = 0
            log_json_event(
                log_path,
                {
                    "event": "phase0_early_stopping_state_reset",
                    "reason": "checkpoint_used_legacy_batch_loss_patience",
                    "resume_checkpoint": str(resume_path),
                    "previous_mode": resumed_mode or "legacy",
                    "new_mode": PHASE0_EARLY_STOPPING_MODE,
                },
            )
        scheduler_state = resume_checkpoint.get("scheduler_state_dict")
        if isinstance(scheduler_state, dict) and scheduler_state:
            scheduler.load_state_dict(scheduler_state)
        else:
            scheduler.step_index = global_step
        scheduler.apply_current_lrs()
        log_json_event(
            log_path,
            {
                "event": "phase0_run_resumed",
                "resume_checkpoint": str(resume_path),
                "start_epoch": start_epoch,
                "resume_epoch": resume_epoch,
                "resume_batch_index": resume_batch_index,
                "resume_epoch_complete": resume_epoch_complete,
                "global_step": global_step,
                "best_loss": best_loss,
                "train_loss_window_batch_count": train_loss_window_batch_count,
                "train_loss_window_best_loss": train_loss_window_best_loss,
                "loss_plateau_windows_without_improvement": loss_plateau_windows_without_improvement,
                "phase0_base_learning_rate": float(args.learning_rate),
                "phase0_scaled_learning_rate": float(scaled_learning_rate),
                "phase0_lr_scale_base_batch_size": int(args.lr_scale_base_batch_size),
                "phase0_warmup_epochs": int(args.warmup_epochs),
                "phase0_warmup_steps": int(warmup_steps),
                "phase0_scheduler_mode": str(args.scheduler_mode),
                "phase0_total_steps": int(args.total_steps),
                "phase0_adamw_betas": [float(args.adamw_beta1), float(args.adamw_beta2)],
            },
        )
    else:
        scheduler.apply_current_lrs()

    mask_generator = SpatialMaskGenerator(args.image_size, args.patch_size, args.mask_ratio)
    last_checkpoint = output_dir / "last.pt"
    best_checkpoint = output_dir / "best.pt"
    phase0_encoder_export = output_dir / "phase0_encoder_final.pth"
    reconstruction_preview_dir = output_dir / "reconstruction_previews"
    stop_training = False
    have_resume_epoch_batch_offset = start_epoch == resume_epoch and resume_batch_index > 0

    epoch_iterator = range(start_epoch, args.epochs) if args.epochs > 0 else itertools.count(start_epoch)
    for epoch in epoch_iterator:
        phase0_dataset.set_epoch(epoch)
        epoch_batch_offset = resume_batch_index if have_resume_epoch_batch_offset and epoch == resume_epoch else 0
        if epoch_batch_offset > 0:
            phase0_sampler.set_start_index(epoch_batch_offset * args.batch_size)
            log_json_event(
                log_path,
                {
                    "event": "phase0_epoch_resume_offset_applied",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "resume_batch_index": epoch_batch_offset,
                    "resume_sampler_start_index": epoch_batch_offset * args.batch_size,
                },
            )
        else:
            phase0_sampler.set_start_index(0)

        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        epoch_masked_mse_sum = 0.0
        epoch_masked_mae_sum = 0.0
        epoch_masked_psnr_sum = 0.0
        epoch_full_mse_sum = 0.0
        epoch_full_psnr_sum = 0.0
        epoch_pred_std_sum = 0.0
        epoch_grad_norm_sum = 0.0
        epoch_grad_norm_count = 0
        epoch_lr_last = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float(args.learning_rate)
        epoch_pred_norm_mean_sum = 0.0
        epoch_pred_norm_std_sum = 0.0
        epoch_target_norm_mean_sum = 0.0
        epoch_target_norm_std_sum = 0.0
        epoch_pred_rgb_after_unnorm_std_sum = 0.0
        epoch_masked_psnr_after_unnorm_sum = 0.0
        effective_batch_loss_sum = 0.0
        effective_batch_microbatch_count = 0
        last_completed_epoch_batch_index = epoch_batch_offset
        epoch_started_at = time.time()
        latest_grad_norm = float("nan")
        latest_effective_batch_loss = float("nan")
        latest_amp_scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
        latest_skipped_optimizer_step = False
        latest_step_time_sec = 0.0
        latest_gpu_memory_allocated = float(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0.0
        optimizer.zero_grad(set_to_none=True)
        progress_total = len(loader)
        progress_desc = f"Phase0 epoch {epoch + 1}" if args.epochs <= 0 else f"Phase0 epoch {epoch + 1}/{args.epochs}"
        current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float(args.learning_rate)
        log_phase0_state(
            log_path,
            event="phase0_epoch_started",
            epoch=epoch + 1,
            global_step=global_step,
            best_loss=best_loss,
            train_loss_window_best_loss=train_loss_window_best_loss,
            loss_plateau_windows_without_improvement=loss_plateau_windows_without_improvement,
            optimizer_lr=current_lr,
            args=args,
        )
        progress = tqdm(loader, total=progress_total, desc=progress_desc, dynamic_ncols=True, leave=False)
        last_preview_images: torch.Tensor | None = None
        last_preview_pixel_mask: torch.Tensor | None = None
        last_preview_reconstructed: torch.Tensor | None = None
        last_preview_batch_index = 0

        for step_index, batch in enumerate(progress, start=1):
            step_started_at = time.time()
            if len(batch) == 3:
                images, _, _ = batch
            elif len(batch) == 2:
                images, _ = batch
            else:
                raise ValueError(f"Unexpected Phase 0 batch structure with {len(batch)} items.")
            images = images.to(device, non_blocking=True)
            pixel_mask, _ = mask_generator(images.shape[0], device)
            non_finite_reason = ""

            if not phase0_tensor_is_finite(images):
                non_finite_reason = "non_finite_input_images"
            elif not phase0_tensor_is_finite(pixel_mask):
                non_finite_reason = "non_finite_pixel_mask"

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                reconstructed = model(images, pixel_mask)
                if not non_finite_reason and not phase0_tensor_is_finite(reconstructed):
                    non_finite_reason = "non_finite_reconstruction"
                loss = compute_phase0_reconstruction_loss(
                    reconstructed,
                    images,
                    pixel_mask,
                    patch_size=args.patch_size,
                    loss_mode=args.loss_mode,
                )
                if not non_finite_reason and not phase0_tensor_is_finite(loss):
                    non_finite_reason = "non_finite_loss"
                loss = loss / args.grad_accum_steps

            if non_finite_reason:
                save_phase0_checkpoint(
                    output_dir / "step_last.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    epoch=epoch,
                    epoch_batch_index=epoch_batch_index,
                    epoch_complete=False,
                    step=global_step,
                    best_loss=best_loss,
                    train_loss_window_best_loss=train_loss_window_best_loss,
                    train_loss_window_batch_count=train_loss_window_batch_count,
                    loss_plateau_windows_without_improvement=loss_plateau_windows_without_improvement,
                    args=args,
                )
                log_json_event(
                    log_path,
                    {
                        "event": "phase0_non_finite_guard_triggered",
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "microbatch_index": step_index,
                        "reason": non_finite_reason,
                        "loss_mode": args.loss_mode,
                        "mask_ratio": float(args.mask_ratio),
                        "patch_size": int(args.patch_size),
                    },
                )
                stop_training = True
                break

            step_loss = float(loss.detach().item()) * args.grad_accum_steps
            pred_stats = reconstructed.detach().float()
            target_stats = images.detach().float()
            pixel_mask_stats = pixel_mask.detach().float()
            prediction_mean = float(pred_stats.mean().item())
            prediction_std = float(pred_stats.std(unbiased=False).item())
            target_mean = float(target_stats.mean().item())
            target_std = float(target_stats.std(unbiased=False).item())
            masked_ratio_actual = float(pixel_mask_stats[:, :1].mean().item())
            monitoring_metrics = _phase0_monitoring_metrics(
                images,
                reconstructed,
                pixel_mask,
                patch_size=args.patch_size,
                loss_mode=args.loss_mode,
            )
            if args.loss_mode == PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE:
                monitoring_metrics.setdefault("pred_norm_mean", float("nan"))
                monitoring_metrics.setdefault("pred_norm_std", float("nan"))
                monitoring_metrics.setdefault("target_norm_mean", float("nan"))
                monitoring_metrics.setdefault("target_norm_std", float("nan"))
                monitoring_metrics.setdefault("pred_rgb_after_unnorm_std", float("nan"))
                monitoring_metrics.setdefault("masked_psnr_after_unnorm", float("nan"))
            amp_scale_before_step = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
            scaler.scale(loss).backward()
            batch_size = int(images.shape[0])
            epoch_batch_index = epoch_batch_offset + step_index
            last_completed_epoch_batch_index = epoch_batch_index
            epoch_loss_sum += step_loss * batch_size
            epoch_sample_count += batch_size
            epoch_masked_mse_sum += float(monitoring_metrics["masked_mse"]) * batch_size
            epoch_masked_mae_sum += float(monitoring_metrics["masked_mae"]) * batch_size
            epoch_masked_psnr_sum += float(monitoring_metrics["masked_psnr"]) * batch_size
            epoch_full_mse_sum += float(monitoring_metrics["full_mse"]) * batch_size
            epoch_full_psnr_sum += float(monitoring_metrics["full_psnr"]) * batch_size
            epoch_pred_std_sum += prediction_std * batch_size
            if "pred_norm_mean" in monitoring_metrics:
                epoch_pred_norm_mean_sum += float(monitoring_metrics["pred_norm_mean"]) * batch_size
                epoch_pred_norm_std_sum += float(monitoring_metrics["pred_norm_std"]) * batch_size
                epoch_target_norm_mean_sum += float(monitoring_metrics["target_norm_mean"]) * batch_size
                epoch_target_norm_std_sum += float(monitoring_metrics["target_norm_std"]) * batch_size
                epoch_pred_rgb_after_unnorm_std_sum += float(monitoring_metrics["pred_rgb_after_unnorm_std"]) * batch_size
                epoch_masked_psnr_after_unnorm_sum += float(monitoring_metrics["masked_psnr_after_unnorm"]) * batch_size
            effective_batch_loss_sum += step_loss
            effective_batch_microbatch_count += 1
            last_preview_images = images.detach()
            last_preview_pixel_mask = pixel_mask.detach()
            last_preview_reconstructed = reconstructed.detach()
            last_preview_batch_index = epoch_batch_index
            latest_amp_scale = amp_scale_before_step
            latest_skipped_optimizer_step = False
            latest_step_time_sec = max(time.time() - step_started_at, 0.0)
            latest_gpu_memory_allocated = float(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0.0

            current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float(args.learning_rate)
            log_phase0_state(
                log_path,
                event="phase0_microbatch_processed",
                epoch=epoch + 1,
                global_step=global_step,
                microbatch_index=step_index,
                microbatches_in_effective_batch=effective_batch_microbatch_count,
                samples_seen=epoch_sample_count,
                epoch_loss_sum=epoch_loss_sum,
                epoch_sample_count=epoch_sample_count,
                microbatch_loss=step_loss,
                train_loss_window_best_loss=train_loss_window_best_loss,
                best_loss=best_loss,
                loss_plateau_windows_without_improvement=loss_plateau_windows_without_improvement,
                optimizer_lr=current_lr,
                amp_scale=latest_amp_scale,
                skipped_optimizer_step=latest_skipped_optimizer_step,
                step_time_sec=latest_step_time_sec,
                gpu_memory_allocated=latest_gpu_memory_allocated,
                prediction_mean=prediction_mean,
                prediction_std=prediction_std,
                target_mean=target_mean,
                target_std=target_std,
                masked_ratio_actual=masked_ratio_actual,
                masked_mse=monitoring_metrics["masked_mse"],
                masked_mae=monitoring_metrics["masked_mae"],
                masked_psnr=monitoring_metrics["masked_psnr"],
                full_mse=monitoring_metrics["full_mse"],
                full_psnr=monitoring_metrics["full_psnr"],
                pred_norm_mean=monitoring_metrics.get("pred_norm_mean"),
                pred_norm_std=monitoring_metrics.get("pred_norm_std"),
                target_norm_mean=monitoring_metrics.get("target_norm_mean"),
                target_norm_std=monitoring_metrics.get("target_norm_std"),
                pred_rgb_after_unnorm_std=monitoring_metrics.get("pred_rgb_after_unnorm_std"),
                masked_psnr_after_unnorm=monitoring_metrics.get("masked_psnr_after_unnorm"),
                args=args,
            )

            progress.set_postfix(
                build_progress_postfix(
                    global_step if global_step > 0 else step_index,
                    None,
                    epoch=epoch + 1,
                    loss=step_loss,
                    lr=current_lr,
                    grad_norm=latest_grad_norm,
                    pred_mean=prediction_mean,
                    pred_std=prediction_std,
                    target_mean=target_mean,
                    target_std=target_std,
                    mask_ratio_actual=masked_ratio_actual,
                    amp_scale=latest_amp_scale,
                    skipped_optimizer_step=latest_skipped_optimizer_step,
                    step_time_sec=latest_step_time_sec,
                    gpu_memory_allocated=latest_gpu_memory_allocated,
                )
            )

            if step_index % args.grad_accum_steps == 0 or step_index == len(loader):
                scaler.unscale_(optimizer)
                grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm)))
                if not phase0_scalar_is_finite(grad_norm):
                    save_phase0_checkpoint(
                        output_dir / "step_last.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        epoch=epoch,
                        epoch_batch_index=epoch_batch_index,
                        epoch_complete=False,
                        step=global_step,
                        best_loss=best_loss,
                        train_loss_window_best_loss=train_loss_window_best_loss,
                        train_loss_window_batch_count=train_loss_window_batch_count,
                        loss_plateau_windows_without_improvement=loss_plateau_windows_without_improvement,
                        args=args,
                    )
                    log_json_event(
                        log_path,
                        {
                            "event": "phase0_non_finite_guard_triggered",
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "microbatch_index": step_index,
                            "reason": "non_finite_grad_norm",
                            "grad_clip_norm": float(args.grad_clip_norm),
                            "mask_ratio": float(args.mask_ratio),
                            "patch_size": int(args.patch_size),
                        },
                    )
                    stop_training = True
                    break
                scaler.step(optimizer)
                scaler.update()
                latest_amp_scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
                latest_skipped_optimizer_step = latest_amp_scale < amp_scale_before_step
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                # Advance the schedule once per optimizer step, not per microbatch.
                scheduler.step()
                completed_microbatches = effective_batch_microbatch_count
                effective_batch_loss = effective_batch_loss_sum / max(1, completed_microbatches)
                latest_effective_batch_loss = effective_batch_loss
                latest_grad_norm = grad_norm
                effective_batch_loss_sum = 0.0
                effective_batch_microbatch_count = 0
                train_loss_window_best_loss = min(train_loss_window_best_loss, effective_batch_loss)
                train_loss_window_batch_count += 1
                current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float(args.learning_rate)
                epoch_lr_last = current_lr
                latest_step_time_sec = max(time.time() - step_started_at, 0.0)
                latest_gpu_memory_allocated = float(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0.0
                epoch_grad_norm_sum += grad_norm
                epoch_grad_norm_count += 1

                log_phase0_state(
                    log_path,
                    event="phase0_optimizer_step",
                    epoch=epoch + 1,
                    global_step=global_step,
                    microbatches_in_effective_batch=completed_microbatches,
                    samples_seen=epoch_sample_count,
                    epoch_loss_sum=epoch_loss_sum,
                    epoch_sample_count=epoch_sample_count,
                    effective_batch_loss=effective_batch_loss,
                    train_loss_window_best_loss=train_loss_window_best_loss,
                    best_loss=best_loss,
                    loss_plateau_windows_without_improvement=loss_plateau_windows_without_improvement,
                    optimizer_lr=current_lr,
                    grad_norm=grad_norm,
                    amp_scale=latest_amp_scale,
                    skipped_optimizer_step=latest_skipped_optimizer_step,
                    step_time_sec=latest_step_time_sec,
                    gpu_memory_allocated=latest_gpu_memory_allocated,
                    prediction_mean=prediction_mean,
                    prediction_std=prediction_std,
                    target_mean=target_mean,
                    target_std=target_std,
                    masked_ratio_actual=masked_ratio_actual,
                    masked_mse=monitoring_metrics["masked_mse"],
                    masked_mae=monitoring_metrics["masked_mae"],
                    masked_psnr=monitoring_metrics["masked_psnr"],
                    full_mse=monitoring_metrics["full_mse"],
                    full_psnr=monitoring_metrics["full_psnr"],
                    pred_norm_mean=monitoring_metrics.get("pred_norm_mean"),
                    pred_norm_std=monitoring_metrics.get("pred_norm_std"),
                    target_norm_mean=monitoring_metrics.get("target_norm_mean"),
                    target_norm_std=monitoring_metrics.get("target_norm_std"),
                    pred_rgb_after_unnorm_std=monitoring_metrics.get("pred_rgb_after_unnorm_std"),
                    masked_psnr_after_unnorm=monitoring_metrics.get("masked_psnr_after_unnorm"),
                    args=args,
                )
                log_json_event(
                    log_path,
                    {
                        "event": "phase0_gradient_clipped",
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "grad_norm": grad_norm,
                        "grad_clip_norm": float(args.grad_clip_norm),
                        "microbatches_in_effective_batch": completed_microbatches,
                        "prediction_mean": prediction_mean,
                        "prediction_std": prediction_std,
                        "target_mean": target_mean,
                        "target_std": target_std,
                        "masked_ratio_actual": masked_ratio_actual,
                        "mask_patch_size": int(args.patch_size),
                        "amp_scale": latest_amp_scale,
                        "skipped_optimizer_step": latest_skipped_optimizer_step,
                        "step_time_sec": latest_step_time_sec,
                        "gpu_memory_allocated": latest_gpu_memory_allocated,
                    },
                )

                step_postfix = build_progress_postfix(
                    global_step,
                    None,
                    epoch=epoch + 1,
                    loss=effective_batch_loss,
                    lr=current_lr,
                    grad_norm=grad_norm,
                    pred_mean=prediction_mean,
                    pred_std=prediction_std,
                    target_mean=target_mean,
                    target_std=target_std,
                    mask_ratio_actual=masked_ratio_actual,
                    amp_scale=latest_amp_scale,
                    skipped_optimizer_step=latest_skipped_optimizer_step,
                    step_time_sec=latest_step_time_sec,
                    gpu_memory_allocated=latest_gpu_memory_allocated,
                )
                progress.set_postfix(step_postfix)

                if train_loss_window_batch_count >= args.train_loss_window:
                    window_best_loss = train_loss_window_best_loss
                    window_improved = window_best_loss < best_loss - args.early_stopping_min_delta
                    if window_improved:
                        best_loss = window_best_loss
                        loss_plateau_windows_without_improvement = 0
                        save_phase0_checkpoint(
                            best_checkpoint,
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            scheduler=scheduler,
                            epoch=epoch,
                            epoch_batch_index=epoch_batch_index,
                            epoch_complete=False,
                            step=global_step,
                            best_loss=best_loss,
                            train_loss_window_best_loss=train_loss_window_best_loss,
                            train_loss_window_batch_count=train_loss_window_batch_count,
                            loss_plateau_windows_without_improvement=loss_plateau_windows_without_improvement,
                            args=args,
                        )
                    else:
                        loss_plateau_windows_without_improvement += 1

                    log_json_event(
                        log_path,
                        {
                            "event": "phase0_train_loss_window_finished",
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "window_effective_batches": train_loss_window_batch_count,
                            "window_best_effective_batch_loss": window_best_loss,
                            "best_loss": best_loss,
                            "improved": window_improved,
                            "loss_plateau_windows_without_improvement": loss_plateau_windows_without_improvement,
                            "early_stopping_patience": args.early_stopping_patience,
                            "early_stopping_min_delta": args.early_stopping_min_delta,
                            "batch_size": int(args.batch_size),
                            "grad_accum_steps": int(args.grad_accum_steps),
                            "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
                            "mask_ratio": float(args.mask_ratio),
                            "patch_size": int(args.patch_size),
                        },
                    )

                    train_loss_window_best_loss = math.inf
                    train_loss_window_batch_count = 0

                    if loss_plateau_windows_without_improvement >= args.early_stopping_patience:
                        save_phase0_checkpoint(
                            output_dir / "step_last.pt",
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            scheduler=scheduler,
                            epoch=epoch,
                            epoch_batch_index=epoch_batch_index,
                            epoch_complete=False,
                            step=global_step,
                            best_loss=best_loss,
                            train_loss_window_best_loss=train_loss_window_best_loss,
                            train_loss_window_batch_count=train_loss_window_batch_count,
                            loss_plateau_windows_without_improvement=loss_plateau_windows_without_improvement,
                            args=args,
                        )
                        log_json_event(
                            log_path,
                            {
                                "event": "phase0_early_stopping_triggered",
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "patience_reached": loss_plateau_windows_without_improvement,
                                "train_loss_window_effective_batches": args.train_loss_window,
                                "best_loss": best_loss,
                                "window_best_effective_batch_loss": window_best_loss,
                                "early_stopping_patience": args.early_stopping_patience,
                                "early_stopping_min_delta": args.early_stopping_min_delta,
                                "early_stopping_mode": PHASE0_EARLY_STOPPING_MODE,
                            },
                        )
                        stop_training = True
                        break

                save_phase0_checkpoint(
                    output_dir / "step_last.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    epoch=epoch,
                    epoch_batch_index=epoch_batch_index,
                    epoch_complete=False,
                    step=global_step,
                    best_loss=best_loss,
                    train_loss_window_best_loss=train_loss_window_best_loss,
                    train_loss_window_batch_count=train_loss_window_batch_count,
                    loss_plateau_windows_without_improvement=loss_plateau_windows_without_improvement,
                    args=args,
                )
                log_json_event(
                    log_path,
                    {
                        "event": "phase0_train_step",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": effective_batch_loss,
                        "microbatch_loss": step_loss,
                        "micro_batch_size": int(images.shape[0]),
                        "effective_batch_size": int(args.batch_size * completed_microbatches),
                        "mask_ratio": float(args.mask_ratio),
                        "backbone": args.backbone,
                        "best_loss": best_loss,
                        "train_loss_window_batch_count": train_loss_window_batch_count,
                        "train_loss_window_best_loss": train_loss_window_best_loss,
                        "loss_plateau_windows_without_improvement": loss_plateau_windows_without_improvement,
                        "optimizer_lr": current_lr,
                        "microbatches_in_effective_batch": completed_microbatches,
                        "epoch_loss_sum": epoch_loss_sum,
                        "epoch_sample_count": epoch_sample_count,
                        "epoch_batch_index": epoch_batch_index,
                        "prediction_mean": prediction_mean,
                        "prediction_std": prediction_std,
                        "target_mean": target_mean,
                        "target_std": target_std,
                        "masked_ratio_actual": masked_ratio_actual,
                        "grad_norm": grad_norm,
                        "mask_patch_size": int(args.patch_size),
                    },
                )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            epoch_elapsed = max(time.time() - epoch_started_at, 1e-8)
            progress.set_postfix(
                build_progress_postfix(
                    step_index,
                    progress_total,
                    micro_loss=step_loss,
                    eff_loss=latest_effective_batch_loss,
                    epoch_loss=epoch_loss_sum / max(1, epoch_sample_count),
                    best=best_loss,
                    window_best=train_loss_window_best_loss,
                    grad_norm=latest_grad_norm,
                    lr=current_lr,
                    samples=epoch_sample_count,
                    throughput=epoch_sample_count / epoch_elapsed,
                    batch_window=train_loss_window_batch_count,
                    plateaus=loss_plateau_windows_without_improvement,
                    mb=f"{step_index}/{len(loader)}",
                    mask_ratio=float(args.mask_ratio),
                    mask_ratio_actual=masked_ratio_actual,
                    batch_size=int(args.batch_size),
                    eff_bs=int(args.batch_size * args.grad_accum_steps),
                    pred_std=prediction_std,
                    tgt_std=target_std,
                )
            )

        epoch_loss = epoch_loss_sum / max(1, epoch_sample_count)
        epoch_elapsed = max(time.time() - epoch_started_at, 1e-8)
        epoch_throughput = epoch_sample_count / epoch_elapsed
        epoch_masked_mse = epoch_masked_mse_sum / max(1, epoch_sample_count)
        epoch_masked_mae = epoch_masked_mae_sum / max(1, epoch_sample_count)
        epoch_masked_psnr = epoch_masked_psnr_sum / max(1, epoch_sample_count)
        epoch_full_mse = epoch_full_mse_sum / max(1, epoch_sample_count)
        epoch_full_psnr = epoch_full_psnr_sum / max(1, epoch_sample_count)
        epoch_pred_std_mean = epoch_pred_std_sum / max(1, epoch_sample_count)
        epoch_grad_norm_mean = epoch_grad_norm_sum / max(1, epoch_grad_norm_count)
        epoch_pred_norm_mean = epoch_pred_norm_mean_sum / max(1, epoch_sample_count)
        epoch_pred_norm_std = epoch_pred_norm_std_sum / max(1, epoch_sample_count)
        epoch_target_norm_mean = epoch_target_norm_mean_sum / max(1, epoch_sample_count)
        epoch_target_norm_std = epoch_target_norm_std_sum / max(1, epoch_sample_count)
        epoch_pred_rgb_after_unnorm_std = epoch_pred_rgb_after_unnorm_std_sum / max(1, epoch_sample_count)
        epoch_masked_psnr_after_unnorm = epoch_masked_psnr_after_unnorm_sum / max(1, epoch_sample_count)
        log_json_event(
            log_path,
            {
                "event": "phase0_epoch_summary",
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss_mean": epoch_loss,
                "val_loss_mean": None,
                "masked_mse": epoch_masked_mse,
                "masked_mae": epoch_masked_mae,
                "masked_psnr": epoch_masked_psnr,
                "full_mse": epoch_full_mse,
                "full_psnr": epoch_full_psnr,
                "pred_std_mean": epoch_pred_std_mean,
                "grad_norm_mean": epoch_grad_norm_mean,
                "lr_last": epoch_lr_last,
                "epoch_time_sec": epoch_elapsed,
                "epoch_sample_count": epoch_sample_count,
                "epoch_elapsed_seconds": epoch_elapsed,
                "epoch_samples_per_second": epoch_throughput,
                "best_loss": best_loss,
                "train_loss_window_best_loss": train_loss_window_best_loss,
                "train_loss_window_batch_count": train_loss_window_batch_count,
                "loss_plateau_windows_without_improvement": loss_plateau_windows_without_improvement,
                "optimizer_lr": epoch_lr_last,
                "last_completed_epoch_batch_index": last_completed_epoch_batch_index,
                "mask_ratio": float(args.mask_ratio),
                "batch_size": int(args.batch_size),
                "grad_accum_steps": int(args.grad_accum_steps),
                "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
            },
        )
        epoch_summary_message = (
            f"[phase0][epoch {epoch + 1}] "
            f"train_loss_mean={epoch_loss:.4f} "
            f"val_loss_mean=n/a "
            f"masked_mse={epoch_masked_mse:.4f} "
            f"masked_mae={epoch_masked_mae:.4f} "
            f"masked_psnr={epoch_masked_psnr:.2f} "
            f"full_mse={epoch_full_mse:.4f} "
            f"full_psnr={epoch_full_psnr:.2f} "
            f"pred_std_mean={epoch_pred_std_mean:.4f} "
            f"grad_norm_mean={epoch_grad_norm_mean:.4f} "
            f"lr_last={epoch_lr_last:.2e} "
            f"epoch_time_sec={epoch_elapsed:.1f}"
        )
        if args.loss_mode == PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE:
            epoch_summary_message += (
                f" pred_norm_mean={epoch_pred_norm_mean:.4f}"
                f" pred_norm_std={epoch_pred_norm_std:.4f}"
                f" target_norm_mean={epoch_target_norm_mean:.4f}"
                f" target_norm_std={epoch_target_norm_std:.4f}"
                f" pred_rgb_after_unnorm_std={epoch_pred_rgb_after_unnorm_std:.4f}"
                f" masked_psnr_after_unnorm={epoch_masked_psnr_after_unnorm:.2f}"
            )
        tqdm.write(epoch_summary_message)
        if args.max_steps > 0 and global_step >= args.max_steps:
            save_phase0_checkpoint(
                last_checkpoint,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                epoch=epoch,
                epoch_batch_index=last_completed_epoch_batch_index,
                epoch_complete=False,
                step=global_step,
                best_loss=best_loss,
                train_loss_window_best_loss=train_loss_window_best_loss,
                train_loss_window_batch_count=train_loss_window_batch_count,
                loss_plateau_windows_without_improvement=loss_plateau_windows_without_improvement,
                args=args,
            )
            log_json_event(
                log_path,
                {
                    "event": "phase0_step_cap_reached",
                    "max_steps": args.max_steps,
                    "global_step": global_step,
                },
            )
            break

        save_phase0_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            epoch=epoch,
            epoch_batch_index=epoch_batch_offset + len(loader),
            epoch_complete=True,
            step=global_step,
            best_loss=best_loss,
            train_loss_window_best_loss=train_loss_window_best_loss,
            train_loss_window_batch_count=train_loss_window_batch_count,
            loss_plateau_windows_without_improvement=loss_plateau_windows_without_improvement,
            args=args,
        )
        log_json_event(
            log_path,
            {
                "event": "phase0_epoch_finished",
                "epoch": epoch + 1,
                "epoch_loss": epoch_loss,
                "best_loss": best_loss,
                "train_loss_window_batch_count": train_loss_window_batch_count,
                "train_loss_window_best_loss": train_loss_window_best_loss,
                "loss_plateau_windows_without_improvement": loss_plateau_windows_without_improvement,
                "global_step": global_step,
                "epoch_sample_count": epoch_sample_count,
                "effective_batches_completed": train_loss_window_batch_count,
                "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
                "epoch_batch_index": epoch_batch_offset + len(loader),
            },
        )
        if (
            args.reconstruction_preview_interval > 0
            and (epoch + 1) % args.reconstruction_preview_interval == 0
            and last_preview_images is not None
            and last_preview_pixel_mask is not None
            and last_preview_reconstructed is not None
        ):
            preview_path = reconstruction_preview_dir / f"epoch_{epoch + 1:04d}_step_{last_preview_batch_index:06d}.png"
            full_output_preview_path, masked_fill_preview_path = save_phase0_reconstruction_preview(
                preview_path,
                originals=last_preview_images,
                pixel_mask=last_preview_pixel_mask,
                reconstructed=last_preview_reconstructed,
                patch_size=args.patch_size,
                loss_mode=args.loss_mode,
                epoch=epoch + 1,
                global_step=global_step,
                sample_count=args.reconstruction_preview_count,
            )
            log_json_event(
                log_path,
                {
                    "event": "phase0_reconstruction_preview_saved",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "preview_path": str(full_output_preview_path),
                    "preview_masked_fill_path": str(masked_fill_preview_path),
                    "preview_count": int(args.reconstruction_preview_count),
                },
            )
        if stop_training:
            break

    export_state = model.encoder.state_dict()
    if best_checkpoint.exists():
        best_payload = torch.load(best_checkpoint, map_location="cpu")
        export_state = best_payload.get("encoder_state_dict", export_state)
    torch.save(export_state, phase0_encoder_export)
    log_json_event(
        log_path,
        {
            "event": "phase0_finished",
            "best_loss": best_loss,
            "train_loss_window_batch_count": train_loss_window_batch_count,
            "train_loss_window_best_loss": train_loss_window_best_loss,
            "loss_plateau_windows_without_improvement": loss_plateau_windows_without_improvement,
            "encoder_export": str(phase0_encoder_export),
            "best_checkpoint": str(best_checkpoint),
            "exported_best_checkpoint_encoder": best_checkpoint.exists(),
            "stop_reason": "early_stopping" if stop_training else "epoch_limit_or_step_cap",
        },
    )
    print(f"[phase0] complete. Encoder exported to {phase0_encoder_export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
