#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PHASE0_EARLY_STOPPING_MODE = "effective_batch_window_best_v3"

try:
    from metric_learning_pipeline import (
        DEFAULT_BACKBONE_NAME,
        BACKBONE_REGISTRY,
        CAMERA_COLOR_CAST_PROBABILITY,
        CAMERA_COLOR_CAST_STRENGTH,
        CAMERA_COLOR_CAST_EVAL,
        build_datasets,
        make_balanced_sampler,
        evaluation_tensor_from_image,
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
        build_datasets,
        make_balanced_sampler,
        evaluation_tensor_from_image,
        load_resume_checkpoint,
        log_json_event,
        seed_everything,
    )


class Phase0WasteDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, samples: list[tuple[str, int]], classes: list[str], image_size: int) -> None:
        self.samples = list(samples)
        self.classes = list(classes)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = evaluation_tensor_from_image(image, self.image_size)
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


def build_llrd_optimizer(model: RepoSafeConvNeXtMIM, base_lr: float, weight_decay: float) -> torch.optim.Optimizer:
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

    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, foreach=False)


def save_phase0_checkpoint(
    path: Path,
    *,
    model: RepoSafeConvNeXtMIM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
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
        "epoch": int(epoch),
        "step": int(step),
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
    parser.add_argument("--augment-repeats", type=int, default=16)
    parser.add_argument("--augment-gaussian-sigmas", type=float, default=0.5)
    parser.add_argument("--camera-color-cast-probability", type=float, default=CAMERA_COLOR_CAST_PROBABILITY)
    parser.add_argument("--camera-color-cast-strength", type=float, default=CAMERA_COLOR_CAST_STRENGTH)
    parser.add_argument("--camera-color-cast-eval", action=argparse.BooleanOptionalAction, default=CAMERA_COLOR_CAST_EVAL)
    parser.add_argument("--class-mapping", type=str, default="")
    parser.add_argument("--auto-split-ratios", default="0.7,0.2,0.1")
    parser.add_argument(
        "--runtime-bad-sample-cleanup",
        action="store_true",
        help="Mirror the main trainer's runtime bad-sample cleanup flag for dataset construction.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
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
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--train-loss-window",
        type=int,
        default=5000,
        help=(
            "Phase 0 plateau window in optimizer-step batches. With batch-size 128 and "
            "grad-accum-steps 2, one window is 5000 effective batches of 256 images."
        ),
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

    log_json_event(
        log_path,
        {
            "event": "phase0_run_started",
            "output_dir": str(output_dir),
            "log_file": str(log_path),
            "resume_path": str(resume_path),
            "phase0_source": None,
            "args": vars(args),
        },
    )

    train_dataset, _, _, _, _ = build_datasets(args)
    phase0_dataset = train_dataset
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
            "train_loss_window_effective_batches": int(args.train_loss_window),
            "early_stopping_patience": int(args.early_stopping_patience),
        },
    )
    model.train()
    if device.type == "cuda" and hasattr(model.encoder, "set_grad_checkpointing"):
        try:
            model.encoder.set_grad_checkpointing(True)
        except TypeError:
            model.encoder.set_grad_checkpointing(enable=True)

    optimizer = build_llrd_optimizer(model, args.learning_rate, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    global_step = 0
    best_loss = math.inf
    train_loss_window_best_loss = math.inf
    train_loss_window_batch_count = 0
    loss_plateau_windows_without_improvement = 0
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
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
        log_json_event(
            log_path,
            {
                "event": "phase0_run_resumed",
                "resume_checkpoint": str(resume_path),
                "start_epoch": start_epoch,
                "global_step": global_step,
                "best_loss": best_loss,
                "train_loss_window_batch_count": train_loss_window_batch_count,
                "train_loss_window_best_loss": train_loss_window_best_loss,
                "loss_plateau_windows_without_improvement": loss_plateau_windows_without_improvement,
            },
        )

    mask_generator = SpatialMaskGenerator(args.image_size, args.patch_size, args.mask_ratio)
    last_checkpoint = output_dir / "last.pt"
    best_checkpoint = output_dir / "best.pt"
    phase0_encoder_export = output_dir / "phase0_encoder_final.pth"
    stop_training = False

    epoch_iterator = range(start_epoch, args.epochs) if args.epochs > 0 else itertools.count(start_epoch)
    for epoch in epoch_iterator:
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        effective_batch_loss_sum = 0.0
        effective_batch_microbatch_count = 0
        optimizer.zero_grad(set_to_none=True)
        progress_total = len(loader)
        progress_desc = f"Phase0 epoch {epoch + 1}" if args.epochs <= 0 else f"Phase0 epoch {epoch + 1}/{args.epochs}"
        progress = tqdm(loader, total=progress_total, desc=progress_desc)

        for step_index, batch in enumerate(progress, start=1):
            if len(batch) == 3:
                images, _, _ = batch
            elif len(batch) == 2:
                images, _ = batch
            else:
                raise ValueError(f"Unexpected Phase 0 batch structure with {len(batch)} items.")
            images = images.to(device, non_blocking=True)
            pixel_mask, _ = mask_generator(images.shape[0], device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                reconstructed = model(images, pixel_mask)
                loss = ((reconstructed - images) ** 2 * pixel_mask).sum() / (pixel_mask.sum() * images.shape[1] + 1e-8)
                loss = loss / args.grad_accum_steps

            step_loss = float(loss.detach().item()) * args.grad_accum_steps
            scaler.scale(loss).backward()
            batch_size = int(images.shape[0])
            epoch_loss_sum += step_loss * batch_size
            epoch_sample_count += batch_size
            effective_batch_loss_sum += step_loss
            effective_batch_microbatch_count += 1

            if step_index % args.grad_accum_steps == 0 or step_index == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                completed_microbatches = effective_batch_microbatch_count
                effective_batch_loss = effective_batch_loss_sum / max(1, completed_microbatches)
                effective_batch_loss_sum = 0.0
                effective_batch_microbatch_count = 0
                train_loss_window_best_loss = min(train_loss_window_best_loss, effective_batch_loss)
                train_loss_window_batch_count += 1

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
                            epoch=epoch,
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
                            epoch=epoch,
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
                    epoch=epoch,
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
                    },
                )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            progress.set_postfix(
                micro_loss=step_loss,
                best=best_loss,
                window_best=train_loss_window_best_loss,
                batch_window=train_loss_window_batch_count,
                plateaus=loss_plateau_windows_without_improvement,
            )

        epoch_loss = epoch_loss_sum / max(1, epoch_sample_count)
        if args.max_steps > 0 and global_step >= args.max_steps:
            save_phase0_checkpoint(
                last_checkpoint,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
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
            epoch=epoch,
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
    print(f"Phase 0 complete. Encoder exported to {phase0_encoder_export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
