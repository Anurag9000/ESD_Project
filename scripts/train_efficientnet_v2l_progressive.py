#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_V2_L_Weights, efficientnet_v2_l
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PhaseSpec:
    name: str
    unfrozen_backbone_modules: int
    epochs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Progressively fine-tune a pretrained EfficientNet V2-L on Dataset_Final"
    )
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--stage-epochs", type=int, default=1)
    parser.add_argument("--unfreeze-chunk-size", type=int, default=20)
    parser.add_argument("--max-progressive-phases", type=int, default=0)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--weighted-sampling", action="store_true")
    parser.add_argument("--weights", choices=("default", "none"), default="default")
    parser.add_argument("--output-dir", default="runs/efficientnet_v2_l_progressive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def class_counts(dataset: datasets.ImageFolder) -> dict[str, int]:
    counts = {name: 0 for name in dataset.classes}
    for target in dataset.targets:
        counts[dataset.classes[target]] += 1
    return counts


def make_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    counts = class_counts(dataset)
    weights = [1.0 / counts[dataset.classes[target]] for target in dataset.targets]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)


def make_loader(
    dataset: datasets.ImageFolder,
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


def load_datasets(args: argparse.Namespace) -> tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder, DataLoader, DataLoader, DataLoader]:
    root = Path(args.dataset_root)
    train_transform, eval_transform = build_transforms(args.image_size)
    train_dataset = datasets.ImageFolder(root / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(root / "test", transform=eval_transform)

    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError("Class folders differ across train/val/test")

    sampler = make_weighted_sampler(train_dataset) if args.weighted_sampling else None
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=not args.weighted_sampling, sampler=sampler)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, shuffle=False)
    test_loader = make_loader(test_dataset, args.batch_size, args.num_workers, shuffle=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_model(num_classes: int, weights_mode: str) -> nn.Module:
    weights = EfficientNet_V2_L_Weights.DEFAULT if weights_mode == "default" else None
    model = efficientnet_v2_l(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def backbone_leaf_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    modules: list[tuple[str, nn.Module]] = []
    for name, module in model.features.named_modules():
        if not name:
            continue
        if any(True for _ in module.parameters(recurse=False)):
            modules.append((f"features.{name}", module))
    return modules


def build_phase_plan(total_modules: int, args: argparse.Namespace) -> list[PhaseSpec]:
    phases: list[PhaseSpec] = []
    if args.head_epochs > 0:
        phases.append(PhaseSpec(name="head_only", unfrozen_backbone_modules=0, epochs=args.head_epochs))

    count = 0
    phase_index = 0
    while count < total_modules and args.stage_epochs > 0:
        count = min(total_modules, count + args.unfreeze_chunk_size)
        phase_index += 1
        if args.max_progressive_phases and phase_index > args.max_progressive_phases:
            break
        phases.append(
            PhaseSpec(
                name=f"backbone_last_{count}_modules",
                unfrozen_backbone_modules=count,
                epochs=args.stage_epochs,
            )
        )
    return phases


def set_progressive_trainability(
    model: nn.Module,
    backbone_modules: list[tuple[str, nn.Module]],
    unfrozen_backbone_modules: int,
) -> list[str]:
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    if unfrozen_backbone_modules <= 0:
        return []

    thawed = backbone_modules[-unfrozen_backbone_modules:]
    for _, module in thawed:
        for param in module.parameters(recurse=False):
            param.requires_grad = True
    return [name for name, _ in thawed]


def freeze_frozen_batchnorms(backbone_modules: list[tuple[str, nn.Module]]) -> None:
    for _, module in backbone_modules:
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        if any(param.requires_grad for param in params):
            continue
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> AdamW:
    head_params = [param for param in model.classifier.parameters() if param.requires_grad]
    head_ids = {id(param) for param in head_params}
    backbone_params = [
        param for param in model.features.parameters() if param.requires_grad and id(param) not in head_ids
    ]

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    return AdamW(param_groups, weight_decay=args.weight_decay)


def limited_batches(loader: DataLoader, max_batches: int):
    if max_batches <= 0:
        yield from loader
        return
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        yield batch


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int,
    backbone_modules: list[tuple[str, nn.Module]],
    max_batches: int,
) -> tuple[float, float]:
    model.train()
    freeze_frozen_batchnorms(backbone_modules)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    optimizer.zero_grad(set_to_none=True)

    total_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    progress = tqdm(enumerate(limited_batches(loader, max_batches), start=1), total=total_batches, leave=False)
    for step, (images, labels) in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels) / grad_accum_steps

        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0 or step == total_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_loss = loss.item() * grad_accum_steps
        total_loss += batch_loss * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += labels.size(0)
        progress.set_postfix(loss=f"{total_loss / total_seen:.4f}", acc=f"{total_correct / total_seen:.4f}")

    return total_loss / total_seen, total_correct / total_seen


def evaluate(
    model: nn.Module,
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
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += labels.size(0)
    return total_loss / total_seen, total_correct / total_seen


def save_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.unfreeze_chunk_size < 1:
        raise ValueError("--unfreeze-chunk-size must be >= 1")

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = load_datasets(args)
    model = build_model(num_classes=len(train_dataset.classes), weights_mode=args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    backbone_modules = backbone_leaf_modules(model)
    phases = build_phase_plan(len(backbone_modules), args)

    total_params, _ = parameter_counts(model)
    history: list[dict[str, object]] = []
    best_val_acc = -1.0

    for phase_index, phase in enumerate(phases, start=1):
        thawed_names = set_progressive_trainability(model, backbone_modules, phase.unfrozen_backbone_modules)
        optimizer = build_optimizer(model, args)
        _, trainable_params = parameter_counts(model)

        for epoch in range(1, phase.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                grad_accum_steps=args.grad_accum_steps,
                backbone_modules=backbone_modules,
                max_batches=args.max_train_batches,
            )
            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                max_batches=args.max_eval_batches,
            )

            row = {
                "phase_index": phase_index,
                "phase_name": phase.name,
                "epoch_in_phase": epoch,
                "unfrozen_backbone_modules": phase.unfrozen_backbone_modules,
                "newly_unfrozen_tail_modules": thawed_names[-args.unfreeze_chunk_size:],
                "trainable_params": trainable_params,
                "total_params": total_params,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            history.append(row)
            print(row)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx,
                "args": vars(args),
                "history": history,
                "current_phase": asdict(phase),
            }
            torch.save(checkpoint, output_dir / "last.pt")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(checkpoint, output_dir / "best.pt")

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        max_batches=args.max_eval_batches,
    )

    metrics = {
        "model_name": "efficientnet_v2_l",
        "weights": args.weights,
        "device": str(device),
        "class_names": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "train_class_counts": class_counts(train_dataset),
        "val_class_counts": class_counts(val_dataset),
        "test_class_counts": class_counts(test_dataset),
        "phase_plan": [asdict(phase) for phase in phases],
        "history": history,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    save_json(output_dir / "metrics.json", metrics)
    print({"test_loss": test_loss, "test_acc": test_acc})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
