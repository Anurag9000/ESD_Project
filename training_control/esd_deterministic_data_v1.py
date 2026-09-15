#!/usr/bin/env python3
"""Restart-addressable ESD data-stream shim used by individual and cohort training.

The historical image dataset class combined the scientific seed/index/epoch identity
with os.urandom(), process id, worker id and a mutable draw counter.  That made a
checkpointed sampler cursor insufficient to reconstruct the next augmented image.
This shim removes only those process-local entropy terms.  The augmentation policy,
its parameter ranges, class sampling and variant/view identities remain owned by
``scripts/metric_learning_pipeline.py``.

Persistent DataLoader workers are also disabled on this exact path so a restarted
worker pool cannot carry hidden process-local state across checkpoint boundaries.
"""
from __future__ import annotations

from typing import Any

SCHEMA = "esd-restart-addressable-data/v1"
_INSTALLED = False


def derive_augmentation_seed(
    *,
    seed: int,
    split_offset: int,
    source_index: int,
    variant_index: int,
    epoch: int,
    view_offset: int = 0,
    attempt: int = 0,
) -> int:
    values = (seed, split_offset, source_index, variant_index, epoch, view_offset, attempt)
    if any(int(value) < 0 for value in values):
        raise ValueError("augmentation seed coordinates must be non-negative")
    return (
        int(seed) * 1_000_003
        + int(split_offset)
        + int(source_index) * 9_973
        + int(variant_index) * 99_991
        + int(epoch) * 104_729
        + int(view_offset) * 1_299_721
        + (int(attempt) + 1) * 15_485_863
    ) & ((1 << 63) - 1)


def install(pipeline: Any) -> None:
    """Install deterministic augmentation + restart-clean DataLoader construction."""
    global _INSTALLED
    if _INSTALLED:
        return
    required = (
        "DeterministicAugmentedImageFolder",
        "SPLIT_OFFSETS",
        "DataLoader",
        "torch",
    )
    missing = [name for name in required if not hasattr(pipeline, name)]
    if missing:
        raise RuntimeError("ESD pipeline source drifted; missing: " + ", ".join(missing))

    def exact_runtime_seed(
        self: Any,
        source_index: int,
        variant_index: int,
        view_offset: int = 0,
        attempt: int = 0,
    ) -> int:
        return derive_augmentation_seed(
            seed=int(self.seed),
            split_offset=int(pipeline.SPLIT_OFFSETS[self.split_name]),
            source_index=int(source_index),
            variant_index=int(variant_index),
            epoch=int(self.current_epoch),
            view_offset=int(view_offset),
            attempt=int(attempt),
        )

    def exact_loader(
        dataset: Any,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int | None,
        shuffle: bool,
        sampler: Any = None,
    ) -> Any:
        kwargs: dict[str, object] = {
            "dataset": dataset,
            "batch_size": int(batch_size),
            "shuffle": bool(shuffle) if sampler is None else False,
            "sampler": sampler,
            "num_workers": int(num_workers),
            "pin_memory": bool(pipeline.torch.cuda.is_available()),
            # Exact replay intentionally recreates workers after interruption.
            "persistent_workers": False,
        }
        if int(num_workers) > 0 and prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
        return pipeline.DataLoader(**kwargs)

    pipeline.DeterministicAugmentedImageFolder._runtime_augmentation_seed = exact_runtime_seed
    pipeline.make_loader = exact_loader
    pipeline.ESD_RESTART_ADDRESSABLE_DATA_SCHEMA = SCHEMA
    _INSTALLED = True


__all__ = ["SCHEMA", "derive_augmentation_seed", "install"]
