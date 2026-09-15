#!/usr/bin/env python3
"""Native-sampler -> shared-broker iterator bridge for ESD cohort training.

This module is deliberately small and science-free.  It does not calculate a loss,
construct an optimizer, change early stopping, or own a stage transition.  Its only
job is to turn the exact deterministic sampler coordinates already used by the ESD
trainer into the one-item iterator accepted by ``train_supcon_steps`` and
``train_classifier_steps``.

For one physical lane/stage:

* every model keeps its own native sampler object;
* the requested ``epoch`` and ``batch_index`` are translated to the same sampler
  start coordinate using the lane's uniform physical ``batch_size``;
* each model independently derives the next index slice, so a sampler/config drift
  remains detectable rather than being hidden by one leader model;
* ``SharedBatchBroker.acquire`` compares those coordinates and materializes the
  dataset/collated/device view only for the first compatible consumer;
* every sibling receives the exact same cached tensor storage;
* the returned iterator contains exactly one batch, allowing the repository-owned
  step engines to run with ``step_limit=1`` and preserve SAM/AdamW/AMP/scheduler
  semantics unchanged.

The shared cohort cursor must be committed only after all active consumers have
completed their one native step and the broker reports a clean batch boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Iterator, Sequence

SCHEMA = "esd-native-shared-batch-iterator/v1"


@dataclass(frozen=True, slots=True)
class PhysicalBatchCoordinate:
    epoch: int
    batch_index: int
    batch_size: int
    start_index: int
    indices: tuple[int, ...]

    @property
    def is_full(self) -> bool:
        return len(self.indices) == self.batch_size


def derive_batch_coordinate(
    sampler: Any,
    *,
    epoch: int,
    batch_index: int,
    batch_size: int,
) -> PhysicalBatchCoordinate:
    """Derive one exact native sampler slice without materializing dataset values."""
    epoch = int(epoch)
    batch_index = int(batch_index)
    batch_size = int(batch_size)
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    set_epoch = getattr(sampler, "set_epoch", None)
    set_start_index = getattr(sampler, "set_start_index", None)
    if not callable(set_epoch) or not callable(set_start_index):
        raise TypeError(
            "ESD cohort sampler must expose set_epoch() and set_start_index(); "
            "fall back to the native non-cohort job for unsupported samplers"
        )
    start_index = batch_index * batch_size
    set_epoch(epoch)
    set_start_index(start_index)
    indices = tuple(int(value) for value in islice(iter(sampler), batch_size))
    if not indices:
        raise StopIteration
    if len(set(indices)) != len(indices):
        # Replacement sampling is not part of the currently admitted ESD samplers.
        # Fail closed rather than silently changing sample multiplicity semantics.
        raise RuntimeError(
            "ESD native sampler emitted duplicate indices inside one physical batch; "
            "this stream is not certified for shared-batch execution"
        )
    return PhysicalBatchCoordinate(
        epoch=epoch,
        batch_index=batch_index,
        batch_size=batch_size,
        start_index=start_index,
        indices=indices,
    )


class NativeSharedBatchIterator(Iterator[Any]):
    """Exactly-one-batch iterator backed by the completion-aware shared broker."""

    def __init__(
        self,
        *,
        broker: Any,
        consumer_id: str,
        dataset: Any,
        sampler: Any,
        epoch: int,
        batch_index: int,
        batch_size: int,
        collate_fn: Callable[[Sequence[Any]], Any] | None = None,
        require_full_batch: bool = False,
    ) -> None:
        if not str(consumer_id):
            raise ValueError("consumer_id must be non-empty")
        self.broker = broker
        self.consumer_id = str(consumer_id)
        self.dataset = dataset
        self.sampler = sampler
        self.collate_fn = collate_fn
        self.require_full_batch = bool(require_full_batch)
        self.coordinate = derive_batch_coordinate(
            sampler,
            epoch=int(epoch),
            batch_index=int(batch_index),
            batch_size=int(batch_size),
        )
        if self.require_full_batch and not self.coordinate.is_full:
            raise RuntimeError(
                f"{self.consumer_id}: final partial batch has {len(self.coordinate.indices)} "
                f"samples but lane requires {self.coordinate.batch_size}; split/fallback "
                "rather than changing effective batch semantics"
            )
        self._yielded = False

    def __iter__(self) -> "NativeSharedBatchIterator":
        return self

    def __next__(self) -> Any:
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.broker.acquire(
            consumer_id=self.consumer_id,
            epoch=self.coordinate.epoch,
            batch_index=self.coordinate.batch_index,
            indices=self.coordinate.indices,
            dataset=self.dataset,
            collate_fn=self.collate_fn,
        )


def one_native_shared_step(
    *,
    native_step: Callable[..., Any],
    broker: Any,
    consumer_id: str,
    dataset: Any,
    sampler: Any,
    epoch: int,
    batch_index: int,
    batch_size: int,
    native_kwargs: dict[str, Any],
    collate_fn: Callable[[Sequence[Any]], Any] | None = None,
    require_full_batch: bool = False,
) -> tuple[Any, PhysicalBatchCoordinate]:
    """Call one repository-owned native step on one broker-shared physical batch.

    ``native_kwargs`` must contain the model/loss/optimizer/scaler/scheduler and
    stage metadata expected by the live ESD step function.  This wrapper injects
    only ``batch_iterator`` and ``step_limit=1`` and refuses conflicting values.
    """
    if "batch_iterator" in native_kwargs or "step_limit" in native_kwargs:
        raise ValueError("native_kwargs must not override batch_iterator or step_limit")
    iterator = NativeSharedBatchIterator(
        broker=broker,
        consumer_id=consumer_id,
        dataset=dataset,
        sampler=sampler,
        epoch=epoch,
        batch_index=batch_index,
        batch_size=batch_size,
        collate_fn=collate_fn,
        require_full_batch=require_full_batch,
    )
    result = native_step(
        batch_iterator=iterator,
        step_limit=1,
        **native_kwargs,
    )
    return result, iterator.coordinate


def assert_lane_batch_boundary_clean(broker: Any) -> None:
    """Require all active models to have consumed the physical batch before commit."""
    checker = getattr(broker, "assert_batch_boundary_clean", None)
    if not callable(checker):
        raise TypeError("ESD cohort broker must expose assert_batch_boundary_clean()")
    checker()


__all__ = [
    "SCHEMA",
    "PhysicalBatchCoordinate",
    "NativeSharedBatchIterator",
    "derive_batch_coordinate",
    "one_native_shared_step",
    "assert_lane_batch_boundary_clean",
]
