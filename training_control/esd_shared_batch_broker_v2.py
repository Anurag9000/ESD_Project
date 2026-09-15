#!/usr/bin/env python3
"""Completion-aware shared physical batch broker for ESD cohorts.

This supersedes v1's static consumer count with explicit consumer identities.
That distinction is required for semantic early stopping and multi-stage ESD
training: a model may complete, or leave SupCon and enter CE, while sibling
models remain active.  A cached GPU view must then release the departed consumer
rather than wait forever for an acquisition that will never happen.

The broker owns only batch/view residency.  Native ESD model adapters continue
to own optimizer/scaler/scheduler/RNG/checkpoint/early-stopping state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import threading
from typing import Any, Callable, Mapping, Sequence

from esd_shared_batch_broker_v1 import move_to_device

SCHEMA = "esd-shared-batch-broker/v2"


def _torch():
    import torch
    return torch


def _default_collate(samples: Sequence[Any]) -> Any:
    from torch.utils.data import default_collate
    return default_collate(list(samples))


def _coordinate_digest(indices: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in indices], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ConsumerRegistration:
    consumer_id: str
    stream_key: str
    view_key: str
    batch_size: int


@dataclass(slots=True)
class _CachedView:
    coordinate_digest: str
    value: Any
    expected: set[str] = field(default_factory=set)
    seen: set[str] = field(default_factory=set)

    def complete(self) -> bool:
        return self.expected <= self.seen


class SharedBatchBroker:
    def __init__(self, *, device: str) -> None:
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError(f"unsupported ESD cohort device: {device!r}")
        torch = _torch()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA cohort requested but torch.cuda is unavailable")
        self.device = str(device)
        self._lock = threading.RLock()
        self._registrations: dict[str, ConsumerRegistration] = {}
        self._active: set[str] = set()
        self._batch_coordinates: dict[tuple[str, int, int], str] = {}
        self._cache: dict[tuple[str, str, int, int], _CachedView] = {}
        self._closed = False

    def register_consumer(
        self,
        *,
        consumer_id: str,
        stream_key: str,
        view_key: str,
        batch_size: int,
        active: bool = True,
    ) -> ConsumerRegistration:
        row = ConsumerRegistration(
            str(consumer_id), str(stream_key), str(view_key), int(batch_size)
        )
        if not row.consumer_id or not row.stream_key or not row.view_key:
            raise ValueError("consumer_id, stream_key and view_key must be non-empty")
        if row.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with self._lock:
            self._ensure_open()
            previous = self._registrations.get(row.consumer_id)
            if previous is not None and previous != row:
                raise RuntimeError(
                    f"ESD cohort consumer registration drift for {row.consumer_id}: "
                    f"{previous!r} != {row!r}"
                )
            self._registrations[row.consumer_id] = row
            if active:
                self._active.add(row.consumer_id)
        return row

    def activate(self, consumer_id: str) -> None:
        with self._lock:
            self._ensure_open()
            if consumer_id not in self._registrations:
                raise KeyError(f"unknown ESD cohort consumer {consumer_id!r}")
            self._active.add(consumer_id)

    def deactivate(self, consumer_id: str) -> None:
        """Retire a completed/stage-transitioned consumer from pending views."""
        with self._lock:
            self._ensure_open()
            self._active.discard(consumer_id)
            for key, cached in list(self._cache.items()):
                if consumer_id in cached.expected and consumer_id not in cached.seen:
                    cached.expected.discard(consumer_id)
                if cached.complete():
                    del self._cache[key]
                    self._release_coordinate_if_unused(key[0], key[2], key[3])

    def transition(
        self,
        consumer_id: str,
        *,
        stream_key: str,
        view_key: str,
        batch_size: int,
    ) -> ConsumerRegistration:
        """Atomically leave the old stage/view and enter a new compatible one."""
        with self._lock:
            self.deactivate(consumer_id)
            self._registrations.pop(consumer_id, None)
            return self.register_consumer(
                consumer_id=consumer_id,
                stream_key=stream_key,
                view_key=view_key,
                batch_size=batch_size,
                active=True,
            )

    def _expected_consumers(self, stream_key: str, view_key: str) -> set[str]:
        return {
            consumer_id
            for consumer_id in self._active
            if (row := self._registrations.get(consumer_id)) is not None
            and row.stream_key == stream_key
            and row.view_key == view_key
        }

    def acquire(
        self,
        *,
        consumer_id: str,
        epoch: int,
        batch_index: int,
        indices: Sequence[int],
        dataset: Any,
        collate_fn: Callable[[Sequence[Any]], Any] | None = None,
    ) -> Any:
        indices = tuple(int(value) for value in indices)
        if not indices:
            raise ValueError("cannot acquire an empty ESD cohort batch")
        digest = _coordinate_digest(indices)
        with self._lock:
            self._ensure_open()
            registration = self._registrations.get(consumer_id)
            if registration is None:
                raise KeyError(f"unknown ESD cohort consumer {consumer_id!r}")
            if consumer_id not in self._active:
                raise RuntimeError(f"inactive ESD cohort consumer {consumer_id!r} attempted batch acquisition")
            if len(indices) > registration.batch_size:
                raise RuntimeError(
                    f"{consumer_id}: physical batch has {len(indices)} samples, exceeds "
                    f"uniform lane size {registration.batch_size}"
                )

            coordinate_key = (registration.stream_key, int(epoch), int(batch_index))
            previous_digest = self._batch_coordinates.get(coordinate_key)
            if previous_digest is None:
                self._batch_coordinates[coordinate_key] = digest
            elif previous_digest != digest:
                raise RuntimeError(
                    "ESD cohort sampler divergence: supposedly compatible consumers selected "
                    f"different indices at stream={registration.stream_key!r}, epoch={epoch}, "
                    f"batch={batch_index}"
                )

            cache_key = (
                registration.stream_key,
                registration.view_key,
                int(epoch),
                int(batch_index),
            )
            cached = self._cache.get(cache_key)
            if cached is None:
                expected = self._expected_consumers(
                    registration.stream_key, registration.view_key
                )
                if consumer_id not in expected:
                    raise RuntimeError(f"{consumer_id}: missing from its active view snapshot")
                samples = [dataset[index] for index in indices]
                collated = (collate_fn or _default_collate)(samples)
                cached = _CachedView(
                    coordinate_digest=digest,
                    value=move_to_device(collated, self.device),
                    expected=expected,
                )
                self._cache[cache_key] = cached
            elif cached.coordinate_digest != digest:
                raise RuntimeError(
                    f"{consumer_id}: cached view coordinate digest drift"
                )
            if consumer_id not in cached.expected:
                raise RuntimeError(
                    f"{consumer_id}: joined a view after the physical batch snapshot; "
                    "stage transitions must occur between committed batches"
                )
            if consumer_id in cached.seen:
                raise RuntimeError(
                    f"{consumer_id}: attempted to consume the same physical batch twice"
                )
            cached.seen.add(consumer_id)
            value = cached.value
            if cached.complete():
                del self._cache[cache_key]
                self._release_coordinate_if_unused(
                    registration.stream_key, int(epoch), int(batch_index)
                )
            return value

    def _release_coordinate_if_unused(
        self, stream_key: str, epoch: int, batch_index: int
    ) -> None:
        if any(
            cache_stream == stream_key
            and cache_epoch == int(epoch)
            and cache_batch == int(batch_index)
            for cache_stream, _view, cache_epoch, cache_batch in self._cache
        ):
            return
        self._batch_coordinates.pop((stream_key, int(epoch), int(batch_index)), None)

    def pending(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": SCHEMA,
                "registered_consumers": len(self._registrations),
                "active_consumers": len(self._active),
                "cached_views": len(self._cache),
                "open_coordinates": len(self._batch_coordinates),
                "device": self.device,
            }

    def assert_batch_boundary_clean(self) -> None:
        with self._lock:
            if self._cache or self._batch_coordinates:
                raise RuntimeError(
                    "ESD shared-batch broker reached a transaction boundary with "
                    f"pending state: {self.pending()}"
                )

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
            self._batch_coordinates.clear()
            self._active.clear()
            self._closed = True
        torch = _torch()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)
            torch.cuda.empty_cache()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ESD shared-batch broker is closed")


__all__ = ["SCHEMA", "ConsumerRegistration", "SharedBatchBroker"]
