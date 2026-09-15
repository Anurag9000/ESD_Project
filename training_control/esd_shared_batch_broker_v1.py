#!/usr/bin/env python3
"""Thread-safe shared physical batch broker for ESD cohort execution.

The broker is intentionally independent of model science.  Native ESD trainers
still own model construction, losses, optimizers, SAM, AMP, schedulers, phase
transitions, validation and semantic early stopping.  Cohort loader adapters use
this broker to make the expensive *data* side physical:

1. a stream registers one uniform physical batch size;
2. each model supplies the deterministic dataset indices selected by its native
   sampler for the next batch;
3. the first consumer materializes/collates a view exactly once;
4. that view is moved to the selected CPU/CUDA device exactly once;
5. every other compatible model receives the same Python object / tensor storage;
6. the cache entry is released after all registered consumers have acquired it.

The broker fails closed if supposedly compatible models disagree on batch
coordinates.  This is critical for ESD because balanced/weighted/shuffle streams
must never be merged merely because they originate from the same folder tree.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import threading
from typing import Any, Callable, Mapping, Sequence

SCHEMA = "esd-shared-batch-broker/v1"


def _torch():
    import torch
    return torch


def _default_collate(samples: Sequence[Any]) -> Any:
    from torch.utils.data import default_collate
    return default_collate(list(samples))


def _coordinate_digest(indices: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in indices], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def move_to_device(value: Any, device: str) -> Any:
    """Recursively place a collated view on one backend without duplicating it."""
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.startswith("cuda"))
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class ViewRegistration:
    stream_key: str
    view_key: str
    batch_size: int
    consumers: int


@dataclass(slots=True)
class _CachedView:
    coordinate_digest: str
    value: Any
    consumers_expected: int
    consumers_seen: int = 0


class SharedBatchBroker:
    def __init__(self, *, device: str) -> None:
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError(f"unsupported ESD cohort device: {device!r}")
        if device.startswith("cuda") and not _torch().cuda.is_available():
            raise RuntimeError("CUDA cohort requested but torch.cuda is unavailable")
        self.device = device
        self._lock = threading.RLock()
        self._registrations: dict[tuple[str, str], ViewRegistration] = {}
        self._batch_coordinates: dict[tuple[str, int, int], str] = {}
        self._cache: dict[tuple[str, str, int, int], _CachedView] = {}
        self._closed = False

    def register_view(
        self,
        *,
        stream_key: str,
        view_key: str,
        batch_size: int,
        consumers: int,
    ) -> ViewRegistration:
        if not stream_key or not view_key:
            raise ValueError("stream_key and view_key must be non-empty")
        if int(batch_size) <= 0 or int(consumers) <= 0:
            raise ValueError("batch_size and consumers must be positive")
        key = (str(stream_key), str(view_key))
        row = ViewRegistration(str(stream_key), str(view_key), int(batch_size), int(consumers))
        with self._lock:
            self._ensure_open()
            previous = self._registrations.get(key)
            if previous is not None and previous != row:
                raise RuntimeError(
                    f"ESD shared-view registration drift for {key}: {previous!r} != {row!r}"
                )
            self._registrations[key] = row
        return row

    def acquire(
        self,
        *,
        stream_key: str,
        view_key: str,
        epoch: int,
        batch_index: int,
        indices: Sequence[int],
        dataset: Any,
        collate_fn: Callable[[Sequence[Any]], Any] | None = None,
    ) -> Any:
        """Return one shared collated/device view for deterministic coordinates."""
        key = (str(stream_key), str(view_key))
        indices = tuple(int(value) for value in indices)
        if not indices:
            raise ValueError("cannot acquire an empty ESD cohort batch")
        digest = _coordinate_digest(indices)
        with self._lock:
            self._ensure_open()
            registration = self._registrations.get(key)
            if registration is None:
                raise RuntimeError(f"unregistered ESD shared view: {key}")
            if len(indices) > registration.batch_size:
                raise RuntimeError(
                    f"{key}: batch has {len(indices)} samples, exceeds uniform size {registration.batch_size}"
                )
            coordinate_key = (registration.stream_key, int(epoch), int(batch_index))
            previous_digest = self._batch_coordinates.get(coordinate_key)
            if previous_digest is None:
                self._batch_coordinates[coordinate_key] = digest
            elif previous_digest != digest:
                raise RuntimeError(
                    "ESD cohort sampler divergence: supposedly compatible models selected "
                    f"different indices at stream={registration.stream_key!r}, epoch={epoch}, "
                    f"batch={batch_index}"
                )

            cache_key = (registration.stream_key, registration.view_key, int(epoch), int(batch_index))
            cached = self._cache.get(cache_key)
            if cached is None:
                samples = [dataset[index] for index in indices]
                collated = (collate_fn or _default_collate)(samples)
                value = move_to_device(collated, self.device)
                cached = _CachedView(
                    coordinate_digest=digest,
                    value=value,
                    consumers_expected=registration.consumers,
                )
                self._cache[cache_key] = cached
            elif cached.coordinate_digest != digest:
                raise RuntimeError(f"{key}: cached view coordinate digest drift")

            cached.consumers_seen += 1
            if cached.consumers_seen > cached.consumers_expected:
                raise RuntimeError(
                    f"{key}: batch acquired by more than {cached.consumers_expected} registered consumers"
                )
            value = cached.value
            if cached.consumers_seen == cached.consumers_expected:
                del self._cache[cache_key]
                # A stream coordinate can be released only when no cached view for
                # that physical batch remains.  This supports CE and SupCon view
                # caches with different consumer counts over one raw coordinate.
                if not any(
                    cache_stream == registration.stream_key
                    and cache_epoch == int(epoch)
                    and cache_batch == int(batch_index)
                    for cache_stream, _view, cache_epoch, cache_batch in self._cache
                ):
                    self._batch_coordinates.pop(coordinate_key, None)
            return value

    def pending(self) -> dict[str, int]:
        with self._lock:
            return {
                "registered_views": len(self._registrations),
                "cached_views": len(self._cache),
                "open_coordinates": len(self._batch_coordinates),
            }

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
            self._batch_coordinates.clear()
            self._closed = True
        torch = _torch()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)
            torch.cuda.empty_cache()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ESD shared-batch broker is closed")


__all__ = [
    "SCHEMA",
    "SharedBatchBroker",
    "ViewRegistration",
    "move_to_device",
]
