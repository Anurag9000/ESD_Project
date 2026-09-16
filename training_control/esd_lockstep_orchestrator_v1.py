#!/usr/bin/env python3
"""Cooperative lockstep orchestration for ESD native batch-step training.

This module supplies the missing concurrency/control seam between the already
source-proven ESD native one-step functions and the completion-aware shared-batch
broker.  It intentionally owns no model science.

Each resident model runs its existing ``run_experiment`` lifecycle in a worker
thread.  When that lifecycle reaches ``train_supcon_steps`` or
``train_classifier_steps``, an adapter submits exactly one native step here.  The
coordinator waits until every currently-live resident has either submitted its
next native step or completed, groups submissions by their *actual next native
coordinate*, and executes each compatible group against one broker-cached batch.
Models whose early stopping / phase progression has diverged simply land in a
different dynamic group; they are never forced back into false lockstep.

The coordinator also serializes the code between native steps.  This is required
because Python/NumPy/PyTorch RNGs are process-global.  A caller may additionally
provide RNG capture/restore callbacks so stochastic model trajectories remain
model-local even though model steps share one process and one GPU.

``before_round`` and ``after_round`` hooks are transaction boundaries for a real
worker.  They are where the worker backs up native ``step_last.pt`` files, writes
an inflight cohort journal, commits a clean shared cursor, and acknowledges OPF
checkpoint-pressure requests.  A crash while a round is inflight can therefore
restore every model to the same pre-round transaction instead of resuming a
partially advanced cohort.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import threading
from typing import Any, Callable, Mapping

from esd_native_batch_iterator_v1 import NativeSharedBatchIterator

SCHEMA = "esd-lockstep-orchestrator/v1"


class CohortExecutionError(RuntimeError):
    pass


@dataclass(slots=True)
class StepRequest:
    consumer_id: str
    stage: str
    native_step: Callable[..., Any]
    native_arguments: dict[str, Any]
    loader: Any
    epoch: int
    batch_index: int
    batch_size: int
    view_key: str
    stream_signature: str
    rng_state: Any = None


@dataclass(slots=True)
class _Result:
    generation: int
    value: Any = None
    error: BaseException | None = None


def _sampler(loader: Any) -> Any:
    sampler = getattr(loader, "sampler", None)
    if sampler is None:
        batch_sampler = getattr(loader, "batch_sampler", None)
        sampler = getattr(batch_sampler, "sampler", None)
    if sampler is None:
        raise TypeError("ESD cohort loader does not expose a native sampler")
    if not hasattr(sampler, "epoch") or not hasattr(sampler, "start_index"):
        raise TypeError(
            "ESD cohort sampler must expose epoch and start_index; unsupported "
            "samplers must remain on the native non-cohort path"
        )
    return sampler


def describe_next_coordinate(loader: Any, *, stage: str, lane_key: str) -> dict[str, Any]:
    sampler = _sampler(loader)
    batch_size = int(getattr(loader, "batch_size", 0) or 0)
    if batch_size <= 0:
        raise ValueError("ESD cohort loader must expose a positive batch_size")
    epoch = int(sampler.epoch)
    start_index = int(sampler.start_index)
    if start_index < 0 or start_index % batch_size:
        raise RuntimeError(
            f"ESD native cursor {start_index} is not aligned to physical batch size {batch_size}"
        )
    batch_index = start_index // batch_size
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        raise TypeError("ESD cohort loader does not expose dataset")
    collate = getattr(loader, "collate_fn", None)
    view_payload = {
        "stage": str(stage),
        "dataset_type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "collate": (
            f"{getattr(collate, '__module__', type(collate).__module__)}."
            f"{getattr(collate, '__qualname__', type(collate).__qualname__)}"
        ),
    }
    view_key = hashlib.sha256(
        json.dumps(view_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    stream_payload = {
        "lane_key": str(lane_key),
        "stage": str(stage),
        "epoch": epoch,
        "batch_index": batch_index,
        "batch_size": batch_size,
        "view_key": view_key,
    }
    stream_signature = hashlib.sha256(
        json.dumps(stream_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **stream_payload,
        "stream_signature": stream_signature,
        "dataset": dataset,
        "sampler": sampler,
        "collate_fn": collate,
    }


class InspectableLimitedBatches:
    """Drop-in replacement for the native ``limited_batches`` generator.

    Standalone iteration is unchanged.  The cohort proxy additionally gets the
    original DataLoader through ``.loader`` without consuming/decode of a batch.
    """

    def __init__(self, loader: Any, max_batches: int) -> None:
        self.loader = loader
        self.max_batches = int(max_batches)
        self._iterator = iter(loader)
        self._count = 0

    def __iter__(self) -> "InspectableLimitedBatches":
        return self

    def __next__(self) -> Any:
        if self.max_batches > 0 and self._count >= self.max_batches:
            raise StopIteration
        value = next(self._iterator)
        self._count += 1
        return value


def inspectable_limited_batches(loader: Any, max_batches: int) -> InspectableLimitedBatches:
    return InspectableLimitedBatches(loader, max_batches)


class LockstepCoordinator:
    """Round barrier + dynamic compatibility grouping for one resident lane window."""

    def __init__(
        self,
        *,
        consumer_ids: list[str],
        lane_key: str,
        broker: Any,
        capture_rng: Callable[[], Any] | None = None,
        restore_rng: Callable[[Any], None] | None = None,
        before_round: Callable[[int, Mapping[str, StepRequest]], None] | None = None,
        after_round: Callable[[int, Mapping[str, StepRequest]], None] | None = None,
    ) -> None:
        ids = [str(value) for value in consumer_ids]
        if not ids or len(ids) != len(set(ids)) or any(not value for value in ids):
            raise ValueError("consumer_ids must be a non-empty unique list")
        self.lane_key = str(lane_key)
        if not self.lane_key:
            raise ValueError("lane_key must be non-empty")
        self.broker = broker
        self.capture_rng = capture_rng
        self.restore_rng = restore_rng
        self.before_round = before_round
        self.after_round = after_round
        self._condition = threading.Condition(threading.RLock())
        self._active = set(ids)
        self._arrivals: dict[str, StepRequest] = {}
        self._results: dict[str, _Result] = {}
        self._rng: dict[str, Any] = {}
        self._generation = 0
        self._between_owner: str | None = None
        self._registrations: dict[str, tuple[str, str, int]] = {}
        self._fatal: BaseException | None = None
        self._first_arrival: dict[str, threading.Event] = {
            value: threading.Event() for value in ids
        }

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    def first_arrival_event(self, consumer_id: str) -> threading.Event:
        return self._first_arrival[str(consumer_id)]

    def _release_between_turn_locked(self, consumer_id: str) -> None:
        if self._between_owner == consumer_id:
            self._between_owner = None
            self._condition.notify_all()

    def finish(self, consumer_id: str) -> None:
        consumer_id = str(consumer_id)
        with self._condition:
            self._release_between_turn_locked(consumer_id)
            self._active.discard(consumer_id)
            self._arrivals.pop(consumer_id, None)
            deactivate = getattr(self.broker, "deactivate", None)
            if callable(deactivate):
                deactivate(consumer_id)
            self._maybe_dispatch_locked()
            self._condition.notify_all()

    def abort(self, error: BaseException) -> None:
        with self._condition:
            if self._fatal is None:
                self._fatal = error
            self._condition.notify_all()

    def submit_native_step(
        self,
        *,
        consumer_id: str,
        stage: str,
        native_step: Callable[..., Any],
        native_arguments: Mapping[str, Any],
        batch_iterator: Any,
    ) -> Any:
        consumer_id = str(consumer_id)
        if not isinstance(batch_iterator, InspectableLimitedBatches):
            raise TypeError(
                "ESD physical cohort requires InspectableLimitedBatches; the worker "
                "must install inspectable_limited_batches before run_experiment"
            )
        coordinate = describe_next_coordinate(
            batch_iterator.loader, stage=str(stage), lane_key=self.lane_key
        )
        request = StepRequest(
            consumer_id=consumer_id,
            stage=str(stage),
            native_step=native_step,
            native_arguments=dict(native_arguments),
            loader=batch_iterator.loader,
            epoch=int(coordinate["epoch"]),
            batch_index=int(coordinate["batch_index"]),
            batch_size=int(coordinate["batch_size"]),
            view_key=str(coordinate["view_key"]),
            stream_signature=str(coordinate["stream_signature"]),
        )
        with self._condition:
            if consumer_id not in self._active:
                raise CohortExecutionError(f"inactive ESD consumer {consumer_id!r}")
            self._release_between_turn_locked(consumer_id)
            if self._fatal is not None:
                raise CohortExecutionError("ESD cohort aborted") from self._fatal
            if consumer_id in self._arrivals:
                raise CohortExecutionError(
                    f"{consumer_id}: submitted two native steps in one cohort generation"
                )
            if consumer_id not in self._rng and self.capture_rng is not None:
                self._rng[consumer_id] = self.capture_rng()
            generation = self._generation
            self._arrivals[consumer_id] = request
            self._first_arrival[consumer_id].set()
            self._maybe_dispatch_locked()
            while True:
                if self._fatal is not None:
                    raise CohortExecutionError("ESD cohort aborted") from self._fatal
                result = self._results.get(consumer_id)
                if result is not None and result.generation == generation:
                    break
                self._condition.wait()

            # Serialize all code between native steps. A model keeps this turn
            # until it either submits its next native step or calls finish().
            while self._between_owner not in (None, consumer_id):
                if self._fatal is not None:
                    raise CohortExecutionError("ESD cohort aborted") from self._fatal
                self._condition.wait()
            self._between_owner = consumer_id
            self._results.pop(consumer_id, None)
            if result.error is not None:
                raise CohortExecutionError(
                    f"native ESD step failed for {consumer_id}"
                ) from result.error
            return result.value

    def _maybe_dispatch_locked(self) -> None:
        if self._fatal is not None:
            return
        if not self._active:
            self._condition.notify_all()
            return
        if set(self._arrivals) != self._active:
            return
        generation = self._generation
        arrivals = dict(self._arrivals)
        try:
            if self.before_round is not None:
                self.before_round(generation, arrivals)
            grouped: dict[str, list[StepRequest]] = {}
            for request in arrivals.values():
                grouped.setdefault(request.stream_signature, []).append(request)
            for signature in sorted(grouped):
                self._execute_group_locked(grouped[signature])
            checker = getattr(self.broker, "assert_batch_boundary_clean", None)
            if callable(checker):
                checker()
            if self.after_round is not None:
                self.after_round(generation, arrivals)
        except BaseException as exc:
            self._fatal = exc
            for consumer_id in arrivals:
                self._results[consumer_id] = _Result(generation, error=exc)
            self._condition.notify_all()
            return
        self._arrivals.clear()
        self._generation += 1
        self._condition.notify_all()

    def _execute_group_locked(self, requests: list[StepRequest]) -> None:
        if not requests:
            return
        leader = requests[0]
        stream_key = f"{self.lane_key}:{leader.stream_signature}"
        for request in requests:
            desired = (stream_key, request.view_key, request.batch_size)
            current = self._registrations.get(request.consumer_id)
            if current is None:
                self.broker.register_consumer(
                    consumer_id=request.consumer_id,
                    stream_key=stream_key,
                    view_key=request.view_key,
                    batch_size=request.batch_size,
                    active=True,
                )
            elif current != desired:
                self.broker.transition(
                    request.consumer_id,
                    stream_key=stream_key,
                    view_key=request.view_key,
                    batch_size=request.batch_size,
                )
            self._registrations[request.consumer_id] = desired

        for request in sorted(requests, key=lambda row: row.consumer_id):
            if self.restore_rng is not None and request.consumer_id in self._rng:
                self.restore_rng(self._rng[request.consumer_id])
            loader = request.loader
            sampler = _sampler(loader)
            iterator = NativeSharedBatchIterator(
                broker=self.broker,
                consumer_id=request.consumer_id,
                dataset=loader.dataset,
                sampler=sampler,
                epoch=request.epoch,
                batch_index=request.batch_index,
                batch_size=request.batch_size,
                collate_fn=getattr(loader, "collate_fn", None),
            )
            arguments = dict(request.native_arguments)
            arguments["batch_iterator"] = iterator
            arguments["step_limit"] = 1
            try:
                value = request.native_step(**arguments)
            except BaseException as exc:
                self._results[request.consumer_id] = _Result(
                    self._generation, error=exc
                )
                raise
            finally:
                if self.capture_rng is not None:
                    self._rng[request.consumer_id] = self.capture_rng()
            self._results[request.consumer_id] = _Result(
                self._generation, value=value
            )


def make_native_step_proxy(
    *,
    coordinator: LockstepCoordinator,
    consumer_id: str,
    stage: str,
    native_step: Callable[..., Any],
) -> Callable[..., Any]:
    """Return a signature-agnostic proxy that forces one native step per round."""
    signature = inspect.signature(native_step)

    def proxy(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        batch_iterator = bound.arguments.pop("batch_iterator")
        # The native outer run_experiment loop owns the requested step_limit and
        # checkpoint cadence. Returning after one step makes that outer loop save
        # its normal step checkpoint before this consumer can join the next round.
        bound.arguments.pop("step_limit")
        return coordinator.submit_native_step(
            consumer_id=consumer_id,
            stage=stage,
            native_step=native_step,
            native_arguments=bound.arguments,
            batch_iterator=batch_iterator,
        )

    proxy.__name__ = f"cohort_{getattr(native_step, '__name__', stage)}"
    proxy.__doc__ = getattr(native_step, "__doc__", None)
    return proxy


__all__ = [
    "SCHEMA",
    "CohortExecutionError",
    "StepRequest",
    "InspectableLimitedBatches",
    "inspectable_limited_batches",
    "describe_next_coordinate",
    "LockstepCoordinator",
    "make_native_step_proxy",
]
