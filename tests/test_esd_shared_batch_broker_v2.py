from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "training_control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

import esd_shared_batch_broker_v2 as broker_module


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def synchronize(*args, **kwargs) -> None:
        return None

    @staticmethod
    def empty_cache() -> None:
        return None


class _FakeTorch:
    cuda = _FakeCuda()


class _Dataset:
    def __init__(self) -> None:
        self.fetches: list[int] = []

    def __getitem__(self, index: int):
        self.fetches.append(int(index))
        return {"index": int(index)}


def _install_fakes() -> None:
    broker_module._torch = lambda: _FakeTorch
    broker_module.move_to_device = lambda value, device: value


def _collate(samples):
    return {"indices": tuple(item["index"] for item in samples)}


def test_shared_view_is_materialized_once_and_identity_shared() -> None:
    _install_fakes()
    dataset = _Dataset()
    broker = broker_module.SharedBatchBroker(device="cpu")
    broker.register_consumer(
        consumer_id="a", stream_key="train", view_key="ce", batch_size=2
    )
    broker.register_consumer(
        consumer_id="b", stream_key="train", view_key="ce", batch_size=2
    )
    first = broker.acquire(
        consumer_id="a",
        epoch=1,
        batch_index=0,
        indices=[3, 7],
        dataset=dataset,
        collate_fn=_collate,
    )
    second = broker.acquire(
        consumer_id="b",
        epoch=1,
        batch_index=0,
        indices=[3, 7],
        dataset=dataset,
        collate_fn=_collate,
    )
    assert first is second
    assert first == {"indices": (3, 7)}
    assert dataset.fetches == [3, 7]
    broker.assert_batch_boundary_clean()


def test_deactivate_retires_pending_consumer_without_cache_leak() -> None:
    _install_fakes()
    dataset = _Dataset()
    broker = broker_module.SharedBatchBroker(device="cpu")
    broker.register_consumer(
        consumer_id="active", stream_key="train", view_key="ce", batch_size=2
    )
    broker.register_consumer(
        consumer_id="stops", stream_key="train", view_key="ce", batch_size=2
    )
    broker.acquire(
        consumer_id="active",
        epoch=2,
        batch_index=4,
        indices=[1, 2],
        dataset=dataset,
        collate_fn=_collate,
    )
    pending = broker.pending()
    assert pending["cached_views"] == 1
    assert pending["open_coordinates"] == 1
    broker.deactivate("stops")
    broker.assert_batch_boundary_clean()
    assert broker.pending()["active_consumers"] == 1


def test_sampler_divergence_fails_closed() -> None:
    _install_fakes()
    dataset = _Dataset()
    broker = broker_module.SharedBatchBroker(device="cpu")
    broker.register_consumer(
        consumer_id="a", stream_key="train", view_key="ce", batch_size=2
    )
    broker.register_consumer(
        consumer_id="b", stream_key="train", view_key="ce", batch_size=2
    )
    broker.acquire(
        consumer_id="a",
        epoch=1,
        batch_index=0,
        indices=[0, 1],
        dataset=dataset,
        collate_fn=_collate,
    )
    try:
        broker.acquire(
            consumer_id="b",
            epoch=1,
            batch_index=0,
            indices=[0, 2],
            dataset=dataset,
            collate_fn=_collate,
        )
    except RuntimeError as exc:
        assert "sampler divergence" in str(exc)
    else:
        raise AssertionError("sampler divergence was accepted")


def test_transition_is_allowed_only_between_clean_batch_boundaries() -> None:
    _install_fakes()
    dataset = _Dataset()
    broker = broker_module.SharedBatchBroker(device="cpu")
    broker.register_consumer(
        consumer_id="a", stream_key="train", view_key="supcon", batch_size=2
    )
    broker.register_consumer(
        consumer_id="b", stream_key="train", view_key="supcon", batch_size=2
    )
    broker.acquire(
        consumer_id="a",
        epoch=1,
        batch_index=0,
        indices=[4, 5],
        dataset=dataset,
        collate_fn=_collate,
    )
    # Transitioning a consumer that has already consumed the open batch retires it
    # from future batches but may not invalidate sibling b's snapshot obligation.
    broker.transition(
        "a", stream_key="train", view_key="ce", batch_size=2
    )
    pending = broker.pending()
    assert pending["cached_views"] == 1
    try:
        broker.acquire(
            consumer_id="a",
            epoch=1,
            batch_index=0,
            indices=[4, 5],
            dataset=dataset,
            collate_fn=_collate,
        )
    except RuntimeError as exc:
        assert "joined a view after the physical batch snapshot" in str(exc) or "sampler divergence" in str(exc)
    else:
        raise AssertionError("stage-transitioned consumer joined an already-open batch")
    broker.acquire(
        consumer_id="b",
        epoch=1,
        batch_index=0,
        indices=[4, 5],
        dataset=dataset,
        collate_fn=_collate,
    )
    broker.assert_batch_boundary_clean()


if __name__ == "__main__":
    names = sorted(
        name for name in globals()
        if name.startswith("test_") and callable(globals()[name])
    )
    for name in names:
        globals()[name]()
    print(f"ESD shared batch broker v2 tests: PASS ({len(names)})")
