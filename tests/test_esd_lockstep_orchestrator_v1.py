from __future__ import annotations

import threading
import unittest

from training_control.esd_lockstep_orchestrator_v1 import (
    InspectableLimitedBatches,
    LockstepCoordinator,
    make_native_step_proxy,
)


class FakeSampler:
    def __init__(self, values, *, epoch=0, start_index=0):
        self.values = list(values)
        self.epoch = int(epoch)
        self.start_index = int(start_index)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_start_index(self, start_index):
        self.start_index = int(start_index)

    def __iter__(self):
        offset = self.epoch * 1000
        return iter([value + offset for value in self.values[self.start_index :]])


class FakeDataset:
    def __getitem__(self, index):
        return f"sample-{index}"

    def __len__(self):
        return 10000


class FakeLoader:
    def __init__(self, *, sampler, batch_size=2):
        self.dataset = FakeDataset()
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.collate_fn = tuple

    def __iter__(self):
        raise AssertionError("cohort proxy must not materialize the native loader independently")


class FakeBroker:
    def __init__(self):
        self.registrations = {}
        self.active = set()
        self.cache = {}
        self.seen = {}
        self.acquire_log = []
        self.clean_checks = 0

    def register_consumer(self, *, consumer_id, stream_key, view_key, batch_size, active=True):
        self.registrations[consumer_id] = (stream_key, view_key, int(batch_size))
        if active:
            self.active.add(consumer_id)

    def transition(self, consumer_id, *, stream_key, view_key, batch_size):
        self.register_consumer(
            consumer_id=consumer_id,
            stream_key=stream_key,
            view_key=view_key,
            batch_size=batch_size,
            active=True,
        )

    def deactivate(self, consumer_id):
        self.active.discard(consumer_id)

    def acquire(self, *, consumer_id, epoch, batch_index, indices, dataset, collate_fn=None):
        stream_key, view_key, _batch_size = self.registrations[consumer_id]
        key = (stream_key, view_key, int(epoch), int(batch_index), tuple(indices))
        expected = {
            cid
            for cid in self.active
            if self.registrations.get(cid, ())[:2] == (stream_key, view_key)
        }
        if key not in self.cache:
            samples = [dataset[index] for index in range(len(indices))]
            self.cache[key] = ("shared-batch", key, tuple(samples))
            self.seen[key] = set()
        self.seen[key].add(consumer_id)
        value = self.cache[key]
        self.acquire_log.append((consumer_id, key, id(value)))
        if expected <= self.seen[key]:
            del self.cache[key]
            del self.seen[key]
        return value

    def assert_batch_boundary_clean(self):
        self.clean_checks += 1
        if self.cache:
            raise AssertionError(f"broker cache remained open: {self.cache}")


class LockstepOrchestratorTests(unittest.TestCase):
    def _run_two(self, starts=(0, 0)):
        broker = FakeBroker()
        coordinator = LockstepCoordinator(
            consumer_ids=["a", "b"], lane_key="lane", broker=broker
        )
        results = {}
        errors = []

        def native_step(*, batch_iterator, step_limit, model_name):
            self.assertEqual(step_limit, 1)
            return model_name, next(batch_iterator)

        def worker(consumer_id, start_index):
            try:
                loader = FakeLoader(
                    sampler=FakeSampler(range(20), epoch=1, start_index=start_index),
                    batch_size=2,
                )
                proxy = make_native_step_proxy(
                    coordinator=coordinator,
                    consumer_id=consumer_id,
                    stage="classifier",
                    native_step=native_step,
                )
                results[consumer_id] = proxy(
                    batch_iterator=InspectableLimitedBatches(loader, 0),
                    step_limit=100,
                    model_name=consumer_id,
                )
            except BaseException as exc:  # pragma: no cover - assertion aid
                errors.append(exc)
            finally:
                coordinator.finish(consumer_id)

        threads = [
            threading.Thread(target=worker, args=("a", starts[0]), daemon=True),
            threading.Thread(target=worker, args=("b", starts[1]), daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "cohort worker thread deadlocked")
        if errors:
            raise errors[0]
        return results, broker

    def test_compatible_consumers_receive_same_cached_batch_object(self):
        results, broker = self._run_two((0, 0))
        self.assertIs(results["a"][1], results["b"][1])
        self.assertEqual(len({entry[2] for entry in broker.acquire_log}), 1)
        self.assertEqual(broker.clean_checks, 1)

    def test_divergent_native_cursors_split_dynamic_groups(self):
        results, broker = self._run_two((0, 2))
        self.assertIsNot(results["a"][1], results["b"][1])
        self.assertEqual(len({entry[2] for entry in broker.acquire_log}), 2)
        self.assertEqual(broker.clean_checks, 1)

    def test_native_requested_step_limit_is_forced_to_one(self):
        results, _broker = self._run_two((0, 0))
        self.assertEqual(results["a"][0], "a")
        self.assertEqual(results["b"][0], "b")


if __name__ == "__main__":
    unittest.main()
