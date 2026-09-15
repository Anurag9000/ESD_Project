from __future__ import annotations

import unittest

from training_control.esd_native_batch_iterator_v1 import (
    NativeSharedBatchIterator,
    assert_lane_batch_boundary_clean,
    derive_batch_coordinate,
    one_native_shared_step,
)


class FakeSampler:
    def __init__(self, values):
        self.values = list(values)
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_start_index(self, start):
        self.start = int(start)

    def __iter__(self):
        offset = self.epoch * 1000
        return iter([value + offset for value in self.values[self.start :]])


class FakeBroker:
    def __init__(self):
        self.calls = []
        self.clean_checks = 0
        self.shared = {}

    def acquire(self, *, consumer_id, epoch, batch_index, indices, dataset, collate_fn=None):
        key = (int(epoch), int(batch_index), tuple(indices))
        self.calls.append((str(consumer_id), key))
        if key not in self.shared:
            samples = [dataset[index] for index in range(len(indices))]
            self.shared[key] = ("shared", key, tuple(samples))
        return self.shared[key]

    def assert_batch_boundary_clean(self):
        self.clean_checks += 1


class FakeDataset:
    def __getitem__(self, index):
        return f"sample-{index}"


class UnsupportedSampler:
    def __iter__(self):
        return iter([1, 2, 3])


class NativeBatchIteratorTests(unittest.TestCase):
    def test_coordinate_uses_epoch_batch_size_and_start_cursor(self):
        sampler = FakeSampler(range(20))
        coordinate = derive_batch_coordinate(sampler, epoch=3, batch_index=2, batch_size=4)
        self.assertEqual(coordinate.start_index, 8)
        self.assertEqual(coordinate.indices, (3008, 3009, 3010, 3011))
        self.assertTrue(coordinate.is_full)
        self.assertEqual(sampler.epoch, 3)
        self.assertEqual(sampler.start, 8)

    def test_partial_final_batch_is_preserved_unless_lane_requires_full_batch(self):
        coordinate = derive_batch_coordinate(FakeSampler(range(10)), epoch=0, batch_index=2, batch_size=4)
        self.assertEqual(coordinate.indices, (8, 9))
        self.assertFalse(coordinate.is_full)
        with self.assertRaisesRegex(RuntimeError, "final partial batch"):
            NativeSharedBatchIterator(
                broker=FakeBroker(), consumer_id="m1", dataset=FakeDataset(),
                sampler=FakeSampler(range(10)), epoch=0, batch_index=2, batch_size=4,
                require_full_batch=True,
            )

    def test_unsupported_sampler_fails_closed(self):
        with self.assertRaisesRegex(TypeError, "set_epoch"):
            derive_batch_coordinate(UnsupportedSampler(), epoch=0, batch_index=0, batch_size=2)

    def test_duplicate_replacement_batch_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate indices"):
            derive_batch_coordinate(FakeSampler([0, 0, 1, 2]), epoch=0, batch_index=0, batch_size=4)

    def test_iterator_yields_exactly_one_broker_batch(self):
        broker = FakeBroker()
        iterator = NativeSharedBatchIterator(
            broker=broker, consumer_id="model-a", dataset=FakeDataset(),
            sampler=FakeSampler(range(8)), epoch=1, batch_index=1, batch_size=4,
        )
        first = next(iterator)
        self.assertEqual(first[0], "shared")
        self.assertEqual(broker.calls, [("model-a", (1, 1, (1004, 1005, 1006, 1007)))])
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_two_consumers_receive_same_shared_object(self):
        broker = FakeBroker()
        kwargs = dict(broker=broker, dataset=FakeDataset(), epoch=2, batch_index=0, batch_size=3)
        one = NativeSharedBatchIterator(consumer_id="a", sampler=FakeSampler(range(9)), **kwargs)
        two = NativeSharedBatchIterator(consumer_id="b", sampler=FakeSampler(range(9)), **kwargs)
        value_a = next(one)
        value_b = next(two)
        self.assertEqual(one.coordinate.indices, (2000, 2001, 2002))
        self.assertEqual(two.coordinate.indices, one.coordinate.indices)
        self.assertIs(value_a, value_b)

    def test_one_native_shared_step_injects_iterator_and_one_step(self):
        seen = {}

        def native_step(*, batch_iterator, step_limit, sentinel):
            seen["step_limit"] = step_limit
            seen["batch"] = next(batch_iterator)
            seen["sentinel"] = sentinel
            return "native-result"

        result, coordinate = one_native_shared_step(
            native_step=native_step, broker=FakeBroker(), consumer_id="m",
            dataset=FakeDataset(), sampler=FakeSampler(range(6)), epoch=0,
            batch_index=1, batch_size=3, native_kwargs={"sentinel": 7},
        )
        self.assertEqual(result, "native-result")
        self.assertEqual(seen["step_limit"], 1)
        self.assertEqual(seen["sentinel"], 7)
        self.assertEqual(coordinate.indices, (3, 4, 5))

    def test_native_kwargs_cannot_override_shared_cursor_controls(self):
        with self.assertRaisesRegex(ValueError, "must not override"):
            one_native_shared_step(
                native_step=lambda **_: None, broker=FakeBroker(), consumer_id="m",
                dataset=FakeDataset(), sampler=FakeSampler(range(4)), epoch=0,
                batch_index=0, batch_size=2, native_kwargs={"step_limit": 99},
            )

    def test_boundary_cleanliness_delegates_to_broker(self):
        broker = FakeBroker()
        assert_lane_batch_boundary_clean(broker)
        self.assertEqual(broker.clean_checks, 1)


if __name__ == "__main__":
    unittest.main()
