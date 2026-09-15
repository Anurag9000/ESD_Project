from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "training_control" / "esd_deterministic_data_v1.py"
spec = importlib.util.spec_from_file_location("esd_deterministic_data_v1", MODULE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_augmentation_seed_is_coordinate_addressable() -> None:
    kwargs = dict(
        seed=42,
        split_offset=0,
        source_index=17,
        variant_index=3,
        epoch=9,
        view_offset=1,
        attempt=0,
    )
    first = module.derive_augmentation_seed(**kwargs)
    second = module.derive_augmentation_seed(**kwargs)
    assert first == second


def test_augmentation_seed_changes_for_scientific_coordinates() -> None:
    base = dict(
        seed=42,
        split_offset=0,
        source_index=17,
        variant_index=3,
        epoch=9,
        view_offset=1,
        attempt=0,
    )
    reference = module.derive_augmentation_seed(**base)
    for key in ("source_index", "variant_index", "epoch", "view_offset", "attempt"):
        changed = dict(base)
        changed[key] += 1
        assert module.derive_augmentation_seed(**changed) != reference


def test_negative_seed_coordinate_is_rejected() -> None:
    try:
        module.derive_augmentation_seed(
            seed=42,
            split_offset=0,
            source_index=-1,
            variant_index=0,
            epoch=0,
        )
    except ValueError:
        return
    raise AssertionError("negative source index should be rejected")
