from __future__ import annotations

import unittest

import torch

from scripts.train_phase0_mim import (
    PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE,
    _render_phase0_preview,
    phase0_scalar_is_finite,
    phase0_tensor_is_finite,
)


class Phase0PreviewTests(unittest.TestCase):
    def test_patch_normalized_preview_unnormalizes_and_blends(self) -> None:
        originals = torch.tensor(
            [
                [
                    [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.2, 0.2, 0.8, 0.8], [0.2, 0.2, 0.8, 0.8]],
                    [[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.2, 0.8, 0.2, 0.8], [0.2, 0.8, 0.2, 0.8]],
                    [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.8, 0.2, 0.2, 0.8], [0.8, 0.2, 0.2, 0.8]],
                ]
            ],
            dtype=torch.float32,
        )
        pixel_mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        pixel_mask[:, :, :2, :2] = 1.0
        reconstructed = torch.zeros_like(originals)

        preview = _render_phase0_preview(
            originals=originals,
            pixel_mask=pixel_mask,
            reconstructed=reconstructed,
            patch_size=2,
            loss_mode=PHASE0_LOSS_MODE_PATCH_NORMALIZED_MSE,
        )

        self.assertEqual(tuple(preview.shape), (1, 3, 4, 4))
        self.assertTrue(torch.isfinite(preview).all())
        self.assertGreater(float(preview.mean().item()), 0.0)
        self.assertLessEqual(float(preview.max().item()), 1.0)

    def test_finite_guards(self) -> None:
        self.assertTrue(phase0_tensor_is_finite(torch.ones(2, 3)))
        self.assertFalse(phase0_tensor_is_finite(torch.tensor([1.0, float("nan")])))
        self.assertTrue(phase0_scalar_is_finite(1.0))
        self.assertFalse(phase0_scalar_is_finite(float("inf")))


if __name__ == "__main__":
    unittest.main()
