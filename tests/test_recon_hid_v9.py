from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models_v9 import (
    ReConHIDV9,
    ReConHIDV9Config,
    load_v9_checkpoint,
    save_v9_checkpoint,
)
from src.training_v9 import FeatureNormalizerV2, ReConHIDV9Loss


class ReConHIDV9Test(unittest.TestCase):
    def test_forward_shapes_and_finite_values(self):
        model = ReConHIDV9()
        sequence = torch.randn(4, 50, 12)
        context = torch.randn(4, 12)

        output = model(sequence, context)

        self.assertEqual(
            tuple(output.sequence_reconstruction.shape),
            (4, 50, 12),
        )
        self.assertEqual(
            tuple(output.context_reconstruction.shape),
            (4, 12),
        )
        self.assertEqual(tuple(output.classifier_logit.shape), (4,))
        self.assertEqual(tuple(output.fused_latent.shape), (4, 32))
        self.assertIsNone(output.prototype_distance)
        self.assertTrue(
            torch.isfinite(output.sequence_reconstruction).all()
        )

    def test_loss_backward(self):
        model = ReConHIDV9()
        loss_fn = ReConHIDV9Loss()
        sequence = torch.randn(3, 50, 12)
        context = torch.randn(3, 12)
        labels = torch.tensor([0.0, 1.0, 0.0])

        output = model(sequence, context)
        losses = loss_fn(
            output,
            sequence_target=sequence,
            context_target=context,
            binary_labels=labels,
        )
        losses.total.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(g is not None for g in gradients))
        self.assertTrue(torch.isfinite(losses.total))

    def test_prototype_distance(self):
        model = ReConHIDV9()
        sequence = torch.randn(5, 50, 12)
        context = torch.randn(5, 12)
        first = model(sequence, context)
        model.set_normal_prototype(first.fused_latent.detach())
        second = model(sequence, context)
        self.assertIsNotNone(second.prototype_distance)
        self.assertEqual(tuple(second.prototype_distance.shape), (5,))
        self.assertTrue((second.prototype_distance >= 0).all())

    def test_checkpoint_roundtrip(self):
        model = ReConHIDV9(ReConHIDV9Config())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_v9_checkpoint(path, model=model, epoch=3)
            loaded, payload = load_v9_checkpoint(path)
            self.assertIsInstance(loaded, ReConHIDV9)
            self.assertEqual(payload["epoch"], 3)

    def test_normalizer_roundtrip_and_binary_preservation(self):
        rng = np.random.default_rng(123)
        sequence = rng.normal(size=(8, 50, 12)).astype(np.float32)
        context = rng.normal(size=(8, 12)).astype(np.float32)

        # Binary feature positions.
        binary_indices = [5, 6, 7, 8, 9, 10, 11]
        sequence[..., binary_indices] = rng.integers(
            0, 2, size=(8, 50, len(binary_indices))
        )

        normalizer = FeatureNormalizerV2()
        seq_n, ctx_n = normalizer.fit_transform(sequence, context)
        restored = FeatureNormalizerV2.from_state_dict(
            normalizer.state_dict()
        )
        seq_n2, ctx_n2 = restored.transform(sequence, context)

        self.assertTrue(np.allclose(seq_n, seq_n2))
        self.assertTrue(np.allclose(ctx_n, ctx_n2))
        self.assertTrue(
            np.array_equal(
                seq_n[..., binary_indices],
                sequence[..., binary_indices],
            )
        )


if __name__ == "__main__":
    unittest.main()
