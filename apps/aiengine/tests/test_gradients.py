"""Numerical verification of every hand-derived backward pass.

This is the safety net under Campy AI's claim to own its model stack: if a
gradient is wrong, training silently produces a worse model rather than
failing, so we prove each one against central differences in float64.
"""
from __future__ import annotations

import numpy as np
from django.test import SimpleTestCase

from apps.aiengine.nn import layers as L
from apps.aiengine.nn import losses as LS
from apps.aiengine.nn.config import precision
from apps.aiengine.nn.model import Sequential

EPS = 1e-6
ATOL = 1e-7
RTOL = 1e-5
SAMPLES = 40


def _close(analytic: np.ndarray, numeric: np.ndarray) -> tuple[bool, float]:
    diff = np.abs(analytic - numeric)
    tolerance = ATOL + RTOL * np.abs(numeric)
    return bool(np.all(diff <= tolerance)), float(np.max(diff))


class LayerGradientTests(SimpleTestCase):
    """Every layer's ``backward`` must match central differences."""

    def _check(self, layer, x, rng):
        """Assumes the caller is already inside ``precision(np.float64)``.

        Construction must happen in the same context: a layer built in float32
        cannot represent a 1e-6 perturbation, and the check would compare
        rounding noise instead of gradients.
        """
        if True:
            x = np.asarray(x, dtype=np.float64)
            out = layer.forward(x.copy(), training=True)
            dout = rng.standard_normal(out.shape)
            dx = layer.backward(dout)

            def objective():
                return float(np.sum(layer.forward(x.copy(), training=True) * dout))

            # --- gradient w.r.t. the input --------------------------------
            flat_idx = rng.choice(x.size, min(SAMPLES, x.size), replace=False)
            analytic, numeric = [], []
            xf = x.ravel()
            for i in flat_idx:
                original = xf[i]
                xf[i] = original + EPS
                plus = objective()
                xf[i] = original - EPS
                minus = objective()
                xf[i] = original
                numeric.append((plus - minus) / (2 * EPS))
                analytic.append(dx.ravel()[i])
            ok, worst = _close(np.array(analytic), np.array(numeric))
            self.assertTrue(ok, f"{layer.describe()} dx mismatch (max abs diff {worst:.3e})")

            # --- gradients w.r.t. every parameter -------------------------
            layer.forward(x.copy(), training=True)
            layer.backward(dout)
            for pname, param in layer.params.items():
                grad = layer.grads.get(pname)
                if grad is None:
                    continue
                pf = param.ravel()
                sel = rng.choice(pf.size, min(SAMPLES, pf.size), replace=False)
                analytic, numeric = [], []
                for i in sel:
                    original = pf[i]
                    pf[i] = original + EPS
                    plus = objective()
                    pf[i] = original - EPS
                    minus = objective()
                    pf[i] = original
                    numeric.append((plus - minus) / (2 * EPS))
                    analytic.append(grad.ravel()[i])
                ok, worst = _close(np.array(analytic), np.array(numeric))
                self.assertTrue(
                    ok, f"{layer.describe()} d{pname} mismatch (max abs diff {worst:.3e})"
                )
                layer.forward(x.copy(), training=True)
                layer.backward(dout)

    def test_conv2d_variants(self):
        with precision(np.float64):
            rng = np.random.default_rng(7)
            x = rng.standard_normal((2, 3, 8, 8))
            for layer in (
                L.Conv2D(3, 4, 3, 1, "same", rng=rng),
                L.Conv2D(3, 5, 3, 2, "valid", rng=rng),
                L.Conv2D(3, 6, 1, 1, "same", use_bias=False, rng=rng),
                L.Conv2D(3, 4, 5, 1, "same", rng=rng),
            ):
                with self.subTest(layer=layer.describe()):
                    self._check(layer, x, rng)

    def test_depthwise_conv2d(self):
        with precision(np.float64):
            rng = np.random.default_rng(11)
            x = rng.standard_normal((2, 3, 8, 8))
            for layer in (
                L.DepthwiseConv2D(3, 3, 1, "same", rng=rng),
                L.DepthwiseConv2D(3, 3, 2, "same", rng=rng),
            ):
                with self.subTest(layer=layer.describe()):
                    self._check(layer, x, rng)

    def test_pooling_and_shape_layers(self):
        with precision(np.float64):
            rng = np.random.default_rng(13)
            x = rng.standard_normal((2, 3, 8, 8))
            for layer in (L.MaxPool2D(2), L.GlobalAvgPool2D(), L.Flatten()):
                with self.subTest(layer=layer.describe()):
                    self._check(layer, x, rng)

    def test_activations(self):
        with precision(np.float64):
            rng = np.random.default_rng(17)
            x = rng.standard_normal((2, 3, 6, 6))
            for layer in (L.ReLU(), L.LeakyReLU(0.1), L.Sigmoid()):
                with self.subTest(layer=layer.describe()):
                    self._check(layer, x, rng)
            x2 = rng.standard_normal((6, 5))
            for layer in (L.Softmax(), L.L2Normalize()):
                with self.subTest(layer=layer.describe()):
                    self._check(layer, x2, rng)

    def test_batchnorm_in_training_mode(self):
        with precision(np.float64):
            rng = np.random.default_rng(19)
            self._check(L.BatchNorm2D(3), rng.standard_normal((4, 3, 6, 6)), rng)
            self._check(L.BatchNorm2D(5), rng.standard_normal((6, 5)), rng)

    def test_dense(self):
        with precision(np.float64):
            rng = np.random.default_rng(23)
            x = rng.standard_normal((6, 5))
            self._check(L.Dense(5, 4, rng=rng), x, rng)
            self._check(L.Dense(5, 3, use_bias=False, rng=rng), x, rng)


class LossGradientTests(SimpleTestCase):
    """Every loss must return the true ``dL/dinput``."""

    def _check(self, loss, predictions, targets, rng):
        if True:
            predictions = np.asarray(predictions, dtype=np.float64)
            _, grad = loss(predictions, targets)
            flat = predictions.ravel()
            sel = rng.choice(flat.size, min(SAMPLES, flat.size), replace=False)
            analytic, numeric = [], []
            for i in sel:
                original = flat[i]
                flat[i] = original + EPS
                plus, _ = loss(predictions, targets)
                flat[i] = original - EPS
                minus, _ = loss(predictions, targets)
                flat[i] = original
                numeric.append((plus - minus) / (2 * EPS))
                analytic.append(grad.ravel()[i])
            ok, worst = _close(np.array(analytic), np.array(numeric))
            self.assertTrue(ok, f"{loss.name} gradient mismatch (max abs diff {worst:.3e})")

    def test_classification_losses(self):
        with precision(np.float64):
            rng = np.random.default_rng(29)
            logits = rng.standard_normal((8, 4))
            labels = rng.integers(0, 4, 8)
            self._check(LS.CrossEntropy(), logits, labels, rng)
            self._check(LS.CrossEntropy(label_smoothing=0.1), logits, labels, rng)

    def test_binary_and_focal_losses(self):
        with precision(np.float64):
            rng = np.random.default_rng(31)
            logits = rng.standard_normal((8, 3))
            targets = rng.integers(0, 2, (8, 3)).astype(np.float64)
            self._check(LS.BinaryCrossEntropy(), logits, targets, rng)
            self._check(LS.BinaryCrossEntropy(pos_weight=3.0), logits, targets, rng)
            self._check(LS.FocalLoss(gamma=2.0, alpha=0.25), logits, targets, rng)

    def test_regression_losses(self):
        with precision(np.float64):
            rng = np.random.default_rng(37)
            predictions = rng.standard_normal((8, 2))
            targets = rng.standard_normal((8, 2))
            self._check(LS.MeanSquaredError(), predictions, targets, rng)
            self._check(LS.HuberLoss(), predictions, targets, rng)

    def test_triplet_loss(self):
        with precision(np.float64):
            rng = np.random.default_rng(41)
            embeddings = rng.standard_normal((9, 6))
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
            labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
            self._check(LS.TripletLoss(margin=0.5), embeddings, labels, rng)


class EndToEndGradientTests(SimpleTestCase):
    def test_full_network_backpropagates_correctly(self):
        """A realistic stack: conv → BN → ReLU → pool → separable → GAP → dense."""
        rng = np.random.default_rng(43)
        with precision(np.float64):
            net = Sequential(
                [
                    L.Conv2D(3, 8, 3, 1, "same", rng=rng),
                    L.BatchNorm2D(8),
                    L.ReLU(),
                    L.MaxPool2D(2),
                    L.DepthwiseConv2D(8, 3, 1, "same", rng=rng),
                    L.Conv2D(8, 12, 1, 1, "same", rng=rng),
                    L.BatchNorm2D(12),
                    L.ReLU(),
                    L.GlobalAvgPool2D(),
                    L.Dense(12, 4, rng=rng),
                ]
            )
            x = rng.standard_normal((4, 3, 16, 16))
            y = rng.integers(0, 4, 4)
            loss = LS.CrossEntropy()

            value, dout = loss(net.forward(x, training=True), y)
            net.backward(dout)
            self.assertGreater(value, 0)

            first_conv = net.layers[0]
            grad = first_conv.grads["W"]
            flat = first_conv.params["W"].ravel()
            sel = rng.choice(flat.size, 25, replace=False)
            analytic, numeric = [], []
            for i in sel:
                original = flat[i]
                flat[i] = original + EPS
                plus, _ = loss(net.forward(x, training=True), y)
                flat[i] = original - EPS
                minus, _ = loss(net.forward(x, training=True), y)
                flat[i] = original
                numeric.append((plus - minus) / (2 * EPS))
                analytic.append(grad.ravel()[i])
            ok, worst = _close(np.array(analytic), np.array(numeric))
            self.assertTrue(ok, f"end-to-end gradient mismatch (max abs diff {worst:.3e})")


class LearningTests(SimpleTestCase):
    """Correct gradients are necessary but not sufficient — the model must learn."""

    def test_network_learns_a_separable_pattern(self):
        rng = np.random.default_rng(5)
        # Two classes: bright top half vs bright bottom half.
        n = 96
        x = rng.standard_normal((n, 1, 12, 12)).astype(np.float32) * 0.25
        y = rng.integers(0, 2, n)
        for i, label in enumerate(y):
            if label == 0:
                x[i, 0, :6, :] += 1.6
            else:
                x[i, 0, 6:, :] += 1.6

        net = Sequential(
            [
                L.Conv2D(1, 8, 3, 1, "same", rng=rng),
                L.BatchNorm2D(8),
                L.ReLU(),
                L.MaxPool2D(2),
                L.Conv2D(8, 16, 3, 1, "same", rng=rng),
                L.ReLU(),
                L.GlobalAvgPool2D(),
                L.Dense(16, 2, rng=rng),
            ]
        )
        history = net.fit(
            x, y, loss="cross_entropy", optimizer="adam", epochs=14, batch_size=16, seed=3
        )
        self.assertGreaterEqual(history[-1]["train_accuracy"], 0.9, f"history={history[-1]}")
        self.assertLess(history[-1]["loss"], history[0]["loss"])

    def test_triplet_training_separates_identities(self):
        """The face-embedding objective must pull identities apart."""
        rng = np.random.default_rng(9)
        classes, per_class, dim = 4, 8, 16
        centres = rng.standard_normal((classes, dim)) * 2.0
        x, y = [], []
        for c in range(classes):
            for _ in range(per_class):
                x.append(centres[c] + rng.standard_normal(dim) * 0.35)
                y.append(c)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y)

        net = Sequential(
            [L.Dense(dim, 24, rng=rng), L.ReLU(), L.Dense(24, 8, rng=rng), L.L2Normalize()]
        )
        before = net.evaluate(x, y, loss="triplet", task="embedding")["rank1"]
        net.fit(
            x, y, loss="triplet", optimizer="adam", epochs=25, batch_size=16,
            task="embedding", seed=3,
        )
        after = net.evaluate(x, y, loss="triplet", task="embedding")["rank1"]
        self.assertGreaterEqual(after, 0.95, f"rank-1 went {before} → {after}")


class SerializationTests(SimpleTestCase):
    def test_checkpoint_roundtrip_preserves_predictions(self):
        import tempfile
        from pathlib import Path

        rng = np.random.default_rng(2)
        net = Sequential(
            [
                L.Conv2D(3, 6, 3, 1, "same", rng=rng),
                L.BatchNorm2D(6),
                L.ReLU(),
                L.GlobalAvgPool2D(),
                L.Dense(6, 3, rng=rng),
            ],
            name="roundtrip",
            meta={"classes": ["a", "b", "c"]},
        )
        x = rng.standard_normal((3, 3, 8, 8)).astype(np.float32)
        # Give BatchNorm non-trivial running statistics so the buffers matter.
        net.forward(x, training=True)
        expected = net.predict(x)

        with tempfile.TemporaryDirectory() as tmp:
            path = net.save(Path(tmp) / "model.npz")
            restored = Sequential.load(path)

        np.testing.assert_allclose(restored.predict(x), expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(restored.meta["classes"], ["a", "b", "c"])
        self.assertEqual(restored.param_count, net.param_count)
