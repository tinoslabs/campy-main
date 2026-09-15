"""Loss functions. Each returns ``(loss, dlogits)`` from :meth:`__call__`."""
from __future__ import annotations

import numpy as np

from .config import as_array, default_dtype
from .functional import log_softmax, one_hot, sigmoid, softmax


class Loss:
    name = "loss"

    def __call__(self, predictions: np.ndarray, targets) -> tuple[float, np.ndarray]:
        raise NotImplementedError


class CrossEntropy(Loss):
    """Softmax + categorical cross-entropy fused for numerical stability.

    Expects raw logits ``(N, C)`` and integer class labels ``(N,)``.
    """

    name = "cross_entropy"

    def __init__(self, label_smoothing: float = 0.0, class_weights: np.ndarray | None = None):
        self.label_smoothing = float(label_smoothing)
        self.class_weights = class_weights

    def __call__(self, logits, targets):
        logits = as_array(logits)
        n, num_classes = logits.shape
        targets = np.asarray(targets)
        y = targets if targets.ndim == 2 else one_hot(targets, num_classes)

        if self.label_smoothing > 0:
            y = y * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes

        log_probs = log_softmax(logits, axis=-1)
        per_sample = -np.sum(y * log_probs, axis=-1)

        if self.class_weights is not None:
            weights = as_array(self.class_weights)
            sample_weights = (y * weights).sum(axis=-1)
            loss = float(np.sum(per_sample * sample_weights) / max(np.sum(sample_weights), 1e-9))
            grad = (softmax(logits) - y) * sample_weights[:, None] / max(np.sum(sample_weights), 1e-9)
            return loss, grad.astype(default_dtype())

        loss = float(np.mean(per_sample))
        grad = (softmax(logits) - y) / n
        return loss, grad.astype(default_dtype())


class BinaryCrossEntropy(Loss):
    """Sigmoid + binary cross-entropy on raw logits."""

    name = "binary_cross_entropy"

    def __init__(self, pos_weight: float = 1.0):
        self.pos_weight = float(pos_weight)

    def __call__(self, logits, targets):
        logits = as_array(logits)
        y = as_array(targets).reshape(logits.shape)
        probs = sigmoid(logits)
        eps = 1e-7
        probs_c = np.clip(probs, eps, 1 - eps)
        loss = float(
            np.mean(-(self.pos_weight * y * np.log(probs_c) + (1 - y) * np.log(1 - probs_c)))
        )
        # dL/dlogit for the weighted form, mean-reduced over the batch.
        weight = self.pos_weight * y + (1 - y)
        grad = (weight * (probs - y)) / y.size
        return loss, grad.astype(default_dtype())


class FocalLoss(Loss):
    """Focal loss — keeps rare classes (fire, weapons, theft) from being drowned
    out by the overwhelming "nothing happening" majority."""

    name = "focal"

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def __call__(self, logits, targets):
        logits = as_array(logits)
        y = as_array(targets).reshape(logits.shape)
        p = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)
        pt = np.where(y > 0.5, p, 1 - p)
        alpha_t = np.where(y > 0.5, self.alpha, 1 - self.alpha)
        loss = float(np.mean(-alpha_t * ((1 - pt) ** self.gamma) * np.log(pt)))

        # L  = -a (1-pt)^g log(pt)
        # dL/dpt = a (1-pt)^(g-1) [ g log(pt) - (1-pt)/pt ]
        # dpt/dz = (2y-1) p (1-p)
        dl_dpt = alpha_t * ((1 - pt) ** (self.gamma - 1)) * (
            self.gamma * np.log(pt) - (1 - pt) / pt
        )
        dpt_dz = (2 * y - 1) * p * (1 - p)
        grad = dl_dpt * dpt_dz
        return loss, (grad / y.size).astype(default_dtype())


class MeanSquaredError(Loss):
    """Used for crowd-density regression and bounding-box refinement."""

    name = "mse"

    def __call__(self, predictions, targets):
        predictions = as_array(predictions)
        y = as_array(targets).reshape(predictions.shape)
        diff = predictions - y
        loss = float(np.mean(diff * diff))
        grad = (2.0 * diff / diff.size).astype(default_dtype())
        return loss, grad


class HuberLoss(Loss):
    """Robust regression loss — less sensitive to annotation outliers."""

    name = "huber"

    def __init__(self, delta: float = 1.0):
        self.delta = float(delta)

    def __call__(self, predictions, targets):
        predictions = as_array(predictions)
        y = as_array(targets).reshape(predictions.shape)
        diff = predictions - y
        small = np.abs(diff) <= self.delta
        loss = float(
            np.mean(np.where(small, 0.5 * diff**2, self.delta * (np.abs(diff) - 0.5 * self.delta)))
        )
        grad = np.where(small, diff, self.delta * np.sign(diff)) / diff.size
        return loss, grad.astype(default_dtype())


class TripletLoss(Loss):
    """Batch-hard triplet loss for face / person embeddings.

    Input is a batch of L2-normalised embeddings ``(N, D)`` plus identity labels
    ``(N,)``.  For every anchor we mine the hardest positive and hardest
    negative *inside the batch* — this is what makes the embedding space
    discriminative enough to tell two employees apart.
    """

    name = "triplet"

    def __init__(self, margin: float = 0.3):
        self.margin = float(margin)

    def __call__(self, embeddings, labels):
        emb = as_array(embeddings)
        labels = np.asarray(labels).ravel()
        n = emb.shape[0]

        # Squared euclidean distance matrix.
        sq = np.sum(emb * emb, axis=1)
        dist = np.maximum(sq[:, None] - 2.0 * (emb @ emb.T) + sq[None, :], 0.0)

        same = labels[:, None] == labels[None, :]
        eye = np.eye(n, dtype=bool)
        positive_mask = same & ~eye
        negative_mask = ~same

        grad = np.zeros_like(emb)
        losses: list[float] = []
        for i in range(n):
            pos_idx = np.where(positive_mask[i])[0]
            neg_idx = np.where(negative_mask[i])[0]
            if pos_idx.size == 0 or neg_idx.size == 0:
                continue
            p = pos_idx[np.argmax(dist[i, pos_idx])]   # hardest (furthest) positive
            q = neg_idx[np.argmin(dist[i, neg_idx])]   # hardest (closest) negative
            value = dist[i, p] - dist[i, q] + self.margin
            if value <= 0:
                continue
            losses.append(float(value))
            # d/da (|a-p|^2 - |a-n|^2) = 2(n - p);  d/dp = 2(p - a);  d/dn = 2(a - n)
            grad[i] += 2.0 * (emb[q] - emb[p])
            grad[p] += 2.0 * (emb[p] - emb[i])
            grad[q] += 2.0 * (emb[i] - emb[q])

        active = max(len(losses), 1)
        return float(np.sum(losses) / active), (grad / active).astype(default_dtype())


LOSS_REGISTRY = {
    "cross_entropy": CrossEntropy,
    "binary_cross_entropy": BinaryCrossEntropy,
    "focal": FocalLoss,
    "mse": MeanSquaredError,
    "huber": HuberLoss,
    "triplet": TripletLoss,
}


def build_loss(name: str, **kwargs) -> Loss:
    cls = LOSS_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(LOSS_REGISTRY)}")
    return cls(**kwargs)
