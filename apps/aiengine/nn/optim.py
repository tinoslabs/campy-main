"""Optimisers and learning-rate schedules."""
from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, lr: float = 1e-3, clip_norm: float | None = 5.0):
        self.lr = float(lr)
        self.base_lr = float(lr)
        self.clip_norm = clip_norm
        self.state: dict[int, dict[str, np.ndarray]] = {}
        self.step_count = 0

    def _clip(self, grads: list[np.ndarray]) -> float:
        """Global-norm gradient clipping — keeps rare-class batches from
        blowing the model up."""
        if not self.clip_norm:
            return 1.0
        total = np.sqrt(sum(float(np.sum(g * g)) for g in grads)) if grads else 0.0
        if total > self.clip_norm and total > 0:
            return self.clip_norm / total
        return 1.0

    def step(self, layers) -> None:
        raise NotImplementedError

    def zero_grad(self, layers) -> None:
        for layer in layers:
            layer.grads = {}

    def describe(self) -> str:
        return f"{self.__class__.__name__}(lr={self.lr:g})"


class SGD(Optimizer):
    """Stochastic gradient descent with Nesterov momentum and weight decay."""

    def __init__(self, lr=1e-2, momentum=0.9, weight_decay=0.0, nesterov=True, clip_norm=5.0):
        super().__init__(lr, clip_norm)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)

    def step(self, layers):
        self.step_count += 1
        all_grads = [g for layer in layers for g in layer.grads.values()]
        scale = self._clip(all_grads)

        for layer in layers:
            for key, grad in layer.grads.items():
                param = layer.params[key]
                grad = grad * scale
                if self.weight_decay and key in ("W", "gamma"):
                    grad = grad + self.weight_decay * param
                slot = self.state.setdefault(id(param), {})
                velocity = slot.get("v")
                if velocity is None:
                    velocity = np.zeros_like(param)
                velocity = self.momentum * velocity - self.lr * grad
                slot["v"] = velocity
                if self.nesterov:
                    param += self.momentum * velocity - self.lr * grad
                else:
                    param += velocity


class Adam(Optimizer):
    """Adam with bias correction."""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, clip_norm=5.0):
        super().__init__(lr, clip_norm)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

    def step(self, layers):
        self.step_count += 1
        t = self.step_count
        all_grads = [g for layer in layers for g in layer.grads.values()]
        scale = self._clip(all_grads)

        bias1 = 1.0 - self.beta1 ** t
        bias2 = 1.0 - self.beta2 ** t

        for layer in layers:
            for key, grad in layer.grads.items():
                param = layer.params[key]
                grad = grad * scale
                if self.weight_decay and key in ("W", "gamma"):
                    grad = grad + self.weight_decay * param
                slot = self.state.setdefault(id(param), {})
                m = slot.get("m")
                v = slot.get("v")
                if m is None:
                    m = np.zeros_like(param)
                    v = np.zeros_like(param)
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * (grad * grad)
                slot["m"], slot["v"] = m, v
                param -= self.lr * (m / bias1) / (np.sqrt(v / bias2) + self.eps)


class AdamW(Adam):
    """Adam with *decoupled* weight decay — the better regulariser for our
    small convolutional backbones."""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2, clip_norm=5.0):
        super().__init__(lr, beta1, beta2, eps, weight_decay=0.0, clip_norm=clip_norm)
        self.decoupled_decay = float(weight_decay)

    def step(self, layers):
        for layer in layers:
            for key in list(layer.grads):
                if key in ("W", "gamma") and self.decoupled_decay:
                    layer.params[key] -= self.lr * self.decoupled_decay * layer.params[key]
        super().step(layers)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
class Schedule:
    def __call__(self, epoch: int, base_lr: float) -> float:
        raise NotImplementedError


class ConstantLR(Schedule):
    def __call__(self, epoch, base_lr):
        return base_lr


class StepDecay(Schedule):
    """Cut the learning rate by ``factor`` every ``every`` epochs."""

    def __init__(self, every: int = 10, factor: float = 0.5):
        self.every = max(int(every), 1)
        self.factor = float(factor)

    def __call__(self, epoch, base_lr):
        return base_lr * (self.factor ** (epoch // self.every))


class CosineDecay(Schedule):
    """Cosine annealing with an optional linear warm-up."""

    def __init__(self, total_epochs: int, warmup_epochs: int = 0, min_lr: float = 1e-6):
        self.total_epochs = max(int(total_epochs), 1)
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.min_lr = float(min_lr)

    def __call__(self, epoch, base_lr):
        if self.warmup_epochs and epoch < self.warmup_epochs:
            return base_lr * (epoch + 1) / self.warmup_epochs
        span = max(self.total_epochs - self.warmup_epochs, 1)
        progress = min((epoch - self.warmup_epochs) / span, 1.0)
        return self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))


OPTIMIZER_REGISTRY = {"sgd": SGD, "adam": Adam, "adamw": AdamW}
SCHEDULE_REGISTRY = {"constant": ConstantLR, "step": StepDecay, "cosine": CosineDecay}


def build_optimizer(name: str, **kwargs) -> Optimizer:
    cls = OPTIMIZER_REGISTRY.get(str(name).lower())
    if cls is None:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {sorted(OPTIMIZER_REGISTRY)}")
    return cls(**kwargs)


def build_schedule(name: str, **kwargs) -> Schedule:
    cls = SCHEDULE_REGISTRY.get(str(name).lower(), ConstantLR)
    try:
        return cls(**kwargs)
    except TypeError:
        return ConstantLR()
