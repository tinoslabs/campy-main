"""Weight initialisation strategies."""
from __future__ import annotations

import numpy as np

from .config import default_dtype


def he_normal(shape: tuple[int, ...], fan_in: int, rng: np.random.Generator) -> np.ndarray:
    """Kaiming/He initialisation — the right choice ahead of ReLU."""
    std = np.sqrt(2.0 / max(fan_in, 1))
    return (rng.standard_normal(shape) * std).astype(default_dtype())


def glorot_uniform(shape: tuple[int, ...], fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier/Glorot initialisation — for tanh/sigmoid/linear outputs."""
    limit = np.sqrt(6.0 / max(fan_in + fan_out, 1))
    return rng.uniform(-limit, limit, size=shape).astype(default_dtype())


def zeros(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=default_dtype())


def ones(shape: tuple[int, ...]) -> np.ndarray:
    return np.ones(shape, dtype=default_dtype())


def make_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else 1337)
