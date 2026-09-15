"""Framework-wide numeric configuration.

Production inference and training run in ``float32`` — half the memory and
noticeably faster GEMMs.  The gradient-verification suite flips the whole
framework to ``float64`` so central-difference checks are not swamped by
round-off, which is the only way to prove the hand-derived backward passes are
actually correct.
"""
from __future__ import annotations

import contextlib

import numpy as np

_DEFAULT_DTYPE = np.float32


def default_dtype() -> type:
    return _DEFAULT_DTYPE


def set_default_dtype(dtype) -> None:
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = np.dtype(dtype).type


@contextlib.contextmanager
def precision(dtype):
    """Temporarily switch the framework's working precision."""
    previous = default_dtype()
    set_default_dtype(dtype)
    try:
        yield
    finally:
        set_default_dtype(previous)


def as_array(x) -> np.ndarray:
    """Coerce to the framework's working dtype without needless copies."""
    return np.asarray(x, dtype=_DEFAULT_DTYPE)
