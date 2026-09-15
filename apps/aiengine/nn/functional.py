"""Low-level array kernels shared by the layers (im2col / col2im and friends)."""
from __future__ import annotations

import numpy as np

from .config import default_dtype


def conv_output_size(size: int, kernel: int, stride: int, pad: int) -> int:
    return (size + 2 * pad - kernel) // stride + 1


def im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int):
    """Flatten sliding local blocks so convolution becomes one matrix multiply.

    Returns ``(cols, out_h, out_w)`` where ``cols`` has shape
    ``(N * out_h * out_w, C * kh * kw)``.
    """
    n, c, h, w = x.shape
    out_h = conv_output_size(h, kh, stride, pad)
    out_w = conv_output_size(w, kw, stride, pad)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid conv geometry: input {h}x{w}, kernel {kh}x{kw}, stride {stride}, pad {pad}")

    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    col = np.empty((n, c, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            col[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]

    cols = col.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
    return cols, out_h, out_w


def col2im(
    cols: np.ndarray,
    x_shape: tuple[int, int, int, int],
    kh: int,
    kw: int,
    stride: int,
    pad: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Inverse of :func:`im2col`, accumulating overlapping gradients."""
    n, c, h, w = x_shape
    col = cols.reshape(n, out_h, out_w, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros(
        (n, c, h + 2 * pad + stride - 1, w + 2 * pad + stride - 1), dtype=cols.dtype
    )
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            img[:, :, i:i_max:stride, j:j_max:stride] += col[:, :, i, j, :, :]
    return img[:, :, pad : h + pad, pad : w + pad]


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(np.sum(exp, axis=axis, keepdims=True), 1e-12, None)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Branch-free stable sigmoid."""
    x = np.asarray(x)
    out = np.empty_like(x, dtype=x.dtype if x.dtype.kind == "f" else default_dtype())
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    return shifted - np.log(np.clip(np.sum(np.exp(shifted), axis=axis, keepdims=True), 1e-12, None))


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).ravel()
    out = np.zeros((labels.size, num_classes), dtype=default_dtype())
    valid = (labels >= 0) & (labels < num_classes)
    out[np.arange(labels.size)[valid], labels[valid]] = 1.0
    return out
