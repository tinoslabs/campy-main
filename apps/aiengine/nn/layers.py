"""Neural-network layers implemented from scratch on NumPy.

Every layer follows the same tiny contract::

    y = layer.forward(x, training=True)
    dx = layer.backward(dy)

Learnable state lives in ``layer.params`` (name → array) and the matching
gradients in ``layer.grads``.  Non-learnable state that still has to be
serialised (BatchNorm running statistics) lives in ``layer.buffers``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import init as winit
from .config import as_array, default_dtype
from .functional import col2im, conv_output_size, im2col, sigmoid, softmax


class Layer:
    """Base class. Subclasses implement :meth:`forward` and :meth:`backward`."""

    trainable = True

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.buffers: dict[str, np.ndarray] = {}
        self.cache: Any = None

    # -- interface ---------------------------------------------------------
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        return self.forward(x, training=training)

    # -- introspection -----------------------------------------------------
    @property
    def param_count(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Shape (without the batch dimension) produced from *input_shape*."""
        return input_shape

    def describe(self) -> str:
        return self.name

    def config(self) -> dict:
        """Serialisable constructor arguments (used to rebuild an architecture)."""
        return {}


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------
class Conv2D(Layer):
    """2-D cross-correlation over NCHW tensors, via im2col + GEMM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | str = "same",
        use_bias: bool = True,
        name: str | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(name)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        if padding == "same":
            self.padding = self.kernel_size // 2
        elif padding == "valid":
            self.padding = 0
        else:
            self.padding = int(padding)
        self.use_bias = use_bias

        rng = rng or winit.make_rng()
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        self.params["W"] = winit.he_normal(
            (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size), fan_in, rng
        )
        if use_bias:
            self.params["b"] = winit.zeros((self.out_channels,))

    def forward(self, x, training=False):
        n = x.shape[0]
        cols, out_h, out_w = im2col(
            np.ascontiguousarray(as_array(x)),
            self.kernel_size, self.kernel_size, self.stride, self.padding,
        )
        weight = self.params["W"].reshape(self.out_channels, -1)
        out = cols @ weight.T
        if self.use_bias:
            out += self.params["b"]
        self.cache = (x.shape, cols, out_h, out_w)
        return out.reshape(n, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

    def backward(self, dout):
        x_shape, cols, out_h, out_w = self.cache
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        self.grads["W"] = (dout_flat.T @ cols).reshape(self.params["W"].shape)
        if self.use_bias:
            self.grads["b"] = dout_flat.sum(axis=0)

        dcols = dout_flat @ self.params["W"].reshape(self.out_channels, -1)
        return col2im(
            dcols, x_shape, self.kernel_size, self.kernel_size,
            self.stride, self.padding, out_h, out_w,
        )

    def output_shape(self, input_shape):
        _, h, w = input_shape
        return (
            self.out_channels,
            conv_output_size(h, self.kernel_size, self.stride, self.padding),
            conv_output_size(w, self.kernel_size, self.stride, self.padding),
        )

    def describe(self):
        return f"Conv2D({self.in_channels}→{self.out_channels}, k{self.kernel_size}, s{self.stride}, p{self.padding})"

    def config(self):
        return {
            "in_channels": self.in_channels, "out_channels": self.out_channels,
            "kernel_size": self.kernel_size, "stride": self.stride,
            "padding": self.padding, "use_bias": self.use_bias,
        }


class DepthwiseConv2D(Layer):
    """One filter per input channel — the cheap half of a separable block."""

    def __init__(self, channels, kernel_size=3, stride=1, padding="same", name=None, rng=None):
        super().__init__(name)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = self.kernel_size // 2 if padding == "same" else (0 if padding == "valid" else int(padding))

        rng = rng or winit.make_rng()
        self.params["W"] = winit.he_normal(
            (self.channels, 1, self.kernel_size, self.kernel_size), self.kernel_size ** 2, rng
        )
        self.params["b"] = winit.zeros((self.channels,))

    def forward(self, x, training=False):
        n, c, _, _ = x.shape
        # Fold channels into the batch axis so each one convolves independently.
        x_folded = x.reshape(n * c, 1, x.shape[2], x.shape[3])
        cols, out_h, out_w = im2col(
            np.ascontiguousarray(as_array(x_folded)),
            self.kernel_size, self.kernel_size, self.stride, self.padding,
        )
        # cols: (n*c*out_h*out_w, k*k) — pair each row with its own channel filter.
        weight = self.params["W"].reshape(self.channels, -1)          # (C, k*k)
        cols_view = cols.reshape(n, c, out_h * out_w, -1)
        out = np.einsum("ncpk,ck->ncp", cols_view, weight)
        out += self.params["b"][None, :, None]
        self.cache = (x.shape, cols_view, out_h, out_w)
        return out.reshape(n, c, out_h, out_w)

    def backward(self, dout):
        x_shape, cols_view, out_h, out_w = self.cache
        n, c = x_shape[0], x_shape[1]
        dout_flat = dout.reshape(n, c, out_h * out_w)

        self.grads["W"] = np.einsum("ncp,ncpk->ck", dout_flat, cols_view).reshape(self.params["W"].shape)
        self.grads["b"] = dout_flat.sum(axis=(0, 2))

        weight = self.params["W"].reshape(self.channels, -1)
        dcols = np.einsum("ncp,ck->ncpk", dout_flat, weight).reshape(n * c * out_h * out_w, -1)
        dx = col2im(
            dcols, (n * c, 1, x_shape[2], x_shape[3]),
            self.kernel_size, self.kernel_size, self.stride, self.padding, out_h, out_w,
        )
        return dx.reshape(x_shape)

    def output_shape(self, input_shape):
        c, h, w = input_shape
        return (
            c,
            conv_output_size(h, self.kernel_size, self.stride, self.padding),
            conv_output_size(w, self.kernel_size, self.stride, self.padding),
        )

    def describe(self):
        return f"DepthwiseConv2D({self.channels}, k{self.kernel_size}, s{self.stride})"

    def config(self):
        return {
            "channels": self.channels, "kernel_size": self.kernel_size,
            "stride": self.stride, "padding": self.padding,
        }


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
class BatchNorm2D(Layer):
    """Batch normalisation over the channel axis of an NCHW tensor.

    Also accepts 2-D ``(N, F)`` input, in which case it behaves as BatchNorm1D.
    """

    def __init__(self, channels: int, momentum: float = 0.9, eps: float = 1e-5, name=None):
        super().__init__(name)
        self.channels = int(channels)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.params["gamma"] = winit.ones((self.channels,))
        self.params["beta"] = winit.zeros((self.channels,))
        self.buffers["running_mean"] = winit.zeros((self.channels,))
        self.buffers["running_var"] = winit.ones((self.channels,))

    def _axes(self, x):
        return (0, 2, 3) if x.ndim == 4 else (0,)

    def _shape(self, x):
        return (1, self.channels, 1, 1) if x.ndim == 4 else (1, self.channels)

    def forward(self, x, training=False):
        x = as_array(x)
        axes, shape = self._axes(x), self._shape(x)
        if training:
            mean = x.mean(axis=axes)
            var = x.var(axis=axes)
            self.buffers["running_mean"] = (
                self.momentum * self.buffers["running_mean"] + (1 - self.momentum) * mean
            ).astype(default_dtype())
            self.buffers["running_var"] = (
                self.momentum * self.buffers["running_var"] + (1 - self.momentum) * var
            ).astype(default_dtype())
        else:
            mean = self.buffers["running_mean"]
            var = self.buffers["running_var"]

        std = np.sqrt(var + self.eps)
        x_hat = (x - mean.reshape(shape)) / std.reshape(shape)
        out = self.params["gamma"].reshape(shape) * x_hat + self.params["beta"].reshape(shape)
        self.cache = (x_hat, std, shape, axes, x.shape)
        return out

    def backward(self, dout):
        x_hat, std, shape, axes, _ = self.cache

        self.grads["gamma"] = (dout * x_hat).sum(axis=axes)
        self.grads["beta"] = dout.sum(axis=axes)

        gamma = self.params["gamma"].reshape(shape)
        dx_hat = dout * gamma
        dx = (
            dx_hat
            - dx_hat.mean(axis=axes).reshape(shape)
            - x_hat * (dx_hat * x_hat).mean(axis=axes).reshape(shape)
        ) / std.reshape(shape)
        return dx

    def describe(self):
        return f"BatchNorm2D({self.channels})"

    def config(self):
        return {"channels": self.channels, "momentum": self.momentum, "eps": self.eps}


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
class ReLU(Layer):
    trainable = False

    def forward(self, x, training=False):
        self.cache = x > 0
        return x * self.cache

    def backward(self, dout):
        return dout * self.cache

    def describe(self):
        return "ReLU"


class LeakyReLU(Layer):
    trainable = False

    def __init__(self, slope: float = 0.1, name=None):
        super().__init__(name)
        self.slope = float(slope)

    def forward(self, x, training=False):
        self.cache = x > 0
        return np.where(self.cache, x, x * self.slope)

    def backward(self, dout):
        return np.where(self.cache, dout, dout * self.slope)

    def describe(self):
        return f"LeakyReLU({self.slope})"

    def config(self):
        return {"slope": self.slope}


class Sigmoid(Layer):
    trainable = False

    def forward(self, x, training=False):
        self.cache = sigmoid(x)
        return self.cache

    def backward(self, dout):
        return dout * self.cache * (1.0 - self.cache)

    def describe(self):
        return "Sigmoid"


class Softmax(Layer):
    trainable = False

    def forward(self, x, training=False):
        self.cache = softmax(x, axis=-1)
        return self.cache

    def backward(self, dout):
        probs = self.cache
        dot = np.sum(dout * probs, axis=-1, keepdims=True)
        return probs * (dout - dot)

    def describe(self):
        return "Softmax"


class L2Normalize(Layer):
    """Projects embeddings onto the unit hypersphere — required for FaceNet-style
    cosine matching."""

    trainable = False

    def __init__(self, eps: float = 1e-10, name=None):
        super().__init__(name)
        self.eps = float(eps)

    def forward(self, x, training=False):
        norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + self.eps)
        self.cache = (x, norm)
        return x / norm

    def backward(self, dout):
        x, norm = self.cache
        dot = np.sum(dout * x, axis=-1, keepdims=True)
        return (dout - (x / (norm ** 2)) * dot) / norm

    def describe(self):
        return "L2Normalize"

    def config(self):
        return {"eps": self.eps}


# ---------------------------------------------------------------------------
# Pooling / reshaping
# ---------------------------------------------------------------------------
class MaxPool2D(Layer):
    trainable = False

    def __init__(self, pool_size: int = 2, stride: int | None = None, name=None):
        super().__init__(name)
        self.pool_size = int(pool_size)
        self.stride = int(stride or pool_size)

    def forward(self, x, training=False):
        n, c, h, w = x.shape
        out_h = conv_output_size(h, self.pool_size, self.stride, 0)
        out_w = conv_output_size(w, self.pool_size, self.stride, 0)
        x_folded = x.reshape(n * c, 1, h, w)
        cols, _, _ = im2col(
            np.ascontiguousarray(as_array(x_folded)), self.pool_size, self.pool_size, self.stride, 0
        )
        argmax = np.argmax(cols, axis=1)
        out = cols[np.arange(cols.shape[0]), argmax]
        self.cache = (x.shape, cols.shape, argmax, out_h, out_w)
        return out.reshape(n, c, out_h, out_w)

    def backward(self, dout):
        x_shape, cols_shape, argmax, out_h, out_w = self.cache
        n, c, h, w = x_shape
        dcols = np.zeros(cols_shape, dtype=default_dtype())
        dcols[np.arange(cols_shape[0]), argmax] = dout.reshape(-1)
        dx = col2im(
            dcols, (n * c, 1, h, w), self.pool_size, self.pool_size, self.stride, 0, out_h, out_w
        )
        return dx.reshape(x_shape)

    def output_shape(self, input_shape):
        c, h, w = input_shape
        return (
            c,
            conv_output_size(h, self.pool_size, self.stride, 0),
            conv_output_size(w, self.pool_size, self.stride, 0),
        )

    def describe(self):
        return f"MaxPool2D({self.pool_size}, s{self.stride})"

    def config(self):
        return {"pool_size": self.pool_size, "stride": self.stride}


class GlobalAvgPool2D(Layer):
    trainable = False

    def forward(self, x, training=False):
        self.cache = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, dout):
        n, c, h, w = self.cache
        return np.repeat(np.repeat(dout[:, :, None, None], h, axis=2), w, axis=3) / (h * w)

    def output_shape(self, input_shape):
        return (input_shape[0],)

    def describe(self):
        return "GlobalAvgPool2D"


class Flatten(Layer):
    trainable = False

    def forward(self, x, training=False):
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.cache)

    def output_shape(self, input_shape):
        return (int(np.prod(input_shape)),)

    def describe(self):
        return "Flatten"


class Reshape(Layer):
    trainable = False

    def __init__(self, shape: tuple[int, ...], name=None):
        super().__init__(name)
        self.shape = tuple(shape)

    def forward(self, x, training=False):
        self.cache = x.shape
        return x.reshape((x.shape[0],) + self.shape)

    def backward(self, dout):
        return dout.reshape(self.cache)

    def output_shape(self, input_shape):
        return self.shape

    def describe(self):
        return f"Reshape{self.shape}"

    def config(self):
        return {"shape": list(self.shape)}


# ---------------------------------------------------------------------------
# Fully connected / regularisation
# ---------------------------------------------------------------------------
class Dense(Layer):
    def __init__(self, in_features: int, out_features: int, use_bias: bool = True, name=None, rng=None):
        super().__init__(name)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_bias = use_bias
        rng = rng or winit.make_rng()
        self.params["W"] = winit.glorot_uniform(
            (self.in_features, self.out_features), self.in_features, self.out_features, rng
        )
        if use_bias:
            self.params["b"] = winit.zeros((self.out_features,))

    def forward(self, x, training=False):
        self.cache = x
        out = x @ self.params["W"]
        if self.use_bias:
            out = out + self.params["b"]
        return out

    def backward(self, dout):
        x = self.cache
        self.grads["W"] = x.T @ dout
        if self.use_bias:
            self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T

    def output_shape(self, input_shape):
        return (self.out_features,)

    def describe(self):
        return f"Dense({self.in_features}→{self.out_features})"

    def config(self):
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "use_bias": self.use_bias,
        }


class Dropout(Layer):
    trainable = False

    def __init__(self, rate: float = 0.3, name=None, rng=None):
        super().__init__(name)
        self.rate = float(rate)
        self.rng = rng or winit.make_rng()

    def forward(self, x, training=False):
        if not training or self.rate <= 0:
            self.cache = None
            return x
        # Inverted dropout — no rescaling needed at inference time.
        mask = (self.rng.random(x.shape) >= self.rate).astype(default_dtype()) / (1.0 - self.rate)
        self.cache = mask
        return x * mask

    def backward(self, dout):
        return dout if self.cache is None else dout * self.cache

    def describe(self):
        return f"Dropout({self.rate})"

    def config(self):
        return {"rate": self.rate}


LAYER_REGISTRY = {
    cls.__name__: cls
    for cls in (
        Conv2D, DepthwiseConv2D, BatchNorm2D, ReLU, LeakyReLU, Sigmoid, Softmax,
        L2Normalize, MaxPool2D, GlobalAvgPool2D, Flatten, Reshape, Dense, Dropout,
    )
}
