"""Campy AI's own neural-network framework.

This is deliberately *not* a wrapper around PyTorch or TensorFlow.  Every layer,
loss and optimiser here is implemented from first principles on top of NumPy so
that Campy AI owns its full model stack end to end: architecture, training loop,
weights format and inference runtime.

Design notes
------------
* Explicit ``forward`` / ``backward`` per layer (no tape) — small, auditable and
  fast enough for the compact models we deploy at the edge.
* Tensor layout is NCHW (batch, channels, height, width), matching the
  convention used by the rest of the vision stack.
* Weights serialise to a single ``.npz`` per model version, which the model
  registry stores and the inference workers memory-map.
"""
from .layers import (  # noqa: F401
    BatchNorm2D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Flatten,
    GlobalAvgPool2D,
    L2Normalize,
    Layer,
    LeakyReLU,
    MaxPool2D,
    ReLU,
    Sigmoid,
    Softmax,
)
from .losses import (  # noqa: F401
    BinaryCrossEntropy,
    CrossEntropy,
    FocalLoss,
    MeanSquaredError,
    TripletLoss,
)
from .model import Sequential  # noqa: F401
from .optim import SGD, Adam, AdamW, CosineDecay, StepDecay  # noqa: F401

__all__ = [
    "Layer", "Conv2D", "DepthwiseConv2D", "BatchNorm2D", "ReLU", "LeakyReLU",
    "Sigmoid", "Softmax", "MaxPool2D", "GlobalAvgPool2D", "Flatten", "Dense",
    "Dropout", "L2Normalize", "Sequential",
    "CrossEntropy", "BinaryCrossEntropy", "MeanSquaredError", "TripletLoss", "FocalLoss",
    "SGD", "Adam", "AdamW", "StepDecay", "CosineDecay",
]
