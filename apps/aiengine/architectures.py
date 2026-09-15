"""Campy AI model architectures.

Everything the platform deploys is built from these factories.  The designs are
deliberately small — Campy AI runs many streams per box on ordinary CPU
hardware, so a 200k-parameter separable CNN that holds 6 fps beats a 25M-param
backbone that manages 0.4 fps.

Families
--------
``campynet``    general image classifier (gestures, fire/smoke, object states)
``campyface``   L2-normalised embedding network for face / person recognition
``campydet``    single-shot grid detector for people and objects
``campydense``  density-map regressor for crowd counting
``campyseq``    temporal model over pooled per-frame features (actions)
"""
from __future__ import annotations

import numpy as np

from .nn import layers as L
from .nn.init import make_rng
from .nn.model import Sequential


def separable_block(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    rng: np.random.Generator | None = None,
) -> list[L.Layer]:
    """Depthwise 3×3 → pointwise 1×1 → BN → ReLU.

    ~8-9× fewer multiply-accumulates than a dense 3×3 convolution at the same
    width, which is the whole reason we can run this on a CPU.
    """
    rng = rng or make_rng()
    return [
        L.DepthwiseConv2D(in_channels, 3, stride, "same", rng=rng),
        L.BatchNorm2D(in_channels),
        L.ReLU(),
        L.Conv2D(in_channels, out_channels, 1, 1, "same", use_bias=False, rng=rng),
        L.BatchNorm2D(out_channels),
        L.ReLU(),
    ]


def conv_block(in_channels: int, out_channels: int, stride: int = 1, rng=None) -> list[L.Layer]:
    rng = rng or make_rng()
    return [
        L.Conv2D(in_channels, out_channels, 3, stride, "same", use_bias=False, rng=rng),
        L.BatchNorm2D(out_channels),
        L.ReLU(),
    ]


def campynet_backbone(
    in_channels: int = 3,
    width: float = 1.0,
    depth: int = 4,
    rng: np.random.Generator | None = None,
) -> tuple[list[L.Layer], int]:
    """Shared feature extractor. Returns ``(layers, output_channels)``.

    Stem downsamples ×2, then ``depth`` separable stages each downsample ×2, so
    a 96×96 input with depth 4 leaves a 3×3 feature map.
    """
    rng = rng or make_rng()
    channels = max(int(round(16 * width)), 8)
    layers = conv_block(in_channels, channels, stride=2, rng=rng)

    for stage in range(depth):
        out_channels = max(int(round(channels * 2)), 8)
        layers += separable_block(channels, out_channels, stride=2, rng=rng)
        # A stride-1 refinement block at deeper stages adds capacity cheaply.
        if stage >= 1:
            layers += separable_block(out_channels, out_channels, stride=1, rng=rng)
        channels = out_channels
    return layers, channels


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def build_classifier(
    num_classes: int,
    in_channels: int = 3,
    width: float = 1.0,
    depth: int = 3,
    dropout: float = 0.25,
    name: str = "campynet",
    seed: int | None = 1337,
    input_size: int = 96,
) -> Sequential:
    """General-purpose image classifier (gesture, fire/smoke, object state)."""
    rng = make_rng(seed)
    layers, channels = campynet_backbone(in_channels, width, depth, rng)
    layers += [
        L.GlobalAvgPool2D(),
        L.Dropout(dropout, rng=rng),
        L.Dense(channels, max(int(round(64 * width)), 16), rng=rng),
        L.ReLU(),
        L.Dense(max(int(round(64 * width)), 16), int(num_classes), rng=rng),
    ]
    return Sequential(
        layers,
        name=name,
        meta={
            "family": "campynet",
            "task": "classification",
            "num_classes": int(num_classes),
            "in_channels": in_channels,
            "input_size": int(input_size),
            "width": width,
            "depth": depth,
        },
    )


# ---------------------------------------------------------------------------
# Embedding network (face / person re-identification)
# ---------------------------------------------------------------------------
def build_embedding_network(
    embedding_dim: int = 128,
    in_channels: int = 3,
    width: float = 1.0,
    depth: int = 3,
    name: str = "campyface",
    seed: int | None = 1337,
    input_size: int = 96,
) -> Sequential:
    """Maps a face crop to a unit-norm vector.

    Trained with batch-hard triplet loss, so cosine similarity between two
    embeddings is a direct identity score and enrolment needs only a handful of
    images per person — no retraining to add an employee.
    """
    rng = make_rng(seed)
    layers, channels = campynet_backbone(in_channels, width, depth, rng)
    hidden = max(int(round(128 * width)), 32)
    layers += [
        L.GlobalAvgPool2D(),
        L.Dense(channels, hidden, rng=rng),
        L.BatchNorm2D(hidden),
        L.ReLU(),
        L.Dense(hidden, int(embedding_dim), rng=rng),
        L.L2Normalize(),
    ]
    return Sequential(
        layers,
        name=name,
        meta={
            "family": "campyface",
            "task": "embedding",
            "embedding_dim": int(embedding_dim),
            "in_channels": in_channels,
            "input_size": int(input_size),
            "width": width,
            "depth": depth,
        },
    )


# ---------------------------------------------------------------------------
# Single-shot detector
# ---------------------------------------------------------------------------
def build_detector(
    num_classes: int,
    grid: int = 6,
    in_channels: int = 3,
    width: float = 1.0,
    name: str = "campydet",
    seed: int | None = 1337,
    input_size: int = 96,
) -> Sequential:
    """Grid detector: each cell predicts one box.

    Output is ``(N, 5 + num_classes, grid, grid)`` where the 5 channels are
    ``[tx, ty, tw, th, objectness]``.  ``tx/ty`` are offsets inside the cell
    (sigmoid), ``tw/th`` are log-scale sizes relative to the whole frame.
    """
    rng = make_rng(seed)
    # Choose depth so the feature map lands on the requested grid size.
    depth = max(int(round(np.log2(max(input_size / max(grid, 1), 1)))) - 1, 1)
    layers, channels = campynet_backbone(in_channels, width, depth, rng)
    head_channels = 5 + int(num_classes)
    layers += [
        L.Conv2D(channels, max(channels, 32), 3, 1, "same", use_bias=False, rng=rng),
        L.BatchNorm2D(max(channels, 32)),
        L.ReLU(),
        L.Conv2D(max(channels, 32), head_channels, 1, 1, "same", rng=rng),
    ]
    return Sequential(
        layers,
        name=name,
        meta={
            "family": "campydet",
            "task": "detection",
            "num_classes": int(num_classes),
            "grid": int(grid),
            "in_channels": in_channels,
            "input_size": int(input_size),
            "width": width,
        },
    )


# ---------------------------------------------------------------------------
# Crowd density regressor
# ---------------------------------------------------------------------------
def build_density_network(
    in_channels: int = 3,
    width: float = 1.0,
    name: str = "campydense",
    seed: int | None = 1337,
    input_size: int = 96,
) -> Sequential:
    """Predicts a per-pixel head-density map; the sum is the people count.

    Counting by density rather than by detection is what keeps crowd numbers
    honest once heads start occluding each other in a packed corridor.
    """
    rng = make_rng(seed)
    layers, channels = campynet_backbone(in_channels, width, depth=2, rng=rng)
    layers += [
        L.Conv2D(channels, max(channels // 2, 8), 3, 1, "same", use_bias=False, rng=rng),
        L.BatchNorm2D(max(channels // 2, 8)),
        L.ReLU(),
        L.Conv2D(max(channels // 2, 8), 1, 1, 1, "same", rng=rng),
        L.ReLU(),  # density can never be negative
    ]
    return Sequential(
        layers,
        name=name,
        meta={
            "family": "campydense",
            "task": "density",
            "in_channels": in_channels,
            "input_size": int(input_size),
            "width": width,
            # Density maps are trained at 1/8 resolution; scale accordingly.
            "stride": 8,
        },
    )


# ---------------------------------------------------------------------------
# Temporal / sequence model
# ---------------------------------------------------------------------------
def build_sequence_classifier(
    feature_dim: int,
    num_classes: int,
    hidden: int = 96,
    dropout: float = 0.2,
    name: str = "campyseq",
    seed: int | None = 1337,
) -> Sequential:
    """Classifies a *window* of pooled per-frame features into an action.

    A single frame cannot distinguish "bending to pick up a box" from "bending
    to conceal an item"; the difference is in the sequence. We flatten a fixed
    window of frame descriptors and let an MLP learn the temporal pattern —
    cheap, stable, and trainable from very few labelled clips.
    """
    rng = make_rng(seed)
    return Sequential(
        [
            L.Dense(int(feature_dim), hidden, rng=rng),
            L.BatchNorm2D(hidden),
            L.ReLU(),
            L.Dropout(dropout, rng=rng),
            L.Dense(hidden, hidden // 2, rng=rng),
            L.ReLU(),
            L.Dropout(dropout / 2, rng=rng),
            L.Dense(hidden // 2, int(num_classes), rng=rng),
        ],
        name=name,
        meta={
            "family": "campyseq",
            "task": "classification",
            "feature_dim": int(feature_dim),
            "num_classes": int(num_classes),
        },
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ARCHITECTURES = {
    "campynet": build_classifier,
    "campyface": build_embedding_network,
    "campydet": build_detector,
    "campydense": build_density_network,
    "campyseq": build_sequence_classifier,
}

ARCHITECTURE_CHOICES = [
    ("campynet", "CampyNet — image classifier"),
    ("campyface", "CampyFace — identity embedding"),
    ("campydet", "CampyDet — object detector"),
    ("campydense", "CampyDense — crowd density"),
    ("campyseq", "CampySeq — temporal action"),
]


def build(family: str, **kwargs) -> Sequential:
    factory = ARCHITECTURES.get(family)
    if factory is None:
        raise ValueError(f"Unknown architecture '{family}'. Available: {sorted(ARCHITECTURES)}")
    return factory(**kwargs)
