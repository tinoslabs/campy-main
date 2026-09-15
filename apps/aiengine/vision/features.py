"""Hand-crafted feature descriptors.

Even with our own CNNs, classical descriptors earn their place: they are
deterministic, need no training data, and run in microseconds. Campy AI uses
them for the cheap first-stage filters (region proposals, colour gating) and as
extra input channels alongside learned features.
"""
from __future__ import annotations

import numpy as np

from .image import resize_bilinear, rgb_to_gray, sobel_gradients


def histogram_of_oriented_gradients(
    image: np.ndarray,
    cell_size: int = 8,
    bins: int = 9,
    block_size: int = 2,
    signed: bool = False,
) -> np.ndarray:
    """HOG descriptor — the classic human-shape signature.

    Gradient orientations are pooled into cells, then L2-Hys normalised over
    overlapping blocks to gain invariance to lighting and contrast.
    """
    gray = rgb_to_gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, dtype=np.float32)
    gx, gy = sobel_gradients(gray)
    magnitude = np.hypot(gx, gy)
    span = 360.0 if signed else 180.0
    orientation = (np.degrees(np.arctan2(gy, gx)) % span)

    h, w = gray.shape
    cells_y, cells_x = h // cell_size, w // cell_size
    if cells_y < 1 or cells_x < 1:
        return np.zeros(bins, dtype=np.float32)

    histogram = np.zeros((cells_y, cells_x, bins), dtype=np.float32)
    bin_width = span / bins
    # Bilinear vote between the two neighbouring orientation bins.
    bin_index = orientation / bin_width
    lower = np.floor(bin_index).astype(np.int32) % bins
    upper = (lower + 1) % bins
    upper_weight = bin_index - np.floor(bin_index)
    lower_weight = 1.0 - upper_weight

    for cy in range(cells_y):
        for cx in range(cells_x):
            ys = slice(cy * cell_size, (cy + 1) * cell_size)
            xs = slice(cx * cell_size, (cx + 1) * cell_size)
            cell_magnitude = magnitude[ys, xs].ravel()
            np.add.at(histogram[cy, cx], lower[ys, xs].ravel(), cell_magnitude * lower_weight[ys, xs].ravel())
            np.add.at(histogram[cy, cx], upper[ys, xs].ravel(), cell_magnitude * upper_weight[ys, xs].ravel())

    blocks_y = max(cells_y - block_size + 1, 1)
    blocks_x = max(cells_x - block_size + 1, 1)
    descriptor = []
    for by in range(blocks_y):
        for bx in range(blocks_x):
            block = histogram[by : by + block_size, bx : bx + block_size].ravel()
            norm = np.sqrt(np.sum(block ** 2) + 1e-6)
            block = np.clip(block / norm, 0, 0.2)          # L2-Hys clipping
            block = block / np.sqrt(np.sum(block ** 2) + 1e-6)
            descriptor.append(block)
    return np.concatenate(descriptor).astype(np.float32) if descriptor else np.zeros(bins, dtype=np.float32)


def local_binary_pattern(image: np.ndarray, radius: int = 1) -> np.ndarray:
    """Uniform-ish LBP texture codes — cheap, powerful for smoke vs. steam and
    for spotting printed-photo spoofing during face enrolment."""
    gray = rgb_to_gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, dtype=np.float32)
    padded = np.pad(gray, radius, mode="edge")
    h, w = gray.shape
    centre = gray
    codes = np.zeros((h, w), dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        y0 = radius + dy
        x0 = radius + dx
        neighbour = padded[y0 : y0 + h, x0 : x0 + w]
        codes |= ((neighbour >= centre).astype(np.uint8) << bit)
    return codes


def lbp_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    codes = local_binary_pattern(image)
    histogram, _ = np.histogram(codes, bins=bins, range=(0, 256))
    total = histogram.sum()
    return (histogram / total).astype(np.float32) if total else histogram.astype(np.float32)


def colour_histogram(image: np.ndarray, bins: int = 8) -> np.ndarray:
    """Joint RGB histogram — the appearance signature used for re-identifying a
    person across cameras and for matching an object to the person carrying it."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    array = np.clip(array[:, :, :3], 0, 255)
    quantised = (array / (256.0 / bins)).astype(np.int32).clip(0, bins - 1)
    flat = quantised[:, :, 0] * bins * bins + quantised[:, :, 1] * bins + quantised[:, :, 2]
    histogram = np.bincount(flat.ravel(), minlength=bins ** 3).astype(np.float32)
    total = histogram.sum()
    return histogram / total if total else histogram


def integral_image(image: np.ndarray) -> np.ndarray:
    """Summed-area table — constant-time rectangle sums."""
    array = np.asarray(image, dtype=np.float64)
    padded = np.zeros((array.shape[0] + 1, array.shape[1] + 1), dtype=np.float64)
    padded[1:, 1:] = np.cumsum(np.cumsum(array, axis=0), axis=1)
    return padded


def rectangle_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])


def edge_density(image: np.ndarray, threshold: float = 40.0) -> float:
    """Fraction of pixels on a strong edge. Smoke lowers it; clutter raises it."""
    gray = rgb_to_gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, dtype=np.float32)
    gx, gy = sobel_gradients(gray)
    return float(np.mean(np.hypot(gx, gy) > threshold))


def image_entropy(image: np.ndarray) -> float:
    """Shannon entropy of the intensity histogram — a texture-richness proxy."""
    gray = rgb_to_gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, dtype=np.float32)
    histogram, _ = np.histogram(np.clip(gray, 0, 255), bins=64, range=(0, 256))
    probabilities = histogram / max(histogram.sum(), 1)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def boundary_gradient_support(frame: np.ndarray, box, samples: int = 24) -> float:
    """How much of a box's outline sits on a real image edge, 0-1.

    This is the test that separates a genuine object from a *ghost* — the patch
    of background a subject vacated, which still differs from the background
    model but has no physical boundary in the current image. A real object is
    bounded by edges on most of its perimeter; a ghost's outline falls on
    continuous, edge-free background.

    Without this check, a ghost keeps producing detections, the tracker keeps
    confirming them, and the camera reports a person who is not there.
    """
    frame = np.asarray(frame, dtype=np.float32)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0

    gray = rgb_to_gray(frame) if frame.ndim == 3 else frame
    gx, gy = sobel_gradients(gray)
    magnitude = np.hypot(gx, gy)

    # Reference: how edgy is this image overall? Comparing against the frame's
    # own gradient distribution keeps the test valid for both a busy warehouse
    # and a blank corridor.
    reference = float(np.percentile(magnitude, 75)) + 1e-6

    per_side = max(samples // 4, 3)
    xs = np.linspace(x1, x2 - 1, per_side)
    ys = np.linspace(y1, y2 - 1, per_side)
    points = (
        [(x, y1) for x in xs] + [(x, y2 - 1) for x in xs]
        + [(x1, y) for y in ys] + [(x2 - 1, y) for y in ys]
    )

    hits = 0
    for px, py in points:
        ix = int(np.clip(px, 0, width - 1))
        iy = int(np.clip(py, 0, height - 1))
        # Look in a small neighbourhood — box edges are approximate.
        window = magnitude[
            max(iy - 1, 0) : min(iy + 2, height), max(ix - 1, 0) : min(ix + 2, width)
        ]
        if window.size and float(window.max()) > reference:
            hits += 1
    return float(hits / max(len(points), 1))


def descriptor_bundle(patch: np.ndarray, size: int = 48) -> np.ndarray:
    """Compact fixed-length descriptor combining shape, texture and colour.

    Used as the appearance embedding for tracking when no learned embedding
    model is deployed for a given camera.
    """
    array = np.asarray(patch, dtype=np.float32)
    if array.size == 0:
        return np.zeros(64, dtype=np.float32)
    resized = resize_bilinear(array, size, size)
    parts = [
        histogram_of_oriented_gradients(resized, cell_size=12, bins=9)[:81],
        lbp_histogram(resized, bins=16),
        colour_histogram(resized, bins=4),
    ]
    vector = np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts])
    norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.float32) if norm > 0 else vector.astype(np.float32)
