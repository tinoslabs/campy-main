"""Motion analysis: background modelling, optical flow and morphology.

This is what makes a *dumb* CCTV feed tractable. Running a neural network on
every pixel of every frame is wasteful; instead we maintain a statistical model
of "what this scene normally looks like", extract the handful of regions that
deviate, and only those go to the detectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .image import gaussian_blur, resize_bilinear, rgb_to_gray


@dataclass
class MotionState:
    """Per-frame motion summary handed to the detectors."""

    mask: np.ndarray                      # bool foreground mask (genuinely moving)
    motion_ratio: float                   # fraction of the frame in motion
    regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    energy: float = 0.0                   # mean absolute deviation
    flow: tuple[np.ndarray, np.ndarray] | None = None
    #: Pixels that differ from the background but have stopped changing — an
    #: object that was placed and left, or a ghost being absorbed.
    static_mask: np.ndarray | None = None
    static_regions: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def has_motion(self) -> bool:
        return self.motion_ratio > 0.0005 and bool(self.regions)


class BackgroundModel:
    """Adaptive per-pixel Gaussian background model (a lean MOG).

    Each pixel keeps a running mean and variance.  A pixel is foreground when it
    sits more than ``n_sigma`` standard deviations from its own mean, which
    handles gradual light changes, flickering fluorescent tubes and slow camera
    gain drift far better than naive frame differencing.

    A separate slow-learning ``long_term`` mean lets us spot *static* changes —
    an object that appeared and then stopped moving (abandoned bag, moved stock)
    which the fast model would otherwise absorb into the background.
    """

    def __init__(
        self,
        learning_rate: float = 0.02,
        n_sigma: float = 2.6,
        min_variance: float = 16.0,
        downscale: int = 2,
        long_term_rate: float = 0.0015,
        static_frames: int = 8,
        ghost_absorb_rate: float = 0.25,
    ):
        self.learning_rate = float(learning_rate)
        self.n_sigma = float(n_sigma)
        self.min_variance = float(min_variance)
        self.downscale = max(int(downscale), 1)
        self.long_term_rate = float(long_term_rate)
        #: Frames a foreground pixel must hold still before it counts as settled.
        self.static_frames = int(static_frames)
        #: How aggressively settled pixels are folded back into the background.
        self.ghost_absorb_rate = float(ghost_absorb_rate)

        self.mean: np.ndarray | None = None
        self.variance: np.ndarray | None = None
        self.long_term: np.ndarray | None = None
        self.previous_gray: np.ndarray | None = None
        #: Per-pixel count of consecutive frames a foreground pixel has been
        #: *not changing*. Drives ghost suppression and static-object detection.
        self.static_counter: np.ndarray | None = None
        #: Boxes (full-frame coords) the tracker has confirmed hold a person.
        #: Pixels inside them are never absorbed as background, so somebody
        #: standing perfectly still does not dissolve into the furniture.
        self.protected_boxes: list[tuple[float, float, float, float]] = []
        self.frames_seen = 0

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self.mean = self.variance = self.long_term = self.previous_gray = None
        self.static_counter = None
        self.protected_boxes = []
        self.frames_seen = 0

    @property
    def is_ready(self) -> bool:
        """The model needs a short warm-up before its output is trustworthy."""
        return self.frames_seen >= 12

    def protect(self, boxes) -> None:
        """Tell the model which full-frame boxes currently contain a subject.

        The inference pipeline feeds the tracker's confirmed person boxes back
        in here each frame. This closes the loop between tracking and
        background modelling: detection may drop out for a stationary person,
        but the *track* persists, and the track keeps their pixels out of the
        background.
        """
        self.protected_boxes = [tuple(float(v) for v in box) for box in (boxes or [])]

    def _protection_mask(self, shape, full_shape) -> np.ndarray | None:
        if not self.protected_boxes:
            return None
        mask = np.zeros(shape, dtype=bool)
        sy = shape[0] / max(full_shape[0], 1)
        sx = shape[1] / max(full_shape[1], 1)
        for x1, y1, x2, y2 in self.protected_boxes:
            # Pad slightly — the tracker's box is an estimate, not a matte.
            pad_x = (x2 - x1) * 0.12
            pad_y = (y2 - y1) * 0.12
            ix1 = int(np.clip((x1 - pad_x) * sx, 0, shape[1]))
            ix2 = int(np.clip((x2 + pad_x) * sx, 0, shape[1]))
            iy1 = int(np.clip((y1 - pad_y) * sy, 0, shape[0]))
            iy2 = int(np.clip((y2 + pad_y) * sy, 0, shape[0]))
            if ix2 > ix1 and iy2 > iy1:
                mask[iy1:iy2, ix1:ix2] = True
        return mask

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        gray = rgb_to_gray(frame)
        if self.downscale > 1:
            gray = resize_bilinear(
                gray, max(gray.shape[0] // self.downscale, 1), max(gray.shape[1] // self.downscale, 1)
            )
        return gaussian_blur(gray, sigma=1.0)

    # -- main entry point --------------------------------------------------
    def update(self, frame: np.ndarray, compute_flow: bool = False) -> MotionState:
        gray = self._prepare(frame)
        full_h, full_w = np.asarray(frame).shape[:2]

        if self.mean is None:
            self.mean = gray.copy()
            self.variance = np.full_like(gray, self.min_variance * 4)
            self.long_term = gray.copy()
            self.previous_gray = gray.copy()
            self.static_counter = np.zeros_like(gray, dtype=np.float32)
            self.frames_seen = 1
            return MotionState(
                mask=np.zeros(gray.shape, dtype=bool),
                motion_ratio=0.0,
                static_mask=np.zeros(gray.shape, dtype=bool),
            )

        delta = gray - self.mean
        std = np.sqrt(np.maximum(self.variance, self.min_variance))
        foreground = np.abs(delta) > (self.n_sigma * std)

        # --- separate "genuinely moving" from "different but standing still" ---
        #
        # A pixel differing from the background is not necessarily a moving
        # object.  It is equally often a *ghost*: the patch of background a
        # subject has vacated, or a chair someone pushed aside.  Both differ
        # from the model but stop changing frame to frame.  Without this
        # distinction, every subject leaves a permanent phantom at the position
        # they occupied when the model was first built.
        frame_delta = np.abs(gray - self.previous_gray)
        changing = frame_delta > max(self.n_sigma * 0.55 * np.sqrt(self.min_variance), 3.0)

        self.static_counter = np.where(
            foreground & ~changing, self.static_counter + 1.0, 0.0
        ).astype(np.float32)
        settled = self.static_counter >= self.static_frames

        protection = self._protection_mask(gray.shape, (full_h, full_w))
        if protection is not None:
            # A tracked subject is never "settled background", however still
            # they stand.
            self.static_counter[protection] = 0.0
            settled &= ~protection

        moving = foreground & ~settled

        # Settled pixels are absorbed quickly so ghosts disappear in a couple of
        # seconds instead of lingering for minutes; true movers adapt slowly so
        # a person who pauses is not dissolved into the background.
        rate = np.where(
            settled, self.ghost_absorb_rate,
            np.where(moving, self.learning_rate * 0.05, self.learning_rate),
        ).astype(np.float32)
        if protection is not None:
            # Learning *nothing* inside a protected box is the point: the model
            # keeps the true empty-scene appearance, so a subject who stands
            # still for ten minutes never fades into the furniture, and the
            # moment they leave, those pixels match the background again.
            rate[protection] = 0.0
        self.mean += rate * delta
        self.variance = (1 - rate) * self.variance + rate * (delta ** 2)
        self.long_term += self.long_term_rate * (gray - self.long_term)

        mask = binary_close(binary_open(moving, iterations=1), iterations=2)
        static_mask = binary_close(binary_open(settled, iterations=1), iterations=1)

        flow = None
        if compute_flow and self.previous_gray is not None:
            flow = lucas_kanade_flow(self.previous_gray, gray)
        self.previous_gray = gray.copy()
        self.frames_seen += 1

        min_area = max(int(mask.size * 0.0004), 6)
        regions = [
            _rescale_region(region, gray.shape, (full_h, full_w))
            for region in connected_regions(mask, min_area=min_area)
        ]
        static_regions = [
            _rescale_region(region, gray.shape, (full_h, full_w))
            for region in connected_regions(static_mask, min_area=min_area * 2)
        ]
        return MotionState(
            mask=mask,
            motion_ratio=float(mask.mean()),
            regions=regions,
            energy=float(np.mean(np.abs(delta))),
            flow=flow,
            static_mask=static_mask,
            static_regions=static_regions,
        )

    def static_change_mask(self, frame: np.ndarray) -> np.ndarray:
        """Regions that differ from the *slow* background — i.e. something
        appeared and then stopped moving.  Feeds abandoned-object detection."""
        if self.long_term is None:
            return np.zeros((1, 1), dtype=bool)
        gray = self._prepare(frame)
        diff = np.abs(gray - self.long_term)
        return binary_close(diff > max(self.n_sigma * np.sqrt(self.min_variance), 18.0), iterations=2)


def _rescale_region(region, small_shape, full_shape):
    """Map a box found on the downscaled mask back to full-frame coordinates."""
    sy = full_shape[0] / max(small_shape[0], 1)
    sx = full_shape[1] / max(small_shape[1], 1)
    x1, y1, x2, y2 = region
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------
def _shift_or(mask: np.ndarray, op) -> np.ndarray:
    """Apply a 3×3 cross structuring element via shifted views."""
    padded = np.pad(mask, 1, mode="constant", constant_values=op is np.logical_and)
    h, w = mask.shape
    neighbours = [
        padded[1 : h + 1, 1 : w + 1],
        padded[0:h, 1 : w + 1],
        padded[2 : h + 2, 1 : w + 1],
        padded[1 : h + 1, 0:w],
        padded[1 : h + 1, 2 : w + 2],
    ]
    out = neighbours[0]
    for neighbour in neighbours[1:]:
        out = op(out, neighbour)
    return out


def binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(int(iterations), 0)):
        out = _shift_or(out, np.logical_or)
    return out


def binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(int(iterations), 0)):
        out = _shift_or(out, np.logical_and)
    return out


def binary_open(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erode then dilate — removes salt-and-pepper sensor noise."""
    return binary_dilate(binary_erode(mask, iterations), iterations)


def binary_close(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilate then erode — fills the holes inside one person's silhouette."""
    return binary_erode(binary_dilate(mask, iterations), iterations)


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------
def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Two-pass 8-connected labelling with union-find.

    Returns ``(labels, count)`` where background is 0 and blobs are 1..count.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    next_label = 1
    for y in range(h):
        row = mask[y]
        for x in range(w):
            if not row[x]:
                continue
            neighbours = []
            if y > 0:
                for dx in (-1, 0, 1):
                    nx = x + dx
                    if 0 <= nx < w and labels[y - 1, nx]:
                        neighbours.append(labels[y - 1, nx])
            if x > 0 and labels[y, x - 1]:
                neighbours.append(labels[y, x - 1])

            if not neighbours:
                labels[y, x] = next_label
                parent.append(next_label)
                next_label += 1
            else:
                smallest = min(neighbours)
                labels[y, x] = smallest
                for other in neighbours:
                    union(smallest, other)

    # Second pass: flatten to consecutive ids.
    remap: dict[int, int] = {}
    count = 0
    flat = labels.ravel()
    for index, value in enumerate(flat):
        if value:
            root = find(value)
            if root not in remap:
                count += 1
                remap[root] = count
            flat[index] = remap[root]
    return labels, count


def connected_regions(mask: np.ndarray, min_area: int = 8, max_regions: int = 60):
    """Bounding boxes of foreground blobs, largest first.

    Uses a fast row-run merge rather than per-pixel labelling — a motion mask is
    mostly empty, and this keeps the hot path out of Python loops.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    h, w = mask.shape

    # Collect horizontal runs per row.
    runs: list[tuple[int, int, int]] = []  # (row, x_start, x_end_exclusive)
    for y in range(h):
        row = mask[y]
        if not row.any():
            continue
        edges = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for start, end in zip(starts, ends):
            runs.append((y, int(start), int(end)))

    if not runs:
        return []

    # Union-find over runs that touch (8-connectivity between adjacent rows).
    parent = list(range(len(runs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_row: dict[int, list[int]] = {}
    for index, (y, _, _) in enumerate(runs):
        by_row.setdefault(y, []).append(index)

    for y, indices in by_row.items():
        above = by_row.get(y - 1)
        if not above:
            continue
        for i in indices:
            _, s1, e1 = runs[i]
            for j in above:
                _, s2, e2 = runs[j]
                if s1 <= e2 and s2 <= e1:  # overlapping or diagonally touching
                    union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(len(runs)):
        groups.setdefault(find(index), []).append(index)

    boxes = []
    for indices in groups.values():
        ys = [runs[i][0] for i in indices]
        x1 = min(runs[i][1] for i in indices)
        x2 = max(runs[i][2] for i in indices)
        y1, y2 = min(ys), max(ys) + 1
        area = sum(runs[i][2] - runs[i][1] for i in indices)
        if area >= min_area:
            boxes.append((area, (int(x1), int(y1), int(x2), int(y2))))

    boxes.sort(key=lambda item: -item[0])
    return [box for _, box in boxes[:max_regions]]


# ---------------------------------------------------------------------------
# Frame differencing & optical flow
# ---------------------------------------------------------------------------
def frame_difference(previous: np.ndarray, current: np.ndarray, threshold: float = 22.0):
    """Simple absolute difference — the cheap fallback when no background model
    has warmed up yet."""
    a = rgb_to_gray(previous)
    b = rgb_to_gray(current)
    if a.shape != b.shape:
        b = resize_bilinear(b, a.shape[0], a.shape[1])
    diff = np.abs(a - b)
    return binary_open(diff > threshold), float(diff.mean())


def lucas_kanade_flow(previous: np.ndarray, current: np.ndarray, window: int = 9):
    """Dense Lucas–Kanade optical flow on a coarse grid.

    Solves the 2×2 normal equations per window::

        [Σ Ix²   Σ IxIy] [u]   [-Σ IxIt]
        [Σ IxIy  Σ Iy² ] [v] = [-Σ IyIt]

    Flow direction and magnitude are what separate "walking past a shelf" from
    "reaching in, grabbing and turning away" in theft detection, and what tells
    crowd analytics whether a group is flowing or jammed.
    """
    a = np.asarray(previous, dtype=np.float32)
    b = np.asarray(current, dtype=np.float32)
    if a.ndim == 3:
        a = rgb_to_gray(a)
    if b.ndim == 3:
        b = rgb_to_gray(b)
    if a.shape != b.shape:
        b = resize_bilinear(b, a.shape[0], a.shape[1])

    a = gaussian_blur(a, 1.0)
    b = gaussian_blur(b, 1.0)

    ix = np.zeros_like(a)
    iy = np.zeros_like(a)
    ix[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / 2.0
    iy[1:-1, :] = (a[2:, :] - a[:-2, :]) / 2.0
    it = b - a

    half = max(int(window) // 2, 1)
    step = half
    h, w = a.shape
    u = np.zeros(((h - 1) // step + 1, (w - 1) // step + 1), dtype=np.float32)
    v = np.zeros_like(u)

    for oy, y in enumerate(range(0, h, step)):
        y0, y1 = max(y - half, 0), min(y + half + 1, h)
        for ox, x in enumerate(range(0, w, step)):
            x0, x1 = max(x - half, 0), min(x + half + 1, w)
            wx = ix[y0:y1, x0:x1].ravel()
            wy = iy[y0:y1, x0:x1].ravel()
            wt = it[y0:y1, x0:x1].ravel()

            sxx = float(np.dot(wx, wx))
            syy = float(np.dot(wy, wy))
            sxy = float(np.dot(wx, wy))
            sxt = float(np.dot(wx, wt))
            syt = float(np.dot(wy, wt))

            determinant = sxx * syy - sxy * sxy
            # Skip windows without enough texture — the aperture problem.
            if abs(determinant) < 1e-3 or (sxx + syy) < 1e-2:
                continue
            u[oy, ox] = (-syy * sxt + sxy * syt) / determinant
            v[oy, ox] = (sxy * sxt - sxx * syt) / determinant

    limit = 12.0
    return np.clip(u, -limit, limit), np.clip(v, -limit, limit)


def flow_statistics(flow) -> dict:
    """Summarise a flow field into features a classifier can consume."""
    if flow is None:
        return {"magnitude": 0.0, "direction": 0.0, "coherence": 0.0, "turbulence": 0.0}
    u, v = flow
    magnitude = np.sqrt(u * u + v * v)
    moving = magnitude > 0.35
    if not moving.any():
        return {"magnitude": 0.0, "direction": 0.0, "coherence": 0.0, "turbulence": 0.0}

    mean_u = float(u[moving].mean())
    mean_v = float(v[moving].mean())
    mean_magnitude = float(magnitude[moving].mean())
    resultant = float(np.hypot(mean_u, mean_v))
    return {
        "magnitude": round(mean_magnitude, 4),
        "direction": round(float(np.degrees(np.arctan2(mean_v, mean_u)) % 360), 2),
        # 1.0 = everyone moving the same way, 0.0 = milling about.
        "coherence": round(resultant / max(mean_magnitude, 1e-6), 4),
        "turbulence": round(float(magnitude[moving].std()), 4),
    }
