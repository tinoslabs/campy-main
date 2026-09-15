"""Boxes, polygons and the geometry behind geofencing."""
from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2

EMPTY_POLYGON = np.empty((0, 2), dtype=np.float64)


def as_polygon(polygon) -> np.ndarray:
    """Coerce stored polygon data into an (N, 2) array of finite floats.

    Polygons live in a JSON column, so they arrive in whatever shape a client
    once managed to store: a ragged list, a coordinate that is a string, an odd
    number of values, a NaN. Anything unusable becomes an empty polygon — a
    zone that covers nothing — because a badly drawn zone must never take down
    the page that lists it or the worker that reads it. Well-formed input comes
    back exactly as before.
    """
    if polygon is None:
        return EMPTY_POLYGON
    try:
        points = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError):
        return EMPTY_POLYGON
    if points.size == 0 or points.size % 2:
        return EMPTY_POLYGON
    points = points.reshape(-1, 2)
    if not np.isfinite(points).all():
        return EMPTY_POLYGON
    return points


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_foot(box: Box) -> tuple[float, float]:
    """Bottom-centre point — where a person actually stands.

    Geofencing must test the *feet*, not the centroid: a tall person standing
    just outside a restricted zone has a centroid that can fall inside it.
    """
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def clip_box(box: Box, width: float, height: float) -> Box:
    x1, y1, x2, y2 = box
    return (
        float(max(0.0, min(x1, width))),
        float(max(0.0, min(y1, height))),
        float(max(0.0, min(x2, width))),
        float(max(0.0, min(y2, height))),
    )


def scale_box(box: Box, sx: float, sy: float) -> Box:
    x1, y1, x2, y2 = box
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = box_area(a) + box_area(b) - intersection
    return float(intersection / union) if union > 0 else 0.0


def iou_matrix(boxes_a, boxes_b) -> np.ndarray:
    """Vectorised pairwise IoU — the association step of the tracker."""
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def non_max_suppression(boxes, scores, iou_threshold: float = 0.45, top_k: int = 200) -> list[int]:
    """Greedy NMS. Returns the indices of the boxes that survive."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).ravel()
    if boxes.shape[0] == 0:
        return []

    order = np.argsort(-scores)[:top_k]
    keep: list[int] = []
    while order.size:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        overlaps = iou_matrix(boxes[best : best + 1], boxes[rest])[0]
        order = rest[overlaps <= iou_threshold]
    return keep


def soft_nms(boxes, scores, sigma: float = 0.5, score_threshold: float = 0.1, top_k: int = 200):
    """Gaussian soft-NMS — decays overlapping scores instead of deleting them.

    Better than hard NMS in crowded frames, where two people genuinely do
    overlap and hard NMS would erase one of them.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).ravel().copy()
    indices = np.arange(len(scores))
    keep: list[int] = []

    while scores.size and len(keep) < top_k:
        best_local = int(np.argmax(scores))
        if scores[best_local] < score_threshold:
            break
        keep.append(int(indices[best_local]))

        mask = np.ones(len(scores), dtype=bool)
        mask[best_local] = False
        if not mask.any():
            break
        overlaps = iou_matrix(boxes[best_local : best_local + 1], boxes[mask])[0]
        scores = scores[mask] * np.exp(-(overlaps ** 2) / sigma)
        boxes = boxes[mask]
        indices = indices[mask]
    return keep


# ---------------------------------------------------------------------------
# Polygons — geofencing
# ---------------------------------------------------------------------------
def point_in_polygon(point, polygon) -> bool:
    """Ray-casting point-in-polygon test (handles concave shapes correctly)."""
    polygon = as_polygon(polygon)
    if polygon.shape[0] < 3:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def points_in_polygon(points, polygon) -> np.ndarray:
    """Vectorised even-odd test for many points at once."""
    polygon = as_polygon(polygon)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if polygon.shape[0] < 3 or points.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool)

    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        straddles = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        inside ^= straddles & (x < x_cross)
        j = i
    return inside


def polygon_area(polygon) -> float:
    """Shoelace area (always positive)."""
    polygon = as_polygon(polygon)
    if polygon.shape[0] < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_centroid(polygon) -> tuple[float, float]:
    polygon = as_polygon(polygon)
    if polygon.shape[0] == 0:
        return (0.0, 0.0)
    if polygon.shape[0] < 3:
        return tuple(polygon.mean(axis=0))
    x, y = polygon[:, 0], polygon[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return tuple(polygon.mean(axis=0))
    cx = float(np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area))
    cy = float(np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area))
    return (cx, cy)


def polygon_bounds(polygon) -> Box:
    polygon = as_polygon(polygon)
    if polygon.shape[0] == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(polygon[:, 0].min()), float(polygon[:, 1].min()),
        float(polygon[:, 0].max()), float(polygon[:, 1].max()),
    )


def box_polygon_overlap(box: Box, polygon) -> float:
    """Fraction of *box* lying inside *polygon*, estimated on a sample grid.

    Exact polygon clipping is overkill here — a 12×12 sample is accurate to
    well under a percent and runs in microseconds per box.
    """
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return 0.0
    xs = np.linspace(x1, x2, 12)
    ys = np.linspace(y1, y2, 12)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return float(np.mean(points_in_polygon(grid, polygon)))


def normalise_polygon(polygon, width: float, height: float) -> list[list[float]]:
    """Absolute pixel coordinates → relative [0, 1], so a zone survives a
    change of stream resolution."""
    polygon = as_polygon(polygon)
    return [[float(p[0] / max(width, 1)), float(p[1] / max(height, 1))] for p in polygon]


def denormalise_polygon(polygon, width: float, height: float) -> np.ndarray:
    polygon = as_polygon(polygon)
    return np.stack([polygon[:, 0] * width, polygon[:, 1] * height], axis=-1)


def line_crossing_direction(previous, current, line) -> int:
    """Which way a track crossed a virtual line: ``+1``, ``-1`` or ``0``.

    Used for entry/exit counting at doors and gates.
    """
    (x1, y1), (x2, y2) = line[0], line[1]

    def side(point):
        return np.sign((x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1))

    before, after = side(previous), side(current)
    if before == 0 or after == 0 or before == after:
        return 0
    return int(after)
