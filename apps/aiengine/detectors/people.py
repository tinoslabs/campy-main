"""Person and object detection — the front of every analytic pipeline.

Two paths, chosen automatically:

1. **Learned** — when the workspace has a trained ``campydet`` model deployed,
   we run it and decode the grid.
2. **Motion-derived** — otherwise we propose regions from the background model
   and score them with HOG-based human-shape scoring.  This is what lets a new
   customer point Campy AI at an existing CCTV camera and get useful people
   tracking on day one, before any training has happened.
"""
from __future__ import annotations

import numpy as np

from ..vision.features import (
    boundary_gradient_support,
    descriptor_bundle,
    histogram_of_oriented_gradients,
)
from ..vision.geometry import box_area, non_max_suppression
from ..vision.image import crop, letterbox, normalise, resize_bilinear, to_chw
from .base import Detection

# Typical standing-person aspect ratio (width / height) sits around 0.35-0.55.
PERSON_ASPECT_RANGE = (0.18, 0.95)
PERSON_MIN_AREA_RATIO = 0.0012
PERSON_MAX_AREA_RATIO = 0.75


def decode_detector_output(
    output: np.ndarray,
    class_names: list[str],
    width: int,
    height: int,
    confidence_threshold: float = 0.4,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Decode ``campydet`` grid output into absolute-pixel detections.

    Channel layout per cell: ``[tx, ty, tw, th, objectness, *class_logits]``.
    ``tx/ty`` are sigmoid offsets within the cell; ``tw/th`` are log-space
    sizes relative to the frame.
    """
    from ..nn.functional import sigmoid, softmax

    grid_output = np.asarray(output)
    if grid_output.ndim == 4:
        grid_output = grid_output[0]
    channels, grid_h, grid_w = grid_output.shape

    tx = sigmoid(grid_output[0])
    ty = sigmoid(grid_output[1])
    tw = np.clip(grid_output[2], -6, 3)
    th = np.clip(grid_output[3], -6, 3)
    objectness = sigmoid(grid_output[4])

    if channels > 5:
        class_logits = grid_output[5:].reshape(channels - 5, -1).T
        class_scores = softmax(class_logits, axis=-1)
        class_ids = np.argmax(class_scores, axis=-1)
        class_confidence = class_scores[np.arange(class_scores.shape[0]), class_ids]
    else:
        class_ids = np.zeros(grid_h * grid_w, dtype=int)
        class_confidence = np.ones(grid_h * grid_w, dtype=np.float32)

    boxes, scores, labels = [], [], []
    flat_objectness = objectness.ravel()
    for index in range(grid_h * grid_w):
        score = float(flat_objectness[index] * class_confidence[index])
        if score < confidence_threshold:
            continue
        row, col = divmod(index, grid_w)
        cx = (col + float(tx.ravel()[index])) / grid_w * width
        cy = (row + float(ty.ravel()[index])) / grid_h * height
        bw = float(np.exp(tw.ravel()[index])) * width / grid_w * 2.0
        bh = float(np.exp(th.ravel()[index])) * height / grid_h * 2.0
        boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        scores.append(score)
        class_index = int(class_ids[index])
        labels.append(class_names[class_index] if class_index < len(class_names) else "object")

    if not boxes:
        return []

    keep = non_max_suppression(boxes, scores, iou_threshold=iou_threshold)
    return [
        Detection(
            label=labels[i],
            confidence=scores[i],
            box=(
                max(boxes[i][0], 0), max(boxes[i][1], 0),
                min(boxes[i][2], width), min(boxes[i][3], height),
            ),
        )
        for i in keep
    ]


def human_shape_score(patch: np.ndarray, box) -> float:
    """Heuristic 0-1 score that a motion blob is a standing person.

    Combines aspect ratio, vertical-gradient dominance (torso and legs produce
    strong vertical edges) and vertical mass distribution (a person is wider at
    the shoulders than at the feet).  Deliberately conservative — the tracker
    filters the rest.
    """
    patch = np.asarray(patch, dtype=np.float32)
    if patch.size < 64:
        return 0.0

    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    aspect = width / height

    low, high = PERSON_ASPECT_RANGE
    if aspect < low or aspect > high:
        return 0.0
    ideal = 0.42
    aspect_score = float(np.exp(-((aspect - ideal) ** 2) / (2 * 0.22 ** 2)))

    descriptor = histogram_of_oriented_gradients(
        resize_bilinear(patch, 32, 16), cell_size=8, bins=9
    )
    if descriptor.size < 9:
        return aspect_score * 0.5
    # Bins around 90° are vertical structure (upright limbs and torso).
    orientation_energy = descriptor.reshape(-1, 9).mean(axis=0)
    total = float(orientation_energy.sum()) or 1.0
    vertical_ratio = float(orientation_energy[3:6].sum() / total)
    vertical_score = float(np.clip((vertical_ratio - 0.20) / 0.30, 0, 1))

    return float(np.clip(0.55 * aspect_score + 0.45 * vertical_score, 0, 1))


def propose_from_motion(
    frame: np.ndarray,
    motion,
    min_score: float = 0.35,
    max_detections: int = 30,
    min_boundary_support: float = 0.28,
) -> list[Detection]:
    """Turn motion blobs into scored person/object candidates.

    Blobs whose outline has no image-edge support are discarded as ghosts —
    see :func:`~apps.aiengine.vision.features.boundary_gradient_support`.
    """
    if motion is None or not getattr(motion, "regions", None):
        return []

    height, width = np.asarray(frame).shape[:2]
    frame_area = float(height * width)
    candidates: list[Detection] = []

    for region in motion.regions[:max_detections]:
        x1, y1, x2, y2 = region
        area = box_area((x1, y1, x2, y2))
        ratio = area / frame_area
        if ratio < PERSON_MIN_AREA_RATIO or ratio > PERSON_MAX_AREA_RATIO:
            continue

        support = boundary_gradient_support(frame, (x1, y1, x2, y2))
        if support < min_boundary_support:
            continue

        patch = crop(frame, (x1, y1, x2, y2))
        score = human_shape_score(patch, (x1, y1, x2, y2))
        box_width = max(x2 - x1, 1)
        box_height = max(y2 - y1, 1)
        aspect = box_width / box_height

        if score >= min_score:
            label, confidence = "person", score
        else:
            # Not person-shaped, but something is moving there.
            label = "object"
            confidence = float(np.clip(0.35 + ratio * 2.0, 0.3, 0.8))
            if aspect > 1.6:
                label = "object"

        candidates.append(
            Detection(
                label=label,
                confidence=float(confidence),
                box=(float(x1), float(y1), float(x2), float(y2)),
                embedding=descriptor_bundle(patch),
                attributes={
                    "source": "motion",
                    "aspect": round(float(aspect), 3),
                    "area_ratio": round(float(ratio), 5),
                    "shape_score": round(float(score), 3),
                    "boundary_support": round(float(support), 3),
                },
            )
        )

    if not candidates:
        return []
    keep = non_max_suppression(
        [c.box for c in candidates], [c.confidence for c in candidates], iou_threshold=0.5
    )
    return [candidates[i] for i in keep]


def detect_people_and_objects(context) -> list[Detection]:
    """Entry point used by the pipeline: learned model first, motion fallback."""
    frame = context.frame
    height, width = np.asarray(frame).shape[:2]
    registry = context.registry

    model = registry.get("campydet") if registry is not None else None
    if model is not None:
        meta = getattr(model, "meta", {}) or {}
        size = int(meta.get("input_size", 96))
        class_names = meta.get("classes") or ["person", "object"]
        letterboxed, scale, pad_x, pad_y = letterbox(frame, size, size)
        batch = to_chw(normalise(letterboxed))[None, ...]
        output = model.predict(batch)
        detections = decode_detector_output(
            output, class_names, size, size,
            confidence_threshold=float(context.config.get("detector_threshold", 0.4)),
        )
        # Undo the letterbox so boxes land back on the original frame.
        restored: list[Detection] = []
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            box = (
                (x1 - pad_x) / scale, (y1 - pad_y) / scale,
                (x2 - pad_x) / scale, (y2 - pad_y) / scale,
            )
            box = (
                float(np.clip(box[0], 0, width)), float(np.clip(box[1], 0, height)),
                float(np.clip(box[2], 0, width)), float(np.clip(box[3], 0, height)),
            )
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            detection.box = box
            detection.embedding = descriptor_bundle(crop(frame, box))
            detection.attributes["source"] = "campydet"
            restored.append(detection)
        if restored:
            return restored

    return propose_from_motion(
        frame,
        context.motion,
        min_score=float(context.config.get("person_min_score", 0.35)),
        min_boundary_support=float(context.config.get("min_boundary_support", 0.28)),
    )
