"""Multi-object tracking: Kalman filter + Hungarian assignment.

Detections alone are stateless — they cannot tell you that *the same person*
loitered for four minutes, entered a restricted zone, or picked something up and
walked away with it.  Tracking is what turns per-frame boxes into behaviour.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .geometry import box_center, box_foot, iou_matrix


# ---------------------------------------------------------------------------
# Hungarian algorithm (Jonker-Volgenant style O(n³) implementation)
# ---------------------------------------------------------------------------
def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Optimal assignment minimising total cost.

    Returns a list of ``(row, column)`` pairs.  Implemented here rather than
    pulled from SciPy so the inference worker stays dependency-light.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return []

    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)
    # Pad to a square matrix with zero-cost dummy entries.
    padded = np.zeros((n, n), dtype=np.float64)
    padded[:n_rows, :n_cols] = cost

    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)   # p[j] = row assigned to column j
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                current = padded[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = []
    for j in range(1, n + 1):
        row = p[j] - 1
        col = j - 1
        if 0 <= row < n_rows and 0 <= col < n_cols:
            pairs.append((row, col))
    return sorted(pairs)


# ---------------------------------------------------------------------------
# Kalman filter for a bounding box
# ---------------------------------------------------------------------------
class BoxKalmanFilter:
    """Constant-velocity Kalman filter on ``[cx, cy, area, aspect]``.

    Tracking centre + area rather than raw corners keeps the motion model linear
    while still following a person walking toward or away from the camera.
    """

    def __init__(self, box, process_noise: float = 1.0, measurement_noise: float = 1.0):
        cx, cy = box_center(box)
        width = max(box[2] - box[0], 1.0)
        height = max(box[3] - box[1], 1.0)

        # State: [cx, cy, s, r, vx, vy, vs]
        self.x = np.array([cx, cy, width * height, width / height, 0.0, 0.0, 0.0], dtype=np.float64)

        self.F = np.eye(7)
        for i, j in ((0, 4), (1, 5), (2, 6)):
            self.F[i, j] = 1.0

        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0        # velocity is very uncertain at birth

        self.Q = np.eye(7) * process_noise
        self.Q[4:, 4:] *= 0.01
        self.Q[2, 2] *= 0.01

        self.R = np.eye(4) * measurement_noise
        self.R[2:, 2:] *= 10.0          # area/aspect measurements are noisier

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1e-3)
        return self.to_box()

    def update(self, box) -> None:
        cx, cy = box_center(box)
        width = max(box[2] - box[0], 1.0)
        height = max(box[3] - box[1], 1.0)
        z = np.array([cx, cy, width * height, width / height], dtype=np.float64)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def to_box(self) -> tuple[float, float, float, float]:
        cx, cy, area, aspect = self.x[:4]
        area = max(float(area), 1.0)
        aspect = max(float(aspect), 1e-3)
        width = np.sqrt(area * aspect)
        height = area / max(width, 1e-6)
        return (
            float(cx - width / 2), float(cy - height / 2),
            float(cx + width / 2), float(cy + height / 2),
        )

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self.x[4]), float(self.x[5]))


# ---------------------------------------------------------------------------
# Track & tracker
# ---------------------------------------------------------------------------
@dataclass
class TrackerConfig:
    max_age: int = 30              # frames a track survives without a detection
    min_hits: int = 3              # detections before a track is "confirmed"
    iou_threshold: float = 0.25
    max_distance: float = 220.0    # px — gates absurd associations
    history_length: int = 120
    appearance_weight: float = 0.35
    #: Frames a confirmed track may coast and still be reported to analytics.
    report_grace: int = 3
    #: Proximity-association radius, in multiples of the object's own width.
    #: Covers subjects moving faster than one body-width per frame.
    proximity_widths: float = 2.5


@dataclass
class TrackedObject:
    """A public, serialisable snapshot of a track for the rest of the platform."""

    track_id: int
    box: tuple[float, float, float, float]
    label: str
    confidence: float
    age: int
    hits: int
    time_since_update: int
    velocity: tuple[float, float]
    centroid: tuple[float, float]
    foot: tuple[float, float]
    trail: list[tuple[float, float]] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    max_displacement: float = 0.0

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3

    @property
    def width(self) -> float:
        return max(self.box[2] - self.box[0], 1.0)

    @property
    def has_moved(self) -> bool:
        """Has this track ever actually gone anywhere?

        A real subject walks into view; a background artefact is born where it
        sits and never leaves. This single fact separates the two.
        """
        return self.max_displacement > self.width * 0.5

    @property
    def is_static_scenery(self) -> bool:
        """A long-lived track that has never moved is furniture, not a person.

        Typically a *ghost*: the patch of scene a subject vacated, or an object
        someone repositioned. Excluding these keeps occupancy counts honest.
        """
        return self.age > 45 and self.max_displacement < self.width * 0.25

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.velocity))

    def displacement(self, frames: int = 30) -> float:
        """Straight-line distance travelled over the last *frames* samples.

        Near-zero displacement across a long lifetime is loitering; the
        gesture and theft analytics both key on this.
        """
        if len(self.trail) < 2:
            return 0.0
        recent = self.trail[-max(int(frames), 2):]
        return float(np.hypot(recent[-1][0] - recent[0][0], recent[-1][1] - recent[0][1]))

    def path_length(self, frames: int = 30) -> float:
        """Total distance walked over the last *frames* samples.

        ``displacement / path_length`` gives path straightness: ~1.0 for
        purposeful walking, ~0.0 for milling about on the spot.
        """
        points = self.trail[-max(int(frames), 2):]
        if len(points) < 2:
            return 0.0
        arr = np.asarray(points, dtype=np.float64)
        return float(np.sum(np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))))

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "box": [round(v, 2) for v in self.box],
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "age": self.age,
            "hits": self.hits,
            "velocity": [round(v, 3) for v in self.velocity],
            "speed": round(self.speed, 3),
            "centroid": [round(v, 2) for v in self.centroid],
            "attributes": self.attributes,
            "has_moved": self.has_moved,
        }


class Track:
    """Internal per-object state carried across frames."""

    _next_id = 1

    def __init__(self, box, label: str, confidence: float, config: TrackerConfig, embedding=None):
        self.id = Track._next_id
        Track._next_id += 1

        self.kf = BoxKalmanFilter(box)
        self.label = label
        self.confidence = float(confidence)
        self.config = config

        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.trail: deque = deque(maxlen=config.history_length)
        self.trail.append(box_center(box))
        self.box_history: deque = deque(maxlen=config.history_length)
        self.box_history.append(tuple(box))
        self.embedding = np.asarray(embedding, dtype=np.float32) if embedding is not None else None
        self.attributes: dict = {}
        self.zone_state: dict = {}      # zone_id → dwell frames
        self.first_seen_frame = 0
        self.carried_object_frames = 0
        self.birth_centre = box_center(box)
        #: Greatest distance from the birth position this track has reached.
        #: A ghost is born static and stays there; a real subject walked in.
        self.max_displacement = 0.0

    @classmethod
    def reset_ids(cls) -> None:
        cls._next_id = 1

    def predict(self):
        box = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return box

    def update(self, box, confidence: float, label: str | None = None, embedding=None):
        self.kf.update(box)
        self.hits += 1
        self.time_since_update = 0
        self.confidence = float(confidence)
        if label:
            self.label = label
        centre = box_center(box)
        self.trail.append(centre)
        self.box_history.append(tuple(box))
        self.max_displacement = max(
            self.max_displacement,
            float(np.hypot(centre[0] - self.birth_centre[0], centre[1] - self.birth_centre[1])),
        )
        if embedding is not None:
            new = np.asarray(embedding, dtype=np.float32)
            # Exponential moving average keeps the appearance model stable
            # through partial occlusion.
            self.embedding = new if self.embedding is None else 0.85 * self.embedding + 0.15 * new
            norm = np.linalg.norm(self.embedding)
            if norm > 0:
                self.embedding = self.embedding / norm

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= self.config.min_hits

    @property
    def is_dead(self) -> bool:
        return self.time_since_update > self.config.max_age

    def displacement(self, frames: int = 30) -> float:
        """How far the object actually travelled over the last N frames.

        Near-zero displacement with a long lifetime is loitering.
        """
        if len(self.trail) < 2:
            return 0.0
        recent = list(self.trail)[-frames:]
        return float(np.hypot(recent[-1][0] - recent[0][0], recent[-1][1] - recent[0][1]))

    def path_length(self, frames: int = 30) -> float:
        points = list(self.trail)[-frames:]
        if len(points) < 2:
            return 0.0
        arr = np.asarray(points)
        return float(np.sum(np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))))

    def to_public(self) -> TrackedObject:
        box = self.kf.to_box()
        return TrackedObject(
            track_id=self.id,
            box=box,
            label=self.label,
            confidence=self.confidence,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            velocity=self.kf.velocity,
            centroid=box_center(box),
            foot=box_foot(box),
            trail=[(round(x, 1), round(y, 1)) for x, y in list(self.trail)[-90:]],
            attributes=dict(self.attributes),
            max_displacement=round(self.max_displacement, 2),
        )


class MultiObjectTracker:
    """SORT-style tracker with an optional appearance term (DeepSORT-lite).

    Association cost blends IoU with cosine distance between appearance
    embeddings, which is what lets a track survive someone walking behind a
    pillar and reappearing a second later.
    """

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self.frame_index = 0

    def reset(self) -> None:
        self.tracks = []
        self.frame_index = 0

    def update(self, detections: list[dict]) -> list[TrackedObject]:
        """``detections`` = ``[{"box": (x1,y1,x2,y2), "label": str,
        "confidence": float, "embedding": ndarray|None}, ...]``"""
        self.frame_index += 1

        predicted = []
        for track in self.tracks:
            predicted.append(track.predict())

        matches, unmatched_detections, unmatched_tracks = self._associate(detections, predicted)

        for track_index, detection_index in matches:
            detection = detections[detection_index]
            self.tracks[track_index].update(
                detection["box"],
                float(detection.get("confidence", 1.0)),
                detection.get("label"),
                detection.get("embedding"),
            )
            attributes = detection.get("attributes")
            if attributes:
                self.tracks[track_index].attributes.update(attributes)

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            track = Track(
                detection["box"],
                detection.get("label", "object"),
                float(detection.get("confidence", 1.0)),
                self.config,
                detection.get("embedding"),
            )
            track.first_seen_frame = self.frame_index
            track.attributes.update(detection.get("attributes") or {})
            self.tracks.append(track)

        self.tracks = [track for track in self.tracks if not track.is_dead]
        _ = unmatched_tracks
        # Tracks are *kept* for max_age frames so identity survives an
        # occlusion, but only recently-seen ones are *reported* — otherwise a
        # person who left the frame keeps inflating occupancy counts for
        # several seconds after they are gone.
        return [
            track.to_public()
            for track in self.tracks
            if track.time_since_update == 0
            or (track.is_confirmed and track.time_since_update <= self.config.report_grace)
        ]

    def _associate(self, detections, predicted_boxes):
        if not detections or not self.tracks:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        detection_boxes = [d["box"] for d in detections]
        iou = iou_matrix(predicted_boxes, detection_boxes)

        # Cosine distance between appearance embeddings, where both sides have one.
        appearance = np.zeros_like(iou)
        weight = self.config.appearance_weight
        if weight > 0:
            for i, track in enumerate(self.tracks):
                if track.embedding is None:
                    continue
                for j, detection in enumerate(detections):
                    emb = detection.get("embedding")
                    if emb is None:
                        continue
                    emb = np.asarray(emb, dtype=np.float32)
                    denominator = np.linalg.norm(track.embedding) * np.linalg.norm(emb)
                    if denominator > 0:
                        appearance[i, j] = float(np.dot(track.embedding, emb) / denominator)

        similarity = (1.0 - weight) * iou + weight * appearance
        cost = 1.0 - similarity

        # Gate impossible pairings before the solver ever sees them.
        for i, track_box in enumerate(predicted_boxes):
            tx, ty = box_center(track_box)
            for j, detection_box in enumerate(detection_boxes):
                dx, dy = box_center(detection_box)
                too_far = np.hypot(tx - dx, ty - dy) > self.config.max_distance
                label_mismatch = (
                    self.tracks[i].label != detections[j].get("label", self.tracks[i].label)
                )
                if too_far or label_mismatch or iou[i, j] < self.config.iou_threshold * 0.4:
                    cost[i, j] = 1e6

        pairs = hungarian(cost)
        matches, matched_tracks, matched_detections = [], set(), set()
        for row, col in pairs:
            if cost[row, col] >= 1e5 or iou[row, col] < self.config.iou_threshold:
                continue
            matches.append((row, col))
            matched_tracks.add(row)
            matched_detections.add(col)

        unmatched_detections = [j for j in range(len(detections)) if j not in matched_detections]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]

        # ---- second pass: proximity association --------------------------
        #
        # IoU association silently fails for anything moving faster than its
        # own width per frame — a running person's boxes simply do not overlap
        # between frames, so every frame spawns a fresh track and nothing is
        # ever confirmed. Falling back to centre distance, scaled by the
        # object's own size, recovers exactly those cases.
        if unmatched_tracks and unmatched_detections:
            second_matches = self._associate_by_distance(
                unmatched_tracks, unmatched_detections, predicted_boxes, detection_boxes, detections
            )
            for track_index, detection_index in second_matches:
                matches.append((track_index, detection_index))
                matched_tracks.add(track_index)
                matched_detections.add(detection_index)
            unmatched_detections = [
                j for j in range(len(detections)) if j not in matched_detections
            ]
            unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]

        return matches, unmatched_detections, unmatched_tracks

    def _associate_by_distance(
        self, track_indices, detection_indices, predicted_boxes, detection_boxes, detections
    ):
        """Greedy nearest-centre matching for leftovers, gated by size."""
        candidates = []
        for i in track_indices:
            track = self.tracks[i]
            tx, ty = box_center(predicted_boxes[i])
            track_width = max(predicted_boxes[i][2] - predicted_boxes[i][0], 1.0)
            track_height = max(predicted_boxes[i][3] - predicted_boxes[i][1], 1.0)
            for j in detection_indices:
                if track.label != detections[j].get("label", track.label):
                    continue
                dx, dy = box_center(detection_boxes[j])
                detection_width = max(detection_boxes[j][2] - detection_boxes[j][0], 1.0)
                detection_height = max(detection_boxes[j][3] - detection_boxes[j][1], 1.0)

                # Sizes must be comparable — a head is not a torso.
                size_ratio = min(track_height, detection_height) / max(track_height, detection_height)
                if size_ratio < 0.55:
                    continue

                distance = float(np.hypot(tx - dx, ty - dy))
                limit = min(
                    self.config.max_distance,
                    max(track_width, detection_width) * self.config.proximity_widths,
                )
                if distance <= limit:
                    candidates.append((distance, i, j))

        candidates.sort()
        used_tracks, used_detections, matched = set(), set(), []
        for _, i, j in candidates:
            if i in used_tracks or j in used_detections:
                continue
            used_tracks.add(i)
            used_detections.add(j)
            matched.append((i, j))
        return matched

    @property
    def active_count(self) -> int:
        return sum(1 for track in self.tracks if track.is_confirmed)
