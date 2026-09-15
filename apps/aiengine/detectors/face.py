"""Face detection, enrolment and recognition.

The spec calls for identifying authorised employees, telling employees apart,
and tracking presence and behaviour.  Campy AI does this with its **own**
embedding network (``campyface``), trained with batch-hard triplet loss.

Why an embedding and not a classifier: adding a new employee must not require
retraining.  With an embedding, enrolment is just "average a few crops into a
128-D prototype and store it" — instant, and it scales to thousands of people.

Face *localisation* uses a two-stage cascade:

1. A cheap proposal stage — skin-chroma gating in YCrCb, restricted to the
   upper region of each tracked person, with symmetry and aspect checks.
2. A learned verification stage — the ``campynet_face`` face/not-face
   classifier, when the workspace has trained one.
"""
from __future__ import annotations

import numpy as np

from ..vision.features import lbp_histogram
from ..vision.geometry import non_max_suppression
from ..vision.image import crop, normalise, resize_bilinear, rgb_to_gray, rgb_to_ycrcb, to_chw
from ..vision.motion import binary_close, binary_open, connected_regions
from .base import AnalyticResult, BaseAnalytic, Detection, EventCandidate, Severity


def skin_mask(frame: np.ndarray) -> np.ndarray:
    """YCrCb skin-chroma gate.

    The Cr/Cb bounds below hold across skin tones because chrominance encodes
    hue rather than brightness — the same reason this is the standard basis for
    skin segmentation rather than raw RGB thresholds.
    """
    ycrcb = rgb_to_ycrcb(frame)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    mask = (
        (y > 60) & (y < 250)
        & (cr >= 133) & (cr <= 180)
        & (cb >= 77) & (cb <= 128)
    )
    return binary_close(binary_open(mask, 1), 1)


def propose_faces(frame: np.ndarray, search_box=None, max_faces: int = 8) -> list[tuple]:
    """Candidate face boxes from skin regions, filtered by shape."""
    region = frame if search_box is None else crop(frame, search_box)
    if region.size < 64:
        return []

    mask = skin_mask(region)
    if not mask.any():
        return []

    height, width = mask.shape
    blobs = connected_regions(mask, min_area=max(int(mask.size * 0.004), 12), max_regions=max_faces * 3)

    candidates = []
    for x1, y1, x2, y2 in blobs:
        box_width = max(x2 - x1, 1)
        box_height = max(y2 - y1, 1)
        aspect = box_width / box_height
        # A face is roughly as wide as it is tall (0.65 - 1.3).
        if aspect < 0.6 or aspect > 1.45:
            continue
        fill = float(mask[y1:y2, x1:x2].mean())
        if fill < 0.42:
            continue
        # Faces are broadly left-right symmetric.
        patch = rgb_to_gray(crop(region, (x1, y1, x2, y2)))
        if patch.shape[1] >= 4:
            left = patch[:, : patch.shape[1] // 2]
            right = np.fliplr(patch[:, -(patch.shape[1] // 2) :])
            symmetry = 1.0 - float(np.mean(np.abs(left - right)) / 128.0)
        else:
            symmetry = 0.0
        score = float(np.clip(0.45 * fill + 0.35 * max(symmetry, 0) + 0.20 * min(aspect, 1.0), 0, 1))
        if score < 0.4:
            continue

        offset_x, offset_y = (0, 0) if search_box is None else (search_box[0], search_box[1])
        candidates.append(
            ((x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y), score)
        )

    if not candidates:
        return []
    keep = non_max_suppression([c[0] for c in candidates], [c[1] for c in candidates], 0.4)
    _ = (height, width)
    return [candidates[i] for i in keep][:max_faces]


class FaceGallery:
    """In-memory index of enrolled identities.

    Prototypes are unit-norm, so a matrix-vector product gives cosine similarity
    to every enrolled person at once — matching thousands of employees costs
    microseconds.
    """

    def __init__(self, entries: list[dict] | None = None):
        self.ids: list[int] = []
        self.names: list[str] = []
        self.roles: list[str] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        if entries:
            self.load(entries)

    def load(self, entries: list[dict]) -> None:
        vectors, ids, names, roles = [], [], [], []
        for entry in entries:
            vector = np.asarray(entry.get("embedding") or [], dtype=np.float32).ravel()
            if vector.size == 0:
                continue
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            vectors.append(vector / norm)
            ids.append(int(entry.get("id", 0)))
            names.append(str(entry.get("name", "Unknown")))
            roles.append(str(entry.get("role", "")))
        self.matrix = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, 0), dtype=np.float32)
        self.ids, self.names, self.roles = ids, names, roles

    def __len__(self) -> int:
        return len(self.ids)

    def match(self, embedding: np.ndarray, threshold: float = 0.62):
        """Return ``(id, name, role, similarity)`` or ``None``."""
        if self.matrix.size == 0 or embedding is None:
            return None
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(vector)
        if norm == 0 or vector.shape[0] != self.matrix.shape[1]:
            return None
        similarities = self.matrix @ (vector / norm)
        best = int(np.argmax(similarities))
        score = float(similarities[best])
        if score < threshold:
            return None
        return self.ids[best], self.names[best], self.roles[best], score


class FaceRecognitionAnalytic(BaseAnalytic):
    key = "face"
    label = "Face Recognition"
    feature = "face_recognition"

    defaults = {
        "enabled": True,
        "match_threshold": 0.62,
        "min_face_size": 22,           # px — smaller faces are not reliable
        "search_upper_ratio": 0.45,    # search the top 45% of a person box
        "alert_unknown": True,
        "unknown_frames": 12,
        "cooldown_frames": 400,
        "max_faces_per_frame": 8,
        "verify_with_model": True,
        "verify_threshold": 0.5,
        "attendance": True,
    }

    def _embed(self, context, patch: np.ndarray) -> np.ndarray | None:
        """Embed a face crop with campyface, or fall back to an LBP signature."""
        model = context.registry.get("campyface") if context.registry else None
        if model is not None:
            meta = getattr(model, "meta", {}) or {}
            size = int(meta.get("input_size", 96))
            batch = to_chw(normalise(resize_bilinear(patch, size, size)))[None, ...]
            vector = np.asarray(model.predict(batch)[0], dtype=np.float32)
            norm = np.linalg.norm(vector)
            return vector / norm if norm else vector
        # Deterministic fallback so tracking/attendance still work pre-training.
        histogram = lbp_histogram(resize_bilinear(patch, 64, 64), bins=64)
        norm = np.linalg.norm(histogram)
        return (histogram / norm).astype(np.float32) if norm else histogram

    def _verify(self, context, patch: np.ndarray) -> float:
        if not self.option("verify_with_model") or context.registry is None:
            return 1.0
        model = context.registry.get("campynet_face")
        if model is None:
            return 1.0
        meta = getattr(model, "meta", {}) or {}
        size = int(meta.get("input_size", 96))
        classes = meta.get("classes") or ["not_face", "face"]
        from ..nn.functional import softmax

        batch = to_chw(normalise(resize_bilinear(patch, size, size)))[None, ...]
        probabilities = softmax(model.predict(batch), axis=-1)[0]
        index = classes.index("face") if "face" in classes else 1
        return float(probabilities[index]) if index < len(probabilities) else 1.0

    def analyse(self, context) -> AnalyticResult:
        result = AnalyticResult(analytic=self.key)
        gallery: FaceGallery = (context.config or {}).get("_gallery") or FaceGallery()
        scratch = context.scratch(self.key)
        identity_cache: dict = scratch.setdefault("identities", {})
        seen_today: set = scratch.setdefault("seen", set())

        people = context.people()
        # Fall back to scanning the whole frame when no person track is available.
        search_targets = people if people else [None]

        recognised, unknown = 0, 0
        matched_names: list[str] = []

        for track in search_targets:
            if track is not None:
                x1, y1, x2, y2 = track.box
                upper = y1 + (y2 - y1) * self.option("search_upper_ratio")
                search_box = (x1, y1, x2, upper)
            else:
                search_box = None

            proposals = propose_faces(
                context.frame, search_box, max_faces=self.option("max_faces_per_frame")
            )

            for box, proposal_score in proposals:
                if (box[2] - box[0]) < self.option("min_face_size"):
                    continue
                patch = crop(context.frame, box, padding=0.12)
                if patch.size < 64:
                    continue

                face_score = self._verify(context, patch)
                if face_score < self.option("verify_threshold"):
                    continue

                embedding = self._embed(context, patch)
                match = gallery.match(embedding, threshold=self.option("match_threshold"))

                if match:
                    employee_id, name, role, similarity = match
                    recognised += 1
                    matched_names.append(name)
                    if track is not None:
                        track.attributes["identity"] = name
                        track.attributes["employee_id"] = employee_id
                        track.attributes["employee_role"] = role
                        track.attributes["identity_confidence"] = round(similarity, 3)
                        identity_cache[track.track_id] = (employee_id, name, role)

                    result.detections.append(
                        Detection(
                            label=f"face:{name}",
                            confidence=float(similarity),
                            box=box,
                            track_id=track.track_id if track else None,
                            embedding=embedding,
                            attributes={
                                "employee_id": employee_id,
                                "name": name,
                                "role": role,
                                "similarity": round(float(similarity), 4),
                                "proposal_score": round(float(proposal_score), 3),
                            },
                        )
                    )

                    if self.option("attendance") and employee_id not in seen_today:
                        seen_today.add(employee_id)
                        result.events.append(
                            EventCandidate(
                                analytic=self.key,
                                event_type="employee_present",
                                severity=Severity.INFO,
                                confidence=float(similarity),
                                title=f"{name} identified",
                                description=f"{name} was recognised on this camera.",
                                box=box,
                                track_id=track.track_id if track else None,
                                subject_id=employee_id,
                                metadata={"name": name, "role": role, "similarity": round(float(similarity), 4)},
                                dedupe_key=f"face:present:{employee_id}",
                            )
                        )
                else:
                    unknown += 1
                    if track is not None:
                        track.attributes.setdefault("identity", None)
                    result.detections.append(
                        Detection(
                            label="face:unknown",
                            confidence=float(proposal_score),
                            box=box,
                            track_id=track.track_id if track else None,
                            embedding=embedding,
                            attributes={"proposal_score": round(float(proposal_score), 3)},
                        )
                    )

                    if self.option("alert_unknown") and len(gallery) and self.sustained(
                        context,
                        f"unknown:{track.track_id if track else 'frame'}",
                        True,
                        required_frames=self.option("unknown_frames"),
                        cooldown_frames=self.option("cooldown_frames"),
                    ):
                        result.events.append(
                            EventCandidate(
                                analytic=self.key,
                                event_type="unknown_person",
                                severity=Severity.MEDIUM,
                                confidence=float(proposal_score),
                                title="Unrecognised person",
                                description=(
                                    "A person was observed whose face does not match any "
                                    "enrolled employee."
                                ),
                                box=box,
                                track_id=track.track_id if track else None,
                                metadata={"gallery_size": len(gallery)},
                            )
                        )

        result.metrics = {
            "faces": recognised + unknown,
            "recognised": recognised,
            "unknown": unknown,
            "gallery_size": len(gallery),
            "identities": sorted(set(matched_names)),
            "model": "campyface" if (context.registry and context.registry.get("campyface")) else "lbp-fallback",
        }
        return result
