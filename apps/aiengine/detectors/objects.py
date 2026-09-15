"""Detection of unwanted, dangerous or abandoned objects.

Identifies items that should not be present in an area — sharp tools, restricted
equipment, hazardous materials — and items left unattended.

Two complementary mechanisms:

* **Classified objects** — the ``campydet`` detector, trained on the customer's
  own object classes, flags anything on the workspace's watch-list.
* **Abandoned objects** — geometry, not classification.  A blob appears, stops
  moving, persists against the slow background model, and no person remains
  within a proximity radius of it.  That is an unattended item regardless of
  what it is, which matters because the dangerous thing is usually the one you
  never trained a class for.
"""
from __future__ import annotations

import numpy as np

from ..vision.geometry import box_area, box_center, box_iou, points_in_polygon
from ..vision.image import crop, normalise, resize_bilinear, to_chw
from ..vision.motion import connected_regions
from .base import AnalyticResult, BaseAnalytic, Detection, EventCandidate, Severity

DEFAULT_WATCHLIST = {
    "knife": Severity.CRITICAL,
    "weapon": Severity.CRITICAL,
    "gun": Severity.CRITICAL,
    "sharp_tool": Severity.HIGH,
    "chemical": Severity.HIGH,
    "gas_cylinder": Severity.HIGH,
    "ladder": Severity.MEDIUM,
    "spill": Severity.MEDIUM,
    "cable": Severity.LOW,
    "box": Severity.LOW,
    "bag": Severity.MEDIUM,
}


class DangerousObjectAnalytic(BaseAnalytic):
    key = "object"
    label = "Unwanted & Dangerous Objects"
    feature = "object_detection"

    defaults = {
        "enabled": True,
        "watchlist": list(DEFAULT_WATCHLIST),
        "detector_threshold": 0.5,
        # Abandoned-object rules
        "abandon_seconds": 45,
        "abandon_proximity": 2.4,       # multiples of the object's own width
        "abandon_min_area_ratio": 0.0008,
        "abandon_max_area_ratio": 0.18,
        "static_tolerance": 6.0,        # px of centre drift still counted as static
        "confirm_frames": 4,
        "cooldown_frames": 400,
        "classify_abandoned": True,
    }

    # -- learned classification ------------------------------------------
    def _classify_patch(self, context, box) -> tuple[str, float] | None:
        if not self.option("classify_abandoned") or context.registry is None:
            return None
        model = context.registry.get("campynet_object")
        if model is None:
            return None
        meta = getattr(model, "meta", {}) or {}
        classes = meta.get("classes") or []
        if not classes:
            return None
        size = int(meta.get("input_size", 96))
        patch = crop(context.frame, box, padding=0.1)
        if patch.size < 32:
            return None
        from ..nn.functional import softmax

        batch = to_chw(normalise(resize_bilinear(patch, size, size)))[None, ...]
        probabilities = softmax(model.predict(batch), axis=-1)[0]
        index = int(np.argmax(probabilities))
        return classes[index], float(probabilities[index])

    def _severity_for(self, label: str) -> str:
        overrides = self.option("severity_overrides") or {}
        if label in overrides:
            return overrides[label]
        return DEFAULT_WATCHLIST.get(label, Severity.MEDIUM)

    def _zone_for(self, context, box) -> tuple[int | None, str]:
        centre = np.array([box_center(box)])
        for zone in context.zones:
            polygon = zone.pixel_polygon(context.width, context.height)
            if polygon.shape[0] >= 3 and bool(points_in_polygon(centre, polygon)[0]):
                return zone.id, zone.name
        return None, ""

    # -- main --------------------------------------------------------------
    def analyse(self, context) -> AnalyticResult:
        result = AnalyticResult(analytic=self.key)
        watchlist = set(self.option("watchlist") or [])
        scratch = context.scratch(self.key)
        candidates: dict = scratch.setdefault("candidates", {})

        people = context.people()
        person_boxes = [t.box for t in people]

        # ---------------- 1. classified watch-list objects ----------------
        flagged = 0
        for detection in context.detections:
            label = detection.label
            if label in {"person", "object"} or label not in watchlist:
                continue
            if detection.confidence < self.option("detector_threshold"):
                continue
            flagged += 1
            severity = self._severity_for(label)
            zone_id, zone_name = self._zone_for(context, detection.box)

            result.detections.append(
                Detection(
                    label=label,
                    confidence=detection.confidence,
                    box=detection.box,
                    attributes={"zone": zone_name, "watchlisted": True},
                )
            )
            if self.sustained(
                context, f"watch:{label}:{zone_id}", True,
                required_frames=self.option("confirm_frames"),
                cooldown_frames=self.option("cooldown_frames"),
            ):
                result.events.append(
                    EventCandidate(
                        analytic=self.key,
                        event_type="dangerous_object",
                        severity=severity,
                        confidence=detection.confidence,
                        title=f"{label.replace('_', ' ').title()} detected",
                        description=(
                            f"A '{label.replace('_', ' ')}' was detected"
                            + (f" in {zone_name}" if zone_name else "")
                            + ". Confirm it belongs there and remove it if not."
                        ),
                        box=detection.box,
                        zone_id=zone_id,
                        metadata={"label": label, "zone": zone_name},
                    )
                )

        # ---------------- 2. abandoned / unattended objects ---------------
        frame_area = float(context.width * context.height)
        motion = context.motion
        static_regions = []

        if context.background is not None:
            try:
                change_mask = context.background.static_change_mask(context.frame)
            except Exception:  # noqa: BLE001
                change_mask = None
            if change_mask is not None and change_mask.size > 1:
                ch, cw = change_mask.shape
                sy, sx = context.height / ch, context.width / cw
                for x1, y1, x2, y2 in connected_regions(
                    change_mask, min_area=max(int(change_mask.size * 0.0006), 6), max_regions=20
                ):
                    box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
                    ratio = box_area(box) / frame_area
                    if not (self.option("abandon_min_area_ratio") <= ratio <= self.option("abandon_max_area_ratio")):
                        continue
                    static_regions.append(box)

        # Track each static region across frames.
        matched_keys = set()
        for box in static_regions:
            centre = box_center(box)
            key = None
            for existing_key, record in candidates.items():
                if box_iou(box, record["box"]) > 0.4 or (
                    abs(centre[0] - record["centre"][0]) < self.option("static_tolerance")
                    and abs(centre[1] - record["centre"][1]) < self.option("static_tolerance")
                ):
                    key = existing_key
                    break
            if key is None:
                key = f"obj{context.frame_index}_{int(centre[0])}_{int(centre[1])}"
                candidates[key] = {
                    "box": box, "centre": centre, "frames": 0,
                    "first_frame": context.frame_index, "alerted": False,
                }
            matched_keys.add(key)

            record = candidates[key]
            record["box"] = box
            record["centre"] = centre
            record["frames"] += 1

            # Is anyone standing near it?
            width = max(box[2] - box[0], 1.0)
            radius = width * self.option("abandon_proximity")
            attended = any(
                np.hypot(centre[0] - box_center(pb)[0], centre[1] - box_center(pb)[1]) < radius
                for pb in person_boxes
            )
            if attended:
                record["frames"] = max(record["frames"] - 3, 0)
                continue

            unattended_seconds = record["frames"] / max(context.fps, 0.1)
            if unattended_seconds >= self.option("abandon_seconds") and not record["alerted"]:
                record["alerted"] = True
                classification = self._classify_patch(context, box)
                label = classification[0] if classification else "unattended_object"
                confidence = classification[1] if classification else 0.65
                severity = self._severity_for(label) if classification else Severity.MEDIUM
                if label in watchlist:
                    severity = Severity.max(severity, Severity.HIGH)
                zone_id, zone_name = self._zone_for(context, box)

                result.detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        box=box,
                        attributes={"unattended_seconds": round(unattended_seconds, 1), "zone": zone_name},
                    )
                )
                result.events.append(
                    EventCandidate(
                        analytic=self.key,
                        event_type="unattended_object",
                        severity=severity,
                        confidence=confidence,
                        title="Unattended object",
                        description=(
                            f"An object has been left unattended for about "
                            f"{int(unattended_seconds)} seconds"
                            + (f" in {zone_name}" if zone_name else "")
                            + (f" (classified as '{label}')" if classification else "")
                            + ". Unattended items are both a trip hazard and a security concern."
                        ),
                        box=box,
                        zone_id=zone_id,
                        metadata={
                            "seconds": round(unattended_seconds, 1),
                            "label": label,
                            "classified": bool(classification),
                        },
                        dedupe_key=f"object:abandoned:{key}",
                    )
                )

        # Forget candidates that are no longer present.
        for key in list(candidates):
            if key not in matched_keys:
                candidates[key]["frames"] -= 2
                if candidates[key]["frames"] <= 0:
                    candidates.pop(key)

        result.metrics = {
            "watchlist_size": len(watchlist),
            "flagged_objects": flagged,
            "static_candidates": len(candidates),
            "unattended_alerts": sum(1 for c in candidates.values() if c["alerted"]),
        }
        _ = motion
        return result
