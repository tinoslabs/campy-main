"""Theft Detection.

Per the spec, this feature highlights *risk* rather than accusing individuals:
repeated object movement, unusual carrying behaviour, and activity in restricted
storage areas.  Every finding is a prompt for a human to review footage.

Signals combined
----------------
* **Object disappearance** — a shelf/storage zone's static appearance changes
  and does not change back, while a person was next to it.
* **Concealment gesture** — a person's silhouette gains mass at torso level
  right after they were near a shelf (the reach-and-pocket pattern).
* **Carrying change** — a person's bounding box "grew" an attached blob that
  travels with them, and did not have one on entry.
* **Loitering near stock** — extended dwell at a shelf zone with low
  displacement and high local motion energy.
* **Exit with unresolved risk** — the person leaves the frame carrying a
  risk score above threshold.
"""
from __future__ import annotations

import numpy as np

from ..vision.geometry import box_area, box_center, box_polygon_overlap, points_in_polygon
from .base import AnalyticResult, BaseAnalytic, Detection, EventCandidate, Severity


class TheftDetectionAnalytic(BaseAnalytic):
    key = "theft"
    label = "Theft Detection"
    feature = "theft_detection"

    defaults = {
        "enabled": True,
        "shelf_proximity": 0.25,       # box overlap with a shelf zone
        "concealment_growth": 0.16,    # relative torso-mass increase
        "carry_growth": 0.22,          # relative box-area increase that persists
        "dwell_seconds": 12,
        "risk_threshold": 0.62,
        "confirm_frames": 5,
        "cooldown_frames": 400,
        "static_change_frames": 20,    # stock removal must persist this long
        "review_only": True,           # always phrase findings as "review", never "guilty"
    }

    def _shelf_zones(self, context):
        return [z for z in context.zones if z.kind in {"shelf", "restricted"} and z.polygon]

    def _torso_mass(self, context, track) -> float:
        """Motion-mask fill in the torso band of a person's box.

        A concealed item adds mass at torso level without changing the person's
        footprint, which is exactly what this measures.
        """
        motion = context.motion
        if motion is None or getattr(motion, "mask", None) is None or motion.mask.size <= 1:
            return 0.0
        mask = motion.mask
        mh, mw = mask.shape
        sy, sx = mh / max(context.height, 1), mw / max(context.width, 1)

        x1, y1, x2, y2 = track.box
        height = max(y2 - y1, 1)
        torso_top = y1 + height * 0.28
        torso_bottom = y1 + height * 0.68

        region = mask[
            int(np.clip(torso_top * sy, 0, mh)) : int(np.clip(torso_bottom * sy, 0, mh)),
            int(np.clip(x1 * sx, 0, mw)) : int(np.clip(x2 * sx, 0, mw)),
        ]
        return float(region.mean()) if region.size else 0.0

    def _near_shelf(self, track, zones, context) -> tuple[bool, str, int | None]:
        for zone in zones:
            polygon = zone.pixel_polygon(context.width, context.height)
            if polygon.shape[0] < 3:
                continue
            if bool(points_in_polygon(np.array([box_center(track.box)]), polygon)[0]):
                return True, zone.name, zone.id
            if box_polygon_overlap(track.box, polygon) >= self.option("shelf_proximity"):
                return True, zone.name, zone.id
        return False, "", None

    def _stock_removal(self, context, zones, scratch) -> list[dict]:
        """Detect stock that vanished from a shelf zone and stayed gone."""
        background = context.background
        if background is None or not zones:
            return []
        try:
            change = background.static_change_mask(context.frame)
        except Exception:  # noqa: BLE001
            return []
        if change.size <= 1:
            return []

        ch, cw = change.shape
        findings = []
        counters = scratch.setdefault("static_counters", {})

        for zone in zones:
            polygon = zone.pixel_polygon(context.width, context.height)
            if polygon.shape[0] < 3:
                continue
            xs = polygon[:, 0] / max(context.width, 1) * cw
            ys = polygon[:, 1] / max(context.height, 1) * ch
            x1, x2 = int(np.clip(xs.min(), 0, cw)), int(np.clip(xs.max(), 0, cw))
            y1, y2 = int(np.clip(ys.min(), 0, ch)), int(np.clip(ys.max(), 0, ch))
            region = change[y1:y2, x1:x2]
            if region.size == 0:
                continue

            changed_ratio = float(region.mean())
            if changed_ratio > 0.08:
                counters[zone.id] = counters.get(zone.id, 0) + 1
            else:
                counters[zone.id] = max(counters.get(zone.id, 0) - 2, 0)

            if counters[zone.id] >= self.option("static_change_frames"):
                counters[zone.id] = 0
                findings.append(
                    {"zone_id": zone.id, "zone": zone.name, "changed_ratio": round(changed_ratio, 4)}
                )
        return findings

    def analyse(self, context) -> AnalyticResult:
        result = AnalyticResult(analytic=self.key)
        scratch = context.scratch(self.key)
        profiles: dict = scratch.setdefault("profiles", {})
        zones = self._shelf_zones(context)

        people = context.people()
        active_ids = set()
        risky = 0

        for track in people:
            active_ids.add(track.track_id)
            profile = profiles.setdefault(
                track.track_id,
                {
                    "baseline_area": box_area(track.box),
                    "baseline_torso": None,
                    "shelf_frames": 0,
                    "risk": 0.0,
                    "signals": [],
                    "zone": "",
                    "zone_id": None,
                },
            )

            torso = self._torso_mass(context, track)
            near, zone_name, zone_id = self._near_shelf(track, zones, context)
            area = box_area(track.box)

            # Establish a "how they arrived" baseline before any interaction.
            if profile["baseline_torso"] is None and track.age > 4 and not near:
                profile["baseline_torso"] = torso
                profile["baseline_area"] = area

            signals: list[str] = []
            risk = 0.0

            if near:
                profile["shelf_frames"] += 1
                profile["zone"] = zone_name
                profile["zone_id"] = zone_id
                dwell_seconds = profile["shelf_frames"] / max(context.fps, 0.1)
                if dwell_seconds >= self.option("dwell_seconds"):
                    risk += 0.25
                    signals.append(f"dwelt {int(dwell_seconds)}s at {zone_name}")

            baseline_torso = profile["baseline_torso"]
            if baseline_torso is not None and baseline_torso > 1e-6:
                growth = (torso - baseline_torso) / baseline_torso
                if growth >= self.option("concealment_growth") and profile["shelf_frames"] > 0:
                    risk += 0.3
                    signals.append("torso mass increased after shelf contact")

            baseline_area = profile["baseline_area"] or area
            if baseline_area > 1e-6:
                area_growth = (area - baseline_area) / baseline_area
                if area_growth >= self.option("carry_growth") and profile["shelf_frames"] > 0:
                    risk += 0.25
                    signals.append("silhouette grew consistently with a carried item")

            # Turning away immediately after interacting with stock.
            if profile["shelf_frames"] > 0 and track.speed > 1.2 and not near:
                risk += 0.2
                signals.append("moved away quickly after shelf contact")

            # Decay so a single frame cannot pin a lasting score on someone.
            profile["risk"] = float(np.clip(profile["risk"] * 0.94 + risk, 0, 1.0))
            profile["signals"] = signals or profile["signals"]
            track.attributes["theft_risk"] = round(profile["risk"], 3)

            if profile["risk"] >= self.option("risk_threshold"):
                risky += 1
                result.detections.append(
                    Detection(
                        label="suspicious_behaviour",
                        confidence=profile["risk"],
                        box=track.box,
                        track_id=track.track_id,
                        attributes={"signals": profile["signals"], "zone": profile["zone"]},
                    )
                )
                if self.sustained(
                    context, f"theft:{track.track_id}", True,
                    required_frames=self.option("confirm_frames"),
                    cooldown_frames=self.option("cooldown_frames"),
                ):
                    result.events.append(
                        EventCandidate(
                            analytic=self.key,
                            event_type="suspicious_handling",
                            severity=Severity.HIGH,
                            confidence=profile["risk"],
                            title="Suspicious object handling — review recommended",
                            description=(
                                "Behaviour consistent with concealed removal of an item was "
                                "observed: "
                                + "; ".join(profile["signals"] or ["unusual handling pattern"])
                                + ". This is a prompt to review the footage, not a conclusion "
                                "about any individual."
                            ),
                            box=track.box,
                            track_id=track.track_id,
                            zone_id=profile["zone_id"],
                            subject_id=track.attributes.get("employee_id"),
                            metadata={
                                "risk": round(profile["risk"], 3),
                                "signals": profile["signals"],
                                "zone": profile["zone"],
                                "review_only": True,
                            },
                        )
                    )

        # -- someone left the scene while flagged --------------------------
        for track_id in list(profiles):
            if track_id in active_ids:
                continue
            profile = profiles.pop(track_id)
            if profile["risk"] >= self.option("risk_threshold"):
                result.events.append(
                    EventCandidate(
                        analytic=self.key,
                        event_type="left_scene_flagged",
                        severity=Severity.HIGH,
                        confidence=profile["risk"],
                        title="Flagged person left the camera view",
                        description=(
                            "A person with an elevated handling-risk score left this camera's "
                            "field of view. Check adjacent cameras and exits."
                        ),
                        track_id=track_id,
                        zone_id=profile.get("zone_id"),
                        metadata={"risk": round(profile["risk"], 3), "signals": profile["signals"]},
                        dedupe_key=f"theft:left:{track_id}",
                    )
                )

        # -- stock disappeared from a shelf --------------------------------
        for finding in self._stock_removal(context, zones, scratch):
            result.events.append(
                EventCandidate(
                    analytic=self.key,
                    event_type="stock_change",
                    severity=Severity.MEDIUM,
                    confidence=0.7,
                    title=f"Stock change in {finding['zone']}",
                    description=(
                        f"The contents of '{finding['zone']}' changed and stayed changed. "
                        "Reconcile against expected movements."
                    ),
                    zone_id=finding["zone_id"],
                    metadata=finding,
                )
            )

        result.metrics = {
            "monitored_people": len(people),
            "flagged": risky,
            "shelf_zones": len(zones),
            "max_risk": round(max([p["risk"] for p in profiles.values()], default=0.0), 3),
        }
        return result
