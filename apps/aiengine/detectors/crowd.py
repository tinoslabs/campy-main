"""Crowd Management — how many people gather where, and how they are moving.

Overcrowding in corridors, entrances, exits and common areas creates safety
risk and slows everyone down.  This analytic counts people, measures local
density, and — crucially — measures *flow*: a packed corridor where everyone is
moving is normal; a packed corridor where nobody is moving is a jam.
"""
from __future__ import annotations

import numpy as np

from ..vision.geometry import box_center, points_in_polygon
from ..vision.image import normalise, resize_bilinear, to_chw
from ..vision.motion import flow_statistics
from .base import AnalyticResult, BaseAnalytic, EventCandidate, Severity


class CrowdManagementAnalytic(BaseAnalytic):
    key = "crowd"
    label = "Crowd Management"
    feature = "crowd_management"

    defaults = {
        "enabled": True,
        "crowd_threshold": 8,          # people in view before it counts as a crowd
        "density_threshold": 0.00035,  # people per pixel of frame
        "sustain_frames": 8,
        "cooldown_frames": 300,
        "stagnation_coherence": 0.25,  # below this, movement is disordered
        "stagnation_magnitude": 0.35,  # and below this, barely moving at all
        "queue_alignment": 0.7,
        "use_density_model": True,
        "grid": 4,                     # heatmap resolution (grid × grid cells)
    }

    def _density_count(self, context) -> float | None:
        """Count by density map when a ``campydense`` model is deployed.

        Detection-based counting collapses once heads occlude each other;
        density regression keeps working in a packed frame.
        """
        if not self.option("use_density_model") or context.registry is None:
            return None
        model = context.registry.get("campydense")
        if model is None:
            return None
        meta = getattr(model, "meta", {}) or {}
        size = int(meta.get("input_size", 96))
        resized = resize_bilinear(context.frame, size, size)
        batch = to_chw(normalise(resized))[None, ...]
        density = model.predict(batch)
        return float(np.clip(np.sum(density), 0, 500))

    def _heatmap(self, people, context) -> list[list[float]]:
        """Coarse occupancy grid — drives the dashboard heat overlay."""
        grid = max(int(self.option("grid")), 1)
        heat = np.zeros((grid, grid), dtype=np.float32)
        for track in people:
            cx, cy = box_center(track.box)
            gx = min(int(cx / max(context.width, 1) * grid), grid - 1)
            gy = min(int(cy / max(context.height, 1) * grid), grid - 1)
            heat[gy, gx] += 1
        return [[round(float(value), 2) for value in row] for row in heat]

    def analyse(self, context) -> AnalyticResult:
        result = AnalyticResult(analytic=self.key)
        people = context.people()
        detected_count = len(people)

        density_count = self._density_count(context)
        # Trust the density model when it materially disagrees in a busy frame —
        # that disagreement is exactly the occlusion case it exists for.
        if density_count is not None and density_count > detected_count * 1.4:
            people_count = int(round(density_count))
            count_source = "density"
        else:
            people_count = detected_count
            count_source = "detection"

        frame_area = max(context.width * context.height, 1)
        density = people_count / frame_area
        flow = flow_statistics(getattr(context.motion, "flow", None))

        scratch = context.scratch(self.key)
        history = scratch.setdefault("counts", [])
        history.append(people_count)
        if len(history) > 300:
            del history[:-300]
        peak = int(max(history)) if history else 0
        average = float(np.mean(history[-60:])) if history else 0.0

        # -- per-zone counting -------------------------------------------
        zone_counts: dict[str, int] = {}
        for zone in context.zones:
            if zone.kind not in {"counting", "monitored", "restricted"}:
                continue
            polygon = zone.pixel_polygon(context.width, context.height)
            if polygon.shape[0] < 3:
                continue
            centres = np.array([box_center(t.box) for t in people]) if people else np.zeros((0, 2))
            inside = int(np.sum(points_in_polygon(centres, polygon))) if len(centres) else 0
            zone_counts[zone.name] = inside

            if zone.max_occupancy and inside > zone.max_occupancy and self.sustained(
                context, f"zonecrowd:{zone.id}", True,
                required_frames=self.option("sustain_frames"),
                cooldown_frames=self.option("cooldown_frames"),
            ):
                result.events.append(
                    EventCandidate(
                        analytic=self.key,
                        event_type="area_overcrowding",
                        severity=Severity.HIGH,
                        confidence=0.85,
                        title=f"Overcrowding in {zone.name}",
                        description=(
                            f"{inside} people are in '{zone.name}' against a limit of "
                            f"{zone.max_occupancy}. Consider directing people to another route."
                        ),
                        zone_id=zone.id,
                        metadata={"count": inside, "limit": zone.max_occupancy},
                    )
                )

        # -- frame-level crowding -----------------------------------------
        crowded = (
            people_count >= self.option("crowd_threshold")
            or density >= self.option("density_threshold")
        )
        if crowded and self.sustained(
            context, "crowd", True,
            required_frames=self.option("sustain_frames"),
            cooldown_frames=self.option("cooldown_frames"),
        ):
            result.events.append(
                EventCandidate(
                    analytic=self.key,
                    event_type="crowd_formation",
                    severity=Severity.MEDIUM,
                    confidence=float(np.clip(0.6 + people_count / 40, 0.6, 0.95)),
                    title=f"Crowd forming — {people_count} people",
                    description=(
                        f"{people_count} people have gathered in view "
                        f"({count_source}-based count). Sustained crowding raises safety risk "
                        "and slows movement."
                    ),
                    metadata={
                        "count": people_count,
                        "peak": peak,
                        "density": round(density, 8),
                        "source": count_source,
                    },
                )
            )

        # -- stagnation: crowded *and* not flowing -------------------------
        stagnant = (
            crowded
            and flow["coherence"] < self.option("stagnation_coherence")
            and flow["magnitude"] < self.option("stagnation_magnitude")
        )
        if stagnant and self.sustained(
            context, "stagnation", True, required_frames=12, cooldown_frames=400
        ):
            result.events.append(
                EventCandidate(
                    analytic=self.key,
                    event_type="crowd_stagnation",
                    severity=Severity.HIGH,
                    confidence=0.8,
                    title="Congestion — crowd is not moving",
                    description=(
                        "A crowd has formed and movement has largely stopped. This is the "
                        "pattern that precedes a crush at an exit or a bottleneck at a doorway."
                    ),
                    metadata={
                        "count": people_count,
                        "coherence": flow["coherence"],
                        "magnitude": flow["magnitude"],
                    },
                )
            )

        # -- sudden dispersal often means something frightened people ------
        if len(history) > 20:
            recent = float(np.mean(history[-5:]))
            earlier = float(np.mean(history[-20:-10])) if len(history) >= 20 else recent
            if earlier >= self.option("crowd_threshold") and recent < earlier * 0.4:
                if self.sustained(context, "dispersal", True, required_frames=2, cooldown_frames=600):
                    result.events.append(
                        EventCandidate(
                            analytic=self.key,
                            event_type="sudden_dispersal",
                            severity=Severity.HIGH,
                            confidence=0.7,
                            title="Sudden dispersal of a group",
                            description=(
                                "A gathered group scattered rapidly. Worth reviewing — this "
                                "often follows an incident off camera."
                            ),
                            metadata={"before": round(earlier, 1), "after": round(recent, 1)},
                        )
                    )

        result.metrics = {
            "people_count": people_count,
            "detected_count": detected_count,
            "density_count": round(density_count, 2) if density_count is not None else None,
            "count_source": count_source,
            "density": round(density, 8),
            "peak": peak,
            "rolling_average": round(average, 2),
            "flow_magnitude": flow["magnitude"],
            "flow_coherence": flow["coherence"],
            "flow_direction": flow["direction"],
            "is_crowded": bool(crowded),
            "is_stagnant": bool(stagnant),
            "zone_counts": zone_counts,
            "heatmap": self._heatmap(people, context),
        }
        return result
