"""Geofencing — restricted-area monitoring.

Virtual boundaries are drawn over the camera view to mark machinery zones,
storage and server rooms, cash areas or staff-only sections.  When someone
enters without authorisation, the system raises an event.

Zones are stored in *relative* coordinates so they survive a change of stream
resolution, and can be armed on a schedule (e.g. only outside working hours).
"""
from __future__ import annotations

import numpy as np

from ..vision.geometry import box_foot, box_polygon_overlap, points_in_polygon
from .base import AnalyticResult, BaseAnalytic, EventCandidate, Severity


class GeofenceAnalytic(BaseAnalytic):
    key = "geofence"
    label = "Geofencing & Restricted Areas"
    feature = "geofencing"

    defaults = {
        "enabled": True,
        # A person counts as "inside" when their feet are in the polygon, or
        # enough of their box overlaps it.
        "overlap_threshold": 0.35,
        "entry_frames": 3,
        "dwell_alert_seconds": 30,
        "cooldown_frames": 150,
        "count_occupancy": True,
        "alert_on_exit": False,
    }

    ZONE_SEVERITY = {
        "restricted": Severity.HIGH,
        "hazard": Severity.CRITICAL,
        "monitored": Severity.LOW,
        "counting": Severity.INFO,
        "shelf": Severity.INFO,
        "exclusion": Severity.HIGH,
    }

    def _inside(self, track, polygon) -> tuple[bool, float]:
        """Feet-first containment test, with box overlap as a fallback."""
        foot = box_foot(track.box)
        if bool(points_in_polygon(np.array([foot]), polygon)[0]):
            return True, 1.0
        overlap = box_polygon_overlap(track.box, polygon)
        return overlap >= self.option("overlap_threshold"), overlap

    def analyse(self, context) -> AnalyticResult:
        result = AnalyticResult(analytic=self.key)
        zones = [z for z in context.zones if z.kind in self.ZONE_SEVERITY]
        if not zones:
            result.metrics = {"zones": 0}
            return result

        scratch = context.scratch(self.key)
        occupancy_state: dict = scratch.setdefault("occupancy", {})
        dwell_state: dict = scratch.setdefault("dwell", {})

        people = context.people()
        zone_metrics: dict[str, dict] = {}

        for zone in zones:
            polygon = zone.pixel_polygon(context.width, context.height)
            if polygon.shape[0] < 3:
                continue

            armed = zone.is_active_now()
            inside_ids: list[int] = []

            for track in people:
                is_inside, overlap = self._inside(track, polygon)
                state_key = f"{zone.id}:{track.track_id}"

                if is_inside:
                    inside_ids.append(track.track_id)
                    dwell_state[state_key] = dwell_state.get(state_key, 0) + 1
                    dwell_frames = dwell_state[state_key]
                    dwell_seconds = dwell_frames / max(context.fps, 0.1)
                    track.attributes.setdefault("zones", []).append(zone.name)

                    if not armed or zone.kind in {"counting", "shelf"}:
                        continue

                    severity = zone.severity or self.ZONE_SEVERITY.get(zone.kind, Severity.MEDIUM)

                    # -- entry event ---------------------------------------
                    if self.sustained(
                        context,
                        f"entry:{state_key}",
                        dwell_frames >= self.option("entry_frames"),
                        required_frames=1,
                        cooldown_frames=self.option("cooldown_frames"),
                    ):
                        identity = track.attributes.get("identity")
                        authorised = self._is_authorised(track, zone)
                        result.events.append(
                            EventCandidate(
                                analytic=self.key,
                                event_type="zone_entry" if authorised else "unauthorised_zone_entry",
                                severity=Severity.LOW if authorised else severity,
                                confidence=float(np.clip(0.6 + overlap * 0.35, 0.6, 0.98)),
                                title=(
                                    f"Entry into {zone.name}"
                                    if authorised
                                    else f"Unauthorised entry into {zone.name}"
                                ),
                                description=(
                                    f"{identity or 'A person'} entered the restricted area "
                                    f"'{zone.name}'."
                                    + ("" if authorised else " No matching authorisation was found.")
                                ),
                                box=track.box,
                                track_id=track.track_id,
                                zone_id=zone.id,
                                subject_id=track.attributes.get("employee_id"),
                                metadata={
                                    "zone": zone.name,
                                    "zone_kind": zone.kind,
                                    "overlap": round(float(overlap), 3),
                                    "authorised": authorised,
                                },
                            )
                        )

                    # -- dwell event ---------------------------------------
                    threshold = zone.min_dwell_seconds or self.option("dwell_alert_seconds")
                    if threshold and dwell_seconds >= threshold and self.sustained(
                        context, f"dwell:{state_key}", True, required_frames=1, cooldown_frames=600
                    ):
                        result.events.append(
                            EventCandidate(
                                analytic=self.key,
                                event_type="zone_dwell",
                                severity=Severity.max(severity, Severity.MEDIUM),
                                confidence=0.8,
                                title=f"Extended presence in {zone.name}",
                                description=(
                                    f"A person has remained inside '{zone.name}' for "
                                    f"{int(dwell_seconds)} seconds."
                                ),
                                box=track.box,
                                track_id=track.track_id,
                                zone_id=zone.id,
                                metadata={"seconds": round(dwell_seconds, 1), "zone": zone.name},
                            )
                        )
                else:
                    if dwell_state.pop(state_key, None) and self.option("alert_on_exit") and armed:
                        result.events.append(
                            EventCandidate(
                                analytic=self.key,
                                event_type="zone_exit",
                                severity=Severity.INFO,
                                confidence=0.7,
                                title=f"Exit from {zone.name}",
                                box=track.box,
                                track_id=track.track_id,
                                zone_id=zone.id,
                                metadata={"zone": zone.name},
                            )
                        )

            # -- occupancy -------------------------------------------------
            occupancy = len(inside_ids)
            previous = occupancy_state.get(zone.id, 0)
            occupancy_state[zone.id] = occupancy
            zone_metrics[zone.name] = {
                "zone_id": zone.id,
                "kind": zone.kind,
                "occupancy": occupancy,
                "armed": armed,
                "track_ids": inside_ids,
            }

            if (
                zone.max_occupancy
                and occupancy > zone.max_occupancy
                and self.sustained(
                    context, f"over:{zone.id}", True, required_frames=4, cooldown_frames=300
                )
            ):
                result.events.append(
                    EventCandidate(
                        analytic=self.key,
                        event_type="zone_over_capacity",
                        severity=Severity.HIGH,
                        confidence=0.85,
                        title=f"{zone.name} is over capacity",
                        description=(
                            f"{occupancy} people are inside '{zone.name}', which is limited to "
                            f"{zone.max_occupancy}."
                        ),
                        zone_id=zone.id,
                        metadata={
                            "occupancy": occupancy,
                            "limit": zone.max_occupancy,
                            "previous": previous,
                        },
                    )
                )

        result.metrics = {
            "zones": len(zones),
            "zone_occupancy": zone_metrics,
            "total_inside": sum(z["occupancy"] for z in zone_metrics.values()),
        }
        result.overlays = [
            {
                "type": "polygon",
                "zone_id": zone.id,
                "name": zone.name,
                "kind": zone.kind,
                "points": zone.polygon,
                "occupied": bool(zone_metrics.get(zone.name, {}).get("occupancy")),
            }
            for zone in zones
        ]
        return result

    @staticmethod
    def _is_authorised(track, zone) -> bool:
        """Authorisation comes from face recognition upstream in the pipeline."""
        if not zone.allowed_roles:
            return False
        role = track.attributes.get("employee_role")
        return bool(role and role in zone.allowed_roles)
