"""The Campy AI inference pipeline.

One :class:`CameraPipeline` per camera.  Each frame flows through:

    frame → background model → detection → tracking → analytics → events

Shared stages run **once** per frame and their output is handed to every
analytic.  That is the whole reason seven analytics cost barely more than one:
background subtraction, detection and tracking are the expensive parts, and
they are not repeated.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .detectors import ANALYTIC_REGISTRY, FaceGallery, build_analytic
from .detectors.base import AnalyticResult, EventCandidate, FrameContext, Severity, Zone
from .detectors.people import detect_people_and_objects
from .registry import ModelProvider
from .vision.motion import BackgroundModel
from .vision.tracking import MultiObjectTracker, TrackerConfig

logger = logging.getLogger("campy.pipeline")


@dataclass
class PipelineResult:
    """Everything the pipeline learned about one frame."""

    frame_index: int
    timestamp: float
    tracks: list = field(default_factory=list)
    detections: list = field(default_factory=list)
    results: dict[str, AnalyticResult] = field(default_factory=dict)
    events: list[EventCandidate] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def people_count(self) -> int:
        return sum(1 for track in self.tracks if track.label == "person")

    @property
    def max_severity(self) -> str:
        severity = Severity.INFO
        for event in self.events:
            severity = Severity.max(severity, event.severity)
        return severity

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "people_count": self.people_count,
            "tracks": [t.to_dict() for t in self.tracks],
            "events": [e.to_dict() for e in self.events],
            "metrics": self.metrics,
            "analytics": {key: value.to_dict() for key, value in self.results.items()},
            "elapsed_ms": round(self.elapsed_ms, 2),
            "max_severity": self.max_severity,
        }


class CameraPipeline:
    """Stateful per-camera analytics runner."""

    def __init__(
        self,
        camera_id: int | None = None,
        analytics: list[str] | None = None,
        config: dict | None = None,
        zones: list[Zone] | None = None,
        gallery: FaceGallery | None = None,
        registry: ModelProvider | None = None,
        fps: float = 6.0,
    ):
        self.camera_id = camera_id
        self.config = dict(config or {})
        self.zones = list(zones or [])
        self.gallery = gallery or FaceGallery()
        self.registry = registry or ModelProvider()
        self.fps = float(fps)

        requested = analytics if analytics is not None else list(ANALYTIC_REGISTRY)
        self.analytics = {
            key: build_analytic(key, self.config.get(key))
            for key in requested
            if key in ANALYTIC_REGISTRY
        }

        self.background = BackgroundModel(
            learning_rate=float(self.config.get("background_learning_rate", 0.02)),
            n_sigma=float(self.config.get("background_sigma", 2.6)),
            downscale=int(self.config.get("background_downscale", 2)),
        )
        self.tracker = MultiObjectTracker(
            TrackerConfig(
                max_age=int(self.config.get("track_max_age", 30)),
                min_hits=int(self.config.get("track_min_hits", 3)),
                iou_threshold=float(self.config.get("track_iou", 0.25)),
            )
        )

        self.state: dict = {}
        self.frame_index = 0
        self.previous_frame: np.ndarray | None = None
        self._dedupe: dict[str, int] = {}
        self._timings: dict[str, float] = {}

    # -- configuration -----------------------------------------------------
    def set_zones(self, zones: list[Zone]) -> None:
        self.zones = list(zones or [])

    def set_gallery(self, gallery: FaceGallery) -> None:
        self.gallery = gallery

    def set_analytics(self, keys: list[str]) -> None:
        self.analytics = {
            key: build_analytic(key, self.config.get(key))
            for key in keys
            if key in ANALYTIC_REGISTRY
        }

    def reset(self) -> None:
        self.background.reset()
        self.tracker.reset()
        self.state.clear()
        self._dedupe.clear()
        self.frame_index = 0
        self.previous_frame = None

    @property
    def enabled_analytics(self) -> list[str]:
        return list(self.analytics)

    # -- main --------------------------------------------------------------
    def process(self, frame: np.ndarray, timestamp: float | None = None) -> PipelineResult:
        started = time.perf_counter()
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        self.frame_index += 1
        timestamp = timestamp if timestamp is not None else time.time()

        # ---- stage 1: motion ------------------------------------------
        stage = time.perf_counter()
        needs_flow = bool({"crowd", "gesture"} & set(self.analytics))
        motion = self.background.update(frame, compute_flow=needs_flow)
        self._timings["motion"] = (time.perf_counter() - stage) * 1000

        # ---- stage 2: detection ---------------------------------------
        context = FrameContext(
            frame=frame,
            frame_index=self.frame_index,
            timestamp=timestamp,
            camera_id=self.camera_id,
            fps=self.fps,
            zones=self.zones,
            motion=motion,
            previous_frame=self.previous_frame,
            background=self.background,
            registry=self.registry,
            config={**self.config, "_gallery": self.gallery},
            state=self.state,
        )

        stage = time.perf_counter()
        detections = detect_people_and_objects(context)
        context.detections = detections
        self._timings["detect"] = (time.perf_counter() - stage) * 1000

        # ---- stage 3: tracking ----------------------------------------
        stage = time.perf_counter()
        tracks = self.tracker.update(
            [
                {
                    "box": d.box,
                    "label": d.label,
                    "confidence": d.confidence,
                    "embedding": d.embedding,
                    "attributes": d.attributes,
                }
                for d in detections
            ]
        )
        context.tracks = tracks
        # Feed confirmed person boxes back to the background model so a
        # stationary subject is not absorbed into the scene.
        # Only tracks still backed by a live detection hold protection — a
        # track that has lost its detection must be allowed to fade, otherwise
        # protection and detection sustain each other indefinitely.
        self.background.protect(
            [
                t.box
                for t in tracks
                if t.label == "person"
                and t.is_confirmed
                and t.time_since_update == 0
                and t.has_moved
            ]
        )
        self._timings["track"] = (time.perf_counter() - stage) * 1000

        # ---- stage 4: analytics ---------------------------------------
        # Face recognition runs first: everything downstream benefits from
        # knowing *who* a track is (authorisation, attribution, attendance).
        ordered = sorted(self.analytics.items(), key=lambda item: 0 if item[0] == "face" else 1)

        results: dict[str, AnalyticResult] = {}
        events: list[EventCandidate] = []
        for key, analytic in ordered:
            result = analytic.run(context)
            results[key] = result
            self._timings[key] = result.elapsed_ms
            events.extend(result.events)

        events = self._deduplicate(events)

        self.previous_frame = frame
        elapsed = (time.perf_counter() - started) * 1000

        return PipelineResult(
            frame_index=self.frame_index,
            timestamp=timestamp,
            tracks=tracks,
            detections=detections,
            results=results,
            events=events,
            metrics={
                "motion_ratio": round(float(motion.motion_ratio), 5),
                "motion_energy": round(float(motion.energy), 3),
                "detections": len(detections),
                "tracks": len(tracks),
                "people": sum(1 for t in tracks if t.label == "person"),
                "background_ready": self.background.is_ready,
                "timings_ms": {k: round(v, 2) for k, v in self._timings.items()},
                "analytics": {key: value.metrics for key, value in results.items()},
            },
            elapsed_ms=elapsed,
        )

    # -- helpers -----------------------------------------------------------
    def _deduplicate(self, events: list[EventCandidate], window: int = 60) -> list[EventCandidate]:
        """Collapse the same finding repeating across consecutive frames."""
        kept: list[EventCandidate] = []
        for event in events:
            last = self._dedupe.get(event.dedupe_key)
            if last is not None and self.frame_index - last < window:
                continue
            self._dedupe[event.dedupe_key] = self.frame_index
            kept.append(event)
        if len(self._dedupe) > 500:
            cutoff = self.frame_index - window * 4
            self._dedupe = {k: v for k, v in self._dedupe.items() if v >= cutoff}
        return kept

    def health(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "frames": self.frame_index,
            "background_ready": self.background.is_ready,
            "active_tracks": self.tracker.active_count,
            "analytics": self.enabled_analytics,
            "models": self.registry.available(),
            "timings_ms": {k: round(v, 2) for k, v in self._timings.items()},
        }


def build_pipeline_from_camera(camera, registry: ModelProvider | None = None) -> CameraPipeline:
    """Construct a pipeline from a ``cameras.Camera`` database row."""
    zones = [
        Zone(
            id=zone.id,
            name=zone.name,
            polygon=zone.polygon,
            kind=zone.kind,
            severity=zone.severity,
            schedule=zone.schedule or {},
            max_occupancy=zone.max_occupancy or 0,
            min_dwell_seconds=float(zone.min_dwell_seconds or 0),
            allowed_roles=list(zone.allowed_roles or []),
        )
        for zone in camera.zones.filter(is_active=True)
    ]

    gallery = FaceGallery()
    if "face" in (camera.enabled_analytics or []):
        from apps.cameras.models import Employee

        gallery.load(
            [
                {
                    "id": employee.id,
                    "name": employee.display_name,
                    "role": employee.role,
                    "embedding": employee.face_embedding,
                }
                for employee in Employee.objects.filter(
                    organization=camera.organization, is_active=True
                ).exclude(face_embedding=[])
            ]
        )

    return CameraPipeline(
        camera_id=camera.id,
        analytics=camera.active_analytics(),
        config=camera.analytics_config or {},
        zones=zones,
        gallery=gallery,
        registry=registry or ModelProvider(camera.organization),
        fps=float(camera.target_fps or 6),
    )
