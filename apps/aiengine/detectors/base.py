"""Shared types and the base class for every Campy AI analytic."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Severity & event vocabulary
# ---------------------------------------------------------------------------
class Severity:
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    ORDER = {INFO: 0, LOW: 1, MEDIUM: 2, HIGH: 3, CRITICAL: 4}

    @classmethod
    def max(cls, a: str, b: str) -> str:
        return a if cls.ORDER.get(a, 0) >= cls.ORDER.get(b, 0) else b


ANALYTIC_CHOICES = [
    ("gesture", "Employee Gesture Tracking"),
    ("face", "Face Recognition"),
    ("geofence", "Geofencing / Restricted Areas"),
    ("crowd", "Crowd Management"),
    ("theft", "Theft Detection"),
    ("object", "Unwanted / Dangerous Objects"),
    ("fire", "Fire & Smoke Detection"),
]

ANALYTIC_KEYS = [key for key, _ in ANALYTIC_CHOICES]


@dataclass
class Detection:
    """One thing the engine found in one frame."""

    label: str
    confidence: float
    box: tuple[float, float, float, float]
    track_id: int | None = None
    embedding: np.ndarray | None = None
    attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box": [round(float(v), 2) for v in self.box],
            "track_id": self.track_id,
            "attributes": self.attributes,
        }


@dataclass
class EventCandidate:
    """A finding worth recording. The pipeline turns these into DB events."""

    analytic: str
    event_type: str
    severity: str
    confidence: float
    title: str
    description: str = ""
    box: tuple[float, float, float, float] | None = None
    track_id: int | None = None
    zone_id: int | None = None
    subject_id: int | None = None          # matched employee, when known
    metadata: dict = field(default_factory=dict)
    # Events repeat every frame while a condition holds; this key lets the
    # pipeline de-duplicate them into a single incident.
    dedupe_key: str = ""

    def __post_init__(self):
        if not self.dedupe_key:
            parts = [self.analytic, self.event_type, str(self.track_id or ""), str(self.zone_id or "")]
            self.dedupe_key = ":".join(parts)

    def to_dict(self) -> dict:
        return {
            "analytic": self.analytic,
            "event_type": self.event_type,
            "severity": self.severity,
            "confidence": round(float(self.confidence), 4),
            "title": self.title,
            "description": self.description,
            "box": [round(float(v), 2) for v in self.box] if self.box else None,
            "track_id": self.track_id,
            "zone_id": self.zone_id,
            "subject_id": self.subject_id,
            "metadata": self.metadata,
            "dedupe_key": self.dedupe_key,
        }


@dataclass
class AnalyticResult:
    analytic: str
    detections: list[Detection] = field(default_factory=list)
    events: list[EventCandidate] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    overlays: list[dict] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "analytic": self.analytic,
            "detections": [d.to_dict() for d in self.detections],
            "events": [e.to_dict() for e in self.events],
            "metrics": self.metrics,
            "overlays": self.overlays,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class Zone:
    """A geofence / region of interest, in relative [0,1] coordinates."""

    id: int
    name: str
    polygon: list[list[float]]
    kind: str = "restricted"               # restricted | monitored | counting | exclusion | shelf
    severity: str = Severity.HIGH
    schedule: dict = field(default_factory=dict)   # e.g. {"days": [...], "from": "18:00", "to": "07:00"}
    max_occupancy: int = 0
    min_dwell_seconds: float = 0.0
    allowed_roles: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def pixel_polygon(self, width: int, height: int) -> np.ndarray:
        points = np.asarray(self.polygon, dtype=np.float64).reshape(-1, 2)
        if points.size == 0:
            return points
        # Values <= 1 are relative; anything larger is already in pixels.
        if points.max() <= 1.001:
            points = np.stack([points[:, 0] * width, points[:, 1] * height], axis=-1)
        return points

    def is_active_now(self, moment=None) -> bool:
        """Zones can be armed only outside working hours, on certain days, etc."""
        schedule = self.schedule or {}
        if not schedule:
            return True
        from datetime import datetime

        moment = moment or datetime.now()
        days = schedule.get("days")
        if days and moment.strftime("%a").lower()[:3] not in [str(d).lower()[:3] for d in days]:
            return False
        start, end = schedule.get("from"), schedule.get("to")
        if not start or not end:
            return True
        current = moment.strftime("%H:%M")
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end     # window crosses midnight


@dataclass
class FrameContext:
    """Everything an analytic needs to judge the current frame."""

    frame: np.ndarray                      # HWC RGB uint8
    frame_index: int
    timestamp: float
    camera_id: int | None = None
    fps: float = 6.0
    zones: list[Zone] = field(default_factory=list)
    motion: Any = None                     # MotionState
    tracks: list = field(default_factory=list)          # TrackedObject
    detections: list[Detection] = field(default_factory=list)
    previous_frame: np.ndarray | None = None
    background: Any = None                 # BackgroundModel
    registry: Any = None                   # ModelProvider
    config: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)          # per-camera persistent scratch space

    @property
    def height(self) -> int:
        return int(self.frame.shape[0])

    @property
    def width(self) -> int:
        return int(self.frame.shape[1])

    def people(self) -> list:
        """Confirmed person tracks only.

        Unconfirmed tracks are single-frame noise; counting them would inflate
        occupancy and fire spurious crowd alerts.
        """
        return [
            t
            for t in self.tracks
            if t.label == "person" and t.is_confirmed and not t.is_static_scenery
        ]

    def analytic_config(self, analytic: str) -> dict:
        return dict((self.config or {}).get(analytic) or {})

    def scratch(self, analytic: str) -> dict:
        """Persistent per-analytic state that survives across frames."""
        return self.state.setdefault(analytic, {})


class BaseAnalytic:
    """Base class for the seven Campy AI analytics.

    Subclasses implement :meth:`analyse`.  The base class handles timing,
    enable/disable, config defaults and exception isolation — one failing
    analytic must never take down the whole camera pipeline.
    """

    key = "base"
    label = "Base analytic"
    #: Subscription feature that must be present on the customer's package.
    feature = ""
    #: Sensible defaults, merged under the camera's own configuration.
    defaults: dict = {}

    def __init__(self, config: dict | None = None):
        self.config = {**self.defaults, **(config or {})}
        self.enabled = bool(self.config.get("enabled", True))

    # -- interface ---------------------------------------------------------
    def analyse(self, context: FrameContext) -> AnalyticResult:
        raise NotImplementedError

    def run(self, context: FrameContext) -> AnalyticResult:
        if not self.enabled:
            return AnalyticResult(analytic=self.key)
        started = time.perf_counter()
        try:
            merged = {**self.config, **context.analytic_config(self.key)}
            previous_config = self.config
            self.config = merged
            try:
                result = self.analyse(context)
            finally:
                self.config = previous_config
        except Exception as exc:  # noqa: BLE001 - never break the pipeline
            import logging

            logging.getLogger("campy.analytics").exception("Analytic %s failed", self.key)
            result = AnalyticResult(analytic=self.key, metrics={"error": str(exc)[:200]})
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return result

    # -- helpers -----------------------------------------------------------
    def option(self, key: str, default=None):
        return self.config.get(key, default)

    def sustained(
        self,
        context: FrameContext,
        key: str,
        condition: bool,
        required_frames: int = 3,
        cooldown_frames: int = 90,
    ) -> bool:
        """Fire once when *condition* has held for ``required_frames``.

        Raw per-frame detections are noisy; requiring persistence and then
        enforcing a cooldown is what turns them into alerts a human will trust
        rather than mute.
        """
        scratch = context.scratch(self.key)
        counters = scratch.setdefault("_streaks", {})
        cooldowns = scratch.setdefault("_cooldowns", {})

        last_fired = cooldowns.get(key)
        if last_fired is not None and context.frame_index - last_fired < cooldown_frames:
            if not condition:
                counters[key] = 0
            return False

        if condition:
            counters[key] = counters.get(key, 0) + 1
            if counters[key] >= required_frames:
                counters[key] = 0
                cooldowns[key] = context.frame_index
                return True
        else:
            counters[key] = 0
        return False
