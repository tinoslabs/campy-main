"""The seven Campy AI analytics."""
from .base import (  # noqa: F401
    ANALYTIC_CHOICES,
    ANALYTIC_KEYS,
    AnalyticResult,
    BaseAnalytic,
    Detection,
    EventCandidate,
    FrameContext,
    Severity,
    Zone,
)
from .crowd import CrowdManagementAnalytic  # noqa: F401
from .face import FaceGallery, FaceRecognitionAnalytic  # noqa: F401
from .fire import FireSmokeAnalytic  # noqa: F401
from .geofence import GeofenceAnalytic  # noqa: F401
from .gesture import GestureTrackingAnalytic  # noqa: F401
from .objects import DangerousObjectAnalytic  # noqa: F401
from .theft import TheftDetectionAnalytic  # noqa: F401

ANALYTIC_REGISTRY: dict[str, type[BaseAnalytic]] = {
    GestureTrackingAnalytic.key: GestureTrackingAnalytic,
    FaceRecognitionAnalytic.key: FaceRecognitionAnalytic,
    GeofenceAnalytic.key: GeofenceAnalytic,
    CrowdManagementAnalytic.key: CrowdManagementAnalytic,
    TheftDetectionAnalytic.key: TheftDetectionAnalytic,
    DangerousObjectAnalytic.key: DangerousObjectAnalytic,
    FireSmokeAnalytic.key: FireSmokeAnalytic,
}

ANALYTIC_FEATURES = {key: cls.feature for key, cls in ANALYTIC_REGISTRY.items()}


def build_analytic(key: str, config: dict | None = None) -> BaseAnalytic:
    cls = ANALYTIC_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown analytic '{key}'. Available: {sorted(ANALYTIC_REGISTRY)}")
    return cls(config)


def analytic_metadata() -> list[dict]:
    """Describes every analytic — powers the camera configuration UI."""
    return [
        {
            "key": cls.key,
            "label": cls.label,
            "feature": cls.feature,
            "defaults": dict(cls.defaults),
            "doc": (cls.__doc__ or "").strip().split("\n")[0],
        }
        for cls in ANALYTIC_REGISTRY.values()
    ]


__all__ = [
    "ANALYTIC_REGISTRY", "ANALYTIC_CHOICES", "ANALYTIC_KEYS", "ANALYTIC_FEATURES",
    "build_analytic", "analytic_metadata", "BaseAnalytic", "AnalyticResult",
    "Detection", "EventCandidate", "FrameContext", "Severity", "Zone", "FaceGallery",
    "GestureTrackingAnalytic", "FaceRecognitionAnalytic", "GeofenceAnalytic",
    "CrowdManagementAnalytic", "TheftDetectionAnalytic", "DangerousObjectAnalytic",
    "FireSmokeAnalytic",
]
