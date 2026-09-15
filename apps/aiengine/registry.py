"""Model registry — resolves a logical model slot to a loaded network.

Slots are logical names (``campyface``, ``campydet``, ``campynet_fire``, …).
Resolution order for a given organisation:

1. A ``ModelVersion`` the organisation has deployed to that slot.
2. The platform-wide base model deployed to that slot by Campy AI.
3. Nothing — the analytic falls back to its classical implementation.

Loaded models are cached process-wide and keyed by checkpoint path plus mtime,
so promoting a new version takes effect without a restart.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from django.conf import settings

logger = logging.getLogger("campy.registry")

MODEL_SLOTS = [
    ("campydet", "Person & object detector"),
    ("campyface", "Face identity embedding"),
    ("campynet_face", "Face / not-face verifier"),
    ("campynet_fire", "Fire & smoke classifier"),
    ("campynet_object", "Object classifier"),
    ("campydense", "Crowd density estimator"),
    ("campyseq_gesture", "Gesture & posture sequence model"),
]

SLOT_KEYS = [key for key, _ in MODEL_SLOTS]

# Which architecture family each slot expects.
SLOT_FAMILY = {
    "campydet": "campydet",
    "campyface": "campyface",
    "campynet_face": "campynet",
    "campynet_fire": "campynet",
    "campynet_object": "campynet",
    "campydense": "campydense",
    "campyseq_gesture": "campyseq",
}

_cache: dict[str, object] = {}
_lock = threading.Lock()


def load_checkpoint(path: str | Path):
    """Load and cache a ``.npz`` checkpoint, invalidating on file mtime."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        key = f"{path}:{path.stat().st_mtime_ns}"
    except OSError:
        return None

    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached

    try:
        from .nn.model import Sequential

        model = Sequential.load(path)
    except Exception:  # noqa: BLE001 - a corrupt checkpoint must not crash a worker
        logger.exception("Failed to load model checkpoint %s", path)
        return None

    with _lock:
        # Keep the cache small; workers only ever hold a handful of models.
        if len(_cache) > 32:
            _cache.clear()
        _cache[key] = model
    return model


def clear_cache() -> None:
    with _lock:
        _cache.clear()


class ModelProvider:
    """Per-organisation view of the registry, handed to every ``FrameContext``."""

    def __init__(self, organization=None, overrides: dict[str, str] | None = None):
        self.organization = organization
        self.overrides = dict(overrides or {})
        self._resolved: dict[str, object] = {}
        self._missing: set[str] = set()

    def _checkpoint_path(self, slot: str) -> Path | None:
        if slot in self.overrides:
            return Path(self.overrides[slot])
        try:
            from apps.training.models import ModelVersion
        except Exception:  # noqa: BLE001 - registry is usable outside Django too
            return None

        version = ModelVersion.deployed_for(slot, self.organization)
        if version is None or not version.weights_path:
            return None
        return Path(version.weights_path)

    def get(self, slot: str):
        """Return the loaded model for *slot*, or ``None``."""
        if slot in self._resolved:
            return self._resolved[slot]
        if slot in self._missing:
            return None

        path = self._checkpoint_path(slot)
        model = load_checkpoint(path) if path else None
        if model is None:
            self._missing.add(slot)
            return None
        self._resolved[slot] = model
        return model

    def available(self) -> dict[str, bool]:
        return {slot: self.get(slot) is not None for slot in SLOT_KEYS}

    def describe(self) -> list[dict]:
        rows = []
        for slot, label in MODEL_SLOTS:
            model = self.get(slot)
            rows.append(
                {
                    "slot": slot,
                    "label": label,
                    "family": SLOT_FAMILY.get(slot, ""),
                    "loaded": model is not None,
                    "params": getattr(model, "param_count", 0) if model else 0,
                    "meta": getattr(model, "meta", {}) if model else {},
                }
            )
        return rows

    def invalidate(self) -> None:
        self._resolved.clear()
        self._missing.clear()


def model_root() -> Path:
    path = Path(settings.CAMPY["MODEL_ROOT"])
    path.mkdir(parents=True, exist_ok=True)
    return path
