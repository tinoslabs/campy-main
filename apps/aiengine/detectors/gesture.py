"""Employee Gesture Tracking — the core analytic from the Campy AI spec.

Observes body movement, posture and activity patterns during working hours:
walking, standing, sitting, bending, running, or remaining inactive for long
periods.  The system does not judge individuals; it surfaces *patterns* and
flags genuinely unsafe or unusual actions for a human to review.

How it works
------------
Per tracked person we accumulate a rolling window of kinematic descriptors —
normalised speed, direction change, vertical extent, aspect ratio, area change,
motion energy inside the box — and classify the window.  A trained ``campyseq``
model is used when the workspace has one deployed; otherwise a transparent
rule-based classifier derived from the same features keeps the analytic useful
from day one.
"""
from __future__ import annotations

import numpy as np

from ..vision.motion import flow_statistics
from .base import AnalyticResult, BaseAnalytic, Detection, EventCandidate, Severity

POSTURES = ["standing", "walking", "running", "bending", "sitting", "idle"]

POSTURE_SEVERITY = {
    "running": Severity.MEDIUM,
    "fall": Severity.CRITICAL,
    "bending": Severity.INFO,
    "idle": Severity.LOW,
}

WINDOW = 16          # frames of history per person
FEATURE_DIM = 8      # descriptors per frame


class GestureTrackingAnalytic(BaseAnalytic):
    key = "gesture"
    label = "Employee Gesture Tracking"
    feature = "gesture_tracking"

    defaults = {
        "enabled": True,
        # Speeds are in *body-heights per second*: dividing by the subject's
        # own pixel height makes them independent of camera distance, and
        # dividing by fps makes them independent of frame rate. Reference
        # points: walking ~1.4 m/s and running ~3 m/s against a ~1.7 m person
        # give roughly 0.8 and 1.8 body-heights per second.
        "walk_speed": 0.45,
        "run_speed": 1.45,
        "idle_speed": 0.06,
        "idle_seconds": 240,          # inactive this long → wellbeing flag
        "loiter_seconds": 180,
        "fall_aspect": 1.25,          # box wider than tall → possible fall
        "fall_drop_ratio": 0.45,      # sudden loss of height
        "report_postures": True,
        "min_confidence": 0.45,
    }

    # -- feature extraction ------------------------------------------------
    def _frame_descriptor(self, track, context) -> np.ndarray:
        """Eight scale-invariant numbers describing this person right now."""
        x1, y1, x2, y2 = track.box
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        vx, vy = track.velocity

        # Body-heights per second: distance-invariant and frame-rate invariant.
        fps = max(float(context.fps), 0.1)
        speed = float(np.hypot(vx, vy) / height * fps)
        vertical_speed = float(vy / height * fps)
        aspect = float(width / height)
        relative_height = float(height / max(context.height, 1))

        trail = list(track.trail)[-6:]
        if len(trail) >= 3:
            deltas = np.diff(np.asarray(trail), axis=0)
            angles = np.arctan2(deltas[:, 1], deltas[:, 0])
            direction_change = float(np.mean(np.abs(np.diff(angles)))) if len(angles) > 1 else 0.0
        else:
            direction_change = 0.0

        patch_energy = 0.0
        if context.motion is not None and getattr(context.motion, "mask", None) is not None:
            mask = context.motion.mask
            mh, mw = mask.shape
            sy = mh / max(context.height, 1)
            sx = mw / max(context.width, 1)
            region = mask[
                int(max(y1 * sy, 0)) : int(min(y2 * sy, mh)),
                int(max(x1 * sx, 0)) : int(min(x2 * sx, mw)),
            ]
            if region.size:
                patch_energy = float(region.mean())

        path = track.path_length(12)
        displacement = track.displacement(12)
        # ~1.0 = walking a straight line, ~0.0 = milling on the spot.
        straightness = float(displacement / path) if path > 1e-3 else 0.0

        return np.array(
            [speed, vertical_speed, aspect, relative_height,
             direction_change, patch_energy, straightness, float(min(track.age, 300) / 300)],
            dtype=np.float32,
        )

    # -- classification ----------------------------------------------------
    def _classify_rules(self, window: np.ndarray, track) -> tuple[str, float]:
        """Transparent fallback classifier over the same feature window."""
        recent = window[-6:] if window.shape[0] >= 6 else window
        speed = float(np.mean(recent[:, 0]))
        aspect = float(np.mean(recent[:, 2]))
        straightness = float(np.mean(recent[:, 6]))

        heights = window[:, 3]
        height_drop = float((heights.max() - heights[-1]) / max(heights.max(), 1e-6))

        # Peak speed over the window catches a brief sprint that averaging
        # would smooth away.
        peak_speed = float(np.max(window[-8:, 0])) if window.shape[0] >= 2 else speed

        if aspect > self.option("fall_aspect") and height_drop > self.option("fall_drop_ratio"):
            return "fall", min(0.55 + height_drop, 0.95)
        if speed >= self.option("run_speed") or peak_speed >= self.option("run_speed") * 1.25:
            return "running", float(np.clip(0.55 + speed / 6.0, 0.55, 0.97))
        if speed >= self.option("walk_speed"):
            return "walking", float(np.clip(0.55 + 0.3 * straightness, 0.5, 0.95))
        if aspect > 0.85 and height_drop > 0.18:
            return "bending", 0.6
        # Idle is decided on *movement*, not on mask energy: once the tracker
        # protects a stationary subject from the background model, their mask
        # stays lit even though they are not moving at all.
        if speed <= self.option("idle_speed") and peak_speed <= self.option("walk_speed") * 0.6:
            return "idle", float(np.clip(0.62 + (self.option("idle_speed") - speed) * 2, 0.55, 0.92))
        if aspect > 0.7:
            return "sitting", 0.55
        return "standing", 0.6

    def _classify(self, window: np.ndarray, track, context) -> tuple[str, float]:
        model = context.registry.get("campyseq_gesture") if context.registry else None
        if model is not None:
            padded = self._pad_window(window)
            logits = model.predict(padded[None, :])
            from ..nn.functional import softmax

            probabilities = softmax(logits, axis=-1)[0]
            classes = (getattr(model, "meta", {}) or {}).get("classes") or POSTURES
            index = int(np.argmax(probabilities))
            if index < len(classes):
                return classes[index], float(probabilities[index])
        return self._classify_rules(window, track)

    @staticmethod
    def _pad_window(window: np.ndarray) -> np.ndarray:
        """Pad/trim to a fixed WINDOW×FEATURE_DIM then flatten for campyseq."""
        if window.shape[0] >= WINDOW:
            trimmed = window[-WINDOW:]
        else:
            padding = np.repeat(window[:1], WINDOW - window.shape[0], axis=0)
            trimmed = np.concatenate([padding, window], axis=0)
        return trimmed.reshape(-1)

    # -- main --------------------------------------------------------------
    def analyse(self, context) -> AnalyticResult:
        result = AnalyticResult(analytic=self.key)
        scratch = context.scratch(self.key)
        windows = scratch.setdefault("windows", {})
        posture_since = scratch.setdefault("posture_since", {})

        people = context.people()
        counts: dict[str, int] = {}
        active_ids = set()

        for track in people:
            active_ids.add(track.track_id)
            history = windows.setdefault(track.track_id, [])
            history.append(self._frame_descriptor(track, context))
            if len(history) > WINDOW * 2:
                del history[: len(history) - WINDOW * 2]

            if len(history) < 4:
                continue

            window = np.asarray(history, dtype=np.float32)
            posture, confidence = self._classify(window, track, context)
            counts[posture] = counts.get(posture, 0) + 1

            track.attributes["posture"] = posture
            track.attributes["posture_confidence"] = round(confidence, 3)

            result.detections.append(
                Detection(
                    label=posture,
                    confidence=confidence,
                    box=track.box,
                    track_id=track.track_id,
                    attributes={
                        "speed": round(float(window[-1, 0]), 4),
                        "aspect": round(float(window[-1, 2]), 3),
                        "straightness": round(float(window[-1, 6]), 3),
                    },
                )
            )

            # -- how long has this posture held? ---------------------------
            previous = posture_since.get(track.track_id)
            if previous is None or previous[0] != posture:
                posture_since[track.track_id] = (posture, context.frame_index)
                held_frames = 0
            else:
                held_frames = context.frame_index - previous[1]
            held_seconds = held_frames / max(context.fps, 0.1)

            if confidence < self.option("min_confidence"):
                continue

            # -- events ----------------------------------------------------
            if posture == "fall" and self.sustained(
                context, f"fall:{track.track_id}", True, required_frames=2, cooldown_frames=150
            ):
                result.events.append(
                    EventCandidate(
                        analytic=self.key,
                        event_type="possible_fall",
                        severity=Severity.CRITICAL,
                        confidence=confidence,
                        title="Possible fall detected",
                        description=(
                            "A person's posture changed abruptly from upright to horizontal. "
                            "Review the clip and check on them."
                        ),
                        box=track.box,
                        track_id=track.track_id,
                        metadata={"posture": posture, "held_seconds": round(held_seconds, 1)},
                    )
                )

            # Trigger on the posture persisting, not on a hold-timer: a
            # posture timer resets on any single-frame flicker, and a sprint
            # across one camera lasts only a couple of seconds.
            if posture == "running" and self.sustained(
                context, f"run:{track.track_id}", True, required_frames=4, cooldown_frames=120
            ):
                result.events.append(
                    EventCandidate(
                        analytic=self.key,
                        event_type="running_detected",
                        severity=Severity.MEDIUM,
                        confidence=confidence,
                        title="Running in the workplace",
                        description=(
                            "Sustained running was observed. This can indicate an emergency, "
                            "or an unsafe hurry in a walkway."
                        ),
                        box=track.box,
                        track_id=track.track_id,
                        metadata={"seconds": round(held_seconds, 1)},
                    )
                )

            if posture == "idle" and held_seconds >= self.option("idle_seconds"):
                if self.sustained(
                    context, f"idle:{track.track_id}", True, required_frames=2, cooldown_frames=600
                ):
                    result.events.append(
                        EventCandidate(
                            analytic=self.key,
                            event_type="prolonged_inactivity",
                            severity=Severity.LOW,
                            confidence=confidence,
                            title="Prolonged inactivity",
                            description=(
                                f"A person has been stationary for about "
                                f"{int(held_seconds // 60)} minutes. Worth a wellbeing check "
                                "in areas where that is unusual."
                            ),
                            box=track.box,
                            track_id=track.track_id,
                            metadata={"seconds": round(held_seconds, 1)},
                        )
                    )

            # Loitering: present a long time but going nowhere.
            lifetime_seconds = track.age / max(context.fps, 0.1)
            if (
                lifetime_seconds >= self.option("loiter_seconds")
                and track.displacement(60) < max(context.width * 0.04, 12)
                and self.sustained(
                    context, f"loiter:{track.track_id}", True, required_frames=3, cooldown_frames=600
                )
            ):
                result.events.append(
                    EventCandidate(
                        analytic=self.key,
                        event_type="loitering",
                        severity=Severity.MEDIUM,
                        confidence=0.7,
                        title="Loitering detected",
                        description=(
                            f"A person has remained in roughly the same spot for "
                            f"{int(lifetime_seconds // 60)} minutes."
                        ),
                        box=track.box,
                        track_id=track.track_id,
                        metadata={"seconds": round(lifetime_seconds, 1)},
                    )
                )

        # Drop state for people who have left the scene.
        for track_id in list(windows):
            if track_id not in active_ids:
                windows.pop(track_id, None)
                posture_since.pop(track_id, None)

        flow = flow_statistics(getattr(context.motion, "flow", None))
        result.metrics = {
            "people": len(people),
            "postures": counts,
            "activity_index": round(
                float(np.mean([d.attributes.get("speed", 0) for d in result.detections]))
                if result.detections else 0.0,
                4,
            ),
            "flow_magnitude": flow["magnitude"],
            "flow_coherence": flow["coherence"],
        }
        return result
