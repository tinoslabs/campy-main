"""Fire & Smoke Detection.

Unlike a smoke sensor — which only reacts once combustion products physically
reach it — visual detection sees a fire the moment it is in frame, anywhere in
the room, and works in high-ceiling or outdoor spaces where sensors are useless.

Three independent signals must agree before we raise a critical alert:

1. **Colour** — flame occupies a tight region of YCrCb/HSV space (Cr > Cb,
   high value, hue in the red-yellow band) that ordinary objects rarely hit.
2. **Motion & flicker** — fire flickers at roughly 5-12 Hz. A red jacket or a
   parked forklift does not. We measure per-pixel temporal variance in the
   candidate region.
3. **Growth** — a real fire's candidate area expands over successive frames.

A trained ``campynet`` fire/smoke classifier, when deployed, adjudicates the
final decision on the candidate crop.
"""
from __future__ import annotations

import numpy as np

from ..vision.features import edge_density, image_entropy
from ..vision.image import crop, normalise, resize_bilinear, rgb_to_gray, rgb_to_hsv, rgb_to_ycrcb, to_chw
from ..vision.motion import binary_close, binary_open, connected_regions
from .base import AnalyticResult, BaseAnalytic, Detection, EventCandidate, Severity


class FireSmokeAnalytic(BaseAnalytic):
    key = "fire"
    label = "Fire & Smoke Detection"
    feature = "fire_detection"

    defaults = {
        "enabled": True,
        "flame_min_area_ratio": 0.0012,
        "smoke_min_area_ratio": 0.006,
        "flicker_frames": 10,
        "flicker_threshold": 6.0,       # temporal std of luminance in the region
        "growth_frames": 12,
        "confirm_frames": 4,
        "cooldown_frames": 200,
        "smoke_saturation_max": 0.22,   # smoke is grey: low saturation
        "smoke_value_range": [0.28, 0.82],
        "use_model": True,
        "model_threshold": 0.6,
        "min_confidence": 0.55,
        # Colour analysis runs at 1/N resolution. Flame and smoke regions are
        # large by nature, so half resolution costs nothing in accuracy and
        # cuts the most expensive stage of the whole pipeline by ~4x.
        "colour_downscale": 2,
    }

    def _analysis_frame(self, frame: np.ndarray):
        """Return ``(small_frame, scale_y, scale_x)`` for colour analysis."""
        factor = max(int(self.option("colour_downscale") or 1), 1)
        if factor <= 1:
            return np.asarray(frame, dtype=np.float32), 1.0, 1.0
        height, width = np.asarray(frame).shape[:2]
        target_h = max(height // factor, 32)
        target_w = max(width // factor, 32)
        small = resize_bilinear(frame, target_h, target_w)
        return small, height / target_h, width / target_w

    @staticmethod
    def _upscale(region, scale_y: float, scale_x: float):
        x1, y1, x2, y2 = region
        return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)

    # -- fast pre-gate -----------------------------------------------------
    def _may_contain_fire(self, frame: np.ndarray, motion, stride: int = 4) -> bool:
        """Microsecond test: could this frame possibly contain fire or smoke?

        Full HSV + YCrCb conversion over every pixel dominates the pipeline's
        cost, yet the overwhelming majority of frames contain nothing
        fire-coloured at all. Sampling every 4th pixel is ample — a fire large
        enough to matter spans far more than 16 pixels.
        """
        sample = np.asarray(frame, dtype=np.float32)[::stride, ::stride, :3]
        r, g, b = sample[..., 0], sample[..., 1], sample[..., 2]
        warm = (r > 118) & (r > g + 14) & (g >= b)
        if float(warm.mean()) >= self.option("flame_min_area_ratio") * 0.5:
            return True

        # Smoke carries no warm signature, so it is gated on moving,
        # desaturated pixels instead. No motion at all means no smoke.
        if motion is None or getattr(motion, "motion_ratio", 0.0) < self.option("smoke_min_area_ratio"):
            return False
        spread = np.max(sample, axis=-1) - np.min(sample, axis=-1)
        brightness = np.max(sample, axis=-1)
        grey = (spread < 42) & (brightness > 70) & (brightness < 215)
        return float(grey.mean()) >= self.option("smoke_min_area_ratio")

    # -- colour gating -----------------------------------------------------
    def _flame_mask(self, frame: np.ndarray) -> np.ndarray:
        """Classic Chen/Celik-style fire chromatic rules, combined."""
        ycrcb = rgb_to_ycrcb(frame)
        y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
        hsv = rgb_to_hsv(frame)
        hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        mean_y, mean_cr, mean_cb = float(y.mean()), float(cr.mean()), float(cb.mean())

        rule_luminance = y > mean_y
        rule_chroma = (cr > mean_cr) & (cb < mean_cb)
        rule_separation = (cr - cb) > 26
        rule_hue = ((hue <= 62) | (hue >= 345)) & (saturation >= 0.32) & (value >= 0.55)

        mask = rule_luminance & rule_chroma & rule_separation & rule_hue
        return binary_close(binary_open(mask, 1), 2)

    def _smoke_mask(self, frame: np.ndarray, motion) -> np.ndarray:
        """Smoke: desaturated, mid-brightness, low edge energy, *and* moving.

        The motion requirement is what keeps grey walls, concrete floors and
        overcast sky out of the mask.
        """
        hsv = rgb_to_hsv(frame)
        saturation, value = hsv[..., 1], hsv[..., 2]
        low, high = self.option("smoke_value_range")

        mask = (saturation <= self.option("smoke_saturation_max")) & (value >= low) & (value <= high)

        if motion is not None and getattr(motion, "mask", None) is not None and motion.mask.size > 1:
            motion_mask = motion.mask
            if motion_mask.shape != mask.shape:
                resized = resize_bilinear(motion_mask.astype(np.float32), mask.shape[0], mask.shape[1])
                motion_mask = resized > 0.4
            mask &= motion_mask
        else:
            return np.zeros_like(mask)

        return binary_close(binary_open(mask, 1), 2)

    # -- temporal signals --------------------------------------------------
    def _flicker_score(self, context, region, scratch) -> float:
        """Temporal std of mean luminance inside the candidate region."""
        gray = rgb_to_gray(crop(context.frame, region))
        if gray.size == 0:
            return 0.0
        history = scratch.setdefault("luma_history", [])
        history.append(float(gray.mean()))
        window = int(self.option("flicker_frames"))
        if len(history) > window * 2:
            del history[: len(history) - window * 2]
        if len(history) < 4:
            return 0.0
        return float(np.std(history[-window:]))

    def _growth_score(self, area_ratio: float, scratch) -> float:
        """Positive when the candidate area is trending upward."""
        history = scratch.setdefault("area_history", [])
        history.append(area_ratio)
        window = int(self.option("growth_frames"))
        if len(history) > window * 2:
            del history[: len(history) - window * 2]
        if len(history) < 6:
            return 0.0
        first_half = float(np.mean(history[-window : -window // 2])) if len(history) >= window else float(np.mean(history[: len(history) // 2]))
        second_half = float(np.mean(history[-window // 2 :]))
        if first_half <= 1e-9:
            return 1.0 if second_half > 1e-6 else 0.0
        return float(np.clip((second_half - first_half) / first_half, -1.0, 2.0))

    def _model_score(self, context, region, positive_class: str) -> float | None:
        if not self.option("use_model") or context.registry is None:
            return None
        model = context.registry.get("campynet_fire")
        if model is None:
            return None
        meta = getattr(model, "meta", {}) or {}
        classes = meta.get("classes") or ["normal", "fire", "smoke"]
        if positive_class not in classes:
            return None
        size = int(meta.get("input_size", 96))
        patch = crop(context.frame, region, padding=0.15)
        if patch.size == 0:
            return None
        batch = to_chw(normalise(resize_bilinear(patch, size, size)))[None, ...]
        from ..nn.functional import softmax

        probabilities = softmax(model.predict(batch), axis=-1)[0]
        return float(probabilities[classes.index(positive_class)])

    # -- main --------------------------------------------------------------
    def analyse(self, context) -> AnalyticResult:
        result = AnalyticResult(analytic=self.key)
        frame = context.frame
        frame_area = float(context.width * context.height)
        scratch = context.scratch(self.key)

        # Cheapest gate first: a subsampled warm-pixel / moving-grey test that
        # lets an ordinary frame skip colour analysis entirely.
        if not self._may_contain_fire(frame, context.motion):
            result.metrics = {
                "flame_ratio": 0.0, "smoke_ratio": 0.0,
                "flame_regions": 0, "smoke_regions": 0,
                "flame_confidence": 0.0, "smoke_confidence": 0.0,
                "prefiltered": True,
            }
            # Keep the temporal histories alive so growth tracking stays valid.
            scratch.setdefault("area_history", []).append(0.0)
            return result

        small, scale_y, scale_x = self._analysis_frame(frame)
        small_area = float(small.shape[0] * small.shape[1])

        # ---------------- flame ----------------
        flame_mask = self._flame_mask(small)
        flame_ratio = float(flame_mask.mean())
        flame_regions = [
            self._upscale(region, scale_y, scale_x)
            for region in connected_regions(
                flame_mask, min_area=max(int(small_area * 0.0004), 6), max_regions=6
            )
        ]

        best_flame = None
        if flame_regions and flame_ratio >= self.option("flame_min_area_ratio"):
            region = tuple(float(v) for v in flame_regions[0])
            area_ratio = ((region[2] - region[0]) * (region[3] - region[1])) / frame_area
            flicker = self._flicker_score(context, region, scratch)
            growth = self._growth_score(flame_ratio, scratch)
            model_score = self._model_score(context, region, "fire")

            flicker_norm = float(np.clip(flicker / max(self.option("flicker_threshold"), 1e-6), 0, 1.5))
            growth_norm = float(np.clip(growth, 0, 1))
            colour_norm = float(np.clip(flame_ratio / 0.02, 0, 1))

            confidence = 0.40 * colour_norm + 0.35 * min(flicker_norm, 1.0) + 0.25 * growth_norm
            if model_score is not None:
                # The learned model gets the final say once it is deployed.
                confidence = 0.35 * confidence + 0.65 * model_score
            best_flame = {
                "region": region,
                "confidence": float(np.clip(confidence, 0, 0.99)),
                "flicker": round(flicker, 3),
                "growth": round(growth, 3),
                "area_ratio": round(area_ratio, 5),
                "model_score": round(model_score, 4) if model_score is not None else None,
            }

        # ---------------- smoke ----------------
        smoke_mask = self._smoke_mask(small, context.motion)
        smoke_ratio = float(smoke_mask.mean()) if smoke_mask.size > 1 else 0.0
        smoke_regions = [
            self._upscale(region, scale_y, scale_x)
            for region in connected_regions(
                smoke_mask, min_area=max(int(small_area * 0.002), 12), max_regions=4
            )
        ]

        best_smoke = None
        if smoke_regions and smoke_ratio >= self.option("smoke_min_area_ratio"):
            region = tuple(float(v) for v in smoke_regions[0])
            patch = crop(frame, region)
            # Smoke blurs detail: low edge density, but non-trivial entropy.
            edges = edge_density(patch, threshold=30.0)
            entropy = image_entropy(patch)
            texture_score = float(np.clip((0.22 - edges) / 0.22, 0, 1)) * float(np.clip(entropy / 5.0, 0, 1))
            growth = self._growth_score(smoke_ratio, scratch.setdefault("smoke", {}))
            model_score = self._model_score(context, region, "smoke")

            confidence = 0.45 * float(np.clip(smoke_ratio / 0.05, 0, 1)) + 0.35 * texture_score + 0.20 * float(np.clip(growth, 0, 1))
            if model_score is not None:
                confidence = 0.35 * confidence + 0.65 * model_score
            best_smoke = {
                "region": region,
                "confidence": float(np.clip(confidence, 0, 0.99)),
                "edge_density": round(edges, 4),
                "entropy": round(entropy, 3),
                "growth": round(growth, 3),
                "model_score": round(model_score, 4) if model_score is not None else None,
            }

        threshold = self.option("min_confidence")

        if best_flame and best_flame["confidence"] >= threshold:
            result.detections.append(
                Detection(
                    label="fire",
                    confidence=best_flame["confidence"],
                    box=tuple(float(v) for v in best_flame["region"]),
                    attributes={k: v for k, v in best_flame.items() if k != "region"},
                )
            )
            if self.sustained(
                context, "fire", True,
                required_frames=self.option("confirm_frames"),
                cooldown_frames=self.option("cooldown_frames"),
            ):
                result.events.append(
                    EventCandidate(
                        analytic=self.key,
                        event_type="fire_detected",
                        severity=Severity.CRITICAL,
                        confidence=best_flame["confidence"],
                        title="Fire detected",
                        description=(
                            "Flame-coloured pixels with fire-like flicker and a growing area "
                            "were detected. Verify immediately and trigger your emergency "
                            "procedure if confirmed."
                        ),
                        box=tuple(float(v) for v in best_flame["region"]),
                        metadata=best_flame | {"region": [round(v, 1) for v in best_flame["region"]]},
                    )
                )

        if best_smoke and best_smoke["confidence"] >= threshold:
            result.detections.append(
                Detection(
                    label="smoke",
                    confidence=best_smoke["confidence"],
                    box=tuple(float(v) for v in best_smoke["region"]),
                    attributes={k: v for k, v in best_smoke.items() if k != "region"},
                )
            )
            if self.sustained(
                context, "smoke", True,
                required_frames=self.option("confirm_frames") + 2,
                cooldown_frames=self.option("cooldown_frames"),
            ):
                result.events.append(
                    EventCandidate(
                        analytic=self.key,
                        event_type="smoke_detected",
                        severity=Severity.HIGH,
                        confidence=best_smoke["confidence"],
                        title="Smoke detected",
                        description=(
                            "A spreading, low-texture grey region consistent with smoke was "
                            "detected. Smoke usually precedes visible flame — check now."
                        ),
                        box=tuple(float(v) for v in best_smoke["region"]),
                        metadata=best_smoke | {"region": [round(v, 1) for v in best_smoke["region"]]},
                    )
                )

        result.metrics = {
            "flame_ratio": round(flame_ratio, 6),
            "smoke_ratio": round(smoke_ratio, 6),
            "flame_regions": len(flame_regions),
            "smoke_regions": len(smoke_regions),
            "flame_confidence": round(best_flame["confidence"], 4) if best_flame else 0.0,
            "smoke_confidence": round(best_smoke["confidence"], 4) if best_smoke else 0.0,
        }
        return result
