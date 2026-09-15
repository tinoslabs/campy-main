"""Synthetic scene generator.

Used for three real purposes:

1. **Tests** — deterministic scenes that must produce specific events.
2. **Demo mode** — a new customer can watch the whole platform work before
   connecting a single real camera.
3. **Training data bootstrap** — generates labelled samples for the fire,
   gesture and object classifiers so a workspace can run its first training job
   on day one.

It is not a substitute for real footage; it produces the *structure* the
detectors key on (motion, colour, flicker, geometry), not photorealism.
"""
from __future__ import annotations

import numpy as np


class SceneSimulator:
    """Renders a simple workplace scene with controllable actors."""

    def __init__(
        self,
        width: int = 320,
        height: int = 180,
        seed: int = 7,
        noise: float = 3.0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.rng = np.random.default_rng(seed)
        self.noise = float(noise)
        self.frame_index = 0
        self.background = self._make_background()

    def _make_background(self) -> np.ndarray:
        """A plausible indoor scene: floor, wall, shelving, ceiling lights."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.float32)
        horizon = int(self.height * 0.45)

        canvas[:horizon] = np.array([196, 198, 202], dtype=np.float32)      # wall
        canvas[horizon:] = np.array([132, 130, 126], dtype=np.float32)      # floor

        # Shelving on the right.
        shelf_x = int(self.width * 0.72)
        canvas[int(horizon * 0.5) : int(self.height * 0.8), shelf_x:] = [148, 122, 96]
        for row in range(3):
            y = int(horizon * 0.5) + int((self.height * 0.8 - horizon * 0.5) / 3 * row)
            canvas[y : y + 2, shelf_x:] = [104, 84, 64]

        # Doorway on the left.
        canvas[int(horizon * 0.4) : int(self.height * 0.9), int(self.width * 0.05) : int(self.width * 0.16)] = [
            88, 90, 96
        ]
        # A subtle lighting gradient so the frame is not perfectly flat.
        gradient = np.linspace(1.06, 0.92, self.width, dtype=np.float32)[None, :, None]
        canvas *= gradient
        return np.clip(canvas, 0, 255)

    def _noise(self) -> np.ndarray:
        return self.rng.normal(0, self.noise, (self.height, self.width, 3)).astype(np.float32)

    # -- actors ------------------------------------------------------------
    def _draw_person(self, canvas, cx: float, cy: float, height_px: float, shade=(58, 62, 74), carrying=False, bend=0.0):
        """Draw a simple upright figure: head, torso, legs."""
        # Once the subject's centre leaves the frame they are gone; without
        # this the head disc would clip to a permanent sliver at the edge.
        if not (0 <= cx < self.width):
            return canvas

        height_px = max(height_px, 12)
        width_px = height_px * 0.36
        head_r = height_px * 0.11

        effective_height = height_px * (1.0 - 0.35 * bend)
        top = cy - effective_height
        x1 = int(np.clip(cx - width_px / 2, 0, self.width))
        x2 = int(np.clip(cx + width_px / 2, 0, self.width))
        y1 = int(np.clip(top + head_r * 2, 0, self.height))
        y2 = int(np.clip(cy, 0, self.height))
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = shade

        # Head
        hy = int(np.clip(top + head_r, 0, self.height))
        hx = int(np.clip(cx, 0, self.width))
        r = int(max(head_r, 2))
        ys, xs = np.ogrid[-r : r + 1, -r : r + 1]
        disc = xs * xs + ys * ys <= r * r
        y0, y1h = max(hy - r, 0), min(hy + r + 1, self.height)
        x0, x1h = max(hx - r, 0), min(hx + r + 1, self.width)
        sub = disc[: y1h - y0, : x1h - x0]
        canvas[y0:y1h, x0:x1h][sub] = [186, 152, 128]        # skin tone for face detection

        if carrying:
            bag_w = int(width_px * 0.55)
            bag_h = int(height_px * 0.18)
            by = int(np.clip(cy - height_px * 0.45, 0, self.height))
            bx = int(np.clip(cx + width_px * 0.4, 0, self.width - bag_w))
            canvas[by : by + bag_h, bx : bx + bag_w] = [40, 44, 52]
        return canvas

    def _draw_fire(self, canvas, cx: float, cy: float, scale: float, phase: float):
        """Flickering flame: hot core, orange body, all with strong Cr > Cb."""
        radius = max(scale * (1.0 + 0.28 * np.sin(phase * 2.7)), 3)
        ys, xs = np.mgrid[0 : self.height, 0 : self.width]
        # Flames are taller than they are wide.
        distance = np.sqrt(((xs - cx) / radius) ** 2 + ((ys - cy) / (radius * 1.7)) ** 2)

        body = distance < 1.0
        core = distance < 0.45
        canvas[body] = [242, 128, 26]
        canvas[core] = [255, 224, 96]
        # Flicker the intensity frame to frame — the temporal signature.
        canvas[body] *= 0.86 + 0.22 * abs(np.sin(phase * 3.9))
        return np.clip(canvas, 0, 255)

    def _draw_smoke(self, canvas, cx: float, cy: float, scale: float, phase: float):
        ys, xs = np.mgrid[0 : self.height, 0 : self.width]
        radius = max(scale, 4)
        distance = np.sqrt(((xs - cx) / (radius * 1.5)) ** 2 + ((ys - cy) / radius) ** 2)
        plume = distance < 1.0
        alpha = np.clip(1.0 - distance, 0, 1)[..., None] * 0.75
        grey = np.array([158, 158, 160], dtype=np.float32) * (0.92 + 0.08 * np.sin(phase))
        canvas[plume] = (canvas * (1 - alpha) + grey * alpha)[plume]
        return canvas

    def _draw_object(self, canvas, x, y, w, h, colour=(46, 48, 56)):
        x1, y1 = int(np.clip(x, 0, self.width)), int(np.clip(y, 0, self.height))
        x2, y2 = int(np.clip(x + w, 0, self.width)), int(np.clip(y + h, 0, self.height))
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = colour
        return canvas

    # -- frame generation --------------------------------------------------
    def frame(self, actors: list[dict] | None = None) -> np.ndarray:
        """Render one frame. ``actors`` describes what is in the scene."""
        canvas = self.background.copy() + self._noise()
        phase = self.frame_index * 0.6

        for actor in actors or []:
            kind = actor.get("kind", "person")
            if kind == "person":
                canvas = self._draw_person(
                    canvas,
                    actor["x"], actor["y"],
                    actor.get("height", self.height * 0.42),
                    actor.get("shade", (58, 62, 74)),
                    actor.get("carrying", False),
                    actor.get("bend", 0.0),
                )
            elif kind == "fire":
                canvas = self._draw_fire(canvas, actor["x"], actor["y"], actor.get("scale", 14), phase)
            elif kind == "smoke":
                canvas = self._draw_smoke(canvas, actor["x"], actor["y"], actor.get("scale", 24), phase)
            elif kind == "object":
                canvas = self._draw_object(
                    canvas, actor["x"], actor["y"],
                    actor.get("w", 18), actor.get("h", 14), actor.get("colour", (46, 48, 56)),
                )

        self.frame_index += 1
        return np.clip(canvas, 0, 255).astype(np.uint8)

    # -- ready-made scenarios ---------------------------------------------
    def walk_across(self, frames: int = 60, y_ratio: float = 0.82, speed: float | None = None):
        """A person walking left to right."""
        speed = speed if speed is not None else self.width / max(frames - 10, 1)
        for i in range(frames):
            x = self.width * 0.12 + speed * i
            yield self.frame([{"kind": "person", "x": x, "y": self.height * y_ratio}])

    def idle_person(self, frames: int = 200, x_ratio: float = 0.45, arrive_frames: int = 24):
        """Someone walks in, then stands still for the rest of the clip.

        Note the arrival: a subject present in the *very first* frame is baked
        into the initial background model and — correctly — never registers as
        motion. Real deployments see people arrive; so does this scenario.
        """
        target = self.width * x_ratio
        start = self.width * 0.08
        for i in range(frames):
            if i < arrive_frames:
                x = start + (target - start) * (i / max(arrive_frames - 1, 1))
            else:
                # Tiny postural sway, well under the motion threshold.
                x = target + 0.4 * np.sin(i * 0.25)
            yield self.frame([{"kind": "person", "x": x, "y": self.height * 0.82}])

    def crowd(self, frames: int = 60, people: int = 10, arrive_frames: int = 22):
        """A group walks in from both sides and mills around in the middle."""
        targets = self.rng.uniform(0.16, 0.86, people)
        starts = np.where(targets < 0.5, -0.08, 1.08)
        for i in range(frames):
            progress = min(i / max(arrive_frames - 1, 1), 1.0)
            actors = []
            for index, (start, target) in enumerate(zip(starts, targets)):
                base = start + (target - start) * progress
                # Once arrived, keep milling so the crowd never freezes into
                # the background.
                mill = 0.018 * np.sin(i * 0.32 + index * 1.7) if progress >= 1.0 else 0.0
                actors.append(
                    {
                        "kind": "person",
                        "x": self.width * (base + mill),
                        "y": self.height * (0.72 + 0.12 * ((index % 3) / 3)),
                        "height": self.height * (0.30 + 0.06 * (index % 3)),
                    }
                )
            yield self.frame(actors)

    def running_person(self, frames: int = 40, fps: float = 6.0, laps: int = 3):
        """Someone genuinely running — around 1.8 body-heights per second.

        A 1.7 m person running at ~3 m/s covers ~1.8 of their own heights each
        second, which at 6 fps is a large step per frame. They therefore cross
        a single camera's view in a couple of seconds, so the clip loops them
        back around to give the classifier a usable window.
        """
        person_height = self.height * 0.42
        step = 1.8 * person_height / max(fps, 0.1)      # px per frame
        crossing = int(self.width / step) + 1
        gap = 6                                          # empty frames between passes
        cycle = crossing + gap
        for i in range(frames):
            position = i % cycle
            if position < crossing:
                yield self.frame(
                    [{"kind": "person", "x": step * position, "y": self.height * 0.84}]
                )
            else:
                yield self.frame([])
        _ = laps

    def fire_outbreak(self, frames: int = 40, warmup: int = 12):
        """Quiet scene, then a growing flame."""
        for i in range(frames):
            if i < warmup:
                yield self.frame([])
            else:
                growth = (i - warmup) / max(frames - warmup, 1)
                yield self.frame(
                    [
                        {
                            "kind": "fire",
                            "x": self.width * 0.55,
                            "y": self.height * 0.72,
                            "scale": 9 + 16 * growth,
                        }
                    ]
                )

    def abandoned_object(self, frames: int = 90, drop_at: int = 20, leave_at: int = 34):
        """Someone walks in, puts something down, and leaves it behind."""
        object_x = self.width * 0.52
        object_y = self.height * 0.78
        for i in range(frames):
            actors = []
            if i < leave_at:
                person_x = self.width * 0.15 + (object_x - self.width * 0.15) * min(i / max(drop_at, 1), 1.0)
                actors.append({"kind": "person", "x": person_x, "y": self.height * 0.84})
            elif i < leave_at + 14:
                progress = (i - leave_at) / 14
                actors.append(
                    {"kind": "person", "x": object_x + (self.width - object_x) * progress, "y": self.height * 0.84}
                )
            if i >= drop_at:
                actors.append({"kind": "object", "x": object_x, "y": object_y, "w": 20, "h": 15})
            yield self.frame(actors)

    def restricted_entry(self, frames: int = 50):
        """A person walks into the right-hand third of the frame."""
        for i in range(frames):
            x = self.width * 0.2 + (self.width * 0.62) * (i / max(frames - 1, 1))
            yield self.frame([{"kind": "person", "x": x, "y": self.height * 0.84}])


def synthetic_classification_dataset(
    classes: list[str],
    samples_per_class: int = 40,
    size: int = 48,
    seed: int = 11,
):
    """Generate a small labelled image set for bootstrapping a classifier.

    Each class gets a distinct, learnable visual signature. Real customers
    replace this with their own footage; it exists so that "train a model"
    is a working button on day one rather than a dead end.
    """
    rng = np.random.default_rng(seed)
    images, labels = [], []

    for index, name in enumerate(classes):
        for _ in range(samples_per_class):
            canvas = rng.normal(120, 18, (size, size, 3)).astype(np.float32)
            key = name.lower()
            cx, cy = rng.integers(size // 4, 3 * size // 4, 2)
            r = rng.integers(size // 6, size // 3)
            ys, xs = np.ogrid[0:size, 0:size]
            disc = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r

            if key in {"fire", "flame"}:
                canvas[disc] = [245, 140, 30] + rng.normal(0, 12, 3)
                canvas[(xs - cx) ** 2 + (ys - cy) ** 2 <= (r // 2) ** 2] = [255, 225, 100]
            elif key == "smoke":
                canvas[disc] = [155, 156, 158] + rng.normal(0, 8, 3)
            elif key in {"face", "person"}:
                canvas[disc] = [188, 150, 126] + rng.normal(0, 10, 3)
                eye = max(r // 4, 1)
                canvas[cy - eye : cy, cx - r // 2 : cx - r // 4] = [50, 40, 35]
                canvas[cy - eye : cy, cx + r // 4 : cx + r // 2] = [50, 40, 35]
            elif key in {"knife", "sharp_tool", "weapon"}:
                canvas[cy - 1 : cy + 2, max(cx - r, 0) : cx + r] = [220, 224, 230]
            elif key in {"bag", "box", "object"}:
                canvas[max(cy - r, 0) : cy + r, max(cx - r, 0) : cx + r] = [70, 66, 60] + rng.normal(0, 8, 3)
            else:  # "normal" / background class
                canvas += rng.normal(0, 6, (size, size, 3))

            images.append(np.clip(canvas, 0, 255).astype(np.uint8))
            labels.append(index)

    order = rng.permutation(len(images))
    return np.asarray(images)[order], np.asarray(labels)[order]
