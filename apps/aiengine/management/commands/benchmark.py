"""Measure what one CPU core of this machine can actually do.

    python manage.py benchmark
    python manage.py benchmark --width 320 --height 180
    python manage.py benchmark --analytics gesture,geofence,fire

Capacity planning for a video system is not a guess: the cost of a frame
depends on the analysis resolution, which analytics are on, and how busy the
scene is. This runs the real pipeline over the simulator's scenes and reports
all three, so an installer can size a server for *their* hardware instead of
trusting a number measured on someone else's.
"""
from __future__ import annotations

import platform
import statistics
import time

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.aiengine.detectors import ANALYTIC_REGISTRY
from apps.aiengine.pipeline import CameraPipeline
from apps.aiengine.simulation import SceneSimulator

#: Scenes in rising order of cost — an empty corridor is cheap, a crowd is not.
SCENES = (
    ("empty room", "idle_person", (60,)),
    ("one person walking", "walk_across", (60,)),
    ("six people moving", "crowd", (60, 6)),
    ("fourteen people", "crowd", (60, 14)),
)
WARMUP = 12


class Command(BaseCommand):
    help = "Benchmark the analytics pipeline on this machine."

    def add_arguments(self, parser):
        campy = settings.CAMPY
        parser.add_argument("--width", type=int, default=int(campy["WORKER_FRAME_WIDTH"]))
        parser.add_argument("--height", type=int, default=int(campy["WORKER_FRAME_HEIGHT"]))
        parser.add_argument(
            "--analytics", default="",
            help="Comma-separated analytics to run (default: all seven)",
        )
        parser.add_argument(
            "--fps", type=float, default=float(campy["WORKER_TARGET_FPS"]),
            help="Analysis rate used to convert speed into cameras per core",
        )
        parser.add_argument("--frames", type=int, default=52)

    def handle(self, *args, **options):
        width, height, fps = options["width"], options["height"], options["fps"]
        keys = [k.strip() for k in options["analytics"].split(",") if k.strip()]
        keys = [k for k in keys if k in ANALYTIC_REGISTRY] or list(ANALYTIC_REGISTRY)

        self.stdout.write(self.style.MIGRATE_HEADING("Campy AI pipeline benchmark"))
        self.stdout.write(f"  machine    {platform.processor() or platform.machine()}")
        self.stdout.write(f"  resolution {width}x{height}")
        self.stdout.write(f"  analytics  {', '.join(keys)}")
        self.stdout.write(f"  rate       {fps:g} fps per camera\n")

        header = f"  {'scene':24s}{'ms/frame':>10s}{'fps/core':>10s}{'cameras/core':>14s}"
        self.stdout.write(header)
        self.stdout.write("  " + "-" * (len(header) - 2))

        results = []
        for label, method, args in SCENES:
            median, stages = self.measure(width, height, keys, method, args, options["frames"], fps)
            results.append((label, median))
            self.stdout.write(
                f"  {label:24s}{median:10.1f}{1000 / median:10.1f}{1000 / median / fps:14.1f}"
            )

        self.stdout.write("\n  Where the time goes, busiest scene (ms):")
        for name, value in sorted(stages.items(), key=lambda kv: -kv[1])[:8]:
            self.stdout.write(f"    {name:16s}{value:7.2f}")

        quiet = min(m for _, m in results)
        busy = max(m for _, m in results)
        self.stdout.write(
            self.style.SUCCESS(
                f"\n  Plan for the busy case: {1000 / busy / fps:.1f} cameras per core "
                f"({busy:.0f} ms/frame). A quiet camera costs {quiet:.0f} ms."
            )
        )
        self.stdout.write(
            "  Halving the analysis resolution roughly quarters the cost; turning off\n"
            "  analytics a camera does not need saves the rest."
        )

    def measure(self, width, height, keys, method, args, frames, fps):
        simulator = SceneSimulator(width=width, height=height, seed=3)
        scene = list(getattr(simulator, method)(*args))[:frames]
        pipeline = CameraPipeline(analytics=list(keys), fps=fps)

        for frame in scene[:WARMUP]:      # let the background model settle
            pipeline.process(frame)

        timings, stages = [], {}
        for frame in scene[WARMUP:]:
            started = time.perf_counter()
            result = pipeline.process(frame)
            timings.append((time.perf_counter() - started) * 1000)
            for name, value in result.metrics["timings_ms"].items():
                stages.setdefault(name, []).append(value)

        if not timings:
            return 0.0, {}
        return (
            statistics.median(timings),
            {name: statistics.median(values) for name, values in stages.items()},
        )
