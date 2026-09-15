"""Run the analytics pipeline against one camera, or every active camera.

    python manage.py run_camera --camera <uid>
    python manage.py run_camera --all --max-frames 500
    python manage.py run_camera --site <uid> --workers 4

Each camera runs in its own thread. For more than a handful of streams, run
several processes and shard with ``--site`` so a crash never takes them all
down together.
"""
from __future__ import annotations

import signal
import threading
import time

from django.core.management.base import BaseCommand, CommandError

from apps.cameras.models import Camera
from apps.cameras.services import CameraWorker


class Command(BaseCommand):
    help = "Run the Campy AI analytics pipeline against live camera streams."

    def add_arguments(self, parser):
        parser.add_argument("--camera", help="Camera UID to run")
        parser.add_argument("--site", help="Run every active camera at this site UID")
        parser.add_argument("--organization", help="Run every active camera in this workspace slug")
        parser.add_argument("--all", action="store_true", help="Run every active camera")
        parser.add_argument(
            "--max-frames", type=int, default=None,
            help="Stop after this many frames per camera (default: run until stopped)",
        )
        parser.add_argument(
            "--no-snapshots", action="store_true", help="Do not store evidence images"
        )
        parser.add_argument(
            "--workers", type=int, default=8, help="Maximum concurrent camera threads"
        )

    def handle(self, *args, **options):
        cameras = self.select_cameras(options)
        if not cameras:
            raise CommandError("No matching active cameras. Add one, or check your filters.")

        self.stdout.write(
            self.style.MIGRATE_HEADING(f"Starting {len(cameras)} camera worker(s)")
        )
        for camera in cameras:
            analytics = camera.active_analytics()
            blocked = camera.blocked_analytics
            self.stdout.write(f"  {camera.name} ({camera.site.name}) — {camera.get_protocol_display()}")
            self.stdout.write(f"    analytics: {', '.join(analytics) or 'none enabled'}")
            if blocked:
                self.stdout.write(
                    self.style.WARNING(f"    blocked by plan: {', '.join(blocked)}")
                )

        workers, threads = [], []
        stopping = threading.Event()

        def shutdown(signum, frame):
            self.stdout.write("\nStopping workers…")
            stopping.set()
            for worker in workers:
                worker.stop()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        limit = max(int(options["workers"]), 1)
        for camera in cameras[:limit]:
            worker = CameraWorker(
                camera,
                max_frames=options["max_frames"],
                save_snapshots=not options["no_snapshots"],
            )
            workers.append(worker)
            thread = threading.Thread(target=worker.run, name=f"camera-{camera.pk}", daemon=True)
            thread.start()
            threads.append(thread)

        if len(cameras) > limit:
            self.stdout.write(
                self.style.WARNING(
                    f"  {len(cameras) - limit} camera(s) skipped — raise --workers or run "
                    "another process."
                )
            )

        try:
            while any(thread.is_alive() for thread in threads):
                time.sleep(0.5)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            shutdown(None, None)

        self.stdout.write("")
        total_frames = total_events = 0
        for worker in workers:
            stats = worker.stats
            total_frames += stats.frames
            total_events += stats.events
            self.stdout.write(
                f"  {worker.camera.name}: {stats.frames} frames @ {stats.fps:.1f} fps, "
                f"{stats.events} events, {stats.errors} errors"
            )
        self.stdout.write(
            self.style.SUCCESS(f"\nProcessed {total_frames} frames, generated {total_events} events.")
        )

    def select_cameras(self, options) -> list[Camera]:
        queryset = Camera.objects.filter(is_active=True).select_related("site", "organization")

        if options["camera"]:
            queryset = queryset.filter(uid=options["camera"])
        elif options["site"]:
            queryset = queryset.filter(site__uid=options["site"])
        elif options["organization"]:
            queryset = queryset.filter(organization__slug=options["organization"])
        elif not options["all"]:
            raise CommandError(
                "Choose what to run: --camera, --site, --organization or --all."
            )

        # Skip cameras whose analytics are all blocked by the workspace's plan.
        return [camera for camera in queryset if camera.active_analytics()]
