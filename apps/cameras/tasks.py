"""Background camera tasks."""
from __future__ import annotations

import logging
from datetime import timedelta

from django.db.models import Q
from django.utils import timezone

logger = logging.getLogger("campy.cameras")

try:
    from celery import shared_task
except ImportError:  # pragma: no cover
    def shared_task(*args, **kwargs):
        def decorator(func):
            func.delay = func
            return func

        return decorator(args[0]) if args and callable(args[0]) else decorator


@shared_task(name="apps.cameras.tasks.sweep_camera_health")
def sweep_camera_health() -> dict:
    """Mark cameras offline when they stop delivering frames."""
    from .models import Camera

    cutoff = timezone.now() - timedelta(minutes=10)
    stale = Camera.objects.filter(status=Camera.Status.ONLINE, is_active=True).filter(
        Q(last_frame_at__lt=cutoff) | Q(last_frame_at__isnull=True)
    )
    count = 0
    for camera in stale:
        camera.mark_offline("No frames received in the last 10 minutes")
        count += 1
    return {"marked_offline": count}


@shared_task(name="apps.cameras.tasks.run_camera_worker")
def run_camera_worker(camera_id: int, max_frames: int | None = None) -> dict:
    from .models import Camera
    from .services import CameraWorker

    camera = Camera.objects.filter(pk=camera_id, is_active=True).select_related(
        "organization", "site"
    ).first()
    if camera is None:
        return {"error": "camera not found"}
    worker = CameraWorker(camera, max_frames=max_frames)
    return worker.run().as_dict()
