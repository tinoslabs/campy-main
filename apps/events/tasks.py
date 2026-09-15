"""Background maintenance for events and evidence."""
from __future__ import annotations

import logging
from datetime import timedelta

from django.conf import settings
from django.utils import timezone

logger = logging.getLogger("campy.events")

try:
    from celery import shared_task
except ImportError:  # pragma: no cover - celery is optional
    def shared_task(*args, **kwargs):
        def decorator(func):
            func.delay = func
            return func

        return decorator(args[0]) if args and callable(args[0]) else decorator


@shared_task(name="apps.events.tasks.purge_expired_evidence")
def purge_expired_evidence() -> dict:
    """Delete snapshots and events past the workspace's retention window.

    Retention is not housekeeping — it is a compliance requirement. Holding
    video of identifiable people longer than the stated period is exactly what
    privacy regulators penalise.
    """
    from apps.accounts.models import Organization
    from apps.cameras.models import CameraSnapshot
    from apps.events.models import Event

    removed = {"snapshots": 0, "events": 0}
    default_events = settings.CAMPY["EVENT_RETENTION_DAYS"]
    default_snapshots = settings.CAMPY["SNAPSHOT_RETENTION_DAYS"]

    for organization in Organization.objects.all():
        subscription = organization.active_subscription
        event_days = (
            subscription.quota("retention_days", default_events) if subscription else default_events
        )
        if event_days < 0:
            continue
        snapshot_days = min(default_snapshots, event_days)

        snapshot_cutoff = timezone.now() - timedelta(days=snapshot_days)
        snapshots = CameraSnapshot.objects.filter(
            camera__organization=organization, captured_at__lt=snapshot_cutoff
        )
        for snapshot in snapshots.iterator(chunk_size=200):
            if snapshot.image:
                snapshot.image.delete(save=False)
            snapshot.delete()
            removed["snapshots"] += 1

        event_cutoff = timezone.now() - timedelta(days=event_days)
        deleted, _ = Event.objects.filter(
            organization=organization, occurred_at__lt=event_cutoff
        ).delete()
        removed["events"] += deleted

    logger.info("Evidence purge removed %s", removed)
    return removed


@shared_task(name="apps.events.tasks.run_escalations")
def run_escalations() -> int:
    from .services import escalate_unacknowledged

    return escalate_unacknowledged()


@shared_task(name="apps.events.tasks.deliver_event_alerts")
def deliver_event_alerts(event_id: int) -> int:
    from .models import Event
    from .services import dispatch_alerts

    event = Event.objects.filter(pk=event_id).select_related("organization", "camera").first()
    if event is None:
        return 0
    return len(dispatch_alerts(event))
