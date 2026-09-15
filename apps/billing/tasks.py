"""Scheduled billing maintenance."""
from __future__ import annotations

import logging

from django.utils import timezone

logger = logging.getLogger("campy.billing")

try:
    from celery import shared_task
except ImportError:  # pragma: no cover
    def shared_task(*args, **kwargs):
        def decorator(func):
            func.delay = func
            return func

        return decorator(args[0]) if args and callable(args[0]) else decorator


@shared_task(name="apps.billing.tasks.expire_due_subscriptions")
def expire_due_subscriptions() -> dict:
    """Move ended trials and periods into the right state.

    A trial that runs out becomes *past due*, not *expired*: the customer keeps
    a grace window to pay rather than losing monitoring the moment the clock
    strikes.
    """
    from .models import Subscription

    now = timezone.now()
    counts = {"trials_ended": 0, "periods_ended": 0, "cancelled": 0}

    for subscription in Subscription.objects.filter(
        status=Subscription.Status.TRIALING, trial_ends_at__lt=now
    ):
        subscription.status = Subscription.Status.PAST_DUE
        subscription.save(update_fields=["status", "updated_at"])
        counts["trials_ended"] += 1

    for subscription in Subscription.objects.filter(
        status=Subscription.Status.ACTIVE, current_period_end__lt=now
    ):
        if subscription.cancel_at_period_end:
            subscription.status = Subscription.Status.CANCELLED
            subscription.ended_at = now
            subscription.save(update_fields=["status", "ended_at", "updated_at"])
            counts["cancelled"] += 1
        else:
            subscription.status = Subscription.Status.PAST_DUE
            subscription.save(update_fields=["status", "updated_at"])
            counts["periods_ended"] += 1

    # Past-due workspaces that never paid, past their grace period.
    for subscription in Subscription.objects.filter(status=Subscription.Status.PAST_DUE):
        if not subscription.in_grace_period:
            subscription.status = Subscription.Status.EXPIRED
            subscription.ended_at = now
            subscription.save(update_fields=["status", "ended_at", "updated_at"])
            organization = subscription.organization
            organization.status = organization.Status.SUSPENDED
            organization.save(update_fields=["status", "updated_at"])

    logger.info("Subscription sweep: %s", counts)
    return counts


@shared_task(name="apps.billing.tasks.record_daily_usage")
def record_daily_usage() -> int:
    """Snapshot per-workspace consumption for metering and overage billing."""
    from django.db.models import Sum

    from apps.accounts.models import Organization
    from apps.cameras.models import Camera
    from apps.events.models import Event

    from .models import UsageRecord

    today = timezone.localdate()
    written = 0
    for organization in Organization.objects.all():
        cameras = Camera.objects.filter(organization=organization)
        record, _ = UsageRecord.objects.update_or_create(
            organization=organization,
            date=today,
            defaults={
                "cameras_active": cameras.filter(status=Camera.Status.ONLINE).count(),
                "frames_processed": cameras.aggregate(total=Sum("frames_processed"))["total"] or 0,
                "events_generated": Event.objects.filter(
                    organization=organization, occurred_at__date=today
                ).count(),
            },
        )
        _ = record
        written += 1
    return written
