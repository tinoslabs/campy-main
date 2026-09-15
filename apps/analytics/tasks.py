"""Scheduled analytics roll-ups and report delivery."""
from __future__ import annotations

import logging
from datetime import timedelta

from django.utils import timezone

logger = logging.getLogger("campy.analytics")

try:
    from celery import shared_task
except ImportError:  # pragma: no cover
    def shared_task(*args, **kwargs):
        def decorator(func):
            func.delay = func
            return func

        return decorator(args[0]) if args and callable(args[0]) else decorator


@shared_task(name="apps.analytics.tasks.rollup_daily_metrics")
def rollup_daily_metrics(days: int = 2) -> int:
    """Recompute the last couple of days so late-arriving events are counted."""
    from apps.accounts.models import Organization

    from .services import rollup_metrics_for

    written = 0
    today = timezone.localdate()
    for organization in Organization.objects.all():
        for offset in range(days):
            rollup_metrics_for(organization, today - timedelta(days=offset))
            written += 1
    return written


@shared_task(name="apps.analytics.tasks.run_scheduled_reports")
def run_scheduled_reports() -> int:
    """Generate and email every report that is due."""
    from django.conf import settings
    from django.core.mail import EmailMultiAlternatives
    from django.template.loader import render_to_string

    from .models import Report, ReportRun
    from .services import build_report

    now = timezone.now()
    delivered = 0

    for report in Report.objects.filter(is_active=True).exclude(frequency=Report.Frequency.ONCE):
        if report.next_run_at and report.next_run_at > now:
            continue

        run = ReportRun.objects.create(
            report=report,
            period_start=now - timedelta(days=report.date_range_days or 7),
            period_end=now,
        )
        try:
            payload = build_report(report)
            run.payload = {"summary": payload["summary"]}
            run.status = ReportRun.Status.SUCCEEDED

            if report.recipients:
                html = render_to_string("analytics/report_email.html", payload)
                message = EmailMultiAlternatives(
                    subject=f"{report.name} — {payload['period_start']} to {payload['period_end']}",
                    body=f"Your {report.get_kind_display()} report is attached.",
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    to=report.recipients,
                )
                message.attach_alternative(html, "text/html")
                message.send(fail_silently=True)
                run.delivered_to = report.recipients
                delivered += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Report %s failed", report.pk)
            run.status = ReportRun.Status.FAILED
            run.error = str(exc)[:400]

        run.finished_at = timezone.now()
        run.save(update_fields=["payload", "status", "error", "finished_at", "delivered_to", "updated_at"])

        interval = {"daily": 1, "weekly": 7, "monthly": 30}.get(report.frequency, 7)
        report.last_run_at = now
        report.next_run_at = now + timedelta(days=interval)
        report.run_count += 1
        report.save(update_fields=["last_run_at", "next_run_at", "run_count", "updated_at"])

    return delivered
