"""Report generation and metric roll-ups."""
from __future__ import annotations

from datetime import timedelta

from django.db.models import Count, Q
from django.utils import timezone

from apps.aiengine.detectors import ANALYTIC_CHOICES
from apps.cameras.models import Camera
from apps.events.models import Event, Incident

ANALYTIC_LABELS = dict(ANALYTIC_CHOICES)


def build_report(report) -> dict:
    """Assemble a report's data. Returns a template/CSV-friendly structure."""
    end = timezone.now()
    start = end - timedelta(days=report.date_range_days or 7)

    events = Event.objects.filter(organization=report.organization, occurred_at__range=(start, end))
    site_ids = list(report.sites.values_list("id", flat=True))
    camera_ids = list(report.cameras.values_list("id", flat=True))
    if camera_ids:
        events = events.filter(camera_id__in=camera_ids)
    elif site_ids:
        events = events.filter(camera__site_id__in=site_ids)
    if report.analytics:
        events = events.filter(analytic__in=report.analytics)

    total = events.count()
    sections = []

    # -- headline numbers -------------------------------------------------
    summary = {
        "total_events": total,
        "critical": events.filter(severity__in=["high", "critical"]).count(),
        "resolved": events.filter(status=Event.Status.RESOLVED).count(),
        "false_positives": events.filter(status=Event.Status.FALSE_POSITIVE).count(),
        "open": events.filter(status=Event.Status.OPEN).count(),
        "incidents": Incident.objects.filter(
            organization=report.organization, started_at__range=(start, end)
        ).count(),
        "cameras_reporting": events.values("camera").distinct().count(),
    }
    summary["false_positive_rate"] = round(
        summary["false_positives"] / max(total, 1) * 100, 1
    )
    summary["resolution_rate"] = round(summary["resolved"] / max(total, 1) * 100, 1)

    # -- by analytic ------------------------------------------------------
    rows = []
    for row in events.values("analytic").annotate(
        total=Count("id"),
        critical=Count("id", filter=Q(severity__in=["high", "critical"])),
        false_positive=Count("id", filter=Q(status=Event.Status.FALSE_POSITIVE)),
    ).order_by("-total"):
        rows.append([
            ANALYTIC_LABELS.get(row["analytic"], row["analytic"]),
            row["total"], row["critical"], row["false_positive"],
            f"{round(row['false_positive'] / max(row['total'], 1) * 100, 1)}%",
        ])
    sections.append(
        {
            "title": "Findings by analytic",
            "columns": ["Analytic", "Events", "High/critical", "False alarms", "False-alarm rate"],
            "rows": rows,
        }
    )

    # -- by camera --------------------------------------------------------
    rows = []
    for row in events.values("camera__name", "camera__site__name").annotate(
        total=Count("id"), critical=Count("id", filter=Q(severity__in=["high", "critical"]))
    ).order_by("-total")[:25]:
        rows.append([
            row["camera__name"] or "—", row["camera__site__name"] or "—",
            row["total"], row["critical"],
        ])
    sections.append(
        {"title": "Findings by camera", "columns": ["Camera", "Site", "Events", "High/critical"], "rows": rows}
    )

    # -- by day -----------------------------------------------------------
    rows = []
    days = min(report.date_range_days or 7, 90)
    for offset in range(days - 1, -1, -1):
        day = (end - timedelta(days=offset)).date()
        day_events = events.filter(occurred_at__date=day)
        rows.append([
            day.isoformat(), day_events.count(),
            day_events.filter(severity__in=["high", "critical"]).count(),
        ])
    sections.append({"title": "Daily volume", "columns": ["Date", "Events", "High/critical"], "rows": rows})

    # -- kind-specific ----------------------------------------------------
    if report.kind == report.Kind.ACTIVITY:
        rows = []
        for row in events.exclude(employee__isnull=True).values("employee__full_name", "employee__department").annotate(
            total=Count("id"), days_seen=Count("occurred_at__date", distinct=True)
        ).order_by("-total")[:40]:
            rows.append([
                row["employee__full_name"], row["employee__department"] or "—",
                row["days_seen"], row["total"],
            ])
        sections.append(
            {"title": "Employee activity", "columns": ["Employee", "Department", "Days seen", "Observations"], "rows": rows}
        )

    elif report.kind == report.Kind.INCIDENTS:
        rows = []
        incidents = Incident.objects.filter(
            organization=report.organization, started_at__range=(start, end)
        ).select_related("primary_camera")
        for incident in incidents.order_by("-started_at")[:100]:
            rows.append([
                incident.reference, incident.started_at.strftime("%Y-%m-%d %H:%M"),
                incident.severity, incident.status,
                incident.primary_camera.name if incident.primary_camera else "—",
                incident.event_count, incident.title,
            ])
        sections.append(
            {
                "title": "Incident log",
                "columns": ["Reference", "Started", "Severity", "Status", "Camera", "Events", "Summary"],
                "rows": rows,
            }
        )

    elif report.kind == report.Kind.CROWD:
        rows = []
        for row in events.filter(analytic="crowd").values("camera__name").annotate(
            total=Count("id")
        ).order_by("-total")[:25]:
            rows.append([row["camera__name"] or "—", row["total"]])
        sections.append({"title": "Crowd events by camera", "columns": ["Camera", "Events"], "rows": rows})

    elif report.kind == report.Kind.COMPLIANCE:
        from apps.accounts.models import AuditLog

        rows = []
        for entry in AuditLog.objects.filter(
            organization=report.organization, created_at__range=(start, end), is_sensitive=True
        ).order_by("-created_at")[:200]:
            rows.append([
                entry.created_at.strftime("%Y-%m-%d %H:%M"), entry.actor_label,
                entry.action, entry.target_label, entry.summary,
            ])
        sections.append(
            {
                "title": "Sensitive actions",
                "columns": ["When", "Who", "Action", "Target", "Detail"],
                "rows": rows,
            }
        )

    return {
        "report": report,
        "period_start": start.strftime("%Y-%m-%d %H:%M"),
        "period_end": end.strftime("%Y-%m-%d %H:%M"),
        "generated_at": timezone.now(),
        "summary": summary,
        "sections": sections,
    }


def rollup_metrics_for(organization, date) -> None:
    """Write one :class:`DailyMetric` row for an organisation and date."""
    from .models import DailyMetric

    events = Event.objects.filter(organization=organization, occurred_at__date=date)
    total = events.count()

    hourly = [0] * 24
    for occurred in events.values_list("occurred_at", flat=True):
        hourly[timezone.localtime(occurred).hour] += 1

    acknowledged = [
        (e.acknowledged_at - e.occurred_at).total_seconds()
        for e in events.exclude(acknowledged_at__isnull=True).only("acknowledged_at", "occurred_at")
    ]

    cameras = Camera.objects.filter(organization=organization, is_active=True)
    online = cameras.filter(status=Camera.Status.ONLINE).count()

    DailyMetric.objects.update_or_create(
        organization=organization, date=date, site=None, camera=None,
        defaults={
            "events_total": total,
            "events_critical": events.filter(severity="critical").count(),
            "events_high": events.filter(severity="high").count(),
            "events_by_analytic": {
                row["analytic"]: row["total"]
                for row in events.values("analytic").annotate(total=Count("id"))
            },
            "events_by_hour": hourly,
            "incidents_opened": Incident.objects.filter(
                organization=organization, started_at__date=date
            ).count(),
            "incidents_resolved": Incident.objects.filter(
                organization=organization, resolved_at__date=date
            ).count(),
            "false_positives": events.filter(status=Event.Status.FALSE_POSITIVE).count(),
            "median_response_seconds": (
                sorted(acknowledged)[len(acknowledged) // 2] if acknowledged else 0
            ),
            "unique_employees": events.exclude(employee__isnull=True)
            .values("employee").distinct().count(),
            "camera_uptime_percent": round(online / max(cameras.count(), 1) * 100, 1),
        },
    )
