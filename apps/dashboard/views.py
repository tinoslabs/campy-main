"""The Campy AI operator dashboard."""
from __future__ import annotations

import json
import time
from datetime import timedelta

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Count, Q, Sum
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from apps.accounts.access import platform_team_required, require_organization, require_perm
from apps.accounts.models import APIKey, AuditLog, Invitation, Membership, Organization
from apps.aiengine.detectors import ANALYTIC_CHOICES, analytic_metadata
from apps.aiengine.registry import MODEL_SLOTS, ModelProvider
from apps.analytics.models import Report
from apps.cameras.models import Camera, CameraSnapshot, Employee, Site, Zone
from apps.events.models import (
    AlertRule,
    Event,
    Incident,
    Notification,
    NotificationChannel,
)
from apps.training.models import Dataset, DatasetSample, ModelVersion, TrainingJob

from .forms import (
    AlertRuleForm,
    APIKeyForm,
    CameraForm,
    DatasetForm,
    EmployeeForm,
    EventFilterForm,
    NotificationChannelForm,
    ReportForm,
    SiteForm,
    TrainingJobForm,
    ZoneForm,
)

ANALYTIC_LABELS = dict(ANALYTIC_CHOICES)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _org(request):
    return request.organization


def _base_context(request, active: str, **extra) -> dict:
    """Sidebar counters every dashboard page needs."""
    organization = _org(request)
    context = {"active": active}
    if organization is not None:
        events = Event.objects.filter(organization=organization, status=Event.Status.OPEN)
        context.update(
            {
                "open_event_count": events.count(),
                "critical_event_count": events.filter(severity__in=["high", "critical"]).count(),
                "offline_camera_count": Camera.objects.filter(
                    organization=organization, is_active=True,
                    status__in=[Camera.Status.OFFLINE, Camera.Status.DEGRADED],
                ).count(),
                "running_job_count": TrainingJob.objects.filter(
                    organization=organization, status__in=[TrainingJob.Status.QUEUED, TrainingJob.Status.RUNNING]
                ).count(),
                "unread_notification_count": Notification.objects.filter(
                    recipient=request.user, read_at__isnull=True
                ).count(),
            }
        )
    context.update(extra)
    return context


def _sparkline(values: list[float], width: int = 240, height: int = 44) -> str:
    """Inline SVG polyline — no chart library, no network request."""
    if not values:
        return ""
    peak = max(values) or 1
    step = width / max(len(values) - 1, 1)
    points = " ".join(
        f"{index * step:.1f},{height - (value / peak) * (height - 4) - 2:.1f}"
        for index, value in enumerate(values)
    )
    return points


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
@login_required
@require_organization
def home(request):
    organization = _org(request)
    now = timezone.now()
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    previous_week = now - timedelta(days=14)

    events = Event.objects.filter(organization=organization)
    recent = events.filter(occurred_at__gte=day_ago)
    this_week = events.filter(occurred_at__gte=week_ago)
    last_week = events.filter(occurred_at__gte=previous_week, occurred_at__lt=week_ago)

    cameras = Camera.objects.filter(organization=organization, is_active=True)
    online = cameras.filter(status=Camera.Status.ONLINE).count()
    total_cameras = cameras.count()

    by_analytic = list(
        this_week.values("analytic").annotate(total=Count("id")).order_by("-total")
    )
    for row in by_analytic:
        row["label"] = ANALYTIC_LABELS.get(row["analytic"], row["analytic"])

    by_severity = {
        row["severity"]: row["total"]
        for row in this_week.values("severity").annotate(total=Count("id"))
    }

    # Hourly event volume for the last 24h.
    hourly = [0] * 24
    for occurred in recent.values_list("occurred_at", flat=True):
        hourly[timezone.localtime(occurred).hour] += 1

    # Seven-day trend.
    daily_counts = []
    for offset in range(6, -1, -1):
        day = (now - timedelta(days=offset)).date()
        daily_counts.append(
            events.filter(occurred_at__date=day).count()
        )

    acknowledged = this_week.exclude(acknowledged_at__isnull=True)
    response_times = [
        (e.acknowledged_at - e.occurred_at).total_seconds()
        for e in acknowledged.only("acknowledged_at", "occurred_at")[:500]
    ]
    median_response = sorted(response_times)[len(response_times) // 2] if response_times else 0

    context = _base_context(
        request, "home",
        stats={
            "events_24h": recent.count(),
            "events_7d": this_week.count(),
            "events_prev_7d": last_week.count(),
            "critical_open": events.filter(status=Event.Status.OPEN, severity__in=["high", "critical"]).count(),
            "incidents_open": Incident.objects.filter(
                organization=organization, status__in=[Incident.Status.OPEN, Incident.Status.INVESTIGATING]
            ).count(),
            "cameras_online": online,
            "cameras_total": total_cameras,
            "camera_health": round(online / total_cameras * 100) if total_cameras else 0,
            "sites": Site.objects.filter(organization=organization, is_active=True).count(),
            "employees": Employee.objects.filter(organization=organization, is_active=True).count(),
            "enrolled": Employee.objects.filter(
                organization=organization, is_active=True, consent_given=True
            ).exclude(face_embedding=[]).count(),
            "false_positive_rate": round(
                this_week.filter(status=Event.Status.FALSE_POSITIVE).count()
                / max(this_week.count(), 1) * 100, 1
            ),
            "median_response": int(median_response),
        },
        by_analytic=by_analytic,
        by_severity=by_severity,
        hourly=hourly,
        hourly_max=max(hourly) or 1,
        daily_counts=daily_counts,
        daily_points=_sparkline(daily_counts),
        daily_max=max(daily_counts) or 1,
        recent_events=events.select_related("camera", "camera__site", "zone")[:8],
        open_incidents=Incident.objects.filter(
            organization=organization, status__in=[Incident.Status.OPEN, Incident.Status.INVESTIGATING]
        ).select_related("primary_camera")[:5],
        problem_cameras=cameras.filter(
            status__in=[Camera.Status.OFFLINE, Camera.Status.DEGRADED]
        ).select_related("site")[:5],
        busiest_cameras=list(
            this_week.values("camera__name", "camera__uid")
            .annotate(total=Count("id")).order_by("-total")[:5]
        ),
        model_status=ModelProvider(organization).describe(),
    )
    return render(request, "dashboard/home.html", context)


@login_required
@require_organization
def onboarding(request):
    """Guided setup checklist for a brand-new workspace."""
    organization = _org(request)
    steps = [
        {
            "key": "site", "title": "Add your first site",
            "description": "A site is a physical location — a floor, branch or warehouse.",
            "done": Site.objects.filter(organization=organization).exists(),
            "url": "dashboard:site_create", "cta": "Add a site",
        },
        {
            "key": "camera", "title": "Connect a camera",
            "description": "Point Campy AI at an existing CCTV stream, or start with a simulated camera.",
            "done": Camera.objects.filter(organization=organization).exists(),
            "url": "dashboard:camera_create", "cta": "Connect a camera",
        },
        {
            "key": "zone", "title": "Draw a zone",
            "description": "Mark restricted areas, counting zones or stock shelves on the camera view.",
            "done": Zone.objects.filter(camera__organization=organization).exists(),
            "url": "dashboard:zones", "cta": "Draw a zone",
        },
        {
            "key": "alert", "title": "Set up alerting",
            "description": "Decide who gets told, through which channel, and how quickly.",
            "done": AlertRule.objects.filter(organization=organization).exists(),
            "url": "dashboard:alert_rule_create", "cta": "Create an alert rule",
        },
        {
            "key": "team", "title": "Invite your team",
            "description": "Give each person the role that matches their job.",
            "done": Membership.objects.filter(organization=organization).count() > 1,
            "url": "accounts:invite_member", "cta": "Invite a colleague",
        },
    ]
    completed = sum(1 for step in steps if step["done"])
    return render(
        request,
        "dashboard/onboarding.html",
        _base_context(
            request, "home", steps=steps, completed=completed,
            progress=round(completed / len(steps) * 100),
        ),
    )


# ---------------------------------------------------------------------------
# Live monitoring
# ---------------------------------------------------------------------------
@login_required
@require_organization
@require_perm("live.view")
def live(request):
    organization = _org(request)
    cameras = (
        Camera.objects.filter(organization=organization, is_active=True)
        .select_related("site")
        .prefetch_related("zones")
    )
    site_filter = request.GET.get("site")
    if site_filter:
        cameras = cameras.filter(site__uid=site_filter)

    return render(
        request,
        "dashboard/live.html",
        _base_context(
            request, "live",
            cameras=cameras,
            sites=Site.objects.filter(organization=organization, is_active=True),
            selected_site=site_filter,
            recent_events=Event.objects.filter(
                organization=organization, occurred_at__gte=timezone.now() - timedelta(hours=2)
            ).select_related("camera")[:12],
        ),
    )


@login_required
@require_organization
@require_perm("live.view")
def live_feed(request):
    """Polled HTML fragment — the live wall's event ticker."""
    organization = _org(request)
    events = (
        Event.objects.filter(
            organization=organization, occurred_at__gte=timezone.now() - timedelta(hours=2)
        )
        .select_related("camera", "zone")[:12]
    )
    return render(request, "dashboard/partials/event_ticker.html", {"recent_events": events})


@login_required
@require_organization
@require_perm("live.view")
def camera_preview(request, uid):
    """Return the newest JPEG frame for a camera tile.

    Served from the live reader's slot, so this costs a memory read rather than
    opening and tearing down a stream per image.
    """
    camera = get_object_or_404(Camera, uid=uid, organization=_org(request))
    from apps.cameras.live import first_frame

    frame = first_frame(camera)
    if frame is None:
        return HttpResponse(status=204)

    response = HttpResponse(frame.jpeg, content_type="image/jpeg")
    response["Cache-Control"] = "no-store"
    response["X-Frame-Age"] = f"{frame.age:.2f}"
    return response


@login_required
@require_organization
@require_perm("live.view")
def camera_stream(request, uid):
    """Stream the camera as MJPEG, so the browser shows moving video.

    One long-lived response beats re-requesting a still every few seconds: the
    picture arrives at the camera's own rate instead of being up to an interval
    stale, and the camera still only has the single session the reader holds.
    """
    camera = get_object_or_404(Camera, uid=uid, organization=_org(request))
    from apps.cameras.live import broker

    reader = broker.reader_for(camera)
    if reader is None:
        return HttpResponse(status=503)

    boundary = "campyframe"
    max_seconds = float(settings.CAMPY.get("LIVE_STREAM_SECONDS", 300))

    def frames():
        deadline = time.monotonic() + max_seconds
        seen = 0
        while time.monotonic() < deadline:
            frame = reader.wait_for_frame(after=seen, timeout=5.0)
            if frame is None:
                if not reader.alive:
                    return
                continue          # a quiet camera is not a dead one
            seen = frame.index
            yield (f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                   f"Content-Length: {len(frame.jpeg)}\r\n\r\n").encode()
            yield frame.jpeg
            yield b"\r\n"

    response = StreamingHttpResponse(
        frames(), content_type=f"multipart/x-mixed-replace; boundary={boundary}"
    )
    response["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response["X-Accel-Buffering"] = "no"   # nginx must not buffer a live stream
    return response


# ---------------------------------------------------------------------------
# Events & incidents
# ---------------------------------------------------------------------------
@login_required
@require_organization
@require_perm("events.view")
def events(request):
    organization = _org(request)
    form = EventFilterForm(organization, request.GET or None)
    queryset = Event.objects.filter(organization=organization).select_related(
        "camera", "camera__site", "zone", "employee", "incident"
    )

    if form.is_valid():
        data = form.cleaned_data
        if data.get("q"):
            queryset = queryset.filter(
                Q(title__icontains=data["q"]) | Q(description__icontains=data["q"])
                | Q(event_type__icontains=data["q"])
            )
        if data.get("severity"):
            queryset = queryset.filter(severity=data["severity"])
        if data.get("analytic"):
            queryset = queryset.filter(analytic=data["analytic"])
        if data.get("status"):
            queryset = queryset.filter(status=data["status"])
        if data.get("camera"):
            queryset = queryset.filter(camera_id=data["camera"])
        if data.get("days"):
            queryset = queryset.filter(
                occurred_at__gte=timezone.now() - timedelta(days=int(data["days"]))
            )

    paginator = Paginator(queryset, 30)
    page = paginator.get_page(request.GET.get("page"))

    severity_counts = {
        row["severity"]: row["total"]
        for row in queryset.values("severity").annotate(total=Count("id"))
    }
    return render(
        request,
        "dashboard/events.html",
        _base_context(
            request, "events",
            form=form, page_obj=page, events=page.object_list,
            total=paginator.count,
            severity_counts=severity_counts,
            # Summed here rather than in the template: resolving a missing dict
            # key as a filter *argument* raises instead of defaulting.
            low_info_count=severity_counts.get("low", 0) + severity_counts.get("info", 0),
        ),
    )


@login_required
@require_organization
@require_perm("events.view")
def event_detail(request, uid):
    organization = _org(request)
    event = get_object_or_404(
        Event.objects.select_related("camera", "camera__site", "zone", "employee", "incident", "snapshot"),
        uid=uid, organization=organization,
    )
    related = Event.objects.filter(
        organization=organization, camera=event.camera,
        occurred_at__range=(event.occurred_at - timedelta(minutes=5), event.occurred_at + timedelta(minutes=5)),
    ).exclude(pk=event.pk).select_related("zone")[:10]

    return render(
        request,
        "dashboard/event_detail.html",
        _base_context(
            request, "events", event=event, related=related,
            annotations=json.dumps(event.snapshot.annotations if event.snapshot else {}),
        ),
    )


@login_required
@require_organization
@require_POST
def event_action(request, uid, action: str):
    organization = _org(request)
    event = get_object_or_404(Event, uid=uid, organization=organization)
    note = request.POST.get("note", "")

    permission = {
        "acknowledge": "events.acknowledge",
        "resolve": "events.resolve",
        "false_positive": "events.resolve",
        "dismiss": "events.resolve",
    }.get(action)
    if not permission or not request.user.has_campy_perm(permission, organization):
        messages.error(request, "You do not have permission to do that.")
        return redirect("dashboard:event_detail", uid=uid)

    if action == "acknowledge":
        event.acknowledge(request.user, note)
        messages.success(request, "Event acknowledged.")
    elif action == "resolve":
        event.resolve(request.user, note)
        messages.success(request, "Event resolved.")
    elif action == "false_positive":
        event.mark_false_positive(request.user, note)
        messages.success(
            request,
            "Marked as a false alarm. This frame can now be used as a hard negative when you retrain.",
        )
    elif action == "dismiss":
        event.status = Event.Status.DISMISSED
        event.resolved_by = request.user
        event.resolved_at = timezone.now()
        event.save(update_fields=["status", "resolved_by", "resolved_at", "updated_at"])
        messages.success(request, "Event dismissed.")

    AuditLog.record(
        action=AuditLog.Action.UPDATE, actor=request.user, organization=organization,
        target=event, summary=f"{action.replace('_', ' ').title()} event: {event.title}", request=request,
    )
    return redirect(request.POST.get("next") or "dashboard:events")


@login_required
@require_organization
@require_perm("events.acknowledge")
@require_POST
def events_bulk_action(request):
    organization = _org(request)
    uids = request.POST.getlist("event_uids")
    action = request.POST.get("bulk_action")
    queryset = Event.objects.filter(organization=organization, uid__in=uids)

    count = 0
    for event in queryset:
        if action == "acknowledge":
            event.acknowledge(request.user)
        elif action == "resolve" and request.user.has_campy_perm("events.resolve", organization):
            event.resolve(request.user)
        elif action == "false_positive" and request.user.has_campy_perm("events.resolve", organization):
            event.mark_false_positive(request.user)
        else:
            continue
        count += 1

    messages.success(request, f"{count} event{'s' if count != 1 else ''} updated.")
    return redirect("dashboard:events")


@login_required
@require_organization
@require_perm("events.export")
def events_export(request):
    """CSV export of the current filter selection."""
    import csv

    organization = _org(request)
    queryset = Event.objects.filter(organization=organization).select_related("camera", "zone", "employee")
    days = request.GET.get("days")
    if days:
        queryset = queryset.filter(occurred_at__gte=timezone.now() - timedelta(days=int(days)))
    if request.GET.get("severity"):
        queryset = queryset.filter(severity=request.GET["severity"])
    if request.GET.get("analytic"):
        queryset = queryset.filter(analytic=request.GET["analytic"])

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = (
        f'attachment; filename="campyai-events-{timezone.now():%Y%m%d-%H%M}.csv"'
    )
    writer = csv.writer(response)
    writer.writerow([
        "Reference", "Occurred at", "Severity", "Status", "Analytic", "Event type",
        "Title", "Camera", "Site", "Zone", "Employee", "Confidence", "Occurrences",
    ])
    for event in queryset.iterator(chunk_size=500):
        writer.writerow([
            str(event.uid), event.occurred_at.isoformat(), event.severity, event.status,
            event.analytic_label, event.event_type, event.title,
            event.camera.name if event.camera else "",
            event.camera.site.name if event.camera else "",
            event.zone.name if event.zone else "",
            event.employee.full_name if event.employee else "",
            round(event.confidence, 4), event.occurrence_count,
        ])

    AuditLog.record(
        action=AuditLog.Action.EXPORT, actor=request.user, organization=organization,
        summary=f"Exported {queryset.count()} events to CSV", request=request,
    )
    return response


@login_required
@require_organization
@require_perm("events.view")
def incidents(request):
    organization = _org(request)
    queryset = Incident.objects.filter(organization=organization).select_related(
        "primary_camera", "site", "assigned_to"
    )
    status = request.GET.get("status", "open")
    if status == "open":
        queryset = queryset.filter(status__in=[Incident.Status.OPEN, Incident.Status.INVESTIGATING])
    elif status:
        queryset = queryset.filter(status=status)

    paginator = Paginator(queryset, 25)
    page = paginator.get_page(request.GET.get("page"))
    return render(
        request,
        "dashboard/incidents.html",
        _base_context(
            request, "incidents", page_obj=page, incidents=page.object_list,
            status=status, total=paginator.count,
        ),
    )


@login_required
@require_organization
@require_perm("events.view")
def incident_detail(request, uid):
    organization = _org(request)
    incident = get_object_or_404(
        Incident.objects.select_related("primary_camera", "site", "assigned_to", "resolved_by"),
        uid=uid, organization=organization,
    )
    return render(
        request,
        "dashboard/incident_detail.html",
        _base_context(
            request, "incidents", incident=incident,
            events=incident.events.select_related("zone", "employee", "snapshot").order_by("occurred_at"),
            team=Membership.objects.filter(
                organization=organization, status=Membership.Status.ACTIVE
            ).select_related("user"),
        ),
    )


@login_required
@require_organization
@require_perm("events.resolve")
@require_POST
def incident_action(request, uid, action: str):
    organization = _org(request)
    incident = get_object_or_404(Incident, uid=uid, organization=organization)

    if action == "assign":
        user_id = request.POST.get("assigned_to")
        membership = Membership.objects.filter(
            organization=organization, user_id=user_id, status=Membership.Status.ACTIVE
        ).first()
        incident.assigned_to = membership.user if membership else None
        incident.status = Incident.Status.INVESTIGATING
        incident.save(update_fields=["assigned_to", "status", "updated_at"])
        messages.success(request, "Incident assigned.")
    elif action == "resolve":
        incident.status = Incident.Status.RESOLVED
        incident.resolved_by = request.user
        incident.resolved_at = timezone.now()
        incident.resolution_note = request.POST.get("note", "")
        incident.save(
            update_fields=["status", "resolved_by", "resolved_at", "resolution_note", "updated_at"]
        )
        incident.events.filter(status__in=[Event.Status.OPEN, Event.Status.ACKNOWLEDGED]).update(
            status=Event.Status.RESOLVED, resolved_by=request.user, resolved_at=timezone.now()
        )
        messages.success(request, "Incident resolved and its events closed.")
    elif action == "close":
        incident.status = Incident.Status.CLOSED
        incident.save(update_fields=["status", "updated_at"])
        messages.success(request, "Incident closed.")

    AuditLog.record(
        action=AuditLog.Action.UPDATE, actor=request.user, organization=organization,
        target=incident, summary=f"{action.title()} incident {incident.reference}", request=request,
    )
    return redirect("dashboard:incident_detail", uid=uid)


# ---------------------------------------------------------------------------
# Sites & cameras
# ---------------------------------------------------------------------------
@login_required
@require_organization
@require_perm("cameras.view")
def sites(request):
    organization = _org(request)
    queryset = (
        Site.objects.filter(organization=organization)
        .annotate(
            cameras_count=Count("cameras", filter=Q(cameras__deleted_at__isnull=True)),
            online_count=Count("cameras", filter=Q(cameras__status=Camera.Status.ONLINE)),
        )
        .order_by("name")
    )
    return render(request, "dashboard/sites.html", _base_context(request, "sites", sites=queryset))


@login_required
@require_organization
@require_perm("cameras.manage")
def site_form(request, uid=None):
    organization = _org(request)
    site = get_object_or_404(Site, uid=uid, organization=organization) if uid else None

    if site is None:
        from apps.billing.middleware import can_add

        allowed, reason = can_add(organization, "sites")
        if not allowed:
            messages.error(request, reason)
            return redirect("dashboard:sites")

    form = SiteForm(request.POST or None, instance=site)
    if request.method == "POST" and form.is_valid():
        saved = form.save(commit=False)
        saved.organization = organization
        if site is None:
            saved.created_by = request.user
        saved.updated_by = request.user
        saved.save()
        AuditLog.record(
            action=AuditLog.Action.CREATE if site is None else AuditLog.Action.UPDATE,
            actor=request.user, organization=organization, target=saved,
            summary=f"{'Created' if site is None else 'Updated'} site {saved.name}", request=request,
        )
        messages.success(request, f"Site '{saved.name}' saved.")
        return redirect("dashboard:sites")

    return render(request, "dashboard/site_form.html", _base_context(request, "sites", form=form, site=site))


@login_required
@require_organization
@require_perm("cameras.view")
def site_detail(request, uid):
    organization = _org(request)
    site = get_object_or_404(Site, uid=uid, organization=organization)
    return render(
        request,
        "dashboard/site_detail.html",
        _base_context(
            request, "sites", site=site,
            cameras=site.cameras.filter(deleted_at__isnull=True).order_by("name"),
            recent_events=Event.objects.filter(organization=organization, camera__site=site)[:10],
            employees=site.employees.filter(is_active=True)[:12],
        ),
    )


@login_required
@require_organization
@require_perm("cameras.view")
def cameras(request):
    organization = _org(request)
    queryset = Camera.objects.filter(organization=organization).select_related("site")
    if request.GET.get("site"):
        queryset = queryset.filter(site__uid=request.GET["site"])
    if request.GET.get("status"):
        queryset = queryset.filter(status=request.GET["status"])
    if request.GET.get("q"):
        queryset = queryset.filter(
            Q(name__icontains=request.GET["q"]) | Q(location_note__icontains=request.GET["q"])
        )

    from apps.billing.middleware import quota_usage

    return render(
        request,
        "dashboard/cameras.html",
        _base_context(
            request, "cameras",
            cameras=queryset.order_by("site__name", "name"),
            sites=Site.objects.filter(organization=organization, is_active=True),
            status_choices=Camera.Status.choices,
            quota=quota_usage(organization).get("cameras", {}),
        ),
    )


@login_required
@require_organization
@require_perm("cameras.manage")
def camera_form(request, uid=None):
    organization = _org(request)
    camera = get_object_or_404(Camera, uid=uid, organization=organization) if uid else None

    if camera is None:
        from apps.billing.middleware import can_add

        allowed, reason = can_add(organization, "cameras")
        if not allowed:
            messages.error(request, reason)
            return redirect("dashboard:cameras")
        if not Site.objects.filter(organization=organization).exists():
            messages.info(request, "Create a site before adding cameras.")
            return redirect("dashboard:site_create")

    form = CameraForm(organization, request.POST or None, request.FILES or None, instance=camera)
    if request.method == "POST" and form.is_valid():
        saved = form.save(commit=False)
        if camera is None:
            saved.created_by = request.user
            saved.status = Camera.Status.PENDING
        saved.updated_by = request.user
        saved.save()
        AuditLog.record(
            action=AuditLog.Action.CREATE if camera is None else AuditLog.Action.UPDATE,
            actor=request.user, organization=organization, target=saved,
            summary=f"{'Added' if camera is None else 'Updated'} camera {saved.name}", request=request,
        )
        messages.success(request, f"Camera '{saved.name}' saved.")
        return redirect("dashboard:camera_detail", uid=saved.uid)

    return render(
        request,
        "dashboard/camera_form.html",
        _base_context(
            request, "cameras", form=form, camera=camera, analytic_meta=analytic_metadata(),
            blocked=getattr(form, "blocked", set()),
        ),
    )


@login_required
@require_organization
@require_perm("cameras.view")
def camera_detail(request, uid):
    organization = _org(request)
    camera = get_object_or_404(
        Camera.objects.select_related("site").prefetch_related("zones"), uid=uid, organization=organization
    )
    recent = Event.objects.filter(camera=camera)

    week_ago = timezone.now() - timedelta(days=7)
    by_analytic = list(
        recent.filter(occurred_at__gte=week_ago)
        .values("analytic").annotate(total=Count("id")).order_by("-total")
    )
    for row in by_analytic:
        row["label"] = ANALYTIC_LABELS.get(row["analytic"], row["analytic"])

    return render(
        request,
        "dashboard/camera_detail.html",
        _base_context(
            request, "cameras", camera=camera,
            zones=camera.zones.all().order_by("name"),
            recent_events=recent.select_related("zone")[:12],
            events_7d=recent.filter(occurred_at__gte=week_ago).count(),
            by_analytic=by_analytic,
            snapshots=CameraSnapshot.objects.filter(camera=camera)[:8],
            analytic_labels=ANALYTIC_LABELS,
            health=camera.health or {},
        ),
    )


@login_required
@require_organization
@require_perm("cameras.manage")
@require_POST
def camera_test(request, uid):
    camera = get_object_or_404(Camera, uid=uid, organization=_org(request))
    from apps.cameras.services import check_connection

    ok, detail = check_connection(camera)
    if ok:
        camera.mark_online(detail)
        messages.success(request, detail)
    else:
        camera.mark_offline(detail)
        messages.error(request, detail)
    return redirect("dashboard:camera_detail", uid=uid)


@login_required
@require_organization
@require_perm("cameras.control")
@require_POST
def camera_run(request, uid):
    """Run a short analysis burst — proves the pipeline end to end."""
    camera = get_object_or_404(Camera, uid=uid, organization=_org(request))
    if not camera.active_analytics():
        messages.warning(request, "Enable at least one analytic on this camera first.")
        return redirect("dashboard:camera_detail", uid=uid)

    from apps.cameras.services import CameraWorker

    frames = min(int(request.POST.get("frames", 60) or 60), 400)
    worker = CameraWorker(camera, max_frames=frames)
    stats = worker.run()

    if not stats.frames:
        # A run that read nothing is a failure, however green the banner looks.
        camera.refresh_from_db()
        messages.error(
            request,
            camera.status_detail or "No frames could be read from this camera.",
        )
        return redirect("dashboard:camera_detail", uid=uid)

    messages.success(
        request,
        f"Analysed {stats.frames} frames in {stats.elapsed:.1f}s "
        f"({stats.fps:.1f} fps) and generated {stats.events} event(s).",
    )
    return redirect("dashboard:camera_detail", uid=uid)


@login_required
@require_organization
@require_perm("cameras.manage")
@require_POST
def camera_delete(request, uid):
    camera = get_object_or_404(Camera, uid=uid, organization=_org(request))
    name = camera.name
    camera.delete()
    AuditLog.record(
        action=AuditLog.Action.DELETE, actor=request.user, organization=_org(request),
        summary=f"Removed camera {name}", request=request,
    )
    messages.success(request, f"Camera '{name}' removed.")
    return redirect("dashboard:cameras")


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------
@login_required
@require_organization
@require_perm("zones.view")
def zones(request):
    organization = _org(request)
    queryset = Zone.objects.filter(camera__organization=organization).select_related("camera", "camera__site")
    if request.GET.get("camera"):
        queryset = queryset.filter(camera__uid=request.GET["camera"])
    return render(
        request,
        "dashboard/zones.html",
        _base_context(
            request, "zones", zones=queryset.order_by("camera__name", "name"),
            cameras=Camera.objects.filter(organization=organization, is_active=True),
        ),
    )


@login_required
@require_organization
@require_perm("zones.manage")
def zone_form(request, camera_uid=None, uid=None):
    organization = _org(request)
    zone = (
        get_object_or_404(Zone.objects.select_related("camera"), uid=uid, camera__organization=organization)
        if uid else None
    )
    camera = zone.camera if zone else get_object_or_404(Camera, uid=camera_uid, organization=organization)

    form = ZoneForm(camera, request.POST or None, instance=zone)
    if request.method == "POST" and form.is_valid():
        saved = form.save(commit=False)
        if zone is None:
            saved.created_by = request.user
        saved.updated_by = request.user
        saved.save()
        AuditLog.record(
            action=AuditLog.Action.CREATE if zone is None else AuditLog.Action.UPDATE,
            actor=request.user, organization=organization, target=saved,
            summary=f"{'Drew' if zone is None else 'Updated'} zone {saved.name} on {camera.name}",
            request=request,
        )
        messages.success(request, f"Zone '{saved.name}' saved.")
        return redirect("dashboard:camera_detail", uid=camera.uid)

    return render(
        request,
        "dashboard/zone_form.html",
        _base_context(
            request, "zones", form=form, zone=zone, camera=camera,
            existing_zones=camera.zones.exclude(pk=zone.pk if zone else None),
        ),
    )


@login_required
@require_organization
@require_perm("zones.manage")
@require_POST
def zone_delete(request, uid):
    zone = get_object_or_404(Zone, uid=uid, camera__organization=_org(request))
    camera_uid, name = zone.camera.uid, zone.name
    zone.delete()
    messages.success(request, f"Zone '{name}' deleted.")
    return redirect("dashboard:camera_detail", uid=camera_uid)


# ---------------------------------------------------------------------------
# Employees & face enrolment
# ---------------------------------------------------------------------------
@login_required
@require_organization
@require_perm("employees.view")
def employees(request):
    organization = _org(request)
    queryset = Employee.objects.filter(organization=organization).prefetch_related("sites")
    if request.GET.get("q"):
        term = request.GET["q"]
        queryset = queryset.filter(
            Q(full_name__icontains=term) | Q(employee_code__icontains=term)
            | Q(department__icontains=term) | Q(role__icontains=term)
        )
    if request.GET.get("status") == "enrolled":
        queryset = queryset.exclude(face_embedding=[])
    elif request.GET.get("status") == "pending":
        queryset = queryset.filter(face_embedding=[])

    paginator = Paginator(queryset.order_by("full_name"), 30)
    page = paginator.get_page(request.GET.get("page"))
    return render(
        request,
        "dashboard/employees.html",
        _base_context(
            request, "employees", page_obj=page, employees=page.object_list, total=paginator.count,
            enrolled_count=Employee.objects.filter(organization=organization)
            .exclude(face_embedding=[]).count(),
            face_enabled=organization.has_feature("face_recognition"),
        ),
    )


@login_required
@require_organization
@require_perm("employees.manage")
def employee_form(request, uid=None):
    organization = _org(request)
    employee = get_object_or_404(Employee, uid=uid, organization=organization) if uid else None

    if employee is None:
        from apps.billing.middleware import can_add

        allowed, reason = can_add(organization, "employees")
        if not allowed:
            messages.error(request, reason)
            return redirect("dashboard:employees")

    form = EmployeeForm(organization, request.POST or None, request.FILES or None, instance=employee)
    if request.method == "POST" and form.is_valid():
        saved = form.save(commit=False)
        if employee is None:
            saved.created_by = request.user
        saved.updated_by = request.user
        saved = form.save()
        messages.success(request, f"{saved.full_name} saved.")
        return redirect("dashboard:employee_detail", uid=saved.uid)

    return render(
        request, "dashboard/employee_form.html",
        _base_context(request, "employees", form=form, employee=employee),
    )


@login_required
@require_organization
@require_perm("employees.view")
def employee_detail(request, uid):
    organization = _org(request)
    employee = get_object_or_404(Employee.objects.prefetch_related("sites"), uid=uid, organization=organization)
    return render(
        request,
        "dashboard/employee_detail.html",
        _base_context(
            request, "employees", employee=employee,
            recent_events=Event.objects.filter(employee=employee).select_related("camera")[:15],
            sightings=Event.objects.filter(
                employee=employee, occurred_at__gte=timezone.now() - timedelta(days=30)
            ).count(),
            face_enabled=organization.has_feature("face_recognition"),
        ),
    )


@login_required
@require_organization
@require_perm("employees.enroll")
def employee_enroll(request, uid):
    """Compute and store a face embedding from uploaded photographs."""
    organization = _org(request)
    employee = get_object_or_404(Employee, uid=uid, organization=organization)

    if not employee.consent_given:
        messages.error(
            request,
            "Record this person's written consent before enrolling their face. "
            "Biometric processing without consent is unlawful in most jurisdictions.",
        )
        return redirect("dashboard:employee_detail", uid=uid)

    if request.method == "POST":
        uploads = request.FILES.getlist("photos")
        if not uploads:
            messages.error(request, "Upload at least one clear, front-facing photograph.")
            return redirect("dashboard:employee_enroll", uid=uid)

        import numpy as np

        from apps.aiengine.detectors.face import propose_faces
        from apps.aiengine.vision.features import lbp_histogram
        from apps.aiengine.vision.image import crop, normalise, resize_bilinear, to_chw

        provider = ModelProvider(organization)
        model = provider.get("campyface")
        embeddings, used, skipped = [], 0, 0

        for upload in uploads[:12]:
            try:
                from PIL import Image

                with Image.open(upload) as handle:
                    frame = np.asarray(handle.convert("RGB"), dtype=np.float32)
            except Exception:  # noqa: BLE001
                skipped += 1
                continue

            proposals = propose_faces(frame, None, max_faces=1)
            patch = crop(frame, proposals[0][0], padding=0.15) if proposals else frame
            if patch.size < 64:
                skipped += 1
                continue

            if model is not None:
                size = int((getattr(model, "meta", {}) or {}).get("input_size", 96))
                batch = to_chw(normalise(resize_bilinear(patch, size, size)))[None, ...]
                vector = np.asarray(model.predict(batch)[0], dtype=np.float32)
            else:
                vector = lbp_histogram(resize_bilinear(patch, 64, 64), bins=64)

            norm = float(np.linalg.norm(vector))
            if norm:
                embeddings.append(vector / norm)
                used += 1
            else:
                skipped += 1

        if not embeddings:
            messages.error(
                request,
                "No usable face could be found in those images. Use clear, well-lit, front-facing photos.",
            )
            return redirect("dashboard:employee_enroll", uid=uid)

        # The prototype is the mean of the sample embeddings, renormalised.
        prototype = np.mean(np.stack(embeddings), axis=0)
        employee.set_embedding(prototype, used, "campyface" if model else "lbp-fallback")
        employee.save(
            update_fields=[
                "face_embedding", "enrolment_samples", "enrolled_at", "embedding_model", "updated_at",
            ]
        )
        AuditLog.record(
            action=AuditLog.Action.UPDATE, actor=request.user, organization=organization,
            target=employee, summary=f"Enrolled {employee.full_name} from {used} photo(s)",
            request=request, sensitive=True,
        )
        message = f"{employee.full_name} enrolled from {used} photo{'s' if used != 1 else ''}."
        if skipped:
            message += f" {skipped} image(s) could not be used."
        if model is None:
            message += (
                " Note: no trained CampyFace model is deployed, so a fallback texture "
                "signature was used. Train and deploy a CampyFace model for reliable recognition."
            )
        messages.success(request, message)
        return redirect("dashboard:employee_detail", uid=uid)

    return render(
        request,
        "dashboard/employee_enroll.html",
        _base_context(
            request, "employees", employee=employee,
            has_model=ModelProvider(organization).get("campyface") is not None,
        ),
    )


@login_required
@require_organization
@require_perm("employees.manage")
@require_POST
def employee_unenroll(request, uid):
    employee = get_object_or_404(Employee, uid=uid, organization=_org(request))
    employee.face_embedding = []
    employee.enrolment_samples = 0
    employee.enrolled_at = None
    employee.save(update_fields=["face_embedding", "enrolment_samples", "enrolled_at", "updated_at"])
    AuditLog.record(
        action=AuditLog.Action.DELETE, actor=request.user, organization=_org(request),
        target=employee, summary=f"Deleted biometric data for {employee.full_name}",
        request=request, sensitive=True,
    )
    messages.success(request, f"Biometric data for {employee.full_name} has been erased.")
    return redirect("dashboard:employee_detail", uid=uid)


# ---------------------------------------------------------------------------
# Alert rules & channels
# ---------------------------------------------------------------------------
@login_required
@require_organization
@require_perm("alerts.view")
def alert_rules(request):
    organization = _org(request)
    return render(
        request,
        "dashboard/alert_rules.html",
        _base_context(
            request, "alerts",
            rules=AlertRule.objects.filter(organization=organization).prefetch_related("channels"),
            channels=NotificationChannel.objects.filter(organization=organization),
        ),
    )


@login_required
@require_organization
@require_perm("alerts.manage")
def alert_rule_form(request, uid=None):
    organization = _org(request)
    rule = get_object_or_404(AlertRule, uid=uid, organization=organization) if uid else None
    form = AlertRuleForm(organization, request.POST or None, instance=rule)

    if request.method == "POST" and form.is_valid():
        saved = form.save()
        AuditLog.record(
            action=AuditLog.Action.CREATE if rule is None else AuditLog.Action.UPDATE,
            actor=request.user, organization=organization, target=saved,
            summary=f"{'Created' if rule is None else 'Updated'} alert rule {saved.name}", request=request,
        )
        messages.success(request, f"Alert rule '{saved.name}' saved.")
        return redirect("dashboard:alert_rules")

    return render(
        request, "dashboard/alert_rule_form.html",
        _base_context(request, "alerts", form=form, rule=rule),
    )


@login_required
@require_organization
@require_perm("alerts.manage")
@require_POST
def alert_rule_delete(request, uid):
    rule = get_object_or_404(AlertRule, uid=uid, organization=_org(request))
    name = rule.name
    rule.delete()
    messages.success(request, f"Alert rule '{name}' deleted.")
    return redirect("dashboard:alert_rules")


@login_required
@require_organization
@require_perm("alerts.channels")
def channel_form(request, uid=None):
    organization = _org(request)
    channel = get_object_or_404(NotificationChannel, uid=uid, organization=organization) if uid else None
    form = NotificationChannelForm(organization, request.POST or None, instance=channel)

    if request.method == "POST" and form.is_valid():
        saved = form.save()
        messages.success(request, f"Channel '{saved.name}' saved.")
        return redirect("dashboard:alert_rules")

    return render(
        request, "dashboard/channel_form.html",
        _base_context(request, "alerts", form=form, channel=channel),
    )


@login_required
@require_organization
@require_perm("alerts.channels")
@require_POST
def channel_test(request, uid):
    """Send a synthetic alert so the customer can prove the channel works."""
    organization = _org(request)
    channel = get_object_or_404(NotificationChannel, uid=uid, organization=organization)

    sample = Event.objects.filter(organization=organization).first()
    if sample is None:
        sample = Event(
            organization=organization,
            analytic="fire", event_type="fire_detected", severity="critical",
            title="Test alert from Campy AI",
            description="This is a test notification. If you received it, this channel is working.",
            confidence=0.99, occurred_at=timezone.now(),
        )

    from apps.events.services import deliver

    notification = deliver(channel, sample)
    if notification and notification.status == Notification.Status.SENT:
        messages.success(request, f"Test alert delivered via {channel.name}.")
    else:
        messages.error(
            request, f"Delivery failed: {notification.error if notification else 'unknown error'}"
        )
    return redirect("dashboard:alert_rules")


@login_required
@require_organization
@require_perm("alerts.channels")
@require_POST
def channel_delete(request, uid):
    channel = get_object_or_404(NotificationChannel, uid=uid, organization=_org(request))
    name = channel.name
    channel.delete()
    messages.success(request, f"Channel '{name}' deleted.")
    return redirect("dashboard:alert_rules")


@login_required
def notifications(request):
    queryset = Notification.objects.filter(recipient=request.user).select_related(
        "event", "event__camera", "channel"
    )
    if request.method == "POST":
        queryset.filter(read_at__isnull=True).update(read_at=timezone.now())
        messages.success(request, "All notifications marked as read.")
        return redirect("dashboard:notifications")

    paginator = Paginator(queryset, 30)
    page = paginator.get_page(request.GET.get("page"))
    return render(
        request, "dashboard/notifications.html",
        _base_context(request, "", page_obj=page, notifications=page.object_list),
    )


# ---------------------------------------------------------------------------
# Analytics & reports
# ---------------------------------------------------------------------------
@login_required
@require_organization
@require_perm("analytics.view")
def analytics(request):
    organization = _org(request)
    days = int(request.GET.get("days", 30) or 30)
    since = timezone.now() - timedelta(days=days)
    events = Event.objects.filter(organization=organization, occurred_at__gte=since)

    by_day = []
    for offset in range(days - 1, -1, -1):
        day = (timezone.now() - timedelta(days=offset)).date()
        day_events = events.filter(occurred_at__date=day)
        by_day.append(
            {
                "date": day,
                "total": day_events.count(),
                "critical": day_events.filter(severity__in=["high", "critical"]).count(),
            }
        )

    by_analytic = list(events.values("analytic").annotate(total=Count("id")).order_by("-total"))
    for row in by_analytic:
        row["label"] = ANALYTIC_LABELS.get(row["analytic"], row["analytic"])

    by_camera = list(
        events.values("camera__name", "camera__uid", "camera__site__name")
        .annotate(total=Count("id")).order_by("-total")[:12]
    )
    by_hour = [0] * 24
    for occurred in events.values_list("occurred_at", flat=True):
        by_hour[timezone.localtime(occurred).hour] += 1

    by_weekday = [0] * 7
    for occurred in events.values_list("occurred_at", flat=True):
        by_weekday[timezone.localtime(occurred).weekday()] += 1

    resolved = events.exclude(resolved_at__isnull=True)
    response_times = [
        (e.acknowledged_at - e.occurred_at).total_seconds()
        for e in events.exclude(acknowledged_at__isnull=True).only("acknowledged_at", "occurred_at")[:1000]
    ]

    top_employees = list(
        events.exclude(employee__isnull=True)
        .values("employee__full_name", "employee__uid")
        .annotate(total=Count("id")).order_by("-total")[:8]
    )

    totals = events.count()
    return render(
        request,
        "dashboard/analytics.html",
        _base_context(
            request, "analytics",
            days=days, since=since,
            totals=totals,
            by_day=by_day,
            day_points=_sparkline([d["total"] for d in by_day]),
            day_max=max([d["total"] for d in by_day]) or 1,
            by_analytic=by_analytic,
            analytic_max=max([r["total"] for r in by_analytic]) if by_analytic else 1,
            by_camera=by_camera,
            camera_max=max([r["total"] for r in by_camera]) if by_camera else 1,
            by_hour=by_hour, hour_max=max(by_hour) or 1,
            by_weekday=by_weekday, weekday_max=max(by_weekday) or 1,
            weekday_names=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            by_severity={
                row["severity"]: row["total"]
                for row in events.values("severity").annotate(total=Count("id"))
            },
            resolution_rate=round(resolved.count() / max(totals, 1) * 100, 1),
            false_positive_rate=round(
                events.filter(status=Event.Status.FALSE_POSITIVE).count() / max(totals, 1) * 100, 1
            ),
            median_response=int(sorted(response_times)[len(response_times) // 2]) if response_times else 0,
            top_employees=top_employees,
            can_export=request.user.has_campy_perm("analytics.export", organization),
        ),
    )


@login_required
@require_organization
@require_perm("analytics.view")
def reports(request):
    organization = _org(request)
    return render(
        request,
        "dashboard/reports.html",
        _base_context(
            request, "reports",
            reports=Report.objects.filter(organization=organization).prefetch_related("sites"),
        ),
    )


@login_required
@require_organization
@require_perm("analytics.schedule")
def report_form(request, uid=None):
    organization = _org(request)
    report = get_object_or_404(Report, uid=uid, organization=organization) if uid else None
    form = ReportForm(organization, request.POST or None, instance=report)

    if request.method == "POST" and form.is_valid():
        saved = form.save()
        saved.created_by = saved.created_by or request.user
        saved.save(update_fields=["created_by"])
        messages.success(request, f"Report '{saved.name}' saved.")
        return redirect("dashboard:reports")

    return render(
        request, "dashboard/report_form.html",
        _base_context(request, "reports", form=form, report=report),
    )


@login_required
@require_organization
@require_perm("analytics.view")
def report_detail(request, uid):
    organization = _org(request)
    report = get_object_or_404(Report, uid=uid, organization=organization)
    from apps.analytics.services import build_report

    payload = build_report(report)
    return render(
        request, "dashboard/report_detail.html",
        _base_context(request, "reports", report=report, payload=payload),
    )


@login_required
@require_organization
@require_perm("analytics.export")
def analysis_report(request):
    """Download an analysis report as PDF, for one feature or several.

    Driven by query parameters so the same endpoint serves a button beside a
    single analytic, the combined download, and whatever filters the operator
    already has applied on the events page.
    """
    organization = _org(request)
    from apps.aiengine.detectors import ANALYTIC_CHOICES
    from apps.analytics.pdf import build_analysis_report

    allowed = {key for key, _ in ANALYTIC_CHOICES}
    requested = [a for a in request.GET.get("analytics", "").split(",") if a in allowed]
    # Never report on a feature this workspace has not bought.
    keys = [k for k in (requested or sorted(allowed)) if _analytic_allowed(organization, k)]
    if not keys:
        messages.warning(request, "Your package does not include those features.")
        return redirect("dashboard:analytics")

    cameras = list(Camera.objects.filter(organization=organization,
                                         uid__in=_uuid_list(request.GET.getlist("camera"))))
    sites = list(Site.objects.filter(organization=organization,
                                     uid__in=_uuid_list(request.GET.getlist("site"))))
    try:
        days = min(max(int(request.GET.get("days", 7)), 1), 365)
    except (TypeError, ValueError):
        days = 7
    try:
        evidence = min(max(int(request.GET.get("evidence", 12)), 0), 60)
    except (TypeError, ValueError):
        evidence = 12

    pdf = build_analysis_report(
        organization, analytics=keys, days=days, cameras=cameras, sites=sites,
        evidence_limit=evidence, generated_by=request.user.get_full_name(),
    )
    stem = "combined" if len(keys) > 1 else keys[0]
    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = (
        f'attachment; filename="campy-{stem}-report-{timezone.now():%Y%m%d}.pdf"'
    )
    return response


def _analytic_allowed(organization, key: str) -> bool:
    from apps.aiengine.detectors import ANALYTIC_FEATURES

    feature = ANALYTIC_FEATURES.get(key)
    return not feature or organization.has_feature(feature)


def _uuid_list(values):
    """Query parameters are user input; a malformed uid must not 500 the page."""
    import uuid

    found = []
    for value in values:
        try:
            found.append(uuid.UUID(str(value)))
        except (TypeError, ValueError):
            continue
    return found


@login_required
@require_organization
@require_perm("analytics.export")
def report_export(request, uid):
    """Download a saved report in the format it was defined with."""
    import csv

    organization = _org(request)
    report = get_object_or_404(Report, uid=uid, organization=organization)
    from apps.analytics.services import build_report

    wanted = request.GET.get("format") or report.output_format
    if wanted == Report.Format.PDF:
        from apps.analytics.pdf import build_analysis_report

        pdf = build_analysis_report(
            organization,
            analytics=list(report.analytics or []),
            days=report.date_range_days or 7,
            cameras=list(report.cameras.all()),
            sites=list(report.sites.all()),
            generated_by=request.user.get_full_name(),
            title=report.name,
        )
        response = HttpResponse(pdf, content_type="application/pdf")
        response["Content-Disposition"] = (
            f'attachment; filename="{report.name.replace(" ", "-").lower()}'
            f'-{timezone.now():%Y%m%d}.pdf"'
        )
        return response

    payload = build_report(report)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = (
        f'attachment; filename="{report.name.replace(" ", "-").lower()}-{timezone.now():%Y%m%d}.csv"'
    )
    writer = csv.writer(response)
    writer.writerow(["Campy AI report", report.name])
    writer.writerow(["Period", payload["period_start"], payload["period_end"]])
    writer.writerow([])
    for section in payload["sections"]:
        writer.writerow([section["title"]])
        writer.writerow(section["columns"])
        for row in section["rows"]:
            writer.writerow(row)
        writer.writerow([])
    return response


# ---------------------------------------------------------------------------
# Training platform
# ---------------------------------------------------------------------------
@login_required
@require_organization
@require_perm("training.view")
def datasets(request):
    organization = _org(request)
    return render(
        request,
        "dashboard/datasets.html",
        _base_context(
            request, "datasets",
            datasets=Dataset.objects.filter(organization=organization).order_by("-created_at"),
        ),
    )


@login_required
@require_organization
@require_perm("training.datasets")
def dataset_form(request, uid=None):
    organization = _org(request)
    dataset = get_object_or_404(Dataset, uid=uid, organization=organization) if uid else None
    form = DatasetForm(organization, request.POST or None, instance=dataset)

    if request.method == "POST" and form.is_valid():
        saved = form.save(commit=False)
        if dataset is None:
            saved.created_by = request.user
        saved.updated_by = request.user
        saved = form.save()
        messages.success(request, f"Dataset '{saved.name}' saved.")
        return redirect("dashboard:dataset_detail", uid=saved.uid)

    return render(
        request, "dashboard/dataset_form.html",
        _base_context(request, "datasets", form=form, dataset=dataset),
    )


@login_required
@require_organization
@require_perm("training.view")
def dataset_detail(request, uid):
    organization = _org(request)
    dataset = get_object_or_404(Dataset, uid=uid, organization=organization)
    samples = dataset.samples.select_related("source_camera").order_by("-created_at")

    label_filter = request.GET.get("label")
    if label_filter == "__unlabelled__":
        samples = samples.filter(label="")
    elif label_filter:
        samples = samples.filter(label=label_filter)

    paginator = Paginator(samples, 48)
    page = paginator.get_page(request.GET.get("page"))
    distribution = dataset.class_distribution or {}

    return render(
        request,
        "dashboard/dataset_detail.html",
        _base_context(
            request, "datasets", dataset=dataset, page_obj=page, samples=page.object_list,
            readiness=dataset.readiness,
            distribution=[
                {
                    "name": name,
                    "count": distribution.get(name, 0),
                    "percent": round(distribution.get(name, 0) / max(dataset.labelled_count, 1) * 100),
                }
                for name in (dataset.classes or [])
            ],
            distribution_max=max(distribution.values()) if distribution else 1,
            unlabelled=dataset.samples.filter(label="").count(),
            label_filter=label_filter,
            jobs=dataset.jobs.order_by("-created_at")[:5],
        ),
    )


@login_required
@require_organization
@require_perm("training.annotate")
@require_POST
def dataset_upload(request, uid):
    organization = _org(request)
    dataset = get_object_or_404(Dataset, uid=uid, organization=organization)
    label = request.POST.get("label", "")
    uploads = request.FILES.getlist("images")

    if label and label not in (dataset.classes or []):
        messages.error(request, f"'{label}' is not one of this dataset's classes.")
        return redirect("dashboard:dataset_detail", uid=uid)

    created = 0
    for upload in uploads[:400]:
        DatasetSample.objects.create(
            dataset=dataset, image=upload, label=label,
            labelled_by=request.user if label else None,
            labelled_at=timezone.now() if label else None,
        )
        created += 1

    dataset.refresh_counts()
    messages.success(
        request,
        f"Added {created} sample{'s' if created != 1 else ''}"
        + (f" labelled '{label}'." if label else " for labelling."),
    )
    return redirect("dashboard:dataset_detail", uid=uid)


@login_required
@require_organization
@require_perm("training.annotate")
@require_POST
def dataset_label(request, uid):
    """Bulk-label the selected samples from the labelling grid."""
    organization = _org(request)
    dataset = get_object_or_404(Dataset, uid=uid, organization=organization)
    label = request.POST.get("label", "")
    sample_uids = request.POST.getlist("sample_uids")

    if label not in (dataset.classes or []):
        messages.error(request, "Choose one of this dataset's classes.")
        return redirect("dashboard:dataset_detail", uid=uid)

    updated = DatasetSample.objects.filter(dataset=dataset, uid__in=sample_uids).update(
        label=label, labelled_by=request.user, labelled_at=timezone.now()
    )
    dataset.refresh_counts()
    messages.success(request, f"Labelled {updated} sample{'s' if updated != 1 else ''} as '{label}'.")
    return redirect(request.POST.get("next") or "dashboard:dataset_detail", uid=uid)


@login_required
@require_organization
@require_perm("training.datasets")
@require_POST
def dataset_bootstrap(request, uid):
    """Populate a dataset with generated samples so training can be tried today."""
    organization = _org(request)
    dataset = get_object_or_404(Dataset, uid=uid, organization=organization)

    import io

    import numpy as np
    from django.core.files.base import ContentFile
    from PIL import Image

    from apps.aiengine.simulation import synthetic_classification_dataset

    per_class = min(int(request.POST.get("per_class", 40) or 40), 150)
    images, labels = synthetic_classification_dataset(
        dataset.classes or ["normal", "event"], samples_per_class=per_class,
        size=int(dataset.image_size or 96),
    )

    created = 0
    for array, label_index in zip(images, labels):
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="JPEG", quality=88)
        sample = DatasetSample(
            dataset=dataset,
            label=dataset.classes[int(label_index)],
            labelled_by=request.user,
            labelled_at=timezone.now(),
            notes="Generated sample — replace with real footage before production use.",
        )
        sample.image.save(f"gen-{created}.jpg", ContentFile(buffer.getvalue()), save=False)
        sample.save()
        created += 1

    dataset.refresh_counts()
    messages.success(
        request,
        f"Generated {created} labelled samples. These prove the training pipeline works; "
        "replace them with your own footage for a model you can rely on.",
    )
    return redirect("dashboard:dataset_detail", uid=uid)


@login_required
@require_organization
@require_perm("training.datasets")
@require_POST
def dataset_harvest(request, uid):
    organization = _org(request)
    dataset = get_object_or_404(Dataset, uid=uid, organization=organization)
    from apps.training.services import harvest_false_positives

    added = harvest_false_positives(dataset)
    messages.success(
        request,
        f"Imported {added} frame{'s' if added != 1 else ''} from dismissed false alarms."
        if added else "No new false-positive frames were available to import.",
    )
    return redirect("dashboard:dataset_detail", uid=uid)


@login_required
@require_organization
@require_perm("training.datasets")
@require_POST
def dataset_delete(request, uid):
    dataset = get_object_or_404(Dataset, uid=uid, organization=_org(request))
    name = dataset.name
    dataset.delete()
    messages.success(request, f"Dataset '{name}' archived.")
    return redirect("dashboard:datasets")


@login_required
@require_organization
@require_perm("training.view")
def training_jobs(request):
    organization = _org(request)
    return render(
        request,
        "dashboard/training_jobs.html",
        _base_context(
            request, "training",
            jobs=TrainingJob.objects.filter(organization=organization)
            .select_related("dataset").order_by("-created_at")[:60],
            can_run=request.user.has_campy_perm("training.run", organization),
        ),
    )


@login_required
@require_organization
@require_perm("training.run")
def training_job_form(request):
    organization = _org(request)
    form = TrainingJobForm(organization, request.POST or None)

    if request.method == "POST" and form.is_valid():
        job = form.save(commit=False)
        job.created_by = request.user
        job.save()

        AuditLog.record(
            action=AuditLog.Action.CREATE, actor=request.user, organization=organization,
            target=job, summary=f"Launched training job {job.name}", request=request,
        )

        # Run inline when Celery is eager (the default), otherwise queue it.
        from apps.training.tasks import run_training_job_task

        try:
            run_training_job_task.delay(job.pk)
        except Exception:  # noqa: BLE001 - no broker configured
            run_training_job_task(job.pk)

        messages.success(request, f"Training job '{job.name}' has started.")
        return redirect("dashboard:training_job_detail", uid=job.uid)

    return render(
        request, "dashboard/training_job_form.html",
        _base_context(request, "training", form=form),
    )


@login_required
@require_organization
@require_perm("training.view")
def training_job_detail(request, uid):
    organization = _org(request)
    job = get_object_or_404(
        TrainingJob.objects.select_related("dataset"), uid=uid, organization=organization
    )
    return render(
        request,
        "dashboard/training_job_detail.html",
        _base_context(
            request, "training", job=job, chart=job.chart_data,
            versions=job.versions.all(),
            per_class=(job.metrics or {}).get("per_class", []),
            can_run=request.user.has_campy_perm("training.run", organization),
            can_deploy=request.user.has_campy_perm("models.deploy", organization),
        ),
    )


@login_required
@require_organization
@require_perm("training.view")
def training_job_progress(request, uid):
    """JSON polled by the live training chart."""
    job = get_object_or_404(TrainingJob, uid=uid, organization=_org(request))
    return JsonResponse(
        {
            "status": job.status,
            "progress": job.progress,
            "epoch": job.current_epoch,
            "epochs": job.epochs,
            "history": job.history[-60:],
            "metrics": job.metrics,
            "log": job.log[-4000:],
            "finished": job.is_finished,
        }
    )


@login_required
@require_organization
@require_perm("training.run")
@require_POST
def training_job_cancel(request, uid):
    job = get_object_or_404(TrainingJob, uid=uid, organization=_org(request))
    if job.is_running:
        job.cancel_requested = True
        job.save(update_fields=["cancel_requested", "updated_at"])
        messages.info(request, "Cancellation requested. The job stops after the current epoch.")
    return redirect("dashboard:training_job_detail", uid=uid)


@login_required
@require_organization
@require_perm("models.view")
def models(request):
    organization = _org(request)
    own = ModelVersion.objects.filter(organization=organization).select_related("dataset", "job")
    base = ModelVersion.objects.filter(organization__isnull=True, is_base_model=True)

    provider = ModelProvider(organization)
    slots = []
    for slot, label in MODEL_SLOTS:
        deployed = ModelVersion.deployed_for(slot, organization)
        slots.append(
            {
                "slot": slot, "label": label, "deployed": deployed,
                "loaded": provider.get(slot) is not None,
                "candidates": own.filter(slot=slot, status__in=[
                    ModelVersion.Status.READY, ModelVersion.Status.DEPLOYED
                ]),
            }
        )

    return render(
        request,
        "dashboard/models.html",
        _base_context(
            request, "models", slots=slots, versions=own.order_by("-created_at")[:40],
            base_models=base,
            can_deploy=request.user.has_campy_perm("models.deploy", organization),
        ),
    )


@login_required
@require_organization
@require_perm("models.view")
def model_detail(request, uid):
    organization = _org(request)
    version = get_object_or_404(
        ModelVersion.objects.select_related("dataset", "job"),
        Q(organization=organization) | Q(organization__isnull=True), uid=uid,
    )
    architecture = []
    model = version.load()
    if model is not None:
        architecture = [
            {"name": layer.describe(), "params": layer.param_count} for layer in model.layers
        ]

    return render(
        request,
        "dashboard/model_detail.html",
        _base_context(
            request, "models", version=version, architecture=architecture,
            summary=model.summary() if model else "",
            per_class=(version.evaluation or {}).get("per_class", []),
            can_deploy=request.user.has_campy_perm("models.deploy", organization),
        ),
    )


@login_required
@require_organization
@require_perm("models.deploy")
@require_POST
def model_deploy(request, uid):
    organization = _org(request)
    version = get_object_or_404(ModelVersion, uid=uid, organization=organization)
    if not version.exists_on_disk:
        messages.error(request, "The weights file for this version is missing on disk.")
        return redirect("dashboard:model_detail", uid=uid)

    version.deploy(request.user)
    AuditLog.record(
        action=AuditLog.Action.DEPLOY, actor=request.user, organization=organization,
        target=version, summary=f"Deployed {version.name} v{version.version} to slot '{version.slot}'",
        request=request, sensitive=True,
    )
    messages.success(
        request,
        f"'{version.name}' v{version.version} is now live on every camera using the "
        f"{version.slot_label} slot.",
    )
    return redirect("dashboard:models")


@login_required
@require_organization
@require_perm("models.rollback")
@require_POST
def model_retire(request, uid):
    version = get_object_or_404(ModelVersion, uid=uid, organization=_org(request))
    version.retire()
    AuditLog.record(
        action=AuditLog.Action.DEPLOY, actor=request.user, organization=_org(request),
        target=version, summary=f"Rolled back {version.name} v{version.version}", request=request,
    )
    messages.success(
        request,
        f"'{version.name}' has been rolled back. Cameras fall back to the previously deployed "
        "model, or to the built-in classical detectors.",
    )
    return redirect("dashboard:models")


# ---------------------------------------------------------------------------
# Team, settings, API keys, audit
# ---------------------------------------------------------------------------
@login_required
@require_organization
@require_perm("people.view")
def team(request):
    organization = _org(request)
    from apps.billing.middleware import quota_usage

    return render(
        request,
        "dashboard/team.html",
        _base_context(
            request, "team",
            memberships=Membership.objects.filter(organization=organization)
            .exclude(status=Membership.Status.REMOVED)
            .select_related("user", "role").order_by("role__rank", "user__full_name"),
            invitations=Invitation.objects.filter(
                organization=organization, status=Invitation.Status.PENDING
            ).select_related("role"),
            roles=organization.roles.annotate(
                member_count=Count("memberships", filter=Q(memberships__status="active"))
            ).order_by("rank"),
            quota=quota_usage(organization).get("users", {}),
            can_invite=request.user.has_campy_perm("people.invite", organization),
            can_manage=request.user.has_campy_perm("people.manage", organization),
            can_remove=request.user.has_campy_perm("people.remove", organization),
            can_edit_roles=request.user.has_campy_perm("people.roles", organization),
        ),
    )


@login_required
@require_organization
@require_perm("org.view")
def settings_view(request):
    organization = _org(request)
    from apps.billing.middleware import quota_usage

    return render(
        request,
        "dashboard/settings.html",
        _base_context(
            request, "settings", organization=organization,
            quotas=quota_usage(organization),
            retention_days=organization.active_subscription.quota("retention_days", 90)
            if organization.active_subscription else 90,
            can_manage=request.user.has_campy_perm("org.manage", organization),
        ),
    )


@login_required
@require_organization
@require_perm("org.manage")
@require_POST
def settings_privacy(request):
    """Workspace-wide privacy and retention preferences."""
    organization = _org(request)
    organization.set_setting("blur_unknown_faces", request.POST.get("blur_unknown_faces") == "on")
    organization.set_setting("store_snapshots", request.POST.get("store_snapshots") == "on")
    organization.set_setting(
        "anonymous_analytics", request.POST.get("anonymous_analytics") == "on"
    )
    organization.save(update_fields=["settings", "updated_at"])

    if request.POST.get("blur_unknown_faces") == "on":
        Camera.objects.filter(organization=organization).update(privacy_blur_faces=True)

    AuditLog.record(
        action=AuditLog.Action.UPDATE, actor=request.user, organization=organization,
        summary="Updated privacy settings", changes=dict(organization.settings), request=request,
    )
    messages.success(request, "Privacy settings saved.")
    return redirect("dashboard:settings")


@login_required
@require_organization
@require_perm("api.view")
def api_keys(request):
    organization = _org(request)
    return render(
        request,
        "dashboard/api_keys.html",
        _base_context(
            request, "api",
            keys=APIKey.objects.filter(organization=organization).select_related("created_by"),
            can_manage=request.user.has_campy_perm("api.manage", organization),
            api_enabled=organization.has_feature("api_access"),
        ),
    )


@login_required
@require_organization
@require_perm("api.manage")
def api_key_create(request):
    organization = _org(request)
    form = APIKeyForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        expires_at = None
        if form.cleaned_data.get("expires_in_days"):
            expires_at = timezone.now() + timedelta(days=form.cleaned_data["expires_in_days"])

        api_key, plaintext = APIKey.issue(
            organization, form.cleaned_data["name"], form.cleaned_data["scopes"],
            created_by=request.user, expires_at=expires_at,
        )
        AuditLog.record(
            action=AuditLog.Action.CREATE, actor=request.user, organization=organization,
            target=api_key, summary=f"Issued API key '{api_key.name}'", request=request, sensitive=True,
        )
        # Shown exactly once — we only ever store the hash.
        return render(
            request, "dashboard/api_key_created.html",
            _base_context(request, "api", api_key=api_key, plaintext=plaintext),
        )

    return render(request, "dashboard/api_key_form.html", _base_context(request, "api", form=form))


@login_required
@require_organization
@require_perm("api.manage")
@require_POST
def api_key_revoke(request, uid):
    api_key = get_object_or_404(APIKey, uid=uid, organization=_org(request))
    api_key.revoke()
    AuditLog.record(
        action=AuditLog.Action.SECURITY, actor=request.user, organization=_org(request),
        target=api_key, summary=f"Revoked API key '{api_key.name}'", request=request, sensitive=True,
    )
    messages.success(request, f"API key '{api_key.name}' revoked.")
    return redirect("dashboard:api_keys")


@login_required
@require_organization
@require_perm("audit.view")
def audit(request):
    organization = _org(request)
    queryset = AuditLog.objects.filter(organization=organization).select_related("actor")
    if request.GET.get("action"):
        queryset = queryset.filter(action=request.GET["action"])
    if request.GET.get("q"):
        term = request.GET["q"]
        queryset = queryset.filter(
            Q(summary__icontains=term) | Q(actor_label__icontains=term) | Q(target_label__icontains=term)
        )
    if request.GET.get("sensitive") == "1":
        queryset = queryset.filter(is_sensitive=True)

    paginator = Paginator(queryset, 50)
    page = paginator.get_page(request.GET.get("page"))
    return render(
        request,
        "dashboard/audit.html",
        _base_context(
            request, "audit", page_obj=page, entries=page.object_list,
            action_choices=AuditLog.Action.choices, total=paginator.count,
            can_export=request.user.has_campy_perm("audit.export", organization),
        ),
    )


@login_required
@require_organization
@require_perm("audit.export")
def audit_export(request):
    import csv

    organization = _org(request)
    queryset = AuditLog.objects.filter(organization=organization).select_related("actor")
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = (
        f'attachment; filename="campyai-audit-{timezone.now():%Y%m%d}.csv"'
    )
    writer = csv.writer(response)
    writer.writerow(["Timestamp", "Actor", "Action", "Target type", "Target", "Summary", "IP", "Sensitive"])
    for entry in queryset.iterator(chunk_size=500):
        writer.writerow([
            entry.created_at.isoformat(), entry.actor_label, entry.action,
            entry.target_type, entry.target_label, entry.summary,
            entry.ip_address or "", "yes" if entry.is_sensitive else "no",
        ])
    AuditLog.record(
        action=AuditLog.Action.EXPORT, actor=request.user, organization=organization,
        summary="Exported the audit log", request=request, sensitive=True,
    )
    return response


# ---------------------------------------------------------------------------
# Platform console (Campy AI staff)
# ---------------------------------------------------------------------------
@login_required
@platform_team_required
def admin_home(request):
    from apps.billing.models import Invoice, Package, Payment, Subscription

    now = timezone.now()
    month_ago = now - timedelta(days=30)
    organizations = Organization.objects.all()
    paid = Invoice.objects.filter(status=Invoice.Status.PAID)

    revenue_by_month = []
    for offset in range(5, -1, -1):
        start = (now - timedelta(days=30 * (offset + 1))).date()
        end = (now - timedelta(days=30 * offset)).date()
        total = paid.filter(paid_at__date__gte=start, paid_at__date__lt=end).aggregate(
            total=Sum("total")
        )["total"] or 0
        revenue_by_month.append({"label": end.strftime("%b"), "total": float(total)})

    return render(
        request,
        "dashboard/admin/home.html",
        _base_context(
            request, "admin",
            stats={
                "organizations": organizations.count(),
                "active_orgs": organizations.filter(status=Organization.Status.ACTIVE).count(),
                "trial_orgs": organizations.filter(status=Organization.Status.TRIAL).count(),
                "users": Membership.objects.values("user").distinct().count(),
                "cameras": Camera.objects.count(),
                "events_30d": Event.objects.filter(occurred_at__gte=month_ago).count(),
                "revenue_30d": float(
                    paid.filter(paid_at__gte=month_ago).aggregate(total=Sum("total"))["total"] or 0
                ),
                "revenue_total": float(paid.aggregate(total=Sum("total"))["total"] or 0),
                "mrr": float(
                    Subscription.objects.filter(status=Subscription.Status.ACTIVE)
                    .aggregate(total=Sum("package__price_inr"))["total"] or 0
                ),
                "trials_ending": Subscription.objects.filter(
                    status=Subscription.Status.TRIALING, trial_ends_at__lte=now + timedelta(days=7)
                ).count(),
                "failed_payments": Payment.objects.filter(
                    status=Payment.Status.FAILED, created_at__gte=month_ago
                ).count(),
                "base_models": ModelVersion.objects.filter(is_base_model=True).count(),
            },
            revenue_by_month=revenue_by_month,
            revenue_max=max([r["total"] for r in revenue_by_month]) or 1,
            recent_orgs=organizations.order_by("-created_at")[:8],
            recent_payments=Payment.objects.select_related("organization", "package").order_by("-created_at")[:10],
            packages=Package.objects.annotate(
                live_subs=Count("subscriptions", filter=Q(subscriptions__status__in=["active", "trialing"]))
            ).order_by("sort_order"),
            leads_count=_lead_count(),
        ),
    )


def _lead_count() -> int:
    from apps.cms.models import Lead

    return Lead.objects.filter(status=Lead.Status.NEW).count()


@login_required
@platform_team_required
def admin_organizations(request):
    queryset = Organization.objects.annotate(
        camera_total=Count("cameras", distinct=True),
        member_total=Count("memberships", distinct=True),
    ).order_by("-created_at")

    if request.GET.get("q"):
        queryset = queryset.filter(
            Q(name__icontains=request.GET["q"]) | Q(contact_email__icontains=request.GET["q"])
        )
    if request.GET.get("status"):
        queryset = queryset.filter(status=request.GET["status"])

    paginator = Paginator(queryset, 30)
    page = paginator.get_page(request.GET.get("page"))
    return render(
        request,
        "dashboard/admin/organizations.html",
        _base_context(
            request, "admin_orgs", page_obj=page, organizations=page.object_list,
            status_choices=Organization.Status.choices, total=paginator.count,
        ),
    )


@login_required
@platform_team_required
def admin_organization_detail(request, slug):
    organization = get_object_or_404(Organization, slug=slug)
    from apps.billing.models import Invoice, Payment

    return render(
        request,
        "dashboard/admin/organization_detail.html",
        _base_context(
            request, "admin_orgs", org=organization,
            memberships=organization.memberships.select_related("user", "role"),
            cameras=Camera.objects.filter(organization=organization).select_related("site"),
            subscription=organization.active_subscription,
            invoices=Invoice.objects.filter(organization=organization)[:10],
            payments=Payment.objects.filter(organization=organization)[:10],
            events_30d=Event.objects.filter(
                organization=organization, occurred_at__gte=timezone.now() - timedelta(days=30)
            ).count(),
            can_impersonate=request.user.has_campy_perm("platform.impersonate"),
        ),
    )


@login_required
@platform_team_required
@require_POST
def admin_organization_status(request, slug):
    organization = get_object_or_404(Organization, slug=slug)
    status = request.POST.get("status")
    if status in dict(Organization.Status.choices):
        organization.status = status
        organization.save(update_fields=["status", "updated_at"])
        AuditLog.record(
            action=AuditLog.Action.UPDATE, actor=request.user, organization=organization,
            target=organization, summary=f"Platform set workspace status to {status}",
            request=request, sensitive=True,
        )
        messages.success(request, f"{organization.name} is now {status}.")
    return redirect("dashboard:admin_organization_detail", slug=slug)


@login_required
@platform_team_required
def admin_packages(request):
    from apps.billing.models import Package

    return render(
        request,
        "dashboard/admin/packages.html",
        _base_context(
            request, "admin_packages",
            packages=Package.objects.annotate(
                live_subs=Count("subscriptions", filter=Q(subscriptions__status__in=["active", "trialing"]))
            ).order_by("sort_order"),
        ),
    )


@login_required
@platform_team_required
def admin_package_form(request, slug=None):
    from apps.billing.forms import PackageForm
    from apps.billing.models import Package

    package = get_object_or_404(Package, slug=slug) if slug else None
    form = PackageForm(request.POST or None, instance=package)

    if request.method == "POST" and form.is_valid():
        saved = form.save()
        AuditLog.record(
            action=AuditLog.Action.UPDATE, actor=request.user, target=saved,
            summary=f"{'Updated' if package else 'Created'} package {saved.name}",
            request=request, sensitive=True,
        )
        messages.success(request, f"Package '{saved.name}' saved.")
        return redirect("dashboard:admin_packages")

    from apps.billing.models import FEATURES, QUOTAS

    return render(
        request,
        "dashboard/admin/package_form.html",
        _base_context(
            request, "admin_packages", form=form, package=package,
            features=FEATURES, quotas=QUOTAS,
            current_features=(package.features if package else {}),
            current_quotas=(package.quotas if package else {}),
        ),
    )


@login_required
@platform_team_required
def admin_cms(request):
    from apps.cms.models import FAQ, Lead, MediaAsset, Page, Post, Testimonial

    return render(
        request,
        "dashboard/admin/cms.html",
        _base_context(
            request, "admin_cms",
            pages=Page.objects.order_by("sort_order", "title")[:40],
            posts=Post.objects.order_by("-created_at")[:20],
            leads=Lead.objects.order_by("-created_at")[:20],
            new_leads=Lead.objects.filter(status=Lead.Status.NEW).count(),
            faqs=FAQ.objects.count(),
            testimonials=Testimonial.objects.count(),
            media_count=MediaAsset.objects.count(),
        ),
    )
