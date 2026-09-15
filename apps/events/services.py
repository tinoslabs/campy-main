"""Turning pipeline findings into events, incidents and notifications."""
from __future__ import annotations

import json
import logging
from datetime import timedelta

from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.db import transaction
from django.template.loader import render_to_string
from django.utils import timezone

from .models import (
    SEVERITY_RANK,
    AlertRule,
    Event,
    Incident,
    Notification,
    NotificationChannel,
)

logger = logging.getLogger("campy.events")

#: Events from the same camera within this window join the same incident.
INCIDENT_WINDOW_SECONDS = 180


@transaction.atomic
def record_event(camera, candidate, *, snapshot=None, frame_index: int = 0) -> Event | None:
    """Persist one :class:`~apps.aiengine.detectors.base.EventCandidate`.

    Repeats of the same condition update the existing row's counter instead of
    creating a new one — the difference between an operator seeing one alert
    and seeing four hundred.
    """
    organization = camera.organization
    now = timezone.now()

    dedupe_key = f"{camera.id}:{candidate.dedupe_key}" if candidate.dedupe_key else ""
    if dedupe_key:
        existing = (
            Event.objects.filter(
                organization=organization,
                camera=camera,
                dedupe_key=dedupe_key,
                occurred_at__gte=now - timedelta(seconds=INCIDENT_WINDOW_SECONDS),
            )
            .order_by("-occurred_at")
            .first()
        )
        if existing is not None:
            existing.occurrence_count += 1
            existing.last_seen_at = now
            existing.confidence = max(existing.confidence, float(candidate.confidence))
            existing.save(update_fields=["occurrence_count", "last_seen_at", "confidence", "updated_at"])
            return None

    zone = None
    if candidate.zone_id:
        from apps.cameras.models import Zone as ZoneModel

        zone = ZoneModel.objects.filter(pk=candidate.zone_id, camera=camera).first()

    employee = None
    if candidate.subject_id:
        from apps.cameras.models import Employee

        employee = Employee.objects.filter(pk=candidate.subject_id, organization=organization).first()

    event = Event.objects.create(
        organization=organization,
        camera=camera,
        zone=zone,
        employee=employee,
        analytic=candidate.analytic,
        event_type=candidate.event_type,
        severity=candidate.severity,
        title=candidate.title[:220],
        description=candidate.description,
        confidence=float(candidate.confidence),
        occurred_at=now,
        frame_index=frame_index,
        track_id=candidate.track_id,
        bounding_box=list(candidate.box) if candidate.box else [],
        metadata=_jsonable(candidate.metadata),
        snapshot=snapshot,
        dedupe_key=dedupe_key,
        last_seen_at=now,
    )

    camera.events_generated = (camera.events_generated or 0) + 1
    camera.save(update_fields=["events_generated", "updated_at"])

    attach_to_incident(event)
    return event


def _jsonable(value):
    """Metadata comes from NumPy-heavy code; make it safe to store as JSON."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        import numpy as np

        def convert(item):
            if isinstance(item, dict):
                return {str(k): convert(v) for k, v in item.items()}
            if isinstance(item, (list, tuple)):
                return [convert(v) for v in item]
            if isinstance(item, (np.integer,)):
                return int(item)
            if isinstance(item, (np.floating,)):
                return round(float(item), 6)
            if isinstance(item, np.ndarray):
                return convert(item.tolist())
            if isinstance(item, (str, int, float, bool)) or item is None:
                return item
            return str(item)

        return convert(value)


def attach_to_incident(event: Event) -> Incident:
    """Group an event with recent related events from the same camera."""
    window_start = event.occurred_at - timedelta(seconds=INCIDENT_WINDOW_SECONDS)
    incident = (
        Incident.objects.filter(
            organization=event.organization,
            primary_camera=event.camera,
            status__in=[Incident.Status.OPEN, Incident.Status.INVESTIGATING],
            started_at__gte=window_start,
        )
        .order_by("-started_at")
        .first()
    )

    if incident is None:
        incident = Incident.objects.create(
            organization=event.organization,
            title=event.title[:240],
            summary=event.description,
            severity=event.severity,
            site=event.camera.site if event.camera else None,
            primary_camera=event.camera,
            analytics=[event.analytic],
            started_at=event.occurred_at,
        )
    event.incident = incident
    event.save(update_fields=["incident", "updated_at"])

    # Promote incident severity to the worst member event.
    if SEVERITY_RANK.get(event.severity, 0) > SEVERITY_RANK.get(incident.severity, 0):
        incident.severity = event.severity
        incident.title = event.title[:240]
    if event.analytic not in (incident.analytics or []):
        incident.analytics = sorted(set((incident.analytics or []) + [event.analytic]))
    incident.event_count = incident.events.count()
    incident.ended_at = event.occurred_at
    incident.save(
        update_fields=["severity", "title", "analytics", "event_count", "ended_at", "updated_at"]
    )
    return incident


# ---------------------------------------------------------------------------
# Alert dispatch
# ---------------------------------------------------------------------------
def dispatch_alerts(event: Event) -> list[Notification]:
    """Evaluate every alert rule against *event* and deliver what matches."""
    rules = (
        AlertRule.objects.filter(organization=event.organization, is_active=True)
        .prefetch_related("channels", "recipients", "cameras", "sites", "zones")
        .order_by("priority")
    )

    sent: list[Notification] = []
    for rule in rules:
        if not rule.matches(event) or rule.is_cooling_down:
            continue
        if rule.threshold_count > 1 and not _threshold_met(rule, event):
            continue

        for channel in rule.channels.filter(is_active=True):
            sent.append(deliver(channel, event, rule))

        for user in rule.recipients.all():
            sent.append(
                Notification.objects.create(
                    channel=_inapp_channel(event.organization),
                    rule=rule,
                    event=event,
                    incident=event.incident,
                    recipient=user,
                    subject=event.title[:240],
                    body=event.description,
                    status=Notification.Status.SENT,
                    sent_at=timezone.now(),
                )
            )

        rule.last_fired_at = timezone.now()
        rule.fire_count += 1
        rule.save(update_fields=["last_fired_at", "fire_count", "updated_at"])

        if rule.auto_acknowledge:
            event.status = Event.Status.ACKNOWLEDGED
            event.acknowledged_at = timezone.now()
            event.save(update_fields=["status", "acknowledged_at", "updated_at"])

    if sent:
        event.notified = True
        event.save(update_fields=["notified", "updated_at"])
    return [n for n in sent if n is not None]


def _threshold_met(rule: AlertRule, event: Event) -> bool:
    """Has the rule seen enough matching events inside its window?"""
    window_start = event.occurred_at - timedelta(seconds=rule.threshold_window_seconds)
    query = Event.objects.filter(
        organization=event.organization,
        occurred_at__gte=window_start,
        severity__in=[s for s, rank in SEVERITY_RANK.items() if rank >= rule.min_severity_rank],
    )
    if rule.analytics:
        query = query.filter(analytic__in=rule.analytics)
    if rule.event_types:
        query = query.filter(event_type__in=rule.event_types)
    return query.count() >= rule.threshold_count


def _inapp_channel(organization) -> NotificationChannel:
    channel, _ = NotificationChannel.objects.get_or_create(
        organization=organization,
        kind=NotificationChannel.Kind.INAPP,
        defaults={"name": "In-app notifications"},
    )
    return channel


def deliver(channel: NotificationChannel, event: Event, rule: AlertRule | None = None,
            is_escalation: bool = False) -> Notification | None:
    """Send one notification through one channel, recording the outcome."""
    subject = f"[{event.severity.upper()}] {event.title}"
    if is_escalation:
        subject = f"[ESCALATED] {subject}"

    notification = Notification.objects.create(
        channel=channel,
        rule=rule,
        event=event,
        incident=event.incident,
        subject=subject[:240],
        body=event.description,
        is_escalation=is_escalation,
    )

    try:
        handler = {
            NotificationChannel.Kind.EMAIL: _send_email,
            NotificationChannel.Kind.WEBHOOK: _send_webhook,
            NotificationChannel.Kind.SLACK: _send_slack,
            NotificationChannel.Kind.TELEGRAM: _send_telegram,
            NotificationChannel.Kind.SMS: _send_sms,
            NotificationChannel.Kind.INAPP: _send_inapp,
        }.get(channel.kind, _send_inapp)

        handler(channel, event, notification)
        notification.status = Notification.Status.SENT
        notification.sent_at = timezone.now()
        channel.success_count += 1
        channel.last_error = ""
    except Exception as exc:  # noqa: BLE001 - a failed channel must not lose the event
        logger.exception("Notification delivery failed on channel %s", channel.pk)
        notification.status = Notification.Status.FAILED
        notification.error = str(exc)[:400]
        channel.failure_count += 1
        channel.last_error = str(exc)[:400]

    notification.attempts += 1
    notification.save(update_fields=["status", "sent_at", "error", "attempts", "updated_at"])
    channel.last_used_at = timezone.now()
    channel.save(update_fields=["success_count", "failure_count", "last_error", "last_used_at", "updated_at"])
    return notification


def _event_payload(event: Event) -> dict:
    return {
        "id": str(event.uid),
        "organization": event.organization.slug,
        "camera": event.camera.name if event.camera else None,
        "site": event.camera.site.name if event.camera else None,
        "zone": event.zone.name if event.zone else None,
        "analytic": event.analytic,
        "event_type": event.event_type,
        "severity": event.severity,
        "title": event.title,
        "description": event.description,
        "confidence": round(event.confidence, 4),
        "occurred_at": event.occurred_at.isoformat(),
        "incident": event.incident.reference if event.incident else None,
        "metadata": event.metadata,
    }


def _send_email(channel, event, notification):
    recipients = (channel.config or {}).get("emails") or []
    if not recipients:
        raise ValueError("No email recipients configured on this channel.")
    context = {"event": event, "payload": _event_payload(event), "brand": settings.CAMPY["BRAND_NAME"]}
    text_body = render_to_string("events/email/alert.txt", context)
    html_body = render_to_string("events/email/alert.html", context)
    message = EmailMultiAlternatives(
        notification.subject, text_body, settings.DEFAULT_FROM_EMAIL, recipients
    )
    message.attach_alternative(html_body, "text/html")
    if channel.include_snapshot and event.snapshot and event.snapshot.image:
        try:
            event.snapshot.image.open("rb")
            message.attach(f"snapshot-{event.uid}.jpg", event.snapshot.image.read(), "image/jpeg")
        except Exception:  # noqa: BLE001 - never fail the alert over an attachment
            logger.warning("Could not attach snapshot to alert %s", event.uid)
    message.send(fail_silently=False)


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 10):
    import requests

    response = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
    response.raise_for_status()
    return response


def _send_webhook(channel, event, notification):
    config = channel.config or {}
    url = config.get("url")
    if not url:
        raise ValueError("Webhook channel has no URL configured.")
    headers = {"Content-Type": "application/json", "User-Agent": "CampyAI/1.0"}
    if config.get("secret"):
        import hashlib
        import hmac

        body = json.dumps(_event_payload(event), sort_keys=True)
        headers["X-Campy-Signature"] = hmac.new(
            config["secret"].encode(), body.encode(), hashlib.sha256
        ).hexdigest()
    headers.update(config.get("headers") or {})
    _post_json(url, {"type": "event.created", "data": _event_payload(event)}, headers)


def _send_slack(channel, event, notification):
    config = channel.config or {}
    url = config.get("url")
    if not url:
        raise ValueError("Slack channel has no incoming-webhook URL configured.")
    emoji = {"critical": ":rotating_light:", "high": ":warning:", "medium": ":eyes:"}.get(
        event.severity, ":information_source:"
    )
    _post_json(
        url,
        {
            "text": f"{emoji} *{event.title}*",
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": f"{emoji} *{event.title}*\n{event.description}"}},
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"*Camera:* {event.camera.name if event.camera else '—'}  |  "
                                f"*Severity:* {event.severity}  |  "
                                f"*Confidence:* {event.confidence:.0%}"
                            ),
                        }
                    ],
                },
            ],
        },
    )


def _send_telegram(channel, event, notification):
    config = channel.config or {}
    token, chat_id = config.get("bot_token"), config.get("chat_id")
    if not token or not chat_id:
        raise ValueError("Telegram channel needs both bot_token and chat_id.")
    text = (
        f"<b>{event.title}</b>\n{event.description}\n\n"
        f"Camera: {event.camera.name if event.camera else '—'}\n"
        f"Severity: {event.severity} · Confidence: {event.confidence:.0%}"
    )
    _post_json(
        f"https://api.telegram.org/bot{token}/sendMessage",
        {"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
    )


def _send_sms(channel, event, notification):
    """SMS goes through whatever HTTP gateway the customer has configured.

    Deliberately provider-agnostic: Campy AI sells to customers in many
    countries, each with their own preferred SMS vendor.
    """
    config = channel.config or {}
    url = config.get("gateway_url")
    numbers = config.get("numbers") or []
    if not url or not numbers:
        raise ValueError("SMS channel needs gateway_url and at least one number.")
    body = f"{event.severity.upper()}: {event.title} ({event.camera.name if event.camera else 'camera'})"
    _post_json(url, {"to": numbers, "message": body[:320], **(config.get("extra") or {})})


def _send_inapp(channel, event, notification):
    """Nothing to transmit — the Notification row *is* the delivery."""
    return None


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------
def escalate_unacknowledged() -> int:
    """Escalate events that nobody has acknowledged in time.

    Run periodically. This is what stops a critical alert from sitting unseen
    because the person on shift stepped away.
    """
    escalated = 0
    rules = (
        AlertRule.objects.filter(is_active=True, escalate_after_seconds__gt=0)
        .prefetch_related("escalation_channels")
    )
    for rule in rules:
        cutoff = timezone.now() - timedelta(seconds=rule.escalate_after_seconds)
        stale = Event.objects.filter(
            organization=rule.organization,
            status=Event.Status.OPEN,
            notified=True,
            occurred_at__lte=cutoff,
            severity__in=[s for s, rank in SEVERITY_RANK.items() if rank >= rule.min_severity_rank],
        ).exclude(notifications__is_escalation=True)[:50]

        for event in stale:
            for channel in rule.escalation_channels.filter(is_active=True):
                deliver(channel, event, rule, is_escalation=True)
                escalated += 1
            if event.incident:
                event.incident.escalation_level += 1
                event.incident.save(update_fields=["escalation_level", "updated_at"])
    return escalated
