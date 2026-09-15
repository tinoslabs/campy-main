"""Detections, incidents, alert rules and notification delivery."""
from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.db import models
from django.urls import reverse
from django.utils import timezone

from apps.aiengine.detectors import ANALYTIC_CHOICES
from apps.core.models import (
    OrganizationOwnedModel,
    TimeStampedModel,
    UUIDModel,
)

SEVERITY_CHOICES = [
    ("info", "Info"),
    ("low", "Low"),
    ("medium", "Medium"),
    ("high", "High"),
    ("critical", "Critical"),
]

SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


class EventQuerySet(models.QuerySet):
    def open(self):
        return self.filter(status=Event.Status.OPEN)

    def actionable(self):
        return self.filter(status__in=[Event.Status.OPEN, Event.Status.ACKNOWLEDGED])

    def critical(self):
        return self.filter(severity__in=["high", "critical"])

    def recent(self, hours: int = 24):
        return self.filter(occurred_at__gte=timezone.now() - timedelta(hours=hours))

    def for_site(self, site):
        return self.filter(camera__site=site)


class Event(UUIDModel, OrganizationOwnedModel, TimeStampedModel):
    """One finding from one analytic — the atomic unit of the platform."""

    organization = models.ForeignKey(
        "accounts.Organization", on_delete=models.CASCADE, related_name="events"
    )

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        ACKNOWLEDGED = "acknowledged", "Acknowledged"
        RESOLVED = "resolved", "Resolved"
        DISMISSED = "dismissed", "Dismissed"
        FALSE_POSITIVE = "false_positive", "False positive"

    camera = models.ForeignKey(
        "cameras.Camera", on_delete=models.CASCADE, related_name="events", null=True
    )
    zone = models.ForeignKey(
        "cameras.Zone", on_delete=models.SET_NULL, null=True, blank=True, related_name="events"
    )
    employee = models.ForeignKey(
        "cameras.Employee", on_delete=models.SET_NULL, null=True, blank=True, related_name="events"
    )
    incident = models.ForeignKey(
        "Incident", on_delete=models.SET_NULL, null=True, blank=True, related_name="events"
    )

    analytic = models.CharField(max_length=20, choices=ANALYTIC_CHOICES, db_index=True)
    event_type = models.CharField(max_length=48, db_index=True)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES, default="medium", db_index=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.OPEN, db_index=True)

    title = models.CharField(max_length=220)
    description = models.TextField(blank=True)
    confidence = models.FloatField(default=0.0)

    occurred_at = models.DateTimeField(default=timezone.now, db_index=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    frame_index = models.BigIntegerField(default=0)
    track_id = models.IntegerField(null=True, blank=True)
    bounding_box = models.JSONField(default=list, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    snapshot = models.ForeignKey(
        "cameras.CameraSnapshot", on_delete=models.SET_NULL, null=True, blank=True, related_name="events"
    )
    clip = models.FileField(upload_to="evidence/%Y/%m/%d/", blank=True, null=True)

    #: Repeated occurrences of the same condition collapse into one row.
    dedupe_key = models.CharField(max_length=180, blank=True, db_index=True)
    occurrence_count = models.PositiveIntegerField(default=1)
    last_seen_at = models.DateTimeField(default=timezone.now)

    acknowledged_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="acknowledged_events"
    )
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="resolved_events"
    )
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolution_note = models.TextField(blank=True)

    #: Operator feedback — the training platform mines these to build a
    #: hard-negative dataset, so muting a false alarm actively improves the model.
    feedback_label = models.CharField(max_length=40, blank=True)
    notified = models.BooleanField(default=False)

    objects = EventQuerySet.as_manager()

    class Meta:
        ordering = ["-occurred_at"]
        indexes = [
            models.Index(fields=["organization", "-occurred_at"]),
            models.Index(fields=["organization", "status", "severity"]),
            models.Index(fields=["camera", "-occurred_at"]),
            models.Index(fields=["analytic", "-occurred_at"]),
        ]

    def __str__(self) -> str:
        return f"[{self.severity}] {self.title}"

    def get_absolute_url(self) -> str:
        return reverse("dashboard:event_detail", args=[self.uid])

    @property
    def severity_rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 0)

    @property
    def is_open(self) -> bool:
        return self.status == self.Status.OPEN

    @property
    def is_actionable(self) -> bool:
        return self.status in {self.Status.OPEN, self.Status.ACKNOWLEDGED}

    @property
    def age_seconds(self) -> float:
        return (timezone.now() - self.occurred_at).total_seconds()

    @property
    def response_seconds(self) -> float | None:
        if not self.acknowledged_at:
            return None
        return (self.acknowledged_at - self.occurred_at).total_seconds()

    @property
    def analytic_label(self) -> str:
        return dict(ANALYTIC_CHOICES).get(self.analytic, self.analytic)

    def acknowledge(self, user, note: str = "") -> None:
        self.status = self.Status.ACKNOWLEDGED
        self.acknowledged_by = user
        self.acknowledged_at = timezone.now()
        if note:
            self.resolution_note = note
        self.save(update_fields=["status", "acknowledged_by", "acknowledged_at", "resolution_note", "updated_at"])

    def resolve(self, user, note: str = "", label: str = "") -> None:
        self.status = self.Status.RESOLVED
        self.resolved_by = user
        self.resolved_at = timezone.now()
        if note:
            self.resolution_note = note
        if label:
            self.feedback_label = label
        self.save(
            update_fields=[
                "status", "resolved_by", "resolved_at", "resolution_note",
                "feedback_label", "updated_at",
            ]
        )

    def mark_false_positive(self, user, note: str = "") -> None:
        self.status = self.Status.FALSE_POSITIVE
        self.feedback_label = "false_positive"
        self.resolved_by = user
        self.resolved_at = timezone.now()
        self.resolution_note = note
        self.save(
            update_fields=[
                "status", "feedback_label", "resolved_by", "resolved_at",
                "resolution_note", "updated_at",
            ]
        )


class Incident(UUIDModel, OrganizationOwnedModel, TimeStampedModel):
    """Related events grouped into one thing a human deals with once.

    Ten geofence breaches by the same person over two minutes is one incident,
    not ten alerts. This grouping is the difference between a system people use
    and one they mute.
    """

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        INVESTIGATING = "investigating", "Investigating"
        RESOLVED = "resolved", "Resolved"
        CLOSED = "closed", "Closed"

    organization = models.ForeignKey(
        "accounts.Organization", on_delete=models.CASCADE, related_name="incidents"
    )
    reference = models.CharField(max_length=24, unique=True, db_index=True)
    title = models.CharField(max_length=240)
    summary = models.TextField(blank=True)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES, default="medium", db_index=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.OPEN, db_index=True)

    site = models.ForeignKey("cameras.Site", null=True, blank=True, on_delete=models.SET_NULL, related_name="incidents")
    primary_camera = models.ForeignKey(
        "cameras.Camera", null=True, blank=True, on_delete=models.SET_NULL, related_name="incidents"
    )
    analytics = models.JSONField(default=list, blank=True)

    started_at = models.DateTimeField(default=timezone.now, db_index=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    event_count = models.PositiveIntegerField(default=0)

    assigned_to = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="assigned_incidents"
    )
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="resolved_incidents"
    )
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolution_note = models.TextField(blank=True)
    escalation_level = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-started_at"]
        indexes = [
            models.Index(fields=["organization", "status", "-started_at"]),
            models.Index(fields=["organization", "severity"]),
        ]

    def __str__(self) -> str:
        return f"{self.reference} — {self.title}"

    def get_absolute_url(self) -> str:
        return reverse("dashboard:incident_detail", args=[self.uid])

    def save(self, *args, **kwargs):
        if not self.reference:
            self.reference = self._next_reference()
        super().save(*args, **kwargs)

    def _next_reference(self) -> str:
        prefix = f"INC-{timezone.now():%y%m}"
        last = (
            Incident.objects.filter(reference__startswith=prefix)
            .order_by("-reference")
            .values_list("reference", flat=True)
            .first()
        )
        sequence = int(last.rsplit("-", 1)[-1]) + 1 if last else 1
        return f"{prefix}-{sequence:04d}"

    @property
    def duration_seconds(self) -> float:
        return ((self.ended_at or timezone.now()) - self.started_at).total_seconds()

    @property
    def is_open(self) -> bool:
        return self.status in {self.Status.OPEN, self.Status.INVESTIGATING}

    def refresh_from_events(self) -> None:
        """Recompute rollup fields from the member events."""
        events = list(self.events.all())
        self.event_count = len(events)
        if events:
            self.severity = max((e.severity for e in events), key=lambda s: SEVERITY_RANK.get(s, 0))
            self.analytics = sorted({e.analytic for e in events})
            self.started_at = min(e.occurred_at for e in events)
            self.ended_at = max(e.occurred_at for e in events)
        self.save(update_fields=["event_count", "severity", "analytics", "started_at", "ended_at", "updated_at"])


class AlertRule(UUIDModel, OrganizationOwnedModel, TimeStampedModel):
    """Declarative policy: when an event matches, notify these channels."""

    name = models.CharField(max_length=160)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    priority = models.PositiveIntegerField(default=100, help_text="Lower runs first")

    # -- matching ---------------------------------------------------------
    analytics = models.JSONField(default=list, blank=True, help_text="Empty = all analytics")
    event_types = models.JSONField(default=list, blank=True, help_text="Empty = all event types")
    min_severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES, default="medium")
    min_confidence = models.FloatField(default=0.5)
    sites = models.ManyToManyField("cameras.Site", blank=True, related_name="alert_rules")
    cameras = models.ManyToManyField("cameras.Camera", blank=True, related_name="alert_rules")
    zones = models.ManyToManyField("cameras.Zone", blank=True, related_name="alert_rules")

    #: {"days": [...], "from": "18:00", "to": "07:00"} — e.g. after-hours only.
    active_schedule = models.JSONField(default=dict, blank=True)
    #: Suppress repeats of the same rule for this many seconds.
    cooldown_seconds = models.PositiveIntegerField(default=300)
    #: Require this many matching events within the window before firing.
    threshold_count = models.PositiveIntegerField(default=1)
    threshold_window_seconds = models.PositiveIntegerField(default=300)

    # -- actions ----------------------------------------------------------
    channels = models.ManyToManyField("NotificationChannel", blank=True, related_name="alert_rules")
    recipients = models.ManyToManyField(settings.AUTH_USER_MODEL, blank=True, related_name="alert_rules")
    escalate_after_seconds = models.PositiveIntegerField(
        default=0, help_text="0 = no escalation. Unacknowledged after this long → escalate."
    )
    escalation_channels = models.ManyToManyField(
        "NotificationChannel", blank=True, related_name="escalation_rules"
    )
    create_incident = models.BooleanField(default=True)
    auto_acknowledge = models.BooleanField(default=False)

    last_fired_at = models.DateTimeField(null=True, blank=True)
    fire_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["priority", "name"]
        indexes = [models.Index(fields=["organization", "is_active"])]

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self) -> str:
        return reverse("dashboard:alert_rule_detail", args=[self.uid])

    @property
    def min_severity_rank(self) -> int:
        return SEVERITY_RANK.get(self.min_severity, 0)

    def is_in_schedule(self, moment=None) -> bool:
        schedule = self.active_schedule or {}
        if not schedule:
            return True
        moment = moment or timezone.localtime()
        days = schedule.get("days")
        if days and moment.strftime("%a").lower()[:3] not in [str(d).lower()[:3] for d in days]:
            return False
        start, end = schedule.get("from"), schedule.get("to")
        if not start or not end:
            return True
        current = moment.strftime("%H:%M")
        return start <= current <= end if start <= end else (current >= start or current <= end)

    def matches(self, event: Event) -> bool:
        """Does *event* satisfy every condition on this rule?"""
        if not self.is_active:
            return False
        if SEVERITY_RANK.get(event.severity, 0) < self.min_severity_rank:
            return False
        if event.confidence < self.min_confidence:
            return False
        if self.analytics and event.analytic not in self.analytics:
            return False
        if self.event_types and event.event_type not in self.event_types:
            return False
        if not self.is_in_schedule(timezone.localtime(event.occurred_at)):
            return False

        camera_ids = set(self.cameras.values_list("id", flat=True))
        if camera_ids and event.camera_id not in camera_ids:
            return False
        site_ids = set(self.sites.values_list("id", flat=True))
        if site_ids and (event.camera is None or event.camera.site_id not in site_ids):
            return False
        zone_ids = set(self.zones.values_list("id", flat=True))
        if zone_ids and event.zone_id not in zone_ids:
            return False
        return True

    @property
    def is_cooling_down(self) -> bool:
        if not self.cooldown_seconds or not self.last_fired_at:
            return False
        return (timezone.now() - self.last_fired_at).total_seconds() < self.cooldown_seconds


class NotificationChannel(UUIDModel, OrganizationOwnedModel, TimeStampedModel):
    """Where alerts go: email, webhook, SMS, Slack, Telegram."""

    class Kind(models.TextChoices):
        EMAIL = "email", "Email"
        WEBHOOK = "webhook", "Webhook"
        SLACK = "slack", "Slack"
        TELEGRAM = "telegram", "Telegram"
        SMS = "sms", "SMS"
        INAPP = "inapp", "In-app only"

    name = models.CharField(max_length=140)
    kind = models.CharField(max_length=12, choices=Kind.choices, default=Kind.EMAIL)
    is_active = models.BooleanField(default=True)

    #: Shape depends on ``kind`` — {"emails": [...]}, {"url": ...}, etc.
    config = models.JSONField(default=dict, blank=True)
    include_snapshot = models.BooleanField(default=True)

    last_used_at = models.DateTimeField(null=True, blank=True)
    last_error = models.CharField(max_length=400, blank=True)
    success_count = models.PositiveIntegerField(default=0)
    failure_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return f"{self.name} ({self.get_kind_display()})"

    @property
    def target_summary(self) -> str:
        config = self.config or {}
        if self.kind == self.Kind.EMAIL:
            recipients = config.get("emails") or []
            return ", ".join(recipients[:3]) + ("…" if len(recipients) > 3 else "")
        if self.kind in {self.Kind.WEBHOOK, self.Kind.SLACK}:
            url = config.get("url", "")
            return url[:48] + "…" if len(url) > 48 else url
        if self.kind == self.Kind.TELEGRAM:
            return f"chat {config.get('chat_id', '—')}"
        if self.kind == self.Kind.SMS:
            numbers = config.get("numbers") or []
            return ", ".join(numbers[:3])
        return "In-app"

    @property
    def health(self) -> str:
        total = self.success_count + self.failure_count
        if not total:
            return "untested"
        return "healthy" if self.failure_count / total < 0.2 else "failing"


class Notification(UUIDModel, TimeStampedModel):
    """One delivery attempt — the audit trail for "were we told?"."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        SENT = "sent", "Sent"
        FAILED = "failed", "Failed"
        SKIPPED = "skipped", "Skipped"

    channel = models.ForeignKey(NotificationChannel, on_delete=models.CASCADE, related_name="notifications")
    rule = models.ForeignKey(AlertRule, null=True, blank=True, on_delete=models.SET_NULL, related_name="notifications")
    event = models.ForeignKey(Event, null=True, blank=True, on_delete=models.CASCADE, related_name="notifications")
    incident = models.ForeignKey(
        Incident, null=True, blank=True, on_delete=models.CASCADE, related_name="notifications"
    )
    recipient = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="notifications"
    )

    subject = models.CharField(max_length=240)
    body = models.TextField(blank=True)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING, db_index=True)
    sent_at = models.DateTimeField(null=True, blank=True)
    error = models.CharField(max_length=400, blank=True)
    attempts = models.PositiveIntegerField(default=0)
    is_escalation = models.BooleanField(default=False)
    read_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["recipient", "read_at"]),
            models.Index(fields=["status", "-created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.subject} → {self.channel.name}"

    @property
    def organization(self):
        return self.channel.organization
