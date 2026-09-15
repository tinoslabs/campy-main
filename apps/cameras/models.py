"""Sites, cameras, zones and the employee directory."""
from __future__ import annotations

from datetime import timedelta

from django.db import models
from django.urls import reverse
from django.utils import timezone

from apps.aiengine.detectors import ANALYTIC_CHOICES, ANALYTIC_KEYS
from apps.core.models import (
    ActorStampedModel,
    OrganizationOwnedModel,
    SoftDeleteModel,
    TimeStampedModel,
    UUIDModel,
)


class Site(UUIDModel, OrganizationOwnedModel, TimeStampedModel, SoftDeleteModel, ActorStampedModel):
    """A physical location — a branch, floor, warehouse or campus."""

    organization = models.ForeignKey(
        "accounts.Organization", on_delete=models.CASCADE, related_name="sites"
    )
    name = models.CharField(max_length=160)
    code = models.CharField(max_length=32, blank=True, help_text="Internal reference, e.g. BLR-01")
    description = models.TextField(blank=True)

    address = models.CharField(max_length=300, blank=True)
    city = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=2, blank=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    timezone = models.CharField(max_length=64, default="Asia/Kolkata")

    #: e.g. {"mon": ["09:00", "18:00"], ...} — drives after-hours alerting.
    working_hours = models.JSONField(default=dict, blank=True)
    contact_name = models.CharField(max_length=120, blank=True)
    contact_phone = models.CharField(max_length=32, blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["name"]
        constraints = [
            models.UniqueConstraint(
                fields=["organization", "name"], name="uniq_site_name_per_org"
            )
        ]

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self) -> str:
        return reverse("dashboard:site_detail", args=[self.uid])

    @property
    def camera_count(self) -> int:
        return self.cameras.filter(deleted_at__isnull=True).count()

    @property
    def online_camera_count(self) -> int:
        return self.cameras.filter(status=Camera.Status.ONLINE, deleted_at__isnull=True).count()

    def is_within_working_hours(self, moment=None) -> bool:
        moment = moment or timezone.localtime()
        window = (self.working_hours or {}).get(moment.strftime("%a").lower()[:3])
        if not window or len(window) != 2:
            return True
        current = moment.strftime("%H:%M")
        start, end = window
        return start <= current <= end if start <= end else (current >= start or current <= end)


class Camera(UUIDModel, OrganizationOwnedModel, TimeStampedModel, SoftDeleteModel, ActorStampedModel):
    """A video source. Campy AI is designed to sit on top of the CCTV you
    already own — any camera that can produce RTSP, HTTP/MJPEG or ONVIF."""

    class Status(models.TextChoices):
        ONLINE = "online", "Online"
        OFFLINE = "offline", "Offline"
        DEGRADED = "degraded", "Degraded"
        DISABLED = "disabled", "Disabled"
        PENDING = "pending", "Pending setup"

    class Protocol(models.TextChoices):
        RTSP = "rtsp", "RTSP"
        HTTP = "http", "HTTP / MJPEG"
        ONVIF = "onvif", "ONVIF"
        HLS = "hls", "HLS"
        FILE = "file", "Video file"
        DEMO = "demo", "Simulated (demo)"

    organization = models.ForeignKey(
        "accounts.Organization", on_delete=models.CASCADE, related_name="cameras"
    )
    site = models.ForeignKey(Site, on_delete=models.CASCADE, related_name="cameras")
    name = models.CharField(max_length=160)
    description = models.TextField(blank=True)
    location_note = models.CharField(max_length=200, blank=True, help_text="e.g. 'Above the main entrance'")

    protocol = models.CharField(max_length=10, choices=Protocol.choices, default=Protocol.RTSP)
    stream_url = models.CharField(max_length=500, blank=True)
    #: Credentials are stored separately so ``stream_url`` can be shown in the
    #: UI without leaking a password.
    username = models.CharField(max_length=120, blank=True)
    password = models.CharField(max_length=200, blank=True)
    onvif_host = models.CharField(max_length=200, blank=True)
    onvif_port = models.PositiveIntegerField(default=80)

    status = models.CharField(max_length=12, choices=Status.choices, default=Status.PENDING, db_index=True)
    status_detail = models.CharField(max_length=300, blank=True)
    last_seen_at = models.DateTimeField(null=True, blank=True)
    last_frame_at = models.DateTimeField(null=True, blank=True)
    consecutive_failures = models.PositiveIntegerField(default=0)

    resolution_width = models.PositiveIntegerField(default=1920)
    resolution_height = models.PositiveIntegerField(default=1080)
    source_fps = models.FloatField(default=25.0)
    #: Frames per second actually analysed — the main cost/accuracy dial.
    target_fps = models.FloatField(default=6.0)
    rotation = models.IntegerField(default=0, choices=[(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")])

    enabled_analytics = models.JSONField(default=list, blank=True)
    analytics_config = models.JSONField(default=dict, blank=True)
    #: Per-camera model slot overrides, e.g. {"campyface": "<path>"}.
    model_overrides = models.JSONField(default=dict, blank=True)

    record_evidence = models.BooleanField(default=True)
    evidence_seconds = models.PositiveIntegerField(default=10)
    privacy_blur_faces = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)

    thumbnail = models.ImageField(upload_to="cameras/thumbnails/", blank=True, null=True)
    tags = models.JSONField(default=list, blank=True)

    # Rolling health/telemetry, refreshed by the worker.
    health = models.JSONField(default=dict, blank=True)
    frames_processed = models.BigIntegerField(default=0)
    events_generated = models.BigIntegerField(default=0)

    class Meta:
        ordering = ["site__name", "name"]
        indexes = [
            models.Index(fields=["organization", "status"]),
            models.Index(fields=["site", "is_active"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["site", "name"], name="uniq_camera_name_per_site"
            )
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.site.name})"

    def get_absolute_url(self) -> str:
        return reverse("dashboard:camera_detail", args=[self.uid])

    def save(self, *args, **kwargs):
        self.enabled_analytics = [a for a in (self.enabled_analytics or []) if a in ANALYTIC_KEYS]
        super().save(*args, **kwargs)

    # -- analytics ---------------------------------------------------------
    def active_analytics(self) -> list[str]:
        """Enabled analytics the organisation's package actually permits."""
        from apps.aiengine.detectors import ANALYTIC_FEATURES

        organization = self.organization
        allowed = []
        for key in self.enabled_analytics or []:
            feature = ANALYTIC_FEATURES.get(key)
            if not feature or organization.has_feature(feature):
                allowed.append(key)
        return allowed

    @property
    def blocked_analytics(self) -> list[str]:
        """Enabled but unavailable on the current plan — shown as an upsell."""
        return [key for key in (self.enabled_analytics or []) if key not in self.active_analytics()]

    @property
    def analytics_labels(self) -> list[str]:
        labels = dict(ANALYTIC_CHOICES)
        return [labels.get(key, key) for key in self.enabled_analytics or []]

    # -- connection --------------------------------------------------------
    @property
    def is_demo(self) -> bool:
        return self.protocol == self.Protocol.DEMO

    @property
    def connection_url(self) -> str:
        """Stream URL with credentials injected — never rendered in templates."""
        from apps.cameras.urls_util import inject_credentials

        return inject_credentials(self.stream_url, self.username, self.password)

    @property
    def masked_url(self) -> str:
        """Safe for display: hides any embedded credentials."""
        url = self.stream_url or ""
        if "@" in url and "://" in url:
            scheme, rest = url.split("://", 1)
            return f"{scheme}://•••@{rest.split('@', 1)[1]}"
        return url

    @property
    def is_healthy(self) -> bool:
        if self.status != self.Status.ONLINE:
            return False
        if not self.last_frame_at:
            return False
        return (timezone.now() - self.last_frame_at) < timedelta(minutes=5)

    @property
    def observed_fps(self):
        """Frames per second the worker actually achieved, if it has run."""
        value = (self.health or {}).get("fps")
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @property
    def connection_hint(self) -> str:
        """Why this camera is probably not showing a picture.

        Answered without touching the network, because it is rendered on a page
        load. "Check the connection" sends somebody to inspect a firewall; the
        deployment usually already knows the real reason and should say it.
        """
        if self.is_demo:
            return ""

        from apps.cameras.services import decoder_error

        problem = decoder_error()
        if problem is not None:
            return problem
        if self.protocol == self.Protocol.ONVIF:
            if not (self.onvif_host or self.stream_url):
                return "No ONVIF address is set for this camera. Add its IP address."
        elif not self.stream_url:
            return (
                f"No stream URL is set for this camera. Add the "
                f"{self.get_protocol_display()} address it publishes."
            )
        if self.status_detail:
            return self.status_detail
        if self.status == self.Status.PENDING:
            return "This camera has not been connected yet. Use Test connection."
        return ""

    @property
    def uptime_label(self) -> str:
        if not self.last_seen_at:
            return "Never connected"
        delta = timezone.now() - self.last_seen_at
        if delta < timedelta(minutes=2):
            return "Live"
        from apps.core.utils import humanize_duration

        return f"Last seen {humanize_duration(delta.total_seconds())} ago"

    def mark_online(self, detail: str = "") -> None:
        now = timezone.now()
        self.status = self.Status.ONLINE
        self.status_detail = detail[:300]
        self.last_seen_at = now
        self.last_frame_at = now
        self.consecutive_failures = 0
        self.save(
            update_fields=[
                "status", "status_detail", "last_seen_at", "last_frame_at",
                "consecutive_failures", "updated_at",
            ]
        )

    def mark_offline(self, detail: str = "") -> None:
        self.consecutive_failures += 1
        self.status = self.Status.DEGRADED if self.consecutive_failures < 3 else self.Status.OFFLINE
        self.status_detail = detail[:300]
        self.save(update_fields=["status", "status_detail", "consecutive_failures", "updated_at"])


class Zone(UUIDModel, TimeStampedModel, ActorStampedModel):
    """A polygon drawn over a camera view: geofence, counting line or shelf."""

    class Kind(models.TextChoices):
        RESTRICTED = "restricted", "Restricted area"
        HAZARD = "hazard", "Hazard zone"
        MONITORED = "monitored", "Monitored area"
        COUNTING = "counting", "Counting / occupancy area"
        SHELF = "shelf", "Stock / shelf area"
        EXCLUSION = "exclusion", "Exclusion (ignore this area)"

    class Severity(models.TextChoices):
        INFO = "info", "Info"
        LOW = "low", "Low"
        MEDIUM = "medium", "Medium"
        HIGH = "high", "High"
        CRITICAL = "critical", "Critical"

    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name="zones")
    name = models.CharField(max_length=140)
    kind = models.CharField(max_length=16, choices=Kind.choices, default=Kind.RESTRICTED)
    severity = models.CharField(max_length=10, choices=Severity.choices, default=Severity.HIGH)

    #: ``[[x, y], ...]`` in relative 0-1 coordinates, so a zone survives a
    #: change of stream resolution.
    polygon = models.JSONField(default=list)
    colour = models.CharField(max_length=7, default="#ef4444")

    #: {"days": ["mon", ...], "from": "18:00", "to": "07:00"}
    schedule = models.JSONField(default=dict, blank=True)
    max_occupancy = models.PositiveIntegerField(default=0, help_text="0 = no limit")
    min_dwell_seconds = models.PositiveIntegerField(default=0, help_text="0 = alert on entry")
    allowed_roles = models.JSONField(default=list, blank=True, help_text="Employee roles permitted here")

    is_active = models.BooleanField(default=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["camera", "name"]
        indexes = [models.Index(fields=["camera", "is_active"])]

    def __str__(self) -> str:
        return f"{self.name} @ {self.camera.name}"

    @property
    def organization(self):
        return self.camera.organization

    @property
    def point_count(self) -> int:
        return len(self.polygon or [])

    @property
    def is_valid(self) -> bool:
        return self.point_count >= 3

    @property
    def area_fraction(self) -> float:
        """Share of the camera view this zone covers (0-1)."""
        from apps.aiengine.vision.geometry import polygon_area

        return round(polygon_area(self.polygon or []), 4)

    @property
    def schedule_label(self) -> str:
        schedule = self.schedule or {}
        if not schedule:
            return "Always armed"
        days = schedule.get("days")
        window = f"{schedule.get('from', '00:00')}–{schedule.get('to', '23:59')}"
        return f"{', '.join(d.title() for d in days)} {window}" if days else window


class Employee(UUIDModel, OrganizationOwnedModel, TimeStampedModel, SoftDeleteModel, ActorStampedModel):
    """A person the workspace wants recognised.

    Enrolment stores an averaged embedding, never the face images themselves
    once processed — you cannot reconstruct a face from a 128-D unit vector.
    """

    organization = models.ForeignKey(
        "accounts.Organization", on_delete=models.CASCADE, related_name="employees"
    )
    employee_code = models.CharField(max_length=64, blank=True)
    full_name = models.CharField(max_length=180)
    role = models.CharField(max_length=120, blank=True)
    department = models.CharField(max_length=120, blank=True)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=32, blank=True)
    photo = models.ImageField(upload_to="employees/photos/", blank=True, null=True)

    #: Averaged, L2-normalised face embedding produced by ``campyface``.
    face_embedding = models.JSONField(default=list, blank=True)
    enrolment_samples = models.PositiveIntegerField(default=0)
    enrolled_at = models.DateTimeField(null=True, blank=True)
    embedding_model = models.CharField(max_length=80, blank=True)

    sites = models.ManyToManyField(Site, blank=True, related_name="employees")
    is_active = models.BooleanField(default=True)
    notes = models.TextField(blank=True)

    #: Written consent is a legal prerequisite for biometric processing in
    #: most jurisdictions; the platform records it explicitly.
    consent_given = models.BooleanField(default=False)
    consent_recorded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["full_name"]
        indexes = [models.Index(fields=["organization", "is_active"])]
        constraints = [
            models.UniqueConstraint(
                fields=["organization", "employee_code"],
                condition=~models.Q(employee_code=""),
                name="uniq_employee_code_per_org",
            )
        ]

    def __str__(self) -> str:
        return self.full_name

    def get_absolute_url(self) -> str:
        return reverse("dashboard:employee_detail", args=[self.uid])

    @property
    def display_name(self) -> str:
        return self.full_name

    @property
    def is_enrolled(self) -> bool:
        return bool(self.face_embedding) and self.consent_given

    @property
    def enrolment_status(self) -> str:
        if not self.consent_given:
            return "Consent required"
        if not self.face_embedding:
            return "Not enrolled"
        if self.enrolment_samples < 3:
            return "Weak enrolment"
        return "Enrolled"

    @property
    def initials(self) -> str:
        parts = [p for p in self.full_name.split(" ") if p]
        return (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else self.full_name[:2].upper()

    def set_embedding(self, embedding, samples: int, model_name: str = "campyface") -> None:
        import numpy as np

        vector = np.asarray(embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(vector))
        if norm:
            vector = vector / norm
        self.face_embedding = [round(float(v), 6) for v in vector]
        self.enrolment_samples = int(samples)
        self.enrolled_at = timezone.now()
        self.embedding_model = model_name


class CameraSnapshot(UUIDModel, TimeStampedModel):
    """A stored frame — the visual evidence attached to an event."""

    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name="snapshots")
    image = models.ImageField(upload_to="snapshots/%Y/%m/%d/")
    captured_at = models.DateTimeField(default=timezone.now, db_index=True)
    width = models.PositiveIntegerField(default=0)
    height = models.PositiveIntegerField(default=0)
    reason = models.CharField(max_length=60, blank=True)
    annotations = models.JSONField(default=dict, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)

    class Meta:
        ordering = ["-captured_at"]
        indexes = [models.Index(fields=["camera", "-captured_at"])]

    def __str__(self) -> str:
        return f"Snapshot {self.camera.name} @ {self.captured_at:%Y-%m-%d %H:%M:%S}"

    @property
    def organization(self):
        return self.camera.organization
