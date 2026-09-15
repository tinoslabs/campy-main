"""Pre-aggregated metrics and saved reports."""
from __future__ import annotations

from django.conf import settings
from django.db import models

from apps.core.models import OrganizationOwnedModel, TimeStampedModel, UUIDModel


class DailyMetric(OrganizationOwnedModel, TimeStampedModel):
    """One row per organisation / site / camera / day.

    Rolling these up nightly is what keeps the dashboard instant: charting a
    year of activity reads 365 rows instead of scanning millions of events.
    """

    date = models.DateField(db_index=True)
    site = models.ForeignKey(
        "cameras.Site", null=True, blank=True, on_delete=models.CASCADE, related_name="daily_metrics"
    )
    camera = models.ForeignKey(
        "cameras.Camera", null=True, blank=True, on_delete=models.CASCADE, related_name="daily_metrics"
    )

    events_total = models.PositiveIntegerField(default=0)
    events_critical = models.PositiveIntegerField(default=0)
    events_high = models.PositiveIntegerField(default=0)
    events_by_analytic = models.JSONField(default=dict, blank=True)
    events_by_hour = models.JSONField(default=list, blank=True)

    incidents_opened = models.PositiveIntegerField(default=0)
    incidents_resolved = models.PositiveIntegerField(default=0)
    false_positives = models.PositiveIntegerField(default=0)
    median_response_seconds = models.FloatField(default=0.0)

    people_peak = models.PositiveIntegerField(default=0)
    people_average = models.FloatField(default=0.0)
    unique_employees = models.PositiveIntegerField(default=0)
    frames_processed = models.BigIntegerField(default=0)
    camera_uptime_percent = models.FloatField(default=0.0)

    class Meta:
        ordering = ["-date"]
        constraints = [
            models.UniqueConstraint(
                fields=["organization", "date", "site", "camera"], name="uniq_daily_metric"
            )
        ]
        indexes = [models.Index(fields=["organization", "-date"])]

    def __str__(self) -> str:
        scope = self.camera or self.site or self.organization
        return f"{scope} — {self.date}"

    @property
    def accuracy_signal(self) -> float:
        """Share of events operators did *not* mark as false positives."""
        if not self.events_total:
            return 0.0
        return round((1 - self.false_positives / self.events_total) * 100, 1)


class Report(UUIDModel, OrganizationOwnedModel, TimeStampedModel):
    """A saved or scheduled report definition."""

    class Kind(models.TextChoices):
        SAFETY = "safety", "Safety summary"
        ACTIVITY = "activity", "Activity & attendance"
        INCIDENTS = "incidents", "Incident log"
        CROWD = "crowd", "Crowd & occupancy"
        COMPLIANCE = "compliance", "Compliance / audit"
        CUSTOM = "custom", "Custom"

    class Frequency(models.TextChoices):
        ONCE = "once", "One-off"
        DAILY = "daily", "Daily"
        WEEKLY = "weekly", "Weekly"
        MONTHLY = "monthly", "Monthly"

    class Format(models.TextChoices):
        HTML = "html", "HTML"
        CSV = "csv", "CSV"
        PDF = "pdf", "PDF"

    name = models.CharField(max_length=180)
    kind = models.CharField(max_length=12, choices=Kind.choices, default=Kind.SAFETY)
    frequency = models.CharField(max_length=10, choices=Frequency.choices, default=Frequency.ONCE)
    output_format = models.CharField(max_length=6, choices=Format.choices, default=Format.HTML)

    sites = models.ManyToManyField("cameras.Site", blank=True, related_name="reports")
    cameras = models.ManyToManyField("cameras.Camera", blank=True, related_name="reports")
    analytics = models.JSONField(default=list, blank=True)
    date_range_days = models.PositiveIntegerField(default=7)
    filters = models.JSONField(default=dict, blank=True)

    recipients = models.JSONField(default=list, blank=True, help_text="Email addresses")
    is_active = models.BooleanField(default=True)
    last_run_at = models.DateTimeField(null=True, blank=True)
    next_run_at = models.DateTimeField(null=True, blank=True)
    run_count = models.PositiveIntegerField(default=0)

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="reports"
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self) -> str:
        from django.urls import reverse

        return reverse("dashboard:report_detail", args=[self.uid])

    @property
    def is_scheduled(self) -> bool:
        return self.frequency != self.Frequency.ONCE and self.is_active


class ReportRun(UUIDModel, TimeStampedModel):
    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"

    report = models.ForeignKey(Report, on_delete=models.CASCADE, related_name="runs")
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.RUNNING)
    period_start = models.DateTimeField()
    period_end = models.DateTimeField()
    payload = models.JSONField(default=dict, blank=True)
    file = models.FileField(upload_to="reports/%Y/%m/", blank=True, null=True)
    error = models.CharField(max_length=400, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    delivered_to = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.report.name} @ {self.created_at:%Y-%m-%d %H:%M}"

    @property
    def organization(self):
        return self.report.organization


class SavedView(UUIDModel, OrganizationOwnedModel, TimeStampedModel):
    """A user's saved filter combination on the events list."""

    name = models.CharField(max_length=140)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="saved_views"
    )
    target = models.CharField(max_length=40, default="events")
    filters = models.JSONField(default=dict, blank=True)
    is_shared = models.BooleanField(default=False)
    is_default = models.BooleanField(default=False)
    use_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name

    @property
    def query_string(self) -> str:
        from urllib.parse import urlencode

        return urlencode({k: v for k, v in (self.filters or {}).items() if v not in (None, "")})
