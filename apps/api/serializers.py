"""DRF serializers for the Campy AI public API."""
from __future__ import annotations

from rest_framework import serializers

from apps.cameras.models import Camera, Employee, Site, Zone
from apps.events.models import AlertRule, Event, Incident
from apps.training.models import Dataset, ModelVersion, TrainingJob
from apps.cameras.validators import clean_zone_polygon


class SiteSerializer(serializers.ModelSerializer):
    camera_count = serializers.IntegerField(read_only=True)
    online_camera_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Site
        fields = [
            "uid", "name", "code", "description", "address", "city", "country",
            "latitude", "longitude", "timezone", "is_active",
            "camera_count", "online_camera_count", "created_at",
        ]
        read_only_fields = ["uid", "created_at"]


class ZoneSerializer(serializers.ModelSerializer):
    camera_uid = serializers.UUIDField(source="camera.uid", read_only=True)

    class Meta:
        model = Zone
        fields = [
            "uid", "camera_uid", "name", "kind", "severity", "polygon", "colour",
            "schedule", "max_occupancy", "min_dwell_seconds", "allowed_roles",
            "is_active", "created_at",
        ]
        read_only_fields = ["uid", "camera_uid", "created_at"]

    def validate_polygon(self, value):
        return clean_zone_polygon(value, serializers.ValidationError)


class CameraSerializer(serializers.ModelSerializer):
    site_uid = serializers.UUIDField(source="site.uid", read_only=True)
    site_name = serializers.CharField(source="site.name", read_only=True)
    zones = ZoneSerializer(many=True, read_only=True)
    active_analytics = serializers.SerializerMethodField()
    stream_url = serializers.SerializerMethodField()

    class Meta:
        model = Camera
        fields = [
            "uid", "name", "site_uid", "site_name", "location_note", "description",
            "protocol", "stream_url", "status", "status_detail",
            "resolution_width", "resolution_height", "target_fps",
            "enabled_analytics", "active_analytics", "zones",
            "record_evidence", "privacy_blur_faces", "is_active",
            "last_seen_at", "last_frame_at", "frames_processed", "events_generated",
            "created_at",
        ]
        read_only_fields = [
            "uid", "status", "status_detail", "last_seen_at", "last_frame_at",
            "frames_processed", "events_generated", "created_at",
        ]

    def get_active_analytics(self, obj) -> list:
        return obj.active_analytics()

    def get_stream_url(self, obj) -> str:
        """Never expose embedded credentials over the API."""
        return obj.masked_url


class EmployeeSerializer(serializers.ModelSerializer):
    is_enrolled = serializers.BooleanField(read_only=True)
    enrolment_status = serializers.CharField(read_only=True)

    class Meta:
        model = Employee
        fields = [
            "uid", "employee_code", "full_name", "role", "department",
            "email", "phone", "is_active", "consent_given",
            "is_enrolled", "enrolment_status", "enrolment_samples", "enrolled_at",
            "created_at",
        ]
        # The face embedding is deliberately never serialised: it is biometric
        # data, and nothing outside the inference pipeline needs it.
        read_only_fields = ["uid", "is_enrolled", "enrolment_status", "enrolment_samples", "enrolled_at", "created_at"]


class EventSerializer(serializers.ModelSerializer):
    camera_uid = serializers.UUIDField(source="camera.uid", read_only=True)
    camera_name = serializers.CharField(source="camera.name", read_only=True)
    site_name = serializers.CharField(source="camera.site.name", read_only=True)
    zone_name = serializers.CharField(source="zone.name", read_only=True)
    employee_name = serializers.CharField(source="employee.full_name", read_only=True)
    incident_reference = serializers.CharField(source="incident.reference", read_only=True)
    analytic_label = serializers.CharField(read_only=True)
    snapshot_url = serializers.SerializerMethodField()

    class Meta:
        model = Event
        fields = [
            "uid", "analytic", "analytic_label", "event_type", "severity", "status",
            "title", "description", "confidence", "occurred_at", "last_seen_at",
            "occurrence_count", "bounding_box", "track_id", "metadata",
            "camera_uid", "camera_name", "site_name", "zone_name", "employee_name",
            "incident_reference", "snapshot_url",
            "acknowledged_at", "resolved_at", "feedback_label",
        ]
        read_only_fields = fields

    def get_snapshot_url(self, obj) -> str | None:
        if obj.snapshot and obj.snapshot.image:
            request = self.context.get("request")
            url = obj.snapshot.image.url
            return request.build_absolute_uri(url) if request else url
        return None


class EventIngestSerializer(serializers.Serializer):
    """Payload an edge device posts when it detects something itself."""

    camera = serializers.UUIDField()
    analytic = serializers.CharField(max_length=20)
    event_type = serializers.CharField(max_length=48)
    severity = serializers.ChoiceField(
        choices=["info", "low", "medium", "high", "critical"], default="medium"
    )
    title = serializers.CharField(max_length=220)
    description = serializers.CharField(required=False, allow_blank=True, default="")
    confidence = serializers.FloatField(min_value=0.0, max_value=1.0, default=0.8)
    occurred_at = serializers.DateTimeField(required=False)
    bounding_box = serializers.ListField(
        child=serializers.FloatField(), required=False, allow_empty=True, default=list
    )
    track_id = serializers.IntegerField(required=False, allow_null=True)
    zone = serializers.UUIDField(required=False, allow_null=True)
    metadata = serializers.DictField(required=False, default=dict)

    def validate_analytic(self, value):
        from apps.aiengine.detectors import ANALYTIC_KEYS

        if value not in ANALYTIC_KEYS:
            raise serializers.ValidationError(
                f"Unknown analytic '{value}'. Valid values: {', '.join(ANALYTIC_KEYS)}"
            )
        return value

    def validate_bounding_box(self, value):
        if value and len(value) != 4:
            raise serializers.ValidationError("bounding_box must be [x1, y1, x2, y2].")
        return value


class IncidentSerializer(serializers.ModelSerializer):
    camera_name = serializers.CharField(source="primary_camera.name", read_only=True)
    site_name = serializers.CharField(source="site.name", read_only=True)
    assigned_to_name = serializers.CharField(source="assigned_to.get_full_name", read_only=True)
    duration_seconds = serializers.FloatField(read_only=True)

    class Meta:
        model = Incident
        fields = [
            "uid", "reference", "title", "summary", "severity", "status",
            "analytics", "event_count", "started_at", "ended_at", "duration_seconds",
            "camera_name", "site_name", "assigned_to_name", "escalation_level",
            "resolved_at", "resolution_note",
        ]
        read_only_fields = fields


class AlertRuleSerializer(serializers.ModelSerializer):
    class Meta:
        model = AlertRule
        fields = [
            "uid", "name", "description", "is_active", "priority",
            "analytics", "event_types", "min_severity", "min_confidence",
            "active_schedule", "cooldown_seconds", "threshold_count",
            "threshold_window_seconds", "escalate_after_seconds",
            "create_incident", "auto_acknowledge", "fire_count", "last_fired_at",
        ]
        read_only_fields = ["uid", "fire_count", "last_fired_at"]


class DatasetSerializer(serializers.ModelSerializer):
    is_trainable = serializers.BooleanField(read_only=True)
    readiness = serializers.DictField(read_only=True)

    class Meta:
        model = Dataset
        fields = [
            "uid", "name", "description", "task", "status", "classes",
            "image_size", "validation_split", "augmentation",
            "sample_count", "labelled_count", "class_distribution",
            "is_trainable", "readiness", "created_at",
        ]
        read_only_fields = [
            "uid", "status", "sample_count", "labelled_count",
            "class_distribution", "is_trainable", "readiness", "created_at",
        ]


class TrainingJobSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source="dataset.name", read_only=True)
    best_metric = serializers.FloatField(read_only=True)

    class Meta:
        model = TrainingJob
        fields = [
            "uid", "name", "dataset_name", "architecture", "target_slot",
            "status", "progress", "current_epoch", "epochs",
            "started_at", "finished_at", "duration_seconds",
            "metrics", "best_metric", "error_message",
        ]
        read_only_fields = fields


class ModelVersionSerializer(serializers.ModelSerializer):
    slot_label = serializers.CharField(read_only=True)
    size_label = serializers.CharField(read_only=True)

    class Meta:
        model = ModelVersion
        fields = [
            "uid", "name", "version", "slot", "slot_label", "architecture", "task",
            "status", "classes", "input_size", "parameter_count",
            "size_label", "accuracy", "precision", "recall", "f1_score",
            "inference_ms", "is_base_model", "deployed_at", "created_at",
        ]
        read_only_fields = fields


class AnalyticsSummarySerializer(serializers.Serializer):
    """Read-only shape returned by ``/analytics/summary/``."""

    period_days = serializers.IntegerField()
    total_events = serializers.IntegerField()
    by_severity = serializers.DictField()
    by_analytic = serializers.DictField()
    open_events = serializers.IntegerField()
    open_incidents = serializers.IntegerField()
    cameras_online = serializers.IntegerField()
    cameras_total = serializers.IntegerField()
    false_positive_rate = serializers.FloatField()
