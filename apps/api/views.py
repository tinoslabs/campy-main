"""Campy AI REST API."""
from __future__ import annotations

from datetime import timedelta

from django.db.models import Count, Q
from django.utils import timezone
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework.response import Response

from apps.cameras.models import Camera, Employee, Site, Zone
from apps.events.models import AlertRule, Event, Incident
from apps.training.models import Dataset, ModelVersion, TrainingJob

from .permissions import HasCampyPermission
from .serializers import (
    AlertRuleSerializer,
    AnalyticsSummarySerializer,
    CameraSerializer,
    DatasetSerializer,
    EmployeeSerializer,
    EventIngestSerializer,
    EventSerializer,
    IncidentSerializer,
    ModelVersionSerializer,
    SiteSerializer,
    TrainingJobSerializer,
    ZoneSerializer,
)


class OrganizationScopedViewSet(viewsets.ModelViewSet):
    """Base viewset: everything is filtered to the caller's workspace."""

    permission_classes = [IsAuthenticated, HasCampyPermission]
    lookup_field = "uid"

    def get_organization(self):
        return getattr(self.request, "organization", None)

    def get_queryset(self):
        organization = self.get_organization()
        queryset = super().get_queryset()
        if getattr(self.request.user, "is_superadmin", False) and organization is None:
            return queryset
        if organization is None:
            return queryset.none()
        return queryset.filter(organization=organization)

    def perform_create(self, serializer):
        serializer.save(organization=self.get_organization())


class SiteViewSet(OrganizationScopedViewSet):
    """Physical locations."""

    queryset = Site.objects.all()
    serializer_class = SiteSerializer
    permission_module = "cameras"
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["is_active", "city", "country"]
    search_fields = ["name", "code", "city"]
    ordering_fields = ["name", "created_at"]


class CameraViewSet(OrganizationScopedViewSet):
    """Video sources and their analytics configuration."""

    queryset = Camera.objects.select_related("site").prefetch_related("zones")
    serializer_class = CameraSerializer
    permission_module = "cameras"
    permission_map = {
        "list": "cameras.view", "retrieve": "cameras.view", "health": "cameras.view",
        "create": "cameras.manage", "update": "cameras.manage",
        "partial_update": "cameras.manage", "destroy": "cameras.manage",
        "test_connection": "cameras.control",
    }
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["status", "protocol", "is_active", "site__uid"]
    search_fields = ["name", "location_note"]
    ordering_fields = ["name", "created_at", "last_frame_at"]

    @action(detail=True, methods=["post"], url_path="test")
    def test_connection(self, request, uid=None):
        """Attempt to pull a frame from this camera."""
        camera = self.get_object()
        from apps.cameras.services import check_connection

        ok, detail = check_connection(camera)
        (camera.mark_online if ok else camera.mark_offline)(detail)
        return Response(
            {"connected": ok, "detail": detail, "status": camera.status},
            status=status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @action(detail=True, methods=["get"], url_path="health")
    def health(self, request, uid=None):
        camera = self.get_object()
        return Response(
            {
                "status": camera.status,
                "healthy": camera.is_healthy,
                "detail": camera.status_detail,
                "last_frame_at": camera.last_frame_at,
                "frames_processed": camera.frames_processed,
                "events_generated": camera.events_generated,
                "active_analytics": camera.active_analytics(),
                "blocked_analytics": camera.blocked_analytics,
                "telemetry": camera.health or {},
            }
        )


class ZoneViewSet(OrganizationScopedViewSet):
    """Geofences and regions of interest."""

    queryset = Zone.objects.select_related("camera")
    serializer_class = ZoneSerializer
    permission_module = "zones"
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["kind", "is_active", "camera__uid"]

    def get_queryset(self):
        # Zones reach their organisation through their camera.
        organization = self.get_organization()
        queryset = Zone.objects.select_related("camera", "camera__site")
        if getattr(self.request.user, "is_superadmin", False) and organization is None:
            return queryset
        if organization is None:
            return queryset.none()
        return queryset.filter(camera__organization=organization)

    def perform_create(self, serializer):
        camera_uid = self.request.data.get("camera_uid") or self.request.data.get("camera")
        camera = Camera.objects.filter(
            uid=camera_uid, organization=self.get_organization()
        ).first()
        if camera is None:
            from rest_framework.exceptions import ValidationError

            raise ValidationError({"camera_uid": "Unknown camera for this workspace."})
        serializer.save(camera=camera)


class EmployeeViewSet(OrganizationScopedViewSet):
    """The employee directory. Face embeddings are never exposed."""

    queryset = Employee.objects.all()
    serializer_class = EmployeeSerializer
    permission_module = "employees"
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["is_active", "department", "consent_given"]
    search_fields = ["full_name", "employee_code", "role", "department"]


class EventViewSet(
    mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet
):
    """Detections. Read-only except for the acknowledge/resolve actions."""

    queryset = Event.objects.select_related("camera", "camera__site", "zone", "employee", "incident", "snapshot")
    serializer_class = EventSerializer
    permission_classes = [IsAuthenticated, HasCampyPermission]
    permission_module = "events"
    lookup_field = "uid"
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["analytic", "severity", "status", "event_type", "camera__uid"]
    search_fields = ["title", "description", "event_type"]
    ordering_fields = ["occurred_at", "severity", "confidence"]
    ordering = ["-occurred_at"]

    #: Per-action permissions. Writes to events are not a single "manage"
    #: capability — acknowledging, resolving and ingesting are distinct rights.
    permission_map = {
        "list": "events.view",
        "retrieve": "events.view",
        "acknowledge": "events.acknowledge",
        "resolve": "events.resolve",
        "ingest": "events.view",
    }

    #: Only the high-volume ingest endpoint is rate-limited; reads are not.
    throttle_scope = None

    def get_throttles(self):
        if getattr(self, "action", None) == "ingest":
            self.throttle_scope = "ingest"
        else:
            self.throttle_scope = None
        return super().get_throttles()

    def get_queryset(self):
        organization = getattr(self.request, "organization", None)
        queryset = super().get_queryset()
        if organization is None:
            return queryset.none() if not getattr(self.request.user, "is_superadmin", False) else queryset
        queryset = queryset.filter(organization=organization)

        since = self.request.query_params.get("since_days")
        if since:
            try:
                queryset = queryset.filter(
                    occurred_at__gte=timezone.now() - timedelta(days=int(since))
                )
            except ValueError:
                pass
        return queryset

    @action(detail=True, methods=["post"])
    def acknowledge(self, request, uid=None):
        event = self.get_object()
        if not request.user.has_campy_perm("events.acknowledge", request.organization):
            return Response({"detail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)
        event.acknowledge(
            request.user if getattr(request.user, "pk", None) else None,
            request.data.get("note", ""),
        )
        return Response(EventSerializer(event, context={"request": request}).data)

    @action(detail=True, methods=["post"])
    def resolve(self, request, uid=None):
        event = self.get_object()
        if not request.user.has_campy_perm("events.resolve", request.organization):
            return Response({"detail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)
        event.resolve(
            request.user if getattr(request.user, "pk", None) else None,
            request.data.get("note", ""),
            request.data.get("label", ""),
        )
        return Response(EventSerializer(event, context={"request": request}).data)

    @action(detail=False, methods=["post"], url_path="ingest")
    def ingest(self, request):
        """Accept a detection from an edge device.

        Lets a customer run Campy AI's engine on their own hardware — or their
        own detector entirely — and still get incident grouping, alert routing
        and the full dashboard.
        """
        if not request.user.has_campy_perm("events.view", request.organization):
            return Response({"detail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)

        serializer = EventIngestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        camera = Camera.objects.filter(
            uid=data["camera"], organization=request.organization
        ).first()
        if camera is None:
            return Response(
                {"error": {"code": "unknown_camera", "message": "No such camera in this workspace."}},
                status=status.HTTP_400_BAD_REQUEST,
            )

        zone = None
        if data.get("zone"):
            zone = Zone.objects.filter(uid=data["zone"], camera=camera).first()

        from apps.aiengine.detectors.base import EventCandidate
        from apps.events.services import dispatch_alerts, record_event

        candidate = EventCandidate(
            analytic=data["analytic"],
            event_type=data["event_type"],
            severity=data["severity"],
            confidence=data["confidence"],
            title=data["title"],
            description=data.get("description", ""),
            box=tuple(data["bounding_box"]) if data.get("bounding_box") else None,
            track_id=data.get("track_id"),
            zone_id=zone.pk if zone else None,
            metadata={**data.get("metadata", {}), "source": "api_ingest"},
        )
        event = record_event(camera, candidate)
        if event is None:
            return Response(
                {"deduplicated": True, "detail": "An identical finding was recorded moments ago."},
                status=status.HTTP_200_OK,
            )
        dispatch_alerts(event)
        return Response(
            EventSerializer(event, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


class IncidentViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = Incident.objects.select_related("primary_camera", "site", "assigned_to")
    serializer_class = IncidentSerializer
    permission_classes = [IsAuthenticated, HasCampyPermission]
    permission_module = "events"
    lookup_field = "uid"
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["status", "severity"]
    ordering = ["-started_at"]

    def get_queryset(self):
        organization = getattr(self.request, "organization", None)
        if organization is None:
            return super().get_queryset().none()
        return super().get_queryset().filter(organization=organization)


class AlertRuleViewSet(OrganizationScopedViewSet):
    queryset = AlertRule.objects.all()
    serializer_class = AlertRuleSerializer
    permission_module = "alerts"
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["is_active", "min_severity"]


class DatasetViewSet(OrganizationScopedViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    permission_module = "training"
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["status", "task"]


class TrainingJobViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = TrainingJob.objects.select_related("dataset")
    serializer_class = TrainingJobSerializer
    permission_classes = [IsAuthenticated, HasCampyPermission]
    permission_module = "training"
    lookup_field = "uid"
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["status", "architecture"]

    def get_queryset(self):
        organization = getattr(self.request, "organization", None)
        if organization is None:
            return super().get_queryset().none()
        return super().get_queryset().filter(organization=organization)


class ModelVersionViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = ModelVersion.objects.all()
    serializer_class = ModelVersionSerializer
    permission_classes = [IsAuthenticated, HasCampyPermission]
    permission_module = "models"
    permission_map = {"list": "models.view", "retrieve": "models.view", "deploy": "models.deploy"}
    lookup_field = "uid"
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["slot", "status", "architecture", "is_base_model"]

    def get_queryset(self):
        organization = getattr(self.request, "organization", None)
        queryset = super().get_queryset()
        if organization is None:
            return queryset.filter(is_base_model=True)
        return queryset.filter(Q(organization=organization) | Q(is_base_model=True))

    @action(detail=True, methods=["post"])
    def deploy(self, request, uid=None):
        version = self.get_object()
        if not request.user.has_campy_perm("models.deploy", request.organization):
            return Response({"detail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)
        if version.is_base_model:
            return Response(
                {"detail": "Base models are managed by Campy AI."}, status=status.HTTP_400_BAD_REQUEST
            )
        version.deploy(request.user if getattr(request.user, "pk", None) else None)
        return Response(ModelVersionSerializer(version).data)


@extend_schema(
    responses=AnalyticsSummarySerializer,
    parameters=[
        OpenApiParameter(
            "days", int, description="Window in days (1-365). Defaults to 7.", required=False
        )
    ],
    summary="Headline analytics for the caller's workspace",
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def analytics_summary(request):
    """Headline numbers for the caller's workspace."""
    organization = getattr(request, "organization", None)
    if organization is None:
        return Response({"detail": "No workspace selected."}, status=status.HTTP_400_BAD_REQUEST)
    if not request.user.has_campy_perm("analytics.view", organization):
        return Response({"detail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)

    try:
        days = max(1, min(int(request.query_params.get("days", 7)), 365))
    except ValueError:
        days = 7

    since = timezone.now() - timedelta(days=days)
    events = Event.objects.filter(organization=organization, occurred_at__gte=since)
    cameras = Camera.objects.filter(organization=organization, is_active=True)
    total = events.count()

    payload = {
        "period_days": days,
        "total_events": total,
        "by_severity": {
            row["severity"]: row["total"]
            for row in events.values("severity").annotate(total=Count("id"))
        },
        "by_analytic": {
            row["analytic"]: row["total"]
            for row in events.values("analytic").annotate(total=Count("id"))
        },
        "open_events": Event.objects.filter(organization=organization, status=Event.Status.OPEN).count(),
        "open_incidents": Incident.objects.filter(
            organization=organization, status__in=[Incident.Status.OPEN, Incident.Status.INVESTIGATING]
        ).count(),
        "cameras_online": cameras.filter(status=Camera.Status.ONLINE).count(),
        "cameras_total": cameras.count(),
        "false_positive_rate": round(
            events.filter(status=Event.Status.FALSE_POSITIVE).count() / max(total, 1) * 100, 2
        ),
    }
    return Response(AnalyticsSummarySerializer(payload).data)


@extend_schema(
    responses={200: dict},
    summary="API discovery document",
    description="Lists every endpoint and explains how to authenticate.",
)
@api_view(["GET"])
@permission_classes([AllowAny])
def api_root(request):
    """Discovery document — what this API offers and how to authenticate."""
    from django.urls import reverse

    def url(name):
        return request.build_absolute_uri(reverse(f"api:{name}"))

    return Response(
        {
            "name": "Campy AI Platform API",
            "version": "1.0.0",
            "documentation": request.build_absolute_uri("/api/v1/docs/"),
            "schema": request.build_absolute_uri("/api/v1/schema/"),
            "authentication": {
                "api_key": "Authorization: Api-Key <prefix>.<secret>",
                "jwt": "Authorization: Bearer <access token> — obtain from /api/v1/auth/token/",
                "workspace_header": "X-Campy-Organization: <workspace-slug> (JWT/session callers with several workspaces)",
            },
            "endpoints": {
                "sites": request.build_absolute_uri("/api/v1/sites/"),
                "cameras": request.build_absolute_uri("/api/v1/cameras/"),
                "zones": request.build_absolute_uri("/api/v1/zones/"),
                "employees": request.build_absolute_uri("/api/v1/employees/"),
                "events": request.build_absolute_uri("/api/v1/events/"),
                "event_ingest": request.build_absolute_uri("/api/v1/events/ingest/"),
                "incidents": request.build_absolute_uri("/api/v1/incidents/"),
                "alert_rules": request.build_absolute_uri("/api/v1/alert-rules/"),
                "datasets": request.build_absolute_uri("/api/v1/datasets/"),
                "training_jobs": request.build_absolute_uri("/api/v1/training-jobs/"),
                "models": request.build_absolute_uri("/api/v1/models/"),
                "analytics_summary": url("analytics_summary"),
            },
        }
    )
