from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView, TokenVerifyView

from . import views

app_name = "api"

router = DefaultRouter()
router.register("sites", views.SiteViewSet, basename="site")
router.register("cameras", views.CameraViewSet, basename="camera")
router.register("zones", views.ZoneViewSet, basename="zone")
router.register("employees", views.EmployeeViewSet, basename="employee")
router.register("events", views.EventViewSet, basename="event")
router.register("incidents", views.IncidentViewSet, basename="incident")
router.register("alert-rules", views.AlertRuleViewSet, basename="alertrule")
router.register("datasets", views.DatasetViewSet, basename="dataset")
router.register("training-jobs", views.TrainingJobViewSet, basename="trainingjob")
router.register("models", views.ModelVersionViewSet, basename="modelversion")

urlpatterns = [
    path("", views.api_root, name="root"),
    path("analytics/summary/", views.analytics_summary, name="analytics_summary"),

    # Auth
    path("auth/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("auth/token/verify/", TokenVerifyView.as_view(), name="token_verify"),

    # Schema & docs
    path("schema/", SpectacularAPIView.as_view(), name="schema"),
    path("docs/", SpectacularSwaggerView.as_view(url_name="api:schema"), name="docs"),

    path("", include(router.urls)),
]
