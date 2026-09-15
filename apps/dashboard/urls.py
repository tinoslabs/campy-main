from django.urls import path

from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.home, name="home"),
    path("start/", views.onboarding, name="onboarding"),

    # Live monitoring
    path("live/", views.live, name="live"),
    path("live/feed/", views.live_feed, name="live_feed"),
    path("live/preview/<uuid:uid>.jpg", views.camera_preview, name="camera_preview"),
    path("live/stream/<uuid:uid>.mjpg", views.camera_stream, name="camera_stream"),

    # Events
    path("events/", views.events, name="events"),
    path("events/export/", views.events_export, name="events_export"),
    path("events/bulk/", views.events_bulk_action, name="events_bulk"),
    path("events/<uuid:uid>/", views.event_detail, name="event_detail"),
    path("events/<uuid:uid>/<str:action>/", views.event_action, name="event_action"),

    # Incidents
    path("incidents/", views.incidents, name="incidents"),
    path("incidents/<uuid:uid>/", views.incident_detail, name="incident_detail"),
    path("incidents/<uuid:uid>/<str:action>/", views.incident_action, name="incident_action"),

    # Sites
    path("sites/", views.sites, name="sites"),
    path("sites/new/", views.site_form, name="site_create"),
    path("sites/<uuid:uid>/", views.site_detail, name="site_detail"),
    path("sites/<uuid:uid>/edit/", views.site_form, name="site_edit"),

    # Cameras
    path("cameras/", views.cameras, name="cameras"),
    path("cameras/new/", views.camera_form, name="camera_create"),
    path("cameras/<uuid:uid>/", views.camera_detail, name="camera_detail"),
    path("cameras/<uuid:uid>/edit/", views.camera_form, name="camera_edit"),
    path("cameras/<uuid:uid>/test/", views.camera_test, name="camera_test"),
    path("cameras/<uuid:uid>/run/", views.camera_run, name="camera_run"),
    path("cameras/<uuid:uid>/delete/", views.camera_delete, name="camera_delete"),

    # Zones
    path("zones/", views.zones, name="zones"),
    path("cameras/<uuid:camera_uid>/zones/new/", views.zone_form, name="zone_create"),
    path("zones/<uuid:uid>/edit/", views.zone_form, name="zone_edit"),
    path("zones/<uuid:uid>/delete/", views.zone_delete, name="zone_delete"),

    # Employees
    path("employees/", views.employees, name="employees"),
    path("employees/new/", views.employee_form, name="employee_create"),
    path("employees/<uuid:uid>/", views.employee_detail, name="employee_detail"),
    path("employees/<uuid:uid>/edit/", views.employee_form, name="employee_edit"),
    path("employees/<uuid:uid>/enroll/", views.employee_enroll, name="employee_enroll"),
    path("employees/<uuid:uid>/unenroll/", views.employee_unenroll, name="employee_unenroll"),

    # Alerting
    path("alerts/", views.alert_rules, name="alert_rules"),
    path("alerts/new/", views.alert_rule_form, name="alert_rule_create"),
    path("alerts/<uuid:uid>/edit/", views.alert_rule_form, name="alert_rule_detail"),
    path("alerts/<uuid:uid>/delete/", views.alert_rule_delete, name="alert_rule_delete"),
    path("channels/new/", views.channel_form, name="channel_create"),
    path("channels/<uuid:uid>/edit/", views.channel_form, name="channel_edit"),
    path("channels/<uuid:uid>/test/", views.channel_test, name="channel_test"),
    path("channels/<uuid:uid>/delete/", views.channel_delete, name="channel_delete"),
    path("notifications/", views.notifications, name="notifications"),

    # Analytics & reports
    path("analytics/", views.analytics, name="analytics"),
    path("reports/", views.reports, name="reports"),
    path("reports/new/", views.report_form, name="report_create"),
    path("reports/<uuid:uid>/", views.report_detail, name="report_detail"),
    path("reports/<uuid:uid>/edit/", views.report_form, name="report_edit"),
    path("reports/analysis.pdf", views.analysis_report, name="analysis_report"),
    path("reports/<uuid:uid>/export/", views.report_export, name="report_export"),

    # Training
    path("datasets/", views.datasets, name="datasets"),
    path("datasets/new/", views.dataset_form, name="dataset_create"),
    path("datasets/<uuid:uid>/", views.dataset_detail, name="dataset_detail"),
    path("datasets/<uuid:uid>/edit/", views.dataset_form, name="dataset_edit"),
    path("datasets/<uuid:uid>/upload/", views.dataset_upload, name="dataset_upload"),
    path("datasets/<uuid:uid>/label/", views.dataset_label, name="dataset_label"),
    path("datasets/<uuid:uid>/bootstrap/", views.dataset_bootstrap, name="dataset_bootstrap"),
    path("datasets/<uuid:uid>/harvest/", views.dataset_harvest, name="dataset_harvest"),
    path("datasets/<uuid:uid>/delete/", views.dataset_delete, name="dataset_delete"),

    path("training/", views.training_jobs, name="training_jobs"),
    path("training/new/", views.training_job_form, name="training_job_create"),
    path("training/<uuid:uid>/", views.training_job_detail, name="training_job_detail"),
    path("training/<uuid:uid>/progress/", views.training_job_progress, name="training_job_progress"),
    path("training/<uuid:uid>/cancel/", views.training_job_cancel, name="training_job_cancel"),

    # Models
    path("models/", views.models, name="models"),
    path("models/<uuid:uid>/", views.model_detail, name="model_detail"),
    path("models/<uuid:uid>/deploy/", views.model_deploy, name="model_deploy"),
    path("models/<uuid:uid>/retire/", views.model_retire, name="model_retire"),

    # Workspace
    path("team/", views.team, name="team"),
    path("settings/", views.settings_view, name="settings"),
    path("settings/privacy/", views.settings_privacy, name="settings_privacy"),
    path("api-keys/", views.api_keys, name="api_keys"),
    path("api-keys/new/", views.api_key_create, name="api_key_create"),
    path("api-keys/<uuid:uid>/revoke/", views.api_key_revoke, name="api_key_revoke"),
    path("audit/", views.audit, name="audit"),
    path("audit/export/", views.audit_export, name="audit_export"),

    # Platform console
    path("platform/", views.admin_home, name="admin_home"),
    path("platform/organizations/", views.admin_organizations, name="admin_organizations"),
    path("platform/organizations/<slug:slug>/", views.admin_organization_detail, name="admin_organization_detail"),
    path("platform/organizations/<slug:slug>/status/", views.admin_organization_status, name="admin_organization_status"),
    path("platform/packages/", views.admin_packages, name="admin_packages"),
    path("platform/packages/new/", views.admin_package_form, name="admin_package_create"),
    path("platform/packages/<slug:slug>/edit/", views.admin_package_form, name="admin_package_edit"),
    path("platform/cms/", views.admin_cms, name="admin_cms"),

    # Aliases used by model get_absolute_url()
    path("workspace/<slug:slug>/", views.admin_organization_detail, name="organization_detail"),
]
