from django.apps import AppConfig


class APIConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.api"
    verbose_name = "Public API"

    def ready(self):  # noqa: D102
        from . import schema  # noqa: F401  — registers the OpenAPI extensions
