from django.apps import AppConfig


class AccountsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.accounts"
    verbose_name = "Identity & Access"

    def ready(self):  # noqa: D102
        from . import signals  # noqa: F401
