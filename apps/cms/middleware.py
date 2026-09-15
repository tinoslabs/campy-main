"""Site settings and maintenance mode."""
from __future__ import annotations

from django.http import HttpResponse
from django.template.loader import render_to_string


class SiteSettingsMiddleware:
    """Attaches the CMS singleton to the request and honours maintenance mode."""

    BYPASS_PREFIXES = ("/admin/", "/static/", "/media/", "/accounts/login", "/healthz", "/webhooks/")

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.site_settings = self._settings()

        if (
            request.site_settings
            and request.site_settings.maintenance_mode
            and not request.path.startswith(self.BYPASS_PREFIXES)
        ):
            user = getattr(request, "user", None)
            if user is None or not (user.is_authenticated and user.is_platform_team):
                return HttpResponse(
                    render_to_string(
                        "cms/maintenance.html",
                        {"settings_obj": request.site_settings},
                        request=request,
                    ),
                    status=503,
                )

        return self.get_response(request)

    @staticmethod
    def _settings():
        try:
            from .models import SiteSettings

            return SiteSettings.load()
        except Exception:  # noqa: BLE001 - the table may not exist yet (first migrate)
            return None
