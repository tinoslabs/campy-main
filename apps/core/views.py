"""Operational endpoints and error pages."""
from __future__ import annotations

from django.conf import settings
from django.db import connection
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.cache import cache_control


def healthz(request):
    """Liveness/readiness probe for load balancers and orchestrators."""
    checks = {"database": False, "models_dir": False}
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            checks["database"] = cursor.fetchone() == (1,)
    except Exception:  # noqa: BLE001
        checks["database"] = False

    try:
        checks["models_dir"] = settings.CAMPY["MODEL_ROOT"].exists()
    except Exception:  # noqa: BLE001
        checks["models_dir"] = False

    healthy = all(checks.values())
    return JsonResponse(
        {"status": "ok" if healthy else "degraded", "checks": checks, "version": "1.0.0"},
        status=200 if healthy else 503,
    )


@cache_control(max_age=3600)
def robots_txt(request):
    lines = [
        "User-agent: *",
        "Disallow: /app/",
        "Disallow: /accounts/",
        "Disallow: /billing/",
        "Disallow: /api/",
        "Disallow: /django-admin/",
        "Allow: /",
        "",
        f"Sitemap: {request.build_absolute_uri('/sitemap.xml')}",
    ]
    return HttpResponse("\n".join(lines), content_type="text/plain")


def sitemap_xml(request):
    """Sitemap built from published CMS content."""
    from apps.cms.models import Page, Post

    entries = []
    for page in Page.objects.published():
        entries.append((request.build_absolute_uri(page.get_absolute_url()), page.updated_at, "0.8"))
    for post in Post.objects.published():
        entries.append((request.build_absolute_uri(post.get_absolute_url()), post.updated_at, "0.6"))

    body = ['<?xml version="1.0" encoding="UTF-8"?>',
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for url, updated, priority in entries:
        body.append(
            f"<url><loc>{url}</loc>"
            f"<lastmod>{updated:%Y-%m-%d}</lastmod>"
            f"<priority>{priority}</priority></url>"
        )
    body.append("</urlset>")
    return HttpResponse("\n".join(body), content_type="application/xml")


# ---------------------------------------------------------------------------
# Error pages
# ---------------------------------------------------------------------------
def _error(request, status: int, title: str, message: str):
    if request.path.startswith("/api/"):
        return JsonResponse({"error": {"code": status, "message": message}}, status=status)
    return render(
        request,
        "errors/error.html",
        {"status": status, "title": title, "message": message},
        status=status,
    )


def bad_request(request, exception=None):
    return _error(request, 400, "Bad request", "That request could not be understood.")


def permission_denied(request, exception=None):
    return _error(
        request, 403, "Not permitted",
        str(exception) if exception else "You do not have access to this area of the workspace.",
    )


def page_not_found(request, exception=None):
    return _error(request, 404, "Page not found", "The page you were looking for does not exist.")


def server_error(request):
    return _error(request, 500, "Something went wrong", "An unexpected error occurred. The team has been notified.")
