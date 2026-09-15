"""Subscription enforcement."""
from __future__ import annotations

from django.contrib import messages
from django.shortcuts import redirect
from django.urls import reverse


class SubscriptionGuardMiddleware:
    """Blocks the dashboard when a workspace has no valid subscription.

    Deliberately permissive: a *past-due* subscription keeps working through a
    grace period rather than cutting a security system off the moment a card
    expires. Only genuinely cancelled or expired workspaces are stopped, and
    even then billing, settings and sign-out stay reachable so the customer can
    fix it themselves.
    """

    #: URL namespaces that remain reachable without an active subscription.
    ALWAYS_ALLOWED = {"billing", "accounts", "admin", "cms", "api"}
    ALLOWED_PATH_PREFIXES = ("/static/", "/media/", "/webhooks/", "/healthz")

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.subscription = None
        user = getattr(request, "user", None)
        organization = getattr(request, "organization", None)

        if user is None or not user.is_authenticated or organization is None:
            return self.get_response(request)

        subscription = organization.active_subscription
        request.subscription = subscription

        if user.is_superadmin or self._is_allowed(request):
            return self.get_response(request)

        if subscription is None:
            messages.warning(request, "Choose a package to activate this workspace.")
            return redirect(reverse("billing:packages"))

        if not subscription.is_entitled and not subscription.in_grace_period:
            messages.error(
                request,
                "This workspace's subscription has ended. Renew to restore monitoring.",
            )
            return redirect(reverse("billing:packages"))

        return self.get_response(request)

    def _is_allowed(self, request) -> bool:
        path = request.path
        if any(path.startswith(prefix) for prefix in self.ALLOWED_PATH_PREFIXES):
            return True
        match = getattr(request, "resolver_match", None)
        namespace = getattr(match, "namespace", "") if match else ""
        if namespace in self.ALWAYS_ALLOWED:
            return True
        # resolver_match is not populated yet in process_request order, so fall
        # back to a path check for the namespaces we care about.
        return any(
            path.startswith(prefix)
            for prefix in ("/billing/", "/accounts/", "/admin/", "/api/")
        ) or path == "/" or not path.startswith("/app/")


def quota_usage(organization) -> dict:
    """Current consumption against every quota on the plan."""
    from apps.accounts.models import Membership
    from apps.cameras.models import Camera, Employee, Site

    subscription = organization.active_subscription
    if subscription is None:
        return {}

    usage = {
        "cameras": Camera.objects.filter(organization=organization).count(),
        "sites": Site.objects.filter(organization=organization).count(),
        "users": Membership.objects.filter(organization=organization, status="active").count(),
        "employees": Employee.objects.filter(organization=organization).count(),
    }

    rows = {}
    for key, used in usage.items():
        limit = subscription.quota(key, 0)
        rows[key] = {
            "used": used,
            "limit": limit,
            "unlimited": limit < 0,
            "percent": 0 if limit <= 0 else min(round(used / limit * 100), 100),
            "exceeded": limit >= 0 and used > limit,
            "near_limit": limit > 0 and used >= limit * 0.8,
        }
    return rows


def can_add(organization, quota_key: str, amount: int = 1) -> tuple[bool, str]:
    """Guard used before creating a camera, site, member or employee."""
    subscription = organization.active_subscription
    if subscription is None:
        return False, "This workspace has no active package."
    limit = subscription.quota(quota_key, 0)
    if limit < 0:
        return True, ""
    rows = quota_usage(organization)
    used = rows.get(quota_key, {}).get("used", 0)
    if used + amount > limit:
        label = quota_key.replace("_", " ")
        return False, (
            f"Your {subscription.package.name} package includes {limit} {label}. "
            "Upgrade to add more."
        )
    return True, ""
