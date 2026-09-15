"""Workspace context available to every template."""
from __future__ import annotations

from .models import Organization
from .permissions import MODULES


def workspace(request):
    user = getattr(request, "user", None)
    organization = getattr(request, "organization", None)
    membership = getattr(request, "membership", None)

    if user is None or not user.is_authenticated:
        return {
            "current_organization": None,
            "current_membership": None,
            "available_organizations": [],
            "user_permissions": set(),
            "PERMISSION_MODULES": MODULES,
        }

    return {
        "current_organization": organization,
        "current_membership": membership,
        "available_organizations": Organization.objects.for_user(user).distinct()[:50],
        "user_permissions": user.permission_codes(organization),
        "is_superadmin": user.is_superadmin,
        "is_platform_team": user.is_platform_team,
        "PERMISSION_MODULES": MODULES,
    }
