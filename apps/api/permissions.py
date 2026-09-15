"""DRF permission classes backed by Campy AI's permission registry."""
from __future__ import annotations

from rest_framework import permissions


class HasCampyPermission(permissions.BasePermission):
    """Checks ``required_permission`` (or a per-method map) on the view.

    Works identically for a signed-in user and an API key, because both
    implement ``has_campy_perm``.
    """

    message = "You do not have permission to perform this action."

    METHOD_ACTIONS = {
        "GET": "view", "HEAD": "view", "OPTIONS": "view",
        "POST": "manage", "PUT": "manage", "PATCH": "manage", "DELETE": "manage",
    }

    def has_permission(self, request, view):
        user = request.user
        if user is None or not user.is_authenticated:
            return False
        organization = getattr(request, "organization", None)
        if organization is None and not getattr(user, "is_superadmin", False):
            self.message = (
                "No workspace selected. Send an X-Campy-Organization header, "
                "or use a workspace-scoped API key."
            )
            return False

        code = self.resolve_code(request, view)
        if code is None:
            return True

        if not user.has_campy_perm(code, organization):
            self.message = f"This action requires the '{code}' permission."
            return False
        return True

    def resolve_code(self, request, view) -> str | None:
        """Work out which permission this request needs.

        Resolution order:

        1. ``permission_map`` on the view — per-action codes, the precise option.
        2. ``required_permission`` — one code for the whole view.
        3. ``permission_module`` + the HTTP verb.

        The derived ``<module>.<action>`` form is validated against the
        permission registry: several modules genuinely have no ``manage``
        permission (``events`` has view/acknowledge/resolve/export/delete), and
        inventing one silently rejects every write to them.
        """
        from apps.accounts.permissions import PERMISSION_MAP

        action = getattr(view, "action", None)
        mapping = getattr(view, "permission_map", None) or {}
        if action and action in mapping:
            return mapping[action]

        explicit = getattr(view, "required_permission", None)
        if explicit:
            return explicit

        module = getattr(view, "permission_module", None)
        if not module:
            return None

        derived = f"{module}.{self.METHOD_ACTIONS.get(request.method, 'manage')}"
        if derived in PERMISSION_MAP:
            return derived
        # No such permission exists for this module — fall back to its read
        # permission rather than demanding something nobody can ever hold.
        fallback = f"{module}.view"
        return fallback if fallback in PERMISSION_MAP else None

    def has_object_permission(self, request, view, obj):
        """Objects must belong to the caller's workspace."""
        organization = getattr(request, "organization", None)
        if getattr(request.user, "is_superadmin", False):
            return True
        owner = getattr(obj, "organization", None)
        if owner is None:
            # Nested objects (zones) reach their organisation through a parent.
            camera = getattr(obj, "camera", None)
            owner = getattr(camera, "organization", None)
        return owner is not None and organization is not None and owner.pk == organization.pk


class ReadOnly(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.method in permissions.SAFE_METHODS
