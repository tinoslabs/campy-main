"""Request-scoped tenancy and audit context."""
from __future__ import annotations

from django.contrib.auth import get_user_model
from django.utils import timezone

from .models import Membership, Organization


class CurrentOrganizationMiddleware:
    """Attach ``request.organization`` and ``request.membership`` to every request.

    Resolution order:

    1. ``?org=<slug>`` query parameter (used by the workspace switcher)
    2. ``X-Campy-Organization`` header (API clients)
    3. the user's remembered ``active_organization``
    4. their first workspace
    """

    HEADER = "HTTP_X_CAMPY_ORGANIZATION"

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.organization = None
        request.membership = None
        user = getattr(request, "user", None)

        if user is not None and user.is_authenticated:
            requested = None
            slug = request.GET.get("org") or request.META.get(self.HEADER)
            if slug:
                requested = Organization.objects.filter(slug=slug).first()
            organization = user.resolve_organization(requested)
            request.organization = organization
            if organization is not None:
                request.membership = user.membership_for(organization)
                if requested is not None and user.active_organization_id != organization.pk:
                    user.active_organization = organization
                    user.save(update_fields=["active_organization"])
            self._touch(user, request.membership)

        return self.get_response(request)

    @staticmethod
    def _touch(user, membership) -> None:
        """Throttled last-seen bookkeeping (at most once every 5 minutes).

        ``request.user`` is a ``SimpleLazyObject``. It forwards attribute access
        to the real user, and even reports the wrapped class through
        ``__class__`` — but ``type()`` reads the object's actual type slot and
        so returns the wrapper, which has no manager. Name the models outright.
        """
        now = timezone.now()
        if user.last_seen_at and (now - user.last_seen_at).total_seconds() <= 300:
            return

        get_user_model()._default_manager.filter(pk=user.pk).update(last_seen_at=now)
        user.last_seen_at = now
        if membership is not None:
            Membership.objects.filter(pk=membership.pk).update(last_active_at=now)


class AuditContextMiddleware:
    """Stash the request on a thread-local so model signals can log the actor."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        from .signals import set_audit_context, clear_audit_context

        set_audit_context(request)
        try:
            return self.get_response(request)
        finally:
            clear_audit_context()
