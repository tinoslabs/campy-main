"""Decorators and mixins that enforce Campy AI's role-based access control."""
from __future__ import annotations

from functools import wraps

from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.shortcuts import redirect
from django.urls import reverse


def _deny(request, code: str):
    """Redirect anonymous users to sign-in; 403 everyone else."""
    if not request.user.is_authenticated:
        return redirect(f"{reverse('accounts:login')}?next={request.path}")
    raise PermissionDenied(f"You do not have the '{code}' permission in this workspace.")


def require_perm(*codes: str, require_all: bool = False):
    """View decorator: user must hold the given permission(s) in ``request.organization``."""

    def decorator(view):
        @wraps(view)
        def wrapped(request, *args, **kwargs):
            user = request.user
            if not user.is_authenticated:
                return _deny(request, codes[0])
            organization = getattr(request, "organization", None)
            checker = all if require_all else any
            if not checker(user.has_campy_perm(code, organization) for code in codes):
                return _deny(request, codes[0])
            return view(request, *args, **kwargs)

        return wrapped

    return decorator


def require_organization(view):
    """View decorator: a workspace must be selected."""

    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect(f"{reverse('accounts:login')}?next={request.path}")
        if getattr(request, "organization", None) is None:
            messages.info(request, "Create or join a workspace to continue.")
            return redirect("accounts:organization_create")
        return view(request, *args, **kwargs)

    return wrapped


def superadmin_required(view):
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect(f"{reverse('accounts:login')}?next={request.path}")
        if not request.user.is_superadmin:
            raise PermissionDenied("Super administrator access required.")
        return view(request, *args, **kwargs)

    return wrapped


def platform_team_required(view):
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect(f"{reverse('accounts:login')}?next={request.path}")
        if not request.user.is_platform_team:
            raise PermissionDenied("Campy AI platform team access required.")
        return view(request, *args, **kwargs)

    return wrapped


class PermissionRequiredMixin:
    """Class-based-view counterpart of :func:`require_perm`."""

    required_permissions: tuple[str, ...] = ()
    require_all_permissions = False

    def dispatch(self, request, *args, **kwargs):
        user = request.user
        if not user.is_authenticated:
            return redirect(f"{reverse('accounts:login')}?next={request.path}")
        if self.required_permissions:
            organization = getattr(request, "organization", None)
            checker = all if self.require_all_permissions else any
            if not checker(
                user.has_campy_perm(code, organization) for code in self.required_permissions
            ):
                raise PermissionDenied(
                    f"You do not have the '{self.required_permissions[0]}' permission."
                )
        return super().dispatch(request, *args, **kwargs)


class OrganizationScopedMixin:
    """Restrict a queryset to the active workspace (superadmins see everything)."""

    def get_queryset(self):
        queryset = super().get_queryset()
        if self.request.user.is_superadmin and self.request.GET.get("all") == "1":
            return queryset
        organization = getattr(self.request, "organization", None)
        if organization is None:
            return queryset.none()
        return queryset.filter(organization=organization)

    def form_valid(self, form):
        if hasattr(form.instance, "organization_id") and not form.instance.organization_id:
            form.instance.organization = self.request.organization
        return super().form_valid(form)
