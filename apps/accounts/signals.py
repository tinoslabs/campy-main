"""Audit hooks and workspace bootstrapping."""
from __future__ import annotations

import threading

from django.contrib.auth.signals import user_logged_in, user_logged_out, user_login_failed
from django.db.models.signals import post_save
from django.dispatch import receiver

from apps.core.utils import client_ip

from .models import AuditLog, Organization, User

_local = threading.local()


def set_audit_context(request) -> None:
    _local.request = request


def clear_audit_context() -> None:
    _local.request = None


def current_request():
    return getattr(_local, "request", None)


@receiver(post_save, sender=Organization)
def bootstrap_organization(sender, instance: Organization, created: bool, **kwargs):
    """Every new workspace gets the built-in role set immediately."""
    if created:
        instance.bootstrap_default_roles()


@receiver(user_logged_in)
def on_login(sender, request, user: User, **kwargs):
    user.register_successful_login(client_ip(request) if request else None)
    AuditLog.record(
        action=AuditLog.Action.LOGIN,
        actor=user,
        organization=getattr(request, "organization", None),
        summary=f"{user.email} signed in",
        request=request,
    )


@receiver(user_logged_out)
def on_logout(sender, request, user, **kwargs):
    if user is not None:
        AuditLog.record(
            action=AuditLog.Action.LOGOUT,
            actor=user,
            summary=f"{user.email} signed out",
            request=request,
        )


@receiver(user_login_failed)
def on_login_failed(sender, credentials, request=None, **kwargs):
    identifier = credentials.get("username") or credentials.get("email") or "unknown"
    AuditLog.record(
        action=AuditLog.Action.LOGIN_FAILED,
        summary=f"Failed sign-in attempt for {identifier}",
        request=request,
        sensitive=True,
    )
