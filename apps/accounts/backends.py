"""Authentication backends."""
from __future__ import annotations

from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend
from django.db.models import Q


class EmailOrUsernameBackend(ModelBackend):
    """Let people sign in with either their email address or their username."""

    def authenticate(self, request, username=None, password=None, **kwargs):
        User = get_user_model()
        identifier = (username or kwargs.get("email") or "").strip()
        if not identifier or not password:
            return None
        user = User.objects.filter(
            Q(email__iexact=identifier) | Q(username__iexact=identifier)
        ).first()
        if user is None:
            # Equalise timing against the "user exists" branch.
            User().set_password(password)
            return None
        if user.is_locked:
            return None
        if user.check_password(password) and self.user_can_authenticate(user):
            return user
        user.register_failed_login()
        return None
