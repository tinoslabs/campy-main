"""API key authentication for edge devices and server integrations."""
from __future__ import annotations

from django.utils import timezone
from rest_framework import authentication, exceptions

from apps.accounts.models import APIKey
from apps.core.utils import client_ip, hash_secret


class APIKeyUser:
    """A non-human principal. Quacks enough like a user for DRF permissions."""

    is_authenticated = True
    is_anonymous = False
    is_active = True
    is_superadmin = False
    is_platform_team = False
    is_staff = False

    def __init__(self, api_key: APIKey):
        self.api_key = api_key
        self.organization = api_key.organization
        self.pk = None
        self.id = None

    def __str__(self) -> str:
        return f"api-key:{self.api_key.prefix}"

    def has_campy_perm(self, code: str, organization=None, **kwargs) -> bool:
        if organization is not None and organization.pk != self.organization.pk:
            return False
        return self.api_key.grants(code)

    def has_any_campy_perm(self, codes, organization=None) -> bool:
        return any(self.has_campy_perm(code, organization) for code in codes)

    def permission_codes(self, organization=None) -> set[str]:
        from apps.accounts import permissions as perms

        return perms.expand(self.api_key.scopes)

    def membership_for(self, organization):
        return None


class OrganizationAPIKeyAuthentication(authentication.BaseAuthentication):
    """``Authorization: Api-Key cai_xxxx.<secret>``"""

    keyword = "Api-Key"

    def authenticate(self, request):
        header = authentication.get_authorization_header(request).decode("latin-1")
        if not header or not header.lower().startswith(self.keyword.lower()):
            return None

        parts = header.split()
        if len(parts) != 2:
            raise exceptions.AuthenticationFailed("Malformed Api-Key header.")

        plaintext = parts[1]
        prefix = plaintext.split(".", 1)[0]
        api_key = APIKey.objects.filter(prefix=prefix).select_related("organization").first()
        if api_key is None or api_key.key_hash != hash_secret(plaintext):
            raise exceptions.AuthenticationFailed("Invalid API key.")
        if not api_key.is_active:
            raise exceptions.AuthenticationFailed("This API key has been revoked or has expired.")
        if not api_key.organization.is_operational:
            raise exceptions.AuthenticationFailed("This workspace is not active.")

        # Cheap, non-blocking usage bookkeeping.
        APIKey.objects.filter(pk=api_key.pk).update(
            last_used_at=timezone.now(),
            last_used_ip=client_ip(request) or None,
            request_count=api_key.request_count + 1,
        )

        principal = APIKeyUser(api_key)
        request.organization = api_key.organization
        return (principal, api_key)

    def authenticate_header(self, request):
        return self.keyword
