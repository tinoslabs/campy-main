"""OpenAPI schema extensions.

drf-spectacular cannot introspect a custom authentication class or a plain
function view, so we describe both explicitly. Without this the published
schema silently omits how to authenticate — the single most important thing
an API consumer needs.
"""
from __future__ import annotations

from drf_spectacular.extensions import OpenApiAuthenticationExtension


class APIKeyAuthenticationScheme(OpenApiAuthenticationExtension):
    target_class = "apps.api.authentication.OrganizationAPIKeyAuthentication"
    name = "ApiKeyAuth"

    def get_security_definition(self, auto_schema):
        return {
            "type": "apiKey",
            "in": "header",
            "name": "Authorization",
            "description": (
                "Workspace-scoped machine credential. Send as "
                "`Authorization: Api-Key <prefix>.<secret>`. The key carries only "
                "the permissions granted when it was issued."
            ),
        }
