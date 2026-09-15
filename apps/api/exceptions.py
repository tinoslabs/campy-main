"""Consistent API error envelopes."""
from __future__ import annotations

import logging

from django.core.exceptions import PermissionDenied, ValidationError as DjangoValidationError
from django.http import Http404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import exception_handler

logger = logging.getLogger("campy.api")


def campy_exception_handler(exc, context):
    """Every API error comes back in the same shape.

    Clients should never have to branch on whether a failure produced a DRF
    error, a Django 404 or an unhandled exception.
    """
    if isinstance(exc, Http404):
        return Response(
            {"error": {"code": "not_found", "message": "Resource not found."}},
            status=status.HTTP_404_NOT_FOUND,
        )
    if isinstance(exc, PermissionDenied):
        return Response(
            {"error": {"code": "forbidden", "message": str(exc) or "Permission denied."}},
            status=status.HTTP_403_FORBIDDEN,
        )
    if isinstance(exc, DjangoValidationError):
        return Response(
            {"error": {"code": "invalid", "message": "Validation failed.", "detail": exc.messages}},
            status=status.HTTP_400_BAD_REQUEST,
        )

    response = exception_handler(exc, context)
    if response is None:
        logger.exception("Unhandled API exception", exc_info=exc)
        return Response(
            {"error": {"code": "server_error", "message": "An unexpected error occurred."}},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    detail = response.data
    code = getattr(exc, "default_code", "error")
    message = "Request failed."
    if isinstance(detail, dict) and "detail" in detail:
        message = str(detail["detail"])
        detail = None
    elif isinstance(detail, list) and detail:
        message = str(detail[0])

    payload = {"error": {"code": code, "message": message}}
    if detail:
        payload["error"]["detail"] = detail
    response.data = payload
    return response
