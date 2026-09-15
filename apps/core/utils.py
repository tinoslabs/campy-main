"""Small helpers used across the platform."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import string
from datetime import timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable

from django.utils import timezone

ALPHABET = string.ascii_letters + string.digits


def random_token(length: int = 40) -> str:
    """URL-safe random token (used for invites, API keys, webhooks)."""
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def hash_secret(raw: str) -> str:
    """One-way hash for API keys — we never store the plaintext."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def constant_time_equals(left: str, right: str) -> bool:
    return hmac.compare_digest(str(left), str(right))


def money(value: Any, places: str = "0.01") -> Decimal:
    return Decimal(str(value or 0)).quantize(Decimal(places), rounding=ROUND_HALF_UP)


def to_minor_units(amount: Decimal | float | int, currency: str = "INR") -> int:
    """Convert a decimal amount to the gateway's minor unit (paise / cents)."""
    zero_decimal = {"JPY", "KRW", "VND", "CLP", "ISK"}
    if str(currency).upper() in zero_decimal:
        return int(money(amount, "1"))
    return int((money(amount) * 100).to_integral_value(rounding=ROUND_HALF_UP))


def from_minor_units(amount: int, currency: str = "INR") -> Decimal:
    zero_decimal = {"JPY", "KRW", "VND", "CLP", "ISK"}
    if str(currency).upper() in zero_decimal:
        return money(amount, "1")
    return money(Decimal(amount) / 100)


def percent_change(current: float, previous: float) -> float:
    """Signed percentage change, guarding against a zero baseline."""
    if not previous:
        return 100.0 if current else 0.0
    return round(((current - previous) / abs(previous)) * 100.0, 1)


def humanize_duration(seconds: float | int | None) -> str:
    if not seconds:
        return "0s"
    seconds = int(seconds)
    parts: list[str] = []
    for label, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        if seconds >= size:
            value, seconds = divmod(seconds, size)
            parts.append(f"{value}{label}")
        if len(parts) == 2:
            break
    return " ".join(parts) or "0s"


def daterange(days: int, end=None) -> tuple:
    """Inclusive ``(start, end)`` window ending *now* by default."""
    end = end or timezone.now()
    return end - timedelta(days=days), end


def chunked(items: Iterable, size: int) -> Iterable[list]:
    bucket: list = []
    for item in items:
        bucket.append(item)
        if len(bucket) >= size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def client_ip(request) -> str:
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "") or ""


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
