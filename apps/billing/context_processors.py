"""Subscription state for templates."""
from __future__ import annotations

from django.conf import settings


def subscription(request):
    organization = getattr(request, "organization", None)
    current = getattr(request, "subscription", None)
    if current is None and organization is not None:
        current = organization.active_subscription

    package = current.package if current else None
    return {
        "current_subscription": current,
        "current_package": package,
        "plan_features": (package.features if package else {}) or {},
        "is_trialing": bool(current and current.is_trialing),
        "trial_days_left": current.days_remaining if current and current.is_trialing else 0,
        "subscription_warning": bool(current and (current.is_expiring_soon or current.status == "past_due")),
        "PAYMENTS_ENABLED": bool(settings.RAZORPAY["KEY_ID"] or settings.STRIPE["SECRET_KEY"]),
        "RAZORPAY_ENABLED": bool(settings.RAZORPAY["KEY_ID"]),
        "STRIPE_ENABLED": bool(settings.STRIPE["SECRET_KEY"]),
    }
