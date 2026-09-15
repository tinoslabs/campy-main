"""Brand + build information injected into every template."""
from __future__ import annotations

from django.conf import settings


def branding(request):
    campy = settings.CAMPY
    return {
        "BRAND": {
            "name": campy["BRAND_NAME"],
            "tagline": campy["BRAND_TAGLINE"],
            "support_email": campy["SUPPORT_EMAIL"],
            "logo": campy["LOGO_PATH"],
            "logo_mark": campy["LOGO_MARK_PATH"],
        },
        "CAMPY_VERSION": "1.0.0",
        "DEFAULT_CURRENCY": campy["DEFAULT_CURRENCY"],
        "RAZORPAY_KEY_ID": settings.RAZORPAY["KEY_ID"],
        "STRIPE_PUBLISHABLE_KEY": settings.STRIPE["PUBLISHABLE_KEY"],
    }
