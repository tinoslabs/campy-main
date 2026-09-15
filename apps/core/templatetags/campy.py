"""Template helpers registered as builtins — usable without ``{% load %}``."""
from __future__ import annotations

import json

from django import template
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.utils.timesince import timesince

from apps.core.utils import humanize_duration, percent_change

register = template.Library()

SEVERITY_STYLES = {
    "critical": ("Critical", "sev-critical"),
    "high": ("High", "sev-high"),
    "medium": ("Medium", "sev-medium"),
    "low": ("Low", "sev-low"),
    "info": ("Info", "sev-info"),
}

STATUS_STYLES = {
    "active": "pill-ok",
    "online": "pill-ok",
    "running": "pill-ok",
    "succeeded": "pill-ok",
    "completed": "pill-ok",
    "paid": "pill-ok",
    "resolved": "pill-ok",
    "published": "pill-ok",
    "trialing": "pill-info",
    "queued": "pill-info",
    "pending": "pill-warn",
    "acknowledged": "pill-warn",
    "degraded": "pill-warn",
    "past_due": "pill-warn",
    "draft": "pill-muted",
    "offline": "pill-bad",
    "failed": "pill-bad",
    "cancelled": "pill-bad",
    "canceled": "pill-bad",
    "expired": "pill-bad",
    "open": "pill-bad",
}


@register.filter(name="severity_badge")
def severity_badge(value):
    label, css = SEVERITY_STYLES.get(str(value).lower(), (str(value).title(), "sev-info"))
    return format_html('<span class="badge {}">{}</span>', css, label)


@register.filter(name="status_pill")
def status_pill(value):
    key = str(value).lower().replace(" ", "_")
    css = STATUS_STYLES.get(key, "pill-muted")
    label = str(value).replace("_", " ").title()
    return format_html('<span class="pill {}">{}</span>', css, label)


@register.filter(name="pct")
def pct(value, digits=1):
    try:
        return f"{float(value) * 100:.{int(digits)}f}%"
    except (TypeError, ValueError):
        return "—"


@register.filter(name="pct_raw")
def pct_raw(value, digits=1):
    try:
        return f"{float(value):.{int(digits)}f}%"
    except (TypeError, ValueError):
        return "—"


@register.filter(name="delta")
def delta(current, previous):
    return percent_change(float(current or 0), float(previous or 0))


@register.filter(name="duration")
def duration(value):
    return humanize_duration(value)


@register.filter(name="ago")
def ago(value):
    if not value:
        return "never"
    return f"{timesince(value).split(',')[0]} ago"


@register.filter(name="json_script_safe")
def json_script_safe(value):
    return mark_safe(json.dumps(value))


@register.filter(name="get_item")
def get_item(mapping, key):
    try:
        return mapping.get(key)
    except AttributeError:
        return None


@register.filter(name="subtract")
def subtract(value, arg):
    try:
        return float(value) - float(arg)
    except (TypeError, ValueError):
        return 0


@register.filter(name="mul")
def mul(value, arg):
    try:
        return float(value) * float(arg)
    except (TypeError, ValueError):
        return 0


@register.filter(name="divide")
def divide(value, arg):
    try:
        arg = float(arg)
        if arg == 0:
            return 0
        return float(value) / arg
    except (TypeError, ValueError):
        return 0


@register.simple_tag
def progress_width(value, total):
    """Percentage width (0-100) for meter bars, clamped."""
    try:
        total = float(total)
        if total <= 0:
            return 0
        return max(0, min(100, round(float(value) / total * 100, 2)))
    except (TypeError, ValueError):
        return 0


@register.simple_tag(takes_context=True)
def querystring(context, **kwargs):
    """Rebuild the current querystring with overrides — used for filters/paging."""
    request = context.get("request")
    params = request.GET.copy() if request else {}
    for key, value in kwargs.items():
        if value in (None, ""):
            params.pop(key, None)
        else:
            params[key] = value
    encoded = params.urlencode() if hasattr(params, "urlencode") else ""
    return f"?{encoded}" if encoded else ""
