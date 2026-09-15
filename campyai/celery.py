"""Celery application for background inference, training and billing jobs.

Celery is optional: with ``CELERY_TASK_ALWAYS_EAGER=True`` (the default in
development) every task runs inline, so the platform works without a broker.
"""
from __future__ import annotations

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "campyai.settings.dev")

try:
    from celery import Celery
    from celery.schedules import crontab
except ImportError:  # pragma: no cover - celery is an optional dependency
    app = None
else:
    app = Celery("campyai")
    app.config_from_object("django.conf:settings", namespace="CELERY")
    app.autodiscover_tasks()

    app.conf.beat_schedule = {
        "camera-health-sweep": {
            "task": "apps.cameras.tasks.sweep_camera_health",
            "schedule": crontab(minute="*/5"),
        },
        "rollup-analytics": {
            "task": "apps.analytics.tasks.rollup_daily_metrics",
            "schedule": crontab(minute=10, hour="*"),
        },
        "expire-subscriptions": {
            "task": "apps.billing.tasks.expire_due_subscriptions",
            "schedule": crontab(minute=0, hour=1),
        },
        "purge-old-evidence": {
            "task": "apps.events.tasks.purge_expired_evidence",
            "schedule": crontab(minute=30, hour=2),
        },
    }
