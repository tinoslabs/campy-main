"""Background training tasks."""
from __future__ import annotations

import logging

logger = logging.getLogger("campy.training")

try:
    from celery import shared_task
except ImportError:  # pragma: no cover
    def shared_task(*args, **kwargs):
        def decorator(func):
            func.delay = func
            return func

        return decorator(args[0]) if args and callable(args[0]) else decorator


@shared_task(name="apps.training.tasks.run_training_job")
def run_training_job_task(job_id: int) -> dict:
    from .models import TrainingJob
    from .services import run_training_job

    job = TrainingJob.objects.filter(pk=job_id).select_related("dataset", "organization").first()
    if job is None:
        return {"error": "job not found"}
    version = run_training_job(job)
    return {
        "job": str(job.uid),
        "status": job.status,
        "version": str(version.uid) if version else None,
        "metrics": job.metrics,
    }


@shared_task(name="apps.training.tasks.harvest_datasets")
def harvest_datasets() -> dict:
    from .models import Dataset
    from .services import harvest_false_positives

    total = 0
    for dataset in Dataset.objects.filter(harvest_from_events=True):
        total += harvest_false_positives(dataset)
    return {"harvested": total}
