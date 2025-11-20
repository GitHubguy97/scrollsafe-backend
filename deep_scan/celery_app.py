from __future__ import annotations

from celery import Celery

from .config import settings


celery_app = Celery(
    "scrollsafe_deep_scan",
    broker=settings.celery_broker_url,
)

celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_queue=settings.queue_name,
    task_default_priority=5,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    include=["deep_scan.tasks"],
    timezone="UTC",
)
