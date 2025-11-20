from __future__ import annotations

import json
from typing import Any, Dict, Optional

import redis


def deep_job_key(job_id: str) -> str:
    return f"deep:job:{job_id}"


def write_job_state(redis_client: redis.Redis, job_id: str, payload: Dict[str, Any], ttl_seconds: int) -> None:
    redis_client.set(
        deep_job_key(job_id),
        json.dumps(payload),
        ex=ttl_seconds,
    )


def fetch_job_state(redis_client: redis.Redis, job_id: str) -> Optional[Dict[str, Any]]:
    data = redis_client.get(deep_job_key(job_id))
    if not data:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError("Corrupt job state") from exc


__all__ = ["deep_job_key", "write_job_state", "fetch_job_state"]
