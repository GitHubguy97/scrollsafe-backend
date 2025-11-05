from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .utils import to_float, extract_platform_and_id

logger = logging.getLogger(__name__)


def _scan_priority_keys(redis_conn: redis.Redis, queue_name: str) -> List[str]:
    pattern = f"{queue_name}\x06\x16*"
    keys: List[str] = []
    cursor = 0
    while True:
        cursor, batch = redis_conn.scan(cursor=cursor, match=pattern, count=128)
        if batch:
            keys.extend(batch)
        if cursor == 0:
            break
    return keys


def queue_depth(redis_conn: Optional[redis.Redis], queue_name: str) -> Optional[int]:
    if not redis_conn:
        return None
    total = 0
    try:
        total += redis_conn.llen(queue_name)
    except redis.RedisError:
        pass
    try:
        priority_keys = _scan_priority_keys(redis_conn, queue_name)
        for key in priority_keys:
            try:
                total += redis_conn.llen(key)
            except redis.RedisError:
                continue
    except redis.RedisError as exc:
        logger.warning("Redis error while fetching queue depth for %s: %s", queue_name, exc)
        return None
    return total


def build_admin_metrics(
    db_pool: ConnectionPool,
    redis_client: Optional[redis.Redis],
    queue_redis: Optional[redis.Redis],
    queue_names: List[str],
    verdict_window_hours: int,
    recent_limit: int,
    admin_limit: int,
) -> Dict[str, Any]:
    queues: Dict[str, Optional[int]] = {name: queue_depth(queue_redis, name) for name in queue_names}

    try:
        with db_pool.connection() as conn:
            conn: Any

            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT label, COUNT(*) AS count
                    FROM analyses
                    WHERE analyzed_at >= NOW() - INTERVAL '{verdict_window_hours} hours'
                    GROUP BY label
                    """,
                )
                counts_rows = cur.fetchall()

            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        a.platform,
                        a.video_id,
                        a.label,
                        a.confidence,
                        a.reason,
                        a.analyzed_at,
                        a.frames_count,
                        a.batch_time_ms,
                        a.source_url,
                        v.views_per_hour,
                        v.region,
                        v.title,
                        v.channel
                    FROM analyses AS a
                    LEFT JOIN videos AS v
                      ON a.platform = v.platform AND a.video_id = v.video_id
                    ORDER BY a.analyzed_at DESC
                    LIMIT %s
                    """,
                    (recent_limit,),
                )
                recent_rows = cur.fetchall()

            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT platform, video_id, label, notes, source_url, created_at
                    FROM admin_labels
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (admin_limit,),
                )
                admin_rows = cur.fetchall()
    except Exception as exc:
        logger.exception("Failed to load admin metrics")
        raise

    verdict_counts: Dict[str, int] = {}
    for row in counts_rows:
        label = (row.get("label") or "unknown").lower()
        verdict_counts[label] = int(row.get("count") or 0)
    for default_label in ("real", "artificial", "suspicious", "unknown"):
        verdict_counts.setdefault(default_label, 0)

    recent_analyses = [
        {
            "platform": row["platform"],
            "video_id": row["video_id"],
            "label": row["label"],
            "confidence": to_float(row.get("confidence")),
            "reason": row.get("reason"),
            "analyzed_at": row["analyzed_at"],
            "frames_count": row.get("frames_count"),
            "batch_time_ms": row.get("batch_time_ms"),
            "views_per_hour": to_float(row.get("views_per_hour")),
            "region": row.get("region"),
            "title": row.get("title"),
            "channel": row.get("channel"),
            "source_url": row.get("source_url"),
        }
        for row in recent_rows
    ]

    admin_overrides = [
        {
            "platform": row["platform"],
            "video_id": row["video_id"],
            "label": row["label"],
            "notes": row.get("notes"),
            "source_url": row.get("source_url"),
            "created_at": row["created_at"],
        }
        for row in admin_rows
    ]

    return {
        "queues": queues,
        "verdict_counts": verdict_counts,
        "recent_analyses": recent_analyses,
        "admin_overrides": admin_overrides,
    }


def upsert_admin_label(
    db_pool: ConnectionPool,
    redis_client: Optional[redis.Redis],
    url: str,
    label: str,
    notes: Optional[str],
    cache_ttl_seconds: int,
) -> Dict[str, Any]:
    platform, video_id = extract_platform_and_id(url)
    cleaned_notes = notes.strip() if isinstance(notes, str) and notes.strip() else None

    try:
        with db_pool.connection() as conn:
            conn: Any
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO admin_labels (platform, video_id, label, notes, source_url)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (platform, video_id)
                    DO UPDATE SET
                        label = EXCLUDED.label,
                        notes = COALESCE(EXCLUDED.notes, admin_labels.notes),
                        source_url = COALESCE(EXCLUDED.source_url, admin_labels.source_url),
                        created_at = NOW()
                    RETURNING platform, video_id, label, notes, source_url, created_at
                    """,
                    (platform, video_id, label.strip(), cleaned_notes, url.strip()),
                )
                row = cur.fetchone()
                conn.commit()
    except Exception:
        logger.exception("Failed to upsert admin label for %s:%s", platform, video_id)
        raise

    cached = False
    if redis_client:
        cache_payload = {
            "platform": row["platform"],
            "video_id": row["video_id"],
            "label": row["label"],
            "confidence": 1.0,
            "reason": row.get("notes") or "admin_override",
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "model_version": "admin_manual",
        }
        try:
            redis_client.set(
                f"video:{row['platform']}:{row['video_id']}",
                json.dumps(cache_payload),
                ex=cache_ttl_seconds,
            )
            cached = True
        except Exception as exc:
            logger.warning("Failed to cache admin override for %s:%s - %s", platform, video_id, exc)

    row_dict = dict(row)
    row_dict["cached"] = cached
    return row_dict


__all__ = ["queue_depth", "build_admin_metrics", "upsert_admin_label"]
