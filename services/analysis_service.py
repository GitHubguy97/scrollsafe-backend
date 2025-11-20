from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import redis
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


def format_analysis_result(payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    label = payload.get("label", "unknown")
    confidence_raw = payload.get("confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    reason = payload.get("reason") or "model_vote"
    return {
        "result": label,
        "confidence": confidence,
        "reason": reason,
        "source": source,
    }


def get_cache_hit(redis_client: Optional[redis.Redis], platform: str, video_id: str) -> Optional[Dict[str, Any]]:
    if not redis_client:
        return None
    key = f"video:{platform}:{video_id}"
    try:
        cached = redis_client.get(key)
    except Exception as exc:
        logger.warning("Redis error while fetching %s: %s", key, exc)
        return None

    if not cached:
        return None

    try:
        payload = json.loads(cached)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in cache for key %s", key)
        return None

    return format_analysis_result(payload, source="Doomscroller Cache")


def get_db_hit(db_pool: Optional[ConnectionPool], platform: str, video_id: str) -> Optional[Dict[str, Any]]:
    if not db_pool:
        return None

    query = """
        WITH merged AS (
            SELECT
                'admin' AS source,
                label,
                1.0 AS confidence,
                COALESCE(notes, 'admin_override') AS reason,
                NOW() AS analyzed_at,
                NULL::jsonb AS features,
                source_url
            FROM admin_labels
            WHERE platform = %s AND video_id = %s

            UNION ALL

            SELECT
                'doomscroller' AS source,
                label,
                confidence,
                reason,
                analyzed_at,
                features,
                source_url
            FROM analyses
            WHERE platform = %s AND video_id = %s
        )
        SELECT
            source,
            label,
            confidence,
            reason,
            analyzed_at,
            features,
            source_url
        FROM merged
        ORDER BY (source = 'admin') DESC, analyzed_at DESC
        LIMIT 1;
    """

    try:
        with db_pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (platform, video_id, platform, video_id))
                row = cur.fetchone()
    except Exception as exc:
        logger.warning("Postgres error while fetching %s:%s - %s", platform, video_id, exc)
        return None

    if not row:
        return None

    source = "Admin Override" if row["source"] == "admin" else "Doomscroller DB"
    return format_analysis_result(row, source=source)


__all__ = ["get_cache_hit", "get_db_hit", "format_analysis_result"]

