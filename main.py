from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import time
import json
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from video_utils import get_video_info
from heuristics import check_heuristics
import asyncio
from concurrent.futures import ThreadPoolExecutor
from huggingface_client import analyze_with_huggingface
import redis
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

DEFAULT_PLATFORM = os.getenv("DEFAULT_PLATFORM", "youtube")
REDIS_URL = os.getenv("REDIS_APP_URL")
DATABASE_URL = os.getenv("DATABASE_URL")

redis_client: Optional[redis.Redis] = None
if REDIS_URL:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

db_pool: Optional[ConnectionPool] = None
if DATABASE_URL:
    db_pool = ConnectionPool(
        conninfo=DATABASE_URL,
        min_size=int(os.getenv("DB_POOL_MIN_SIZE", "1")),
        max_size=int(os.getenv("DB_POOL_MAX_SIZE", "10")),
        timeout=float(os.getenv("DB_POOL_TIMEOUT", "30")),
    )

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

app = FastAPI(title="ScrollSafe Backend", version="1.0.0")

# Create thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)  # Handle 4 concurrent requests

def _format_analysis_result(data: Dict[str, Any], source: str) -> AnalysisResult:
    label = data.get("label", "unknown")
    confidence_raw = data.get("confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    reason = data.get("reason") or "model_vote"
    return AnalysisResult(
        result=label,
        confidence=confidence,
        reason=reason,
        source=source,
    )


def _get_cache_hit(platform: str, video_id: str) -> Optional[AnalysisResult]:
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

    return _format_analysis_result(payload, source="Doomscroller Cache")


def _get_db_hit(platform: str, video_id: str) -> Optional[AnalysisResult]:
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
                NULL::text AS source_url
            FROM admin_labels
            WHERE platform = %s AND video_id = %s

            UNION ALL

            SELECT
                'analysis' AS source,
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
            conn: psycopg.Connection
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (platform, video_id, platform, video_id))
                row = cur.fetchone()
    except Exception as exc:
        logger.warning("Postgres error while fetching %s:%s - %s", platform, video_id, exc)
        return None

    if not row:
        return None

    source = "Admin Override" if row["source"] == "admin" else "Doomscroller DB"
    return _format_analysis_result(row, source=source)

# CORS Configuration - Secure settings for Chrome Extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Chrome extensions use chrome-extension://<id> origins
                          # TODO: Replace with specific extension ID once published:
                          # allow_origins=["chrome-extension://your-extension-id-here"]
    allow_credentials=False,  # We don't use cookies or auth headers
    allow_methods=["GET"],    # Only GET requests are used
    allow_headers=["Accept", "Content-Type"],  # Minimal headers needed
    max_age=3600,  # Cache preflight requests for 1 hour
)


class AnalysisResult(BaseModel):
  result: str
  confidence: float
  reason: str
  source: str

# Routes
@app.get("/")
async def root():
    return {"message": "ScrollSafe Backend API"}

@app.get("/api/ds-cache/{video_id}")
async def check_doom_scroller_cache(video_id: str, platform: Optional[str] = None):
    """Check Doomscroller cache and database for a verdict."""
    platform = (platform or DEFAULT_PLATFORM).lower()

    cache_hit = _get_cache_hit(platform, video_id)
    if cache_hit:
        logger.info("DS cache hit for %s:%s", platform, video_id)
        return cache_hit

    db_hit = _get_db_hit(platform, video_id)
    if db_hit:
        logger.info("DS DB hit for %s:%s", platform, video_id)
        return db_hit

    logger.info("DS miss for %s:%s", platform, video_id)
    raise HTTPException(status_code=404, detail="Not in cache")

@app.get("/api/analyze/{video_id}")
async def analyze_video(video_id: str, request: Request):
    """Quick analysis: C2PA + metadata heuristics"""
    
    request_start = time.time()
    
    if await request.is_disconnected():
        print(f"⏭️ Client disconnected for {video_id}, aborting")
        raise HTTPException(status_code=499, detail="Client disconnected")
    
    # Get video metadata
    loop = asyncio.get_event_loop()
    video_info = await loop.run_in_executor(executor, get_video_info, video_id)

     # Check again after slow operation
    if await request.is_disconnected():
        print(f"⏭️ Client disconnected during processing for {video_id}")
        raise HTTPException(status_code=499, detail="Client disconnected")
    
    if not video_info:
        return AnalysisResult(
            result="unknown",
            confidence=0.0,
            reason="Could not fetch video information",
            source="Backend API"
        )
    
    # Run heuristics check
    heuristics_result = check_heuristics(video_info)
    
    # Calculate total time
    total_duration = time.time() - request_start
    
    # Log the complete request
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "video_id": video_id,
        "total_duration": round(total_duration, 3),
        "result": heuristics_result["result"],
        "confidence": heuristics_result["confidence"],
        "reason": heuristics_result["reason"]
    }
    
    with open("api_requests.log", 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    print(f"✅ Total request: {total_duration:.3f}s - Result: {heuristics_result['result']}")
    
    return AnalysisResult(
        result=heuristics_result["result"],
        confidence=heuristics_result["confidence"],
        reason=heuristics_result["reason"],
        source="Backend API"
    )

@app.get("/api/deep-scan/{video_id}")
async def deep_scan_video(video_id: str, request: Request):
    request_start = time.time()
    
    print(f"Starting deep scan for: {video_id}")
    
    if await request.is_disconnected():
        return
    
    # Run Hugging Face analysis
    result = await analyze_with_huggingface(video_id)
    
    total_duration = time.time() - request_start
    print(f"Deep scan complete: {total_duration:.3f}s - Result: {result['result']}")
    
    return AnalysisResult(
        result=result["result"],
        confidence=result["confidence"],
        reason=result["reason"],
        source="Hybrid AI + Heuristics"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        if db_pool:
            db_pool.close()
