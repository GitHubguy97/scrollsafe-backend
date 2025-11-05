from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import time
import os
import logging
from datetime import datetime, timezone
from uuid import uuid4
from dotenv import load_dotenv
from video_utils import get_video_info
from heuristics import check_heuristics
import asyncio
from concurrent.futures import ThreadPoolExecutor
import redis
from psycopg_pool import ConnectionPool
from contextlib import asynccontextmanager

from services.analysis_service import get_cache_hit, get_db_hit
from services.admin_service import build_admin_metrics, upsert_admin_label as admin_upsert_label
from services.deep_scan_service import deep_job_key, write_job_state, fetch_job_state

from deep_scan.config import settings as deep_scan_settings
from deep_scan.tasks import process_deep_scan_job

# Load environment variables
load_dotenv()

DEFAULT_PLATFORM = os.getenv("DEFAULT_PLATFORM", "youtube")
REDIS_URL = os.getenv("REDIS_APP_URL")
DATABASE_URL = os.getenv("DATABASE_URL")
BROKER_URL = os.getenv("CELERY_BROKER_URL")

redis_client: Optional[redis.Redis] = None
if REDIS_URL:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

broker_redis: Optional[redis.Redis] = None
if BROKER_URL and BROKER_URL.startswith(("redis://", "rediss://")):
    try:
        broker_redis = redis.Redis.from_url(BROKER_URL, decode_responses=True)
    except Exception:
        broker_redis = None

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

ADMIN_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 1 week for manual overrides
METRICS_VERDICT_WINDOW_HOURS = 24
RECENT_ANALYSES_LIMIT = 20
RECENT_ADMIN_LIMIT = 20

# CORS Configuration - Secure settings for Chrome Extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Chrome extensions use chrome-extension://<id> origins
                          # TODO: Replace with specific extension ID once published:
                          # allow_origins=["chrome-extension://your-extension-id-here"]
    allow_credentials=False,  # We don't use cookies or auth headers
    allow_methods=["GET", "POST", "OPTIONS"],    # Allow read + deep-scan creation + preflight
    allow_headers=["Accept", "Content-Type", "Access-Control-Request-Private-Network"],  # Include PNA header
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Middleware for Chrome Private Network Access (PNA)
# Required for extensions on public sites (youtube.com) to access localhost
@app.middleware("http")
async def add_private_network_access_headers(request: Request, call_next):
    response = await call_next(request)
    # Add PNA header to allow requests from public sites to local network
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


class AnalysisResult(BaseModel):
  result: str
  confidence: float
  reason: str
  source: str


class DeepScanJobRequest(BaseModel):
    video_id: str
    platform: str
    url: str
    heuristics: Optional[Dict[str, Any]] = None


class DeepScanJobResponse(BaseModel):
    job_id: str
    status: str


class DeepScanPollResponse(BaseModel):
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    updated_at: Optional[str] = None


class RecentAnalysisEntry(BaseModel):
    platform: str
    video_id: str
    label: str
    confidence: Optional[float] = None
    reason: Optional[str] = None
    analyzed_at: datetime
    frames_count: Optional[int] = None
    batch_time_ms: Optional[int] = None
    views_per_hour: Optional[float] = None
    region: Optional[str] = None
    title: Optional[str] = None
    channel: Optional[str] = None
    source_url: Optional[str] = None


class AnalyzeRequest(BaseModel):
    platform: str
    video_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AdminLabelEntry(BaseModel):
    platform: str
    video_id: str
    label: str
    notes: Optional[str] = None
    source_url: Optional[str] = None
    created_at: datetime


class AdminMetricsResponse(BaseModel):
    queues: Dict[str, Optional[int]]
    verdict_counts: Dict[str, int]
    recent_analyses: List[RecentAnalysisEntry]
    admin_overrides: List[AdminLabelEntry]


class AdminLabelRequest(BaseModel):
    url: str
    label: str
    notes: Optional[str] = None


class AdminLabelResponse(BaseModel):
    platform: str
    video_id: str
    label: str
    notes: Optional[str] = None
    source_url: Optional[str] = None
    cached: bool
    created_at: datetime

# Routes
@app.get("/")
async def root():
    return {"message": "ScrollSafe Backend API"}

@app.get("/api/ds-cache/{video_id}")
async def check_doom_scroller_cache(video_id: str, platform: Optional[str] = None):
    """Check Doomscroller cache and database for a verdict."""
    platform = (platform or DEFAULT_PLATFORM).lower()

    cache_hit = get_cache_hit(redis_client, platform, video_id)
    if cache_hit:
        logger.info("DS cache hit for %s:%s", platform, video_id)
        return AnalysisResult(**cache_hit)

    db_hit = get_db_hit(db_pool, platform, video_id)
    if db_hit:
        logger.info("DS DB hit for %s:%s", platform, video_id)
        return AnalysisResult(**db_hit)

    logger.info("DS miss for %s:%s", platform, video_id)
    raise HTTPException(status_code=404, detail="Not in cache")

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_video_post(payload: AnalyzeRequest, request: Request):
    request_start = time.time()
    platform = (payload.platform or DEFAULT_PLATFORM).lower()
    video_id = payload.video_id
    metadata: Dict[str, Any] = payload.metadata or {}

    if await request.is_disconnected():
        raise HTTPException(status_code=499, detail="Client disconnected")

    video_info: Optional[Dict[str, Any]] = None

    if platform == "youtube" and video_id:
        print (payload)
        loop = asyncio.get_event_loop()
        try:
            video_info = await loop.run_in_executor(executor, get_video_info, video_id)
        except Exception:
            video_info = None
    else:
        if isinstance(metadata, dict) and metadata:
            title = (metadata.get("title") or metadata.get("caption") or "")
            description = (metadata.get("description") or title or "")
            tags = metadata.get("hashtags") or metadata.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            channel = metadata.get("channel") or metadata.get("author") or ""
            video_info = {
                "title": title,
                "description": description,
                "tags": tags,
                "channelTitle": channel,
            }
            print("[Heuristics] Received client metadata for {}:{} -> title='{}'".format(platform, video_id, title[:40]))

    if not video_info:
        return AnalysisResult(
            result="unknown",
            confidence=0.0,
            reason="Could not fetch video information",
            source="Backend API",
        )

    heuristics_result = check_heuristics(video_info)
    print(heuristics_result)
    total_duration = time.time() - request_start
    print("[Heuristics] {}:{} -> {} in {:.3f}s".format(platform, video_id, heuristics_result.get("result"), total_duration))

    return AnalysisResult(
        result=heuristics_result["result"],
        confidence=heuristics_result["confidence"],
        reason=heuristics_result["reason"],
        source="Backend API",
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


@app.get("/api/admin/metrics", response_model=AdminMetricsResponse)
async def get_admin_metrics():
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    queue_names = ["analyze", deep_scan_settings.queue_name]
    try:
        metrics = build_admin_metrics(
            db_pool=db_pool,
            redis_client=redis_client,
            queue_redis=broker_redis,
            queue_names=queue_names,
            verdict_window_hours=METRICS_VERDICT_WINDOW_HOURS,
            recent_limit=RECENT_ANALYSES_LIMIT,
            admin_limit=RECENT_ADMIN_LIMIT,
        )
    except Exception as exc:
        logger.exception("Failed to load admin metrics")
        raise HTTPException(status_code=500, detail="Failed to load metrics") from exc

    return AdminMetricsResponse(
        queues=metrics["queues"],
        verdict_counts=metrics["verdict_counts"],
        recent_analyses=[RecentAnalysisEntry(**item) for item in metrics["recent_analyses"]],
        admin_overrides=[AdminLabelEntry(**item) for item in metrics["admin_overrides"]],
    )


@app.post("/api/admin/labels", response_model=AdminLabelResponse)
async def upsert_admin_label(payload: AdminLabelRequest):
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    if not payload.url or not payload.url.strip():
        raise HTTPException(status_code=400, detail="URL must not be empty")

    if not payload.label or not payload.label.strip():
        raise HTTPException(status_code=400, detail="Label must not be empty")

    try:
        result = admin_upsert_label(
            db_pool=db_pool,
            redis_client=redis_client,
            url=payload.url,
            label=payload.label,
            notes=payload.notes,
            cache_ttl_seconds=ADMIN_CACHE_TTL_SECONDS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to upsert admin label")
        raise HTTPException(status_code=500, detail="Failed to upsert admin label") from exc

    return AdminLabelResponse(**result)


@app.post("/api/deep-scan", response_model=DeepScanJobResponse)
async def enqueue_deep_scan(payload: DeepScanJobRequest):
    print(payload.video_id)
    if not redis_client:
        raise HTTPException(status_code=503, detail="Deep scan unavailable")

    job_id = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    write_job_state(
        redis_client,
        job_id,
        {
            "status": "queued",
            "created_at": created_at,
            "updated_at": created_at,
        },
        deep_scan_settings.redis_job_ttl_seconds,
    )

    job_payload: Dict[str, Any] = {
        "platform": payload.platform,
        "video_id": payload.video_id,
        "url": payload.url,
    }
    if payload.heuristics:
        job_payload["client_hints"] = payload.heuristics

    try:
        process_deep_scan_job.apply_async(
            args=[job_id, job_payload],
            queue=deep_scan_settings.queue_name,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to enqueue deep scan job %s", job_id)
        redis_client.delete(deep_job_key(job_id))
        raise HTTPException(status_code=500, detail="Failed to enqueue job") from exc

    logger.info(
        "Deep scan job enqueued job_id=%s platform=%s video_id=%s, url=%s",
        job_id,
        payload.platform,
        payload.video_id,
        payload.url
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/deep-scan/{job_id}", response_model=DeepScanPollResponse)
async def poll_deep_scan(job_id: str):
    if not redis_client:
        raise HTTPException(status_code=503, detail="Deep scan unavailable")

    try:
        payload = fetch_job_state(redis_client, job_id)
    except ValueError:
        raise HTTPException(status_code=500, detail="Corrupt job state")
    if not payload:
        raise HTTPException(status_code=404, detail="Job not found")

    status = payload.get("status") or "queued"
    response: Dict[str, Any] = {"status": status}

    updated_at = payload.get("updated_at")
    if updated_at:
        response["updated_at"] = updated_at

    if status == "done":
        result = payload.get("result") or {}
        response["result"] = {
            "result": result.get("label"),
            "confidence": result.get("confidence"),
            "reason": result.get("reason"),
            "vote_share": result.get("vote_share"),
            "analyzed_at": result.get("analyzed_at"),
            "model_version": result.get("model_version"),
        }
    elif status == "failed":
        response["error"] = payload.get("error", "deep_scan_failed")

    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        if db_pool:
            db_pool.close()


