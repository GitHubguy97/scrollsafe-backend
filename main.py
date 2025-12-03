from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import time
import os
import logging
import requests
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
from pathlib import Path
import base64
import binascii
import json
import shutil

from services.analysis_service import get_cache_hit, get_db_hit
from services.admin_service import build_admin_metrics, upsert_admin_label as admin_upsert_label
from services.deep_scan_service import deep_job_key, write_job_state, fetch_job_state

from deep_scan.config import settings as deep_scan_settings
from deep_scan.tasks import process_deep_scan_job
from huggingface_client import analyze_with_huggingface

# Load environment variables
load_dotenv()

DEFAULT_PLATFORM = os.getenv("DEFAULT_PLATFORM", "youtube")
REDIS_URL = os.getenv("REDIS_APP_URL")
DATABASE_URL = os.getenv("DATABASE_URL")
BROKER_URL = os.getenv("CELERY_BROKER_URL")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")  # Required for admin routes
PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEEP_SCAN_STORAGE_DIR = (OUT_DIR / "deep_scans")
DEEP_SCAN_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

redis_client: Optional[redis.Redis] = None
if REDIS_URL:
    try:
        redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    except Exception as exc:
        print("Failed to initialize Redis client:", exc)
        redis_client = None

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
    allow_headers=["Accept", "Content-Type", "Access-Control-Request-Private-Network", "X-API-Key"],  # Include PNA header and API key
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


# Admin API Key Authentication
def verify_admin_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Verify admin API key from X-API-Key header."""
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Admin API key not configured on server"
        )
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key"
        )
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key


class AnalysisResult(BaseModel):
  result: str
  confidence: float
  reason: str
  source: str


class DeepScanJobRequest(BaseModel):
    video_id: str
    platform: str
    url: Optional[str] = None
    heuristics: Optional[Dict[str, Any]] = None
    frames: Optional[List[str]] = None


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

@app.get("/health")
async def health():
    """Health check endpoint for Docker healthcheck and load balancers."""
    return {
        "status": "healthy",
        "redis": redis_client is not None,
        "database": db_pool is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

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

    # Check cache and database BEFORE running heuristics (important for admin labels)
    if video_id:
        cache_hit = get_cache_hit(redis_client, platform, video_id)
        if cache_hit:
            logger.info("Cache hit for %s:%s in analyze endpoint", platform, video_id)
            return AnalysisResult(**cache_hit)

        db_hit = get_db_hit(db_pool, platform, video_id)
        if db_hit:
            logger.info("DB hit for %s:%s in analyze endpoint", platform, video_id)
            return AnalysisResult(**db_hit)

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
async def get_admin_metrics(api_key: str = Depends(verify_admin_api_key)):
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
async def upsert_admin_label(payload: AdminLabelRequest, api_key: str = Depends(verify_admin_api_key)):
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


def _check_resolver_health(resolver_url: str, timeout: int = 3) -> bool:
    """Check if the resolver service is available by hitting its health endpoint."""
    if not resolver_url:
        return False

    try:
        health_url = f"{resolver_url.rstrip('/')}/health"
        response = requests.get(health_url, timeout=timeout)
        return response.status_code == 200
    except Exception as exc:
        logger.warning("Resolver health check failed: %s", exc)
        return False


@app.post("/api/deep-scan", response_model=DeepScanJobResponse)
async def enqueue_deep_scan(payload: DeepScanJobRequest):
    """Store uploaded frames and enqueue a Celery job to run inference."""
    frames = payload.frames or []
    if not frames:
        raise HTTPException(status_code=400, detail="No frames provided for deep scan")

    if not redis_client:
        raise HTTPException(status_code=503, detail="Deep scan requires Redis connectivity")

    job_id = f"deep-{uuid4().hex}"
    try:
        frame_dir, saved_files = persist_client_frames(job_id, payload.video_id, frames)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    manifest = {
        "job_id": job_id,
        "platform": (payload.platform or DEFAULT_PLATFORM).lower(),
        "video_id": payload.video_id,
        "url": payload.url,
        "frame_dir": str(frame_dir),
        "frames": [Path(path).name for path in saved_files],
        "metadata": payload.heuristics or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(frame_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    write_job_state(
        redis_client,
        job_id,
        {"status": "queued", "updated_at": datetime.now(timezone.utc).isoformat()},
        deep_scan_settings.redis_job_ttl_seconds,
    )

    job_payload = {
        "platform": manifest["platform"],
        "video_id": payload.video_id,
        "url": payload.url,
        "frame_dir": str(frame_dir),
        "frame_count": len(saved_files),
        "metadata": payload.heuristics or {},
    }

    try:
        process_deep_scan_job.delay(job_id, job_payload)
    except Exception as exc:
        logger.exception("Failed to enqueue deep scan job")
        raise HTTPException(status_code=500, detail="Failed to enqueue deep scan") from exc

    logger.info(
        "Enqueued deep scan job %s with %s frames for %s",
        job_id,
        len(saved_files),
        payload.video_id,
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/deep-scan/{job_id}", response_model=DeepScanPollResponse)
async def poll_deep_scan(job_id: str):
    if redis_client:
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

    raise HTTPException(status_code=503, detail="Deep scan storage unavailable")


def decode_data_url(data_url: str) -> bytes:
    if not data_url:
        raise ValueError("empty frame data")
    if isinstance(data_url, bytes):
        return data_url
    if data_url.startswith("data:"):
        try:
            _, encoded = data_url.split(",", 1)
        except ValueError as exc:
            raise ValueError("malformed data URL") from exc
    else:
        encoded = data_url
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 data") from exc


def persist_client_frames(job_id: str, video_id: Optional[str], frames: List[str]) -> tuple[Path, List[str]]:
    job_dir = DEEP_SCAN_STORAGE_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    job_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []
    for index, frame_data in enumerate(frames):
        try:
            binary = decode_data_url(frame_data)
        except ValueError as exc:
            raise ValueError(f"Invalid frame at index {index}: {exc}") from exc
        filename = job_dir / f"frame_{index + 1:03d}.jpg"
        with open(filename, "wb") as handle:
            handle.write(binary)
        saved_files.append(str(filename))

    return job_dir, saved_files


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        if db_pool:
            db_pool.close()


if __name__ == "__main__":
    import uvicorn

    max_bytes = int(os.getenv("MAX_REQUEST_BYTES", str(64 * 1024 * 1024)))
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        h11_max_incomplete_event_size=max_bytes,
    )


