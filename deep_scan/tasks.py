from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import redis
import requests
import yt_dlp
from celery import shared_task
from yt_dlp.utils import DownloadError

from deep_scan.config import settings
from heuristics import check_heuristics
from video_utils import get_video_info
from dotenv import load_dotenv

from pathlib import Path
from typing import List, Tuple, Dict, Any
from yt_dlp import YoutubeDL


load_dotenv()

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(level=settings.log_level)
logger.setLevel(settings.log_level)


def _redis_client() -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def _job_key(job_id: str) -> str:
    return f"deep:job:{job_id}"


def _video_cache_key(platform: str, video_id: str) -> str:
    return f"video:{platform}:{video_id}"


def _lock_key(platform: str, video_id: str) -> str:
    return f"deep:lock:{platform}:{video_id}"


def _store_job_status(job_id: str, status: str, *, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
    payload: Dict[str, Any] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error

    client = _redis_client()
    client.set(_job_key(job_id), json.dumps(payload), ex=settings.redis_job_ttl_seconds)


# def _build_yt_dlp_command(url: str) -> List[str]:
#     cmd = [
#         "yt-dlp",
#         "-f",
#         "bestvideo[height<=1080]/best",
#         "-o",
#         "-",
#         "--quiet",
#         "--no-warnings",
#         "--no-part",
#         url,
#     ]

#     cookie_browser = os.getenv("YTDLP_COOKIES_BROWSER")
#     cookies_path = os.getenv("YTDLP_COOKIES_FILE")
#     if cookie_browser:
#         cmd.extend(["--cookies-from-browser", cookie_browser])
#     elif cookies_path:
#         cmd.extend(["--cookies", cookies_path])

#     return cmd


# def _build_ffmpeg_command(*, duration_seconds: float, target_frames: int, output_pattern: Path) -> List[str]:
#     duration_seconds = max(duration_seconds, 0.001)
#     fps_value = max(target_frames / duration_seconds, 0.01)
#     filters = [
#         f"fps=fps={fps_value:.8f}:round=up",
#         "scale=-2:1080:force_original_aspect_ratio=decrease",
#     ]
#     return [
#         "ffmpeg",
#         "-hide_banner",
#         "-loglevel",
#         "error",
#         "-nostdin",
#         "-i",
#         "pipe:0",
#         "-an",
#         "-vf",
#         ",".join(filters),
#         "-vsync",
#         "vfr",
#         "-frames:v",
#         str(target_frames),
#         "-q:v",
#         "2",
#         str(output_pattern),
#     ]


# def _extract_frames(url: str, target_frames: int, *, timeout: int) -> List[bytes]:
#     metadata = _probe_stream_metadata(url)
#     duration = metadata.get("duration") or 0.0

#     with tempfile.TemporaryDirectory(prefix="deep_frames_") as tmpdir:
#         tmp_path = Path(tmpdir)
#         output_pattern = tmp_path / "frame_%03d.jpg"

#         yt_cmd = _build_yt_dlp_command(url)
#         ff_cmd = _build_ffmpeg_command(
#             duration_seconds=duration,
#             target_frames=target_frames,
#             output_pattern=output_pattern,
#         )

#         try:
#             ydl_proc = subprocess.Popen(yt_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         except FileNotFoundError as exc:
#             raise RuntimeError("yt-dlp executable not found on PATH") from exc

#         try:
#             try:
#                 subprocess.run(
#                     ff_cmd,
#                     stdin=ydl_proc.stdout,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     check=True,
#                     timeout=timeout,
#                 )
#             except FileNotFoundError as exc:
#                 ydl_proc.kill()
#                 raise RuntimeError("ffmpeg executable not found on PATH") from exc
#             except subprocess.TimeoutExpired as exc:
#                 ydl_proc.kill()
#                 raise RuntimeError("ffmpeg timed out while extracting frames") from exc
#             except subprocess.CalledProcessError as exc:
#                 ydl_proc.kill()
#                 stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
#                 raise RuntimeError(f"ffmpeg failed to extract frames: {stderr.strip()}") from exc
#         finally:
#             if ydl_proc.stdout:
#                 ydl_proc.stdout.close()
#             if ydl_proc.stderr:
#                 ydl_proc.stderr.close()
#             try:
#                 ydl_proc.wait(timeout=5)
#             except subprocess.TimeoutExpired:
#                 ydl_proc.kill()
#                 ydl_proc.wait(timeout=5)

#         frame_files = sorted(tmp_path.glob("frame_*.jpg"))
#         frames = [path.read_bytes() for path in frame_files[:target_frames]]
#         if not frames:
#             raise RuntimeError("No frames extracted from stream")
#         return frames


# def _probe_stream_metadata(url: str) -> Dict[str, Any]:
#     try:
#         with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
#             info = ydl.extract_info(url, download=False)
#             duration = info.get("duration")
#             if info.get("_type") == "playlist":
#                 entries = info.get("entries") or []
#                 if entries:
#                     duration = entries[0].get("duration")
#             return {"duration": duration or 0.0}
#     except DownloadError as exc:
#         logger.warning("yt-dlp metadata probe failed: %s", exc)
#     except Exception:
#         logger.exception("Unexpected error probing stream metadata")
#     return {"duration": 0.0}


# fast_extract_frames.py


def _select_video_url(info: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    """Pick a direct playable URL + headers from yt-dlp's info dict."""
    if "entries" in info and info["entries"]:
        info = info["entries"][0]

    headers = info.get("http_headers", {}) or {}

    # If yt-dlp already chose formats, prefer an actual video one
    if info.get("requested_formats"):
        for f in info["requested_formats"]:
            if f.get("vcodec") != "none" and f.get("url"):
                return f["url"], headers or f.get("http_headers", {})

    # Otherwise choose a good candidate from formats
    fmts = [f for f in (info.get("formats") or []) if f.get("url")]
    video_fmts = [f for f in fmts if f.get("vcodec") and f["vcodec"] != "none"]
    candidates = video_fmts or fmts
    if not candidates and info.get("url"):
        return info["url"], headers

    def score(f):
        return (
            1 if f.get("ext") == "mp4" else 0,
            f.get("height") or 0,
            f.get("tbr") or 0,
        )
    candidates.sort(key=score, reverse=True)
    best = candidates[0]
    return best["url"], headers or best.get("http_headers", {})


def _headers_to_ffmpeg_args(headers: Dict[str, str]) -> list:
    args: list = []
    for k, v in (headers or {}).items():
        args += ["-headers", f"{k}: {v}"]
    if headers.get("User-Agent"):
        args += ["-user_agent", headers["User-Agent"]]
    if headers.get("Referer"):
        args += ["-referer", headers["Referer"]]
    return args


def _ffprobe_duration(media_url: str, headers: Dict[str, str]) -> float:
    """Try to get duration via ffprobe; return 0.0 on failure."""
    hdr_args = _headers_to_ffmpeg_args(headers)
    cmd = [
        "ffprobe",
        *hdr_args,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        media_url,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return float(out)
    except Exception:
        return 0.0


def _extract_frames(url: str, target_frames: int, *, timeout: int) -> list[bytes]:
    """
    Resolve a direct media URL with yt-dlp, then have FFmpeg pull frames
    (evenly spaced across the whole duration) **fast** and return them as bytes.
    """
    # Resolve a playable URL + headers
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "format": "bestvideo*[protocol^=http]/best[protocol^=http]/bestvideo/best",
    }
    # Optional cookie support via env vars (useful for IG/TT)
    cookies_file = os.getenv("YTDLP_COOKIES_FILE")
    cookies_browser = os.getenv("YTDLP_COOKIES_BROWSER")
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
    elif cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser,)

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    media_url, headers = _select_video_url(info)

    # Duration: prefer yt-dlp's, else ffprobe, else fallback
    duration = float(info.get("duration") or 0.0)
    if duration <= 0:
        duration = _ffprobe_duration(media_url, headers)
    if duration <= 0:
        duration = 1.0  # avoid div-by-zero; may bias toward early frames

    # Compute fps to get ~target_frames across full length
    fps = max(target_frames / duration, 0.01)
    hdr_args = _headers_to_ffmpeg_args(headers)

    with tempfile.TemporaryDirectory(prefix="fast_frames_") as tmpdir:
        out_dir = Path(tmpdir)
        out_pattern = str(out_dir / "%05d.jpg")

        ff_cmd = [
            "ffmpeg",
            "-y",
            *hdr_args,
            "-i", media_url,
            "-vf", f"fps=fps={fps:.8f}:round=up,scale=-2:1080:force_original_aspect_ratio=decrease",
            "-q:v", "2",
            out_pattern,
        ]

        try:
            subprocess.run(
                ff_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("ffmpeg timed out while extracting frames") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
            raise RuntimeError(f"ffmpeg failed to extract frames: {stderr.strip()}") from exc

        # Read back bytes (keeps the fast path: ffmpeg → files; then load to memory)
        frame_files = sorted(out_dir.glob("*.jpg"))
        if not frame_files:
            raise RuntimeError("No frames produced by ffmpeg")

        # Only take up to target_frames, in order
        frames: List[bytes] = [p.read_bytes() for p in frame_files[:target_frames]]
        return frames



def _call_inference(frames: Sequence[bytes]) -> Dict[str, Any]:
    if not frames:
        raise ValueError("No frames provided to inference")

    endpoint = settings.inference_url
    # settings.inference_url.rstrip("/")
    if not endpoint.endswith("/v1/infer"):
        endpoint = f"{endpoint}/v1/infer"

    files = [
        ("files", (f"frame_{idx:03d}.jpg", blob, "image/jpeg"))
        for idx, blob in enumerate(frames)
    ]

    if settings.inference_api_key:
        # Support both Hugging Face Bearer and custom header usage
        headers = {
        "Authorization": f"Bearer {settings.hf_token}",
        "X-API-Key": settings.inference_api_key,
    }
        
    # Debug logging
    logger.info("Inference URL: %s", endpoint)
    logger.info("API Key present: %s", bool(settings.inference_api_key))
    logger.info("API Key (first 10 chars): %s", settings.inference_api_key[:10] if settings.inference_api_key else "None")
    logger.info("Headers: %s", {k: v[:20] + "..." if len(v) > 20 else v for k, v in headers.items()})

    response = requests.post(
        endpoint,
        headers=headers,
        files=files,
        timeout=settings.inference_timeout,
    )
    response.raise_for_status()
    return response.json()


def _aggregate_inference(inference: Dict[str, Any]) -> Dict[str, Any]:
    results = inference.get("results") or []
    if not results:
        raise ValueError("Inference payload did not contain frame results")

    totals: Counter[str] = Counter()
    max_scores: List[float] = []
    frame_details: List[Dict[str, Any]] = []

    for entry in results:
        scores = entry.get("label_scores") or {}
        if not scores:
            continue
        best_label, best_score = max(scores.items(), key=lambda item: item[1])
        totals[best_label] += 1
        max_scores.append(float(best_score))
        frame_details.append({"label": best_label, "scores": scores})

    if not totals:
        raise ValueError("Inference results contained empty scores")

    total_frames = sum(totals.values())
    vote_share = {label: count / total_frames for label, count in totals.items()}
    majority_label = max(vote_share.items(), key=lambda item: item[1])[0]
    confidence = sum(max_scores) / len(max_scores)

    return {
        "vote_share": vote_share,
        "label": majority_label,
        "confidence": confidence,
        "reason": f"model_vote:{majority_label}",
        "features": {
            "frame_count": total_frames,
            "frame_details": frame_details,
            "batch_time_ms": inference.get("batch_time_ms"),
        },
    }


def _apply_heuristics(
    aggregate: Dict[str, Any], heuristics_result: Optional[Dict[str, Any]], client_hints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    label = aggregate["label"]
    confidence = float(aggregate["confidence"])
    reasons = [aggregate["reason"]]

    features = aggregate["features"]
    if heuristics_result:
        features["heuristics"] = heuristics_result
        h_label = heuristics_result.get("result")
        h_conf = float(heuristics_result.get("confidence", 0.0))
        h_reason = heuristics_result.get("reason")
        if h_reason:
            reasons.append(f"metadata:{h_reason}")
        if h_label == "ai-detected":
            label = "ai-detected"
            confidence = max(confidence, h_conf)
        elif h_label == "suspicious" and label == "verified":
            label = "suspicious"
            confidence = max(confidence, max(h_conf, 0.6))

    if client_hints:
        features["client_hints"] = client_hints
        hint_label = client_hints.get("result")
        hint_conf = float(client_hints.get("confidence", 0.0))
        hint_reason = client_hints.get("reason")
        if hint_reason:
            reasons.append(f"client:{hint_reason}")
        if hint_label == "ai-detected":
            label = "ai-detected"
            confidence = max(confidence, hint_conf)
        elif hint_label == "suspicious" and label == "verified":
            label = "suspicious"
            confidence = max(confidence, max(hint_conf, 0.6))

    reason = "; ".join(reasons)
    return {
        "label": label,
        "confidence": min(max(confidence, 0.0), 1.0),
        "reason": reason,
        "features": features,
    }


def _cache_video_result(
    *,
    platform: str,
    video_id: str,
    result: Dict[str, Any],
    analyzed_at: datetime,
    vote_share: Dict[str, float],
) -> None:
    payload = {
        "platform": platform,
        "video_id": video_id,
        "label": result["label"],
        "confidence": result["confidence"],
        "reason": result["reason"],
        "vote_share": vote_share,
        "analyzed_at": analyzed_at.isoformat(),
        "model_version": settings.model_version,
    }
    client = _redis_client()
    client.set(_video_cache_key(platform, video_id), json.dumps(payload), ex=settings.redis_job_ttl_seconds)


@shared_task(name="deep_scan.tasks.process_job", queue=settings.queue_name)
def process_deep_scan_job(job_id: str, payload: Dict[str, Any]) -> None:
    client = _redis_client()
    platform = (payload.get("platform") or "youtube").lower()
    video_id = payload.get("video_id")
    url = payload.get("url")
    client_hints = payload.get("client_hints")

    if not video_id or not url:
        _store_job_status(job_id, "failed", error="Missing video_id or url")
        return

    lock_key = _lock_key(platform, video_id)
    lock_acquired = client.set(lock_key, job_id, nx=True, ex=settings.redis_lock_ttl_seconds)
    if not lock_acquired:
        logger.info("Deep scan skipped for %s:%s (lock held)", platform, video_id)
        _store_job_status(job_id, "failed", error="duplicate_in_progress")
        return

    started_at = time.perf_counter()
    try:
        _store_job_status(job_id, "running")

        frames = _extract_frames(url, settings.target_frames, timeout=settings.frame_extract_timeout)
        inference = _call_inference(frames)
        aggregate = _aggregate_inference(inference)

        video_info: Optional[Dict[str, Any]] = None
        if platform == "youtube":
            video_info = get_video_info(video_id)

        heuristics_result = check_heuristics(video_info) if video_info else None
        merged = _apply_heuristics(aggregate, heuristics_result, client_hints)

        analyzed_at = datetime.now(timezone.utc)
        final_result = {
            "label": merged["label"],
            "confidence": merged["confidence"],
            "reason": merged["reason"],
            "vote_share": aggregate["vote_share"],
            "features": merged["features"],
            "frames_count": len(frames),
            "batch_time_ms": inference.get("batch_time_ms"),
            "analyzed_at": analyzed_at.isoformat(),
            "model_version": settings.model_version,
            "platform": platform,
            "video_id": video_id,
        }
        logger.info("Deep scan result job_id=%s platform=%s video_id=%s label=%s confidence=%.4f", job_id, platform, video_id, final_result['label'], final_result['confidence'])

        _store_job_status(job_id, "done", result=final_result)
        _cache_video_result(
            platform=platform,
            video_id=video_id,
            result=final_result,
            vote_share=aggregate["vote_share"],
            analyzed_at=analyzed_at,
        )

        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Deep scan job %s finished for %s:%s in %.1f ms",
            job_id,
            platform,
            video_id,
            duration_ms,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Deep scan job %s failed", job_id)
        _store_job_status(job_id, "failed", error=str(exc))
        raise
    finally:
        client.delete(lock_key)
