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


def _build_metadata_for_heuristics(metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not metadata:
        return {}

    tags = metadata.get("hashtags") or metadata.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]

    return {
        "title": metadata.get("title") or metadata.get("channel") or "",
        "description": metadata.get("description") or metadata.get("caption") or "",
        "tags": tags,
    }


def _load_saved_frames(frame_dir: str) -> List[bytes]:
    path = Path(frame_dir)
    if not path.exists():
        raise RuntimeError(f"Frame directory not found: {frame_dir}")

    frame_files = sorted(path.glob("frame_*.jpg"))
    if not frame_files:
        raise RuntimeError(f"No frames found in {frame_dir}")

    return [file.read_bytes() for file in frame_files]


# ==============================================================================
# ROBUST FRAME EXTRACTION PIPELINE
# ==============================================================================
# Fast path: yt-dlp pipe → ffmpeg stdin (keeps speed)
# Fallback A: Stricter progressive format
# Fallback B: Direct URL → ffmpeg with headers
# Fallback C: Temp file download
# ==============================================================================

from enum import Enum
import threading


class ErrorClass(Enum):
    """Classification of frame extraction errors."""
    HLS_PARSE = "hls_parse"
    AUTH_REQUIRED = "auth_required"
    FORBIDDEN_403 = "forbidden_403"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


def _classify_error(stderr: str) -> ErrorClass:
    """Classify error from ffmpeg/yt-dlp stderr."""
    stderr_lower = stderr.lower()

    if "403" in stderr_lower or "forbidden" in stderr_lower:
        return ErrorClass.FORBIDDEN_403
    if "401" in stderr_lower or "unauthorized" in stderr_lower:
        return ErrorClass.AUTH_REQUIRED
    if "429" in stderr_lower or "rate limit" in stderr_lower:
        return ErrorClass.RATE_LIMIT
    if "m3u8" in stderr_lower or "hls" in stderr_lower or "dash" in stderr_lower:
        return ErrorClass.HLS_PARSE

    return ErrorClass.UNKNOWN


def _compute_fps(duration: float, target_frames: int) -> float:
    """Compute FPS to extract target_frames evenly across duration."""
    duration = max(duration, 0.001)
    return max(target_frames / duration, 0.01)


def _get_cookie_config() -> Tuple[str, str]:
    """Get cookie configuration from environment.
    Returns (mode, value) where mode is 'file', 'browser', or 'none'.
    """
    cookies_file = os.getenv("YTDLP_COOKIES_FILE")
    cookies_browser = os.getenv("YTDLP_COOKIES_BROWSER")

    if cookies_file:
        return ("file", cookies_file)
    elif cookies_browser:
        return ("browser", cookies_browser)
    else:
        return ("none", "")


def _probe_metadata(url: str) -> Tuple[float, Dict[str, str]]:
    """Probe video metadata to get duration and headers.
    Returns (duration, http_headers).
    """
    cookie_mode, cookie_value = _get_cookie_config()
    logger.debug("Cookie mode: %s", cookie_mode)

    ydl_opts = {
        "format": "bestvideo*[protocol^=http][ext=mp4]/best[protocol^=http][ext=mp4]/best[protocol^=http]",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": 10,
        "retries": 2,
        "ignore_no_formats_error": True,
    }

    if cookie_mode == "file":
        ydl_opts["cookiefile"] = cookie_value
    elif cookie_mode == "browser":
        ydl_opts["cookiesfrombrowser"] = (cookie_value,)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Handle playlist wrappers
            if info.get("_type") == "playlist":
                entries = info.get("entries") or []
                if entries:
                    info = entries[0]

            # Extract duration (try multiple keys)
            duration = 0.0
            for key in ["duration", "duration_float", "duration_seconds"]:
                val = info.get(key)
                if val and float(val) > 0:
                    duration = float(val)
                    break

            # Fallback to target_frames if no duration found
            if duration <= 0:
                duration = float(settings.target_frames)
                logger.debug("No duration found, using target_frames as fallback: %.1f", duration)

            headers = info.get("http_headers", {}) or {}

            logger.debug("Probed metadata: duration=%.2fs, headers_keys=%s",
                        duration, list(headers.keys()))

            return (duration, headers)

    except Exception as exc:
        logger.warning("Metadata probe failed: %s, using defaults", exc)
        return (float(settings.target_frames), {})


def _build_yt_dlp_command(url: str, format_selector: str) -> List[str]:
    """Build yt-dlp command for streaming to stdout."""
    cookie_mode, cookie_value = _get_cookie_config()

    cmd = [
        "yt-dlp",
        "-f", format_selector,
        "--hls-use-mpegts",
        "--retries", "5",
        "--fragment-retries", "10",
        "--concurrent-fragments", "5",
        "--no-part",
        "--quiet",
        "--no-warnings",
        "-o", "-",
        url,
    ]

    if cookie_mode == "file":
        cmd.extend(["--cookies", cookie_value])
    elif cookie_mode == "browser":
        cmd.extend(["--cookies-from-browser", cookie_value])

    return cmd


def _build_ffmpeg_command(duration: float, target_frames: int, output_pattern: Path) -> List[str]:
    """Build ffmpeg command for extracting frames from stdin."""
    fps = _compute_fps(duration, target_frames)

    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-i", "pipe:0",
        "-an",
        "-vf", f"fps=fps={fps:.8f}:round=up,scale=-2:1080:force_original_aspect_ratio=decrease",
        "-vsync", "vfr",
        "-frames:v", str(target_frames),
        "-q:v", "2",
        str(output_pattern),
    ]


def _drain_stderr(proc: subprocess.Popen, output_list: List[str]):
    """Thread function to drain stderr from a subprocess to avoid deadlock."""
    try:
        if proc.stderr:
            for line in iter(proc.stderr.readline, b""):
                output_list.append(line.decode("utf-8", errors="ignore"))
    except Exception:
        pass


def _try_fast_path(url: str, target_frames: int, duration: float, format_selector: str, timeout: int, tmpdir: Path) -> Tuple[bool, str]:
    """Try fast path: yt-dlp pipe → ffmpeg stdin.
    Returns (success, error_message).
    """
    output_pattern = tmpdir / "frame_%03d.jpg"

    yt_cmd = _build_yt_dlp_command(url, format_selector)
    ff_cmd = _build_ffmpeg_command(duration, target_frames, output_pattern)

    logger.debug("Fast path attempt with format: %s", format_selector)
    logger.debug("yt-dlp command: %s", " ".join(yt_cmd))
    logger.debug("ffmpeg command: %s", " ".join(ff_cmd))

    ydl_stderr_lines: List[str] = []

    try:
        ydl_proc = subprocess.Popen(yt_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        return (False, "yt-dlp executable not found on PATH")

    # Start thread to drain yt-dlp stderr
    stderr_thread = threading.Thread(target=_drain_stderr, args=(ydl_proc, ydl_stderr_lines), daemon=True)
    stderr_thread.start()

    try:
        try:
            result = subprocess.run(
                ff_cmd,
                stdin=ydl_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            ydl_proc.kill()
            return (False, "ffmpeg executable not found on PATH")
        except subprocess.TimeoutExpired:
            ydl_proc.kill()
            return (False, "ffmpeg timed out while extracting frames")
        except subprocess.CalledProcessError as exc:
            ydl_proc.kill()
            stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
            return (False, f"ffmpeg failed: {stderr.strip()}")
    finally:
        if ydl_proc.stdout:
            ydl_proc.stdout.close()
        try:
            ydl_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ydl_proc.kill()
            ydl_proc.wait(timeout=5)

        stderr_thread.join(timeout=1)

    # Check if frames were produced
    frame_files = sorted(tmpdir.glob("frame_*.jpg"))
    if not frame_files:
        ydl_stderr = "".join(ydl_stderr_lines)
        return (False, f"No frames produced. yt-dlp stderr: {ydl_stderr[:500]}")

    return (True, "")


def _select_media_format(info: Dict[str, Any]) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """Select best media format and return (url, headers, format_info)."""
    if "entries" in info and info["entries"]:
        info = info["entries"][0]

    headers = info.get("http_headers", {}) or {}

    # Try requested_formats first
    if info.get("requested_formats"):
        for fmt in info["requested_formats"]:
            if fmt.get("vcodec") != "none" and fmt.get("url"):
                return (fmt["url"], headers or fmt.get("http_headers", {}), fmt)

    # Select from available formats
    fmts = [f for f in (info.get("formats") or []) if f.get("url")]
    video_fmts = [f for f in fmts if f.get("vcodec") and f["vcodec"] != "none"]
    candidates = video_fmts or fmts

    if not candidates and info.get("url"):
        return (info["url"], headers, {})

    def score(f):
        is_mp4 = 1 if f.get("ext") == "mp4" else 0
        is_http = 1 if str(f.get("protocol", "")).startswith("http") else 0
        height = min(f.get("height") or 0, 1080)
        tbr = f.get("tbr") or 0
        return (is_http, is_mp4, height, tbr)

    candidates.sort(key=score, reverse=True)
    best = candidates[0]

    logger.debug("Selected format: id=%s ext=%s height=%s tbr=%s protocol=%s",
                 best.get("format_id"), best.get("ext"), best.get("height"),
                 best.get("tbr"), best.get("protocol"))

    return (best["url"], headers or best.get("http_headers", {}), best)


def _headers_to_ffmpeg_args(headers: Dict[str, str]) -> List[str]:
    """Convert HTTP headers to ffmpeg command-line arguments."""
    args: List[str] = []

    # Build headers string
    header_lines = []
    for k, v in (headers or {}).items():
        header_lines.append(f"{k}: {v}")

    if header_lines:
        args.extend(["-headers", "\r\n".join(header_lines)])

    # Special handling for User-Agent and Referer
    if headers.get("User-Agent"):
        args.extend(["-user_agent", headers["User-Agent"]])
    if headers.get("Referer"):
        args.extend(["-referer", headers["Referer"]])

    return args


def _try_fallback_b(url: str, target_frames: int, timeout: int, tmpdir: Path) -> Tuple[bool, str]:
    """Fallback B: Direct URL → ffmpeg with headers.
    Returns (success, error_message).
    """
    logger.info("Attempting Fallback B: Direct URL to ffmpeg")

    cookie_mode, cookie_value = _get_cookie_config()

    ydl_opts = {
        "format": "bestvideo*[protocol^=http][ext=mp4]/best[protocol^=http][ext=mp4]/best[protocol^=http]",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    if cookie_mode == "file":
        ydl_opts["cookiefile"] = cookie_value
    elif cookie_mode == "browser":
        ydl_opts["cookiesfrombrowser"] = (cookie_value,)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        media_url, headers, fmt_info = _select_media_format(info)

        # Get duration
        duration = 0.0
        for key in ["duration", "duration_float", "duration_seconds"]:
            val = info.get(key)
            if val and float(val) > 0:
                duration = float(val)
                break

        if duration <= 0:
            duration = float(target_frames)

        fps = _compute_fps(duration, target_frames)
        hdr_args = _headers_to_ffmpeg_args(headers)
        output_pattern = tmpdir / "frame_%03d.jpg"

        # Check if HLS - add protocol whitelist
        is_hls = ".m3u8" in media_url or fmt_info.get("protocol") == "m3u8"
        protocol_args = []
        if is_hls:
            protocol_args = ["-protocol_whitelist", "file,http,https,tcp,tls,crypto"]

        ff_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            *protocol_args,
            *hdr_args,
            "-i", media_url,
            "-an",
            "-vf", f"fps=fps={fps:.8f}:round=up,scale=-2:1080:force_original_aspect_ratio=decrease",
            "-vsync", "vfr",
            "-frames:v", str(target_frames),
            "-q:v", "2",
            str(output_pattern),
        ]

        logger.debug("Fallback B ffmpeg command: %s", " ".join(ff_cmd[:10]) + "...")

        subprocess.run(
            ff_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

        frame_files = sorted(tmpdir.glob("frame_*.jpg"))
        if not frame_files:
            return (False, "No frames produced in Fallback B")

        return (True, "")

    except subprocess.TimeoutExpired:
        return (False, "Fallback B: ffmpeg timed out")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        return (False, f"Fallback B failed: {stderr[:500]}")
    except Exception as exc:
        return (False, f"Fallback B exception: {str(exc)}")


def _try_fallback_c(url: str, target_frames: int, timeout: int, tmpdir: Path) -> Tuple[bool, str]:
    """Fallback C: Download temp file, then extract frames.
    Returns (success, error_message).
    """
    logger.info("Attempting Fallback C: Temp file download")

    cookie_mode, cookie_value = _get_cookie_config()

    temp_video = tmpdir / "temp_video.mp4"

    ydl_opts = {
        "format": "best[ext=mp4][protocol^=http]/best[protocol^=http]",
        "outtmpl": str(temp_video),
        "no_part": True,
        "quiet": True,
        "no_warnings": True,
    }

    if cookie_mode == "file":
        ydl_opts["cookiefile"] = cookie_value
    elif cookie_mode == "browser":
        ydl_opts["cookiesfrombrowser"] = (cookie_value,)

    try:
        # Download video
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not temp_video.exists():
            return (False, "Fallback C: Download produced no file")

        # Get duration from downloaded file
        try:
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1",
                str(temp_video),
            ]
            duration_str = subprocess.check_output(probe_cmd, stderr=subprocess.STDOUT, text=True).strip()
            duration = float(duration_str)
        except Exception:
            duration = float(target_frames)

        fps = _compute_fps(duration, target_frames)
        output_pattern = tmpdir / "frame_%03d.jpg"

        ff_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(temp_video),
            "-an",
            "-vf", f"fps=fps={fps:.8f}:round=up,scale=-2:1080:force_original_aspect_ratio=decrease",
            "-vsync", "vfr",
            "-frames:v", str(target_frames),
            "-q:v", "2",
            str(output_pattern),
        ]

        subprocess.run(
            ff_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

        frame_files = sorted(tmpdir.glob("frame_*.jpg"))
        if not frame_files:
            return (False, "Fallback C: No frames extracted from temp file")

        return (True, "")

    except subprocess.TimeoutExpired:
        return (False, "Fallback C: ffmpeg timed out")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        return (False, f"Fallback C failed: {stderr[:500]}")
    except Exception as exc:
        return (False, f"Fallback C exception: {str(exc)}")
    finally:
        # Clean up temp video
        if temp_video.exists():
            try:
                temp_video.unlink()
            except Exception:
                pass


def _extract_frames(url: str, target_frames: int, *, timeout: int) -> List[bytes]:
    """
    Robust frame extraction with fast path + fallbacks.

    Pipeline:
    1. Probe metadata (duration + headers)
    2. Fast path: yt-dlp pipe → ffmpeg (progressive MP4 preference)
    3. Fallback A: Stricter progressive format
    4. Fallback B: Direct URL → ffmpeg with headers
    5. Fallback C: Temp file download

    Returns list of JPEG frame bytes, evenly spaced across video duration.
    """
    start_time = time.perf_counter()
    logger.info("Starting frame extraction for %s (target: %d frames)", url, target_frames)

    # Step 1: Probe metadata
    duration, headers = _probe_metadata(url)
    logger.info("Probed duration: %.2fs", duration)

    with tempfile.TemporaryDirectory(prefix="deep_frames_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # Step 2: Try fast path (primary format)
        format_primary = "bestvideo*[protocol^=http][ext=mp4]/best[protocol^=http][ext=mp4]/best[protocol^=http]"
        success, error = _try_fast_path(url, target_frames, duration, format_primary, timeout, tmpdir)

        if success:
            logger.info("Fast path succeeded with primary format")
        else:
            logger.warning("Fast path failed: %s", error[:200])
            error_class = _classify_error(error)
            logger.debug("Error classified as: %s", error_class.value)

            # Step 3: Fallback A - Stricter progressive
            logger.info("Trying Fallback A: Stricter progressive format")
            format_strict = "best[ext=mp4][protocol^=http]/best[protocol^=http]"
            success, error_a = _try_fast_path(url, target_frames, duration, format_strict, timeout, tmpdir)

            if success:
                logger.info("Fallback A succeeded")
            else:
                logger.warning("Fallback A failed: %s", error_a[:200])

                # Step 4: Fallback B - Direct URL
                success, error_b = _try_fallback_b(url, target_frames, timeout, tmpdir)

                if success:
                    logger.info("Fallback B succeeded")
                else:
                    logger.warning("Fallback B failed: %s", error_b[:200])

                    # Step 5: Fallback C - Temp file
                    success, error_c = _try_fallback_c(url, target_frames, timeout, tmpdir)

                    if success:
                        logger.info("Fallback C succeeded")
                    else:
                        # All fallbacks failed
                        error_class = _classify_error(error + error_a + error_b + error_c)
                        raise RuntimeError(
                            f"All extraction attempts failed. Error type: {error_class.value}. "
                            f"Primary: {error[:100]} | FallbackA: {error_a[:100]} | "
                            f"FallbackB: {error_b[:100]} | FallbackC: {error_c[:100]}"
                        )

        # Read frames
        frame_files = sorted(tmpdir.glob("frame_*.jpg"))
        if not frame_files:
            raise RuntimeError("Frame extraction succeeded but no frame files found")

        frames: List[bytes] = [p.read_bytes() for p in frame_files[:target_frames]]

        if len(frames) < target_frames:
            logger.debug("Extracted %d/%d frames (fewer than requested)", len(frames), target_frames)

        elapsed = time.perf_counter() - start_time
        logger.info("Frame extraction completed: %d frames in %.2fs", len(frames), elapsed)

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

    headers: Dict[str, str] = {}
    if settings.hf_token:
        headers["Authorization"] = f"Bearer {settings.hf_token}"
    if settings.inference_api_key:
        headers["X-API-Key"] = settings.inference_api_key
    if "Authorization" not in headers:
        raise RuntimeError("HUGGING_FACE_API_KEY is required for inference")

    # Debug logging
    logger.info("Inference URL: %s", endpoint)
    logger.info("API key present: %s", bool(settings.inference_api_key))

    response = requests.post(
        endpoint,
        headers=headers,
        files=files,
        timeout=settings.inference_timeout,
    )
    response.raise_for_status()
    return response.json()


def _aggregate_inference(inference: Dict[str, Any], heuristics_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Aggregate inference results with conservative classification.
    Integrates heuristics (AI keywords in title/description) into decision logic.
    Skews towards real - only classifies as AI with very strong signals.
    """
    results = inference.get("results") or []
    if not results:
        raise ValueError("Inference payload did not contain frame results")

    # Collect scores from each frame
    label_scores_list: List[Dict[str, float]] = []
    vote_totals = {"real": 0.0, "artificial": 0.0}

    for entry in results:
        scores = entry.get("label_scores", {}) or {}
        real_score = float(scores.get("real", 0.0))
        artificial_score = float(scores.get("artificial", 0.0))

        label_scores_list.append({
            "real": real_score,
            "artificial": artificial_score,
        })
        vote_totals["real"] += real_score
        vote_totals["artificial"] += artificial_score

    # Calculate vote share percentages
    total_votes = vote_totals["real"] + vote_totals["artificial"] or 1.0
    vote_share = {
        "real": vote_totals["real"] / total_votes,
        "artificial": vote_totals["artificial"] / total_votes,
    }

    # Use conservative decision logic with heuristics integration
    decision = _decide_label(label_scores_list, heuristics_result)
    internal_label = decision["label"]

    # Map internal labels to external labels
    label_map = {
        "artificial": "ai-detected",
        "real": "verified",
        "suspicious": "suspicious",
    }
    external_label = label_map.get(internal_label, "verified")  # Default to verified instead of unknown
    confidence = float(decision.get("confidence", 0.0))
    reason = decision.get("reason", "model_vote")
    features = decision.get("features", {})

    # Add batch timing
    features["batch_time_ms"] = inference.get("batch_time_ms")

    return {
        "vote_share": vote_share,
        "label": external_label,
        "confidence": confidence,
        "reason": f"model_vote: {reason}",
        "features": features,
    }


def _decide_label(scores_list: List[Dict[str, float]], heuristics_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Conservative classification logic that skews towards real.
    Only classifies as AI with very strong signals.
    Integrates heuristics for AI keywords in title/description.
    Never returns unknown - always picks real, suspicious, or artificial.

    Decision rules:
    1. With AI keywords: Lower threshold for AI/suspicious
    2. Without AI keywords: Very high threshold for AI (default to real)
    3. Too few frames → default to real
    4. No unknown label - must classify
    """
    total_frames = len(scores_list)
    artificial_scores = [scores["artificial"] for scores in scores_list]

    # Count votes: if artificial_score >= real_score, count as "artificial" vote
    vote_counts = {"real": 0, "artificial": 0}
    for scores in scores_list:
        if scores["artificial"] >= scores["real"]:
            vote_counts["artificial"] += 1
        else:
            vote_counts["real"] += 1

    # Calculate statistics on artificial scores
    if artificial_scores:
        sorted_artificial = sorted(artificial_scores, reverse=True)
        max_artificial = sorted_artificial[0]
        top3 = sorted_artificial[:3]
        top3_mean = sum(top3) / len(top3) if top3 else 0.0
    else:
        sorted_artificial = []
        max_artificial = 0.0
        top3 = []
        top3_mean = 0.0

    # Count frames exceeding thresholds
    count_a90 = sum(score >= 0.90 for score in artificial_scores)
    count_a80 = sum(score >= 0.80 for score in artificial_scores)
    count_a95 = sum(score >= 0.95 for score in artificial_scores)
    frac_a90 = count_a90 / total_frames if total_frames else 0.0
    frac_a80 = count_a80 / total_frames if total_frames else 0.0
    frac_a95 = count_a95 / total_frames if total_frames else 0.0

    majority_label = (
        "artificial"
        if vote_counts["artificial"] >= vote_counts["real"]
        else "real"
    )

    # Check heuristics for AI keywords in title/description
    has_ai_keywords = False
    heuristic_label = None
    if heuristics_result:
        heuristic_label = heuristics_result.get("result")
        # If heuristics detected AI, it means AI keywords in title/description
        if heuristic_label == "ai-detected":
            has_ai_keywords = True

    # Build features object for storage
    features = {
        "majority_label": majority_label,
        "real_votes": vote_counts["real"],
        "artificial_votes": vote_counts["artificial"],
        "total_frames": total_frames,
        "max_artificial": max_artificial,
        "top3_mean_artificial": top3_mean,
        "count_a90": count_a90,
        "count_a80": count_a80,
        "count_a95": count_a95,
        "frac_a90": frac_a90,
        "frac_a80": frac_a80,
        "frac_a95": frac_a95,
        "has_ai_keywords": has_ai_keywords,
        "heuristic_label": heuristic_label,
    }

    # Rule 1: Too few frames - default to real (not unknown)
    if total_frames < 4:
        return {
            "label": "real",
            "confidence": 0.5,
            "reason": "too_few_frames_default_real",
            "features": features,
        }

    # Rule 2: Very strong artificial signal (RAISED thresholds - more conservative)
    if has_ai_keywords:
        # With AI keywords, still require strong visual signals
        if (
            frac_a95 >= 0.40  # 40%+ frames at 0.95+
            or (count_a90 >= 5 and top3_mean >= 0.95)  # 5+ frames at 0.90+, top3 avg 0.95+
            or frac_a90 >= 0.60  # 60%+ frames at 0.90+
        ):
            return {
                "label": "artificial",
                "confidence": max_artificial,
                "reason": "strong_artificial_with_keywords",
                "features": features,
            }
    else:
        # NO AI keywords - need VERY strong signal to classify as AI
        if (
            frac_a95 >= 0.6  # 60%+ frames at 0.95+ (very high threshold)
            or (count_a95 >= 6 and top3_mean >= 0.97)  # 6+ frames at 0.95+, top3 avg 0.97+
            or (
                frac_a90 >= 0.75  # 75%+ frames at 0.90+ (raised from 0.5)
                and min(sorted_artificial[:5] if len(sorted_artificial) >= 5 else sorted_artificial) >= 0.93
            )
        ):
            return {
                "label": "artificial",
                "confidence": max_artificial,
                "reason": "very_strong_artificial_no_keywords",
                "features": features,
            }

    # Rule 3: Suspicious signals
    if has_ai_keywords:
        # With AI keywords, still require more than a single spike
        if (
            count_a90 >= 2  # At least two strong frames
            or frac_a80 >= 0.30  # 30%+ frames at 0.80
            or (max_artificial >= 0.90 and frac_a80 >= 0.15)
        ):
            return {
                "label": "suspicious",
                "confidence": max_artificial,
                "reason": "ai_keywords_with_signals",
                "features": features,
            }
    else:
        # Without AI keywords, need stronger signal for suspicious
        if (
            (4 <= count_a90 <= 6 and top3_mean >= 0.93)  # 4-6 frames at 0.90+, top3 avg 0.93+
            or (0.35 <= frac_a90 <= 0.60 and max_artificial >= 0.93)
            or (frac_a80 >= 0.45 and top3_mean >= 0.90)
        ):
            return {
                "label": "suspicious",
                "confidence": max_artificial,
                "reason": "mixed_signal_no_keywords",
                "features": features,
            }

    # Rule 4: Default to real
    # If we got here and didn't hit artificial or suspicious, it's real
    return {
        "label": "real",
        "confidence": max(1.0 - max_artificial, 0.6),  # At least 0.6 confidence
        "reason": "default_real",
        "features": features,
    }


def _apply_heuristics(
    aggregate: Dict[str, Any], heuristics_result: Optional[Dict[str, Any]], client_hints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply heuristics and client hints to aggregate result.
    Less aggressive than before since heuristics are already integrated into decision logic.
    Mainly boosts confidence and adds reasoning, only overrides in extreme cases.
    """
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

        # Less aggressive - only boost confidence and reasons
        # Labels are already influenced by heuristics in _decide_label
        if h_label == "ai-detected" and label == "ai-detected":
            # Both agree it's AI - boost confidence
            confidence = max(confidence, h_conf)

    if client_hints:
        features["client_hints"] = client_hints
        hint_label = client_hints.get("result")
        hint_conf = float(client_hints.get("confidence", 0.0))
        hint_reason = client_hints.get("reason")
        if hint_reason:
            reasons.append(f"client:{hint_reason}")

        # Client hints can still override - user-reported suspicion is important
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

        metadata = payload.get("metadata") or {}

        # Fetch video metadata for heuristics (YouTube) or fall back to client-provided metadata
        heuristics_source: Optional[Dict[str, Any]] = None
        if platform == "youtube" and video_id:
            video_info = get_video_info(video_id)
            heuristics_source = video_info or None
        if not heuristics_source and metadata:
            heuristics_source = _build_metadata_for_heuristics(metadata)

        heuristics_result = check_heuristics(heuristics_source) if heuristics_source else None

        frame_dir = payload.get("frame_dir")
        if not frame_dir:
            raise RuntimeError("Frame directory not provided in job payload")

        frames = _load_saved_frames(frame_dir)
        inference_start = time.perf_counter()
        inference = _call_inference(frames)
        inference_duration = (time.perf_counter() - inference_start) * 1000
        logger.info(
            "Inference completed for job %s with %d frames in %.1f ms",
            job_id,
            len(frames),
            inference_duration,
        )

        # Aggregate with heuristics integrated into decision logic
        aggregate = _aggregate_inference(inference, heuristics_result)

        # Apply remaining heuristics and client hints (less aggressive now)
        merged = _apply_heuristics(aggregate, heuristics_result, client_hints)

        analyzed_at = datetime.now(timezone.utc)
        final_result = {
            "label": merged["label"],
            "confidence": merged["confidence"],
            "reason": merged["reason"],
            "vote_share": aggregate["vote_share"],
            "features": merged["features"],
            "frames_count": len(inference.get("results", [])),
            "batch_time_ms": inference.get("batch_time_ms"),
            "analyzed_at": analyzed_at.isoformat(),
            "model_version": settings.model_version,
            "platform": platform,
            "video_id": video_id,
        }
        logger.info("Deep scan result job_id=%s platform=%s video_id=%s label=%s confidence=%.4f", job_id, platform, video_id, final_result['label'], final_result['confidence'])

        _store_job_status(job_id, "done", result=final_result)

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
