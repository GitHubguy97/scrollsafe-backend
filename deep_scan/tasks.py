from __future__ import annotations

import json
import logging
import re
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import redis
from celery import shared_task
from google import genai
from google.genai import types as genai_types

from deep_scan.config import settings
from heuristics import check_heuristics
from video_utils import get_video_info

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


def _store_job_status(
    job_id: str,
    status: str,
    *,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
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


def _cleanup_frame_dir(frame_dir: Optional[str]) -> None:
    if not frame_dir:
        return
    try:
        shutil.rmtree(frame_dir, ignore_errors=True)
    except Exception:
        logger.warning("Failed to remove frame directory %s", frame_dir, exc_info=True)


_GEMINI_CLIENT: genai.Client | None = None


def _gemini_client() -> genai.Client:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini deep scan")
    _GEMINI_CLIENT = genai.Client(
        api_key=settings.gemini_api_key,
        http_options={"api_version": settings.gemini_api_version},
    )
    return _GEMINI_CLIENT


def _build_gemini_prompt(frame_count: int) -> str:
    return (
        "You are a forensic visual analyst. You will be given video frames (in order).\n"
        f"There are {frame_count} frames.\n"
        "Task: for EACH frame, output (1) a verdict and (2) a confidence score.\n"
        "Then output ONE short overall summary that synthesizes the evidence across all frames.\n\n"
        'Verdict must be exactly one of: "ai-detected", "real", "suspicious".\n'
        "Confidence must be a number from 0.0 to 1.0.\n\n"
        "Be conservative and filter-aware:\n"
        '- Do NOT classify as "ai-detected" based only on smooth skin, beauty filters, denoise, '
        "compression artifacts, bokeh, cinematic color grading, motion blur, or shallow depth of field.\n"
        '- Use "ai-detected" only when there are clear structural/semantic clues such as impossible anatomy, '
        "warped or unstable text, object merging, identity drift, impossible causality, or scene-logic contradictions.\n"
        '- Evaluate temporal consistency AND semantic/context plausibility together. '
        "A video can be temporally consistent but still synthetic due to implausible context/physics.\n"
        '- If evidence is weak or explainable by filters/compression, prefer "suspicious" over "ai-detected".\n'
        "- If cues are mostly soft visual style cues, cap confidence at 0.7.\n\n"
        "Return a structured response matching this shape (JSON-like is acceptable):\n"
        "{\n"
        '  "frames": [\n'
        '    {"frame": 1, "verdict": "...", "confidence": 0.0, "reason": "max 16 words"},\n'
        "    ...\n"
        f'    {{"frame": {frame_count}, "verdict": "...", "confidence": 0.0, "reason": "max 16 words"}}\n'
        "  ],\n"
        '  "summary": {"overall": "max 140 words"}\n'
        "}\n"
    )


def _parse_gemini_structured_output(raw: str) -> Dict[str, Any]:
    if not raw or not isinstance(raw, str):
        raise ValueError("Gemini returned empty text")

    text = raw.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "frames" in parsed:
                return parsed
        except Exception:
            pass

    if text.startswith('"frames"') or text.startswith("'frames'"):
        try:
            parsed = json.loads("{\n" + text + "\n}")
            if isinstance(parsed, dict) and "frames" in parsed:
                return parsed
        except Exception:
            pass

    frames_section = text
    frames_match = re.search(r'"frames"\s*:\s*\[(.*?)]\s*(?:,|\n|\r|\})', text, flags=re.DOTALL)
    if frames_match:
        frames_section = frames_match.group(1)

    frame_blocks = re.findall(r"\{.*?\}", frames_section, flags=re.DOTALL)
    frames: List[Dict[str, Any]] = []
    for idx, block in enumerate(frame_blocks, start=1):
        frame_num_match = re.search(r'"frame"\s*:\s*(\d+)', block)
        verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', block)
        confidence_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', block)
        reason_match = re.search(r'"reason"\s*:\s*"(.*?)"\s*(?:,|\n\s*\})', block, flags=re.DOTALL)

        if not frame_num_match:
            continue

        frame_entry: Dict[str, Any] = {"frame": int(frame_num_match.group(1))}
        frame_entry["verdict"] = verdict_match.group(1).strip() if verdict_match else "suspicious"
        if confidence_match:
            try:
                frame_entry["confidence"] = float(confidence_match.group(1))
            except ValueError:
                frame_entry["confidence"] = 0.0
        else:
            frame_entry["confidence"] = 0.0
        frame_entry["reason"] = reason_match.group(1).strip() if reason_match else ""
        frames.append(frame_entry)

    summary_overall = ""
    overall_match = re.search(
        r'"summary"\s*:\s*\{.*?"overall"\s*:\s*"(.*?)"\s*\}',
        text,
        flags=re.DOTALL,
    )
    if overall_match:
        summary_overall = overall_match.group(1).strip()

    if not frames:
        raise ValueError("Unable to parse Gemini response into frame results")

    # Normalize ordering to protect downstream deterministic vote behavior.
    frames.sort(key=lambda item: int(item.get("frame", 0)))
    return {"frames": frames, "summary": {"overall": summary_overall}}


def _sanitize_json_like(text: str) -> str:
    sanitized = text.strip()
    sanitized = re.sub(r"^```(?:json)?\s*", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s*```$", "", sanitized)
    sanitized = sanitized.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    # Remove trailing commas before ] or }.
    sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
    return sanitized


def _gemini_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "frame": {"type": "integer"},
                        "verdict": {"type": "string", "enum": ["ai-detected", "real", "suspicious"]},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["frame", "verdict", "confidence", "reason"],
                },
            },
            "summary": {
                "type": "object",
                "properties": {
                    "overall": {"type": "string"},
                },
                "required": ["overall"],
            },
        },
        "required": ["frames", "summary"],
    }


def _extract_response_text(response: Any) -> str:
    raw = getattr(response, "text", None) or ""
    if raw:
        return str(raw)

    # Fallback path for SDK variants where text is not populated.
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        text_parts = [str(getattr(part, "text", "")) for part in parts if getattr(part, "text", None)]
        return "\n".join(text_parts).strip()
    return ""


def _attempt_parse_payload(raw: str) -> Dict[str, Any]:
    sanitized = _sanitize_json_like(raw)
    try:
        parsed = json.loads(sanitized)
        if isinstance(parsed, dict) and "frames" in parsed:
            return parsed
    except Exception:
        pass
    return _parse_gemini_structured_output(sanitized)


def _call_gemini(frames: Sequence[bytes]) -> Dict[str, Any]:
    if not frames:
        raise ValueError("No frames provided to Gemini")

    client = _gemini_client()
    prompt = _build_gemini_prompt(len(frames))
    image_parts = [genai_types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in frames]

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[prompt, *image_parts],
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=1400,
            response_mime_type="application/json",
            response_schema=_gemini_response_schema(),
        ),
    )

    raw = _extract_response_text(response)
    if not raw:
        raise ValueError("Gemini returned empty text")

    try:
        return _attempt_parse_payload(raw)
    except Exception:
        logger.warning("Gemini parse failed on first pass; attempting repair retry")
        repair_prompt = (
            "Convert the following content into valid JSON with this schema only: "
            '{"frames":[{"frame":1,"verdict":"ai-detected|real|suspicious","confidence":0.0,"reason":"..."}],'
            '"summary":{"overall":"..."}}. Return JSON only.\n\n'
            f"CONTENT:\n{raw}"
        )
        repair_response = client.models.generate_content(
            model=settings.gemini_model,
            contents=[repair_prompt],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1400,
                response_mime_type="application/json",
                response_schema=_gemini_response_schema(),
            ),
        )
        repair_raw = _extract_response_text(repair_response)
        if not repair_raw:
            raise ValueError("Gemini repair response was empty")
        return _attempt_parse_payload(repair_raw)


def _aggregate_gemini(gemini_payload: Dict[str, Any], frame_count: int) -> Dict[str, Any]:
    frames = gemini_payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Gemini payload missing 'frames' list")

    normalized: List[Dict[str, Any]] = []
    for idx, entry in enumerate(frames, start=1):
        if not isinstance(entry, dict):
            continue

        verdict = str(entry.get("verdict", "")).strip().lower()
        if verdict not in {"ai-detected", "real", "suspicious"}:
            verdict = "suspicious"
        try:
            conf = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = min(max(conf, 0.0), 1.0)
        reason = str(entry.get("reason", "")).strip()
        frame_raw = entry.get("frame", idx)
        frame_idx = int(frame_raw) if str(frame_raw).isdigit() else idx

        normalized.append(
            {
                "frame": frame_idx,
                "verdict": verdict,
                "confidence": conf,
                "reason": reason[:140],
            }
        )

    if len(normalized) != frame_count:
        logger.warning("Gemini returned %d frame entries for %d frames", len(normalized), frame_count)

    counts = Counter([f["verdict"] for f in normalized]) if normalized else Counter()
    precedence = {"ai-detected": 2, "suspicious": 1, "real": 0}
    chosen = max(counts.items(), key=lambda kv: (kv[1], precedence.get(kv[0], 0)))[0] if counts else "suspicious"

    chosen_confs = [f["confidence"] for f in normalized if f["verdict"] == chosen]
    confidence = sum(chosen_confs) / len(chosen_confs) if chosen_confs else 0.0

    label_map = {"ai-detected": "ai-detected", "suspicious": "suspicious", "real": "verified"}
    external_label = label_map.get(chosen, "suspicious")

    real_votes = float(counts.get("real", 0))
    artificial_votes = float(counts.get("ai-detected", 0))
    total = real_votes + artificial_votes
    vote_share = {"real": (real_votes / total) if total > 0 else 0.5, "artificial": (artificial_votes / total) if total > 0 else 0.5}

    summary = gemini_payload.get("summary") if isinstance(gemini_payload.get("summary"), dict) else {}
    overall = str(summary.get("overall", "")).strip()

    features: Dict[str, Any] = {
        "gemini": {
            "model": settings.gemini_model,
            "api_version": settings.gemini_api_version,
            "frames": normalized,
            "summary": {"overall": overall},
        }
    }

    return {
        "vote_share": vote_share,
        "label": external_label,
        "confidence": confidence,
        "reason": f"gemini: {overall or 'model_vote'}",
        "features": features,
    }


def _apply_heuristics(
    aggregate: Dict[str, Any],
    heuristics_result: Optional[Dict[str, Any]] = None,
    client_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    label = aggregate.get("label", "verified")
    confidence = float(aggregate.get("confidence", 0.0))
    reasons = [aggregate.get("reason", "gemini")]

    features = aggregate.get("features", {}) or {}
    if heuristics_result:
        features["heuristics"] = heuristics_result
        h_label = heuristics_result.get("result")
        h_conf = float(heuristics_result.get("confidence", 0.0))
        h_reason = heuristics_result.get("reason")
        if h_reason:
            reasons.append(f"metadata:{h_reason}")
        if h_label == "ai-detected" and label == "ai-detected":
            confidence = max(confidence, h_conf)

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

    return {
        "label": label,
        "confidence": min(max(confidence, 0.0), 1.0),
        "reason": "; ".join([r for r in reasons if r]),
        "features": features,
    }


@shared_task(name="deep_scan.tasks.process_job", queue=settings.queue_name)
def process_deep_scan_job(job_id: str, payload: Dict[str, Any]) -> None:
    client = _redis_client()
    platform = (payload.get("platform") or "youtube").lower()
    video_id = payload.get("video_id")
    url = payload.get("url")
    client_hints = payload.get("client_hints")
    frame_dir = payload.get("frame_dir")

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

        heuristics_source: Optional[Dict[str, Any]] = None
        if platform == "youtube" and video_id:
            video_info = get_video_info(video_id)
            heuristics_source = video_info or None
        if not heuristics_source and metadata:
            heuristics_source = _build_metadata_for_heuristics(metadata)
        heuristics_result = check_heuristics(heuristics_source) if heuristics_source else None

        if not frame_dir:
            raise RuntimeError("Frame directory not provided in job payload")

        frames = _load_saved_frames(frame_dir)
        inference_start = time.perf_counter()
        gemini_payload: Dict[str, Any]
        try:
            gemini_payload = _call_gemini(frames)
        except Exception:
            logger.exception("Gemini call/parse failed for job %s; using suspicious fallback", job_id)
            gemini_payload = {
                "frames": [],
                "summary": {"overall": "Model response could not be parsed reliably."},
            }
        inference_duration_ms = (time.perf_counter() - inference_start) * 1000
        logger.info(
            "Gemini inference completed for job %s with %d frames in %.1f ms",
            job_id,
            len(frames),
            inference_duration_ms,
        )

        try:
            aggregate = _aggregate_gemini(gemini_payload, frame_count=len(frames))
        except Exception:
            logger.exception("Gemini aggregation failed for job %s; using suspicious fallback", job_id)
            aggregate = {
                "vote_share": {"real": 0.5, "artificial": 0.5},
                "label": "suspicious",
                "confidence": 0.55,
                "reason": "gemini:parse_fallback",
                "features": {
                    "gemini": {
                        "model": settings.gemini_model,
                        "api_version": settings.gemini_api_version,
                        "frames": [],
                        "summary": {"overall": "Model response parsing failed; returned cautious fallback."},
                    }
                },
            }
        merged = _apply_heuristics(aggregate, heuristics_result, client_hints)

        analyzed_at = datetime.now(timezone.utc)
        final_result = {
            "label": merged["label"],
            "confidence": merged["confidence"],
            "reason": merged["reason"],
            "vote_share": aggregate["vote_share"],
            "features": merged["features"],
            "frames_count": len(frames),
            "batch_time_ms": inference_duration_ms,
            "analyzed_at": analyzed_at.isoformat(),
            "model_version": settings.model_version,
            "platform": platform,
            "video_id": video_id,
        }
        logger.info(
            "Deep scan result job_id=%s platform=%s video_id=%s label=%s confidence=%.4f",
            job_id,
            platform,
            video_id,
            final_result["label"],
            final_result["confidence"],
        )

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
        _cleanup_frame_dir(frame_dir)
        client.delete(lock_key)
