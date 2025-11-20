from __future__ import annotations

from decimal import Decimal
from typing import Optional
from urllib.parse import urlparse, parse_qs


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _clean_video_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    for delimiter in ("?", "&", "#"):
        candidate = candidate.split(delimiter, 1)[0]
    return candidate.strip("/") or None


def extract_platform_and_id(url: str) -> tuple[str, str]:
    if not url:
        raise ValueError("URL is required")

    normalized = url.strip()
    if "://" not in normalized:
        normalized = f"https://{normalized}"

    parsed = urlparse(normalized)
    host = parsed.netloc.lower()
    path = (parsed.path or "").strip("/")

    if "youtube" in host or "youtu.be" in host:
        platform = "youtube"
        video_id: Optional[str] = None

        if "youtu.be" in host:
            video_id = _clean_video_id(path)
        else:
            segments = path.split("/")
            if segments and segments[0] == "shorts" and len(segments) > 1:
                video_id = _clean_video_id(segments[1])
            elif segments and segments[0] == "embed" and len(segments) > 1:
                video_id = _clean_video_id(segments[1])
            else:
                query_map = parse_qs(parsed.query or "")
                video_id = _clean_video_id(query_map.get("v", [None])[0])

        if not video_id:
            raise ValueError("Unable to extract YouTube video ID from URL")
        return platform, video_id

    if "instagram" in host:
        platform = "instagram"
        video_id = None

        segments = path.split("/")
        # Handle /p/{id}/ (Open-Post View)
        if len(segments) >= 2 and segments[0] == "p":
            video_id = _clean_video_id(segments[1])
        # Handle /reels/{id}/ or /reel/{id}/
        elif len(segments) >= 2 and segments[0] in ("reels", "reel"):
            video_id = _clean_video_id(segments[1])

        if not video_id:
            raise ValueError("Unable to extract Instagram video ID from URL")
        return platform, video_id

    raise ValueError("Unsupported platform in URL")


__all__ = ["to_float", "extract_platform_and_id"]

