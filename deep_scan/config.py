from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables BEFORE reading them
# Use absolute path to ensure .env is found regardless of working directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float") from exc


@dataclass(frozen=True)
class Settings:
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
    redis_url: str = os.getenv("REDIS_APP_URL", "redis://127.0.0.1:6379/1")
    redis_job_ttl_seconds: int = _int_env("DEEP_SCAN_RESULT_TTL_SECONDS", 900)
    redis_lock_ttl_seconds: int = _int_env("DEEP_SCAN_LOCK_TTL_SECONDS", 300)
    queue_name: str = os.getenv("DEEP_SCAN_QUEUE", "deep_scan")

    # Optional legacy resolver URL (unused when frames come from extension)
    resolver_url: str = os.getenv("DEEPSCAN_RESOLVER_URL", "")

    inference_url: str = os.getenv(
        "INFER_API_URL",
        "https://chkwk82q35esq5v4.us-east-1.aws.endpoints.huggingface.cloud",
    )
    hf_token: str = os.getenv("HUGGING_FACE_API_KEY", "")
    inference_api_key: str = os.getenv("INFER_API_KEY", "")
    target_frames: int = _int_env("DEEP_SCAN_TARGET_FRAMES", _int_env("INFER_TARGET_FRAMES", 8))
    frame_extract_timeout: int = _int_env("DEEP_SCAN_FRAME_TIMEOUT", 180)
    inference_timeout: float = _float_env("DEEP_SCAN_INFER_TIMEOUT", _float_env("INFER_REQUEST_TIMEOUT", 20.0))
    model_version: str = os.getenv("DEEP_SCAN_MODEL_VERSION", "deep_v1")
    log_level: str = os.getenv("DEEP_SCAN_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
    storage_dir: str = os.getenv(
        "DEEP_SCAN_STORAGE_DIR",
        str((Path(__file__).parent.parent / "out" / "deep_scans").resolve()),
    )


settings = Settings()
