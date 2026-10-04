"""
Application Configuration Settings
"""
import os
from pathlib import Path
from typing import List, Set, Dict, Optional


def _load_env_file():
    """Load key-value pairs from root .env file into os.environ if not already set."""
    root = Path(__file__).resolve().parent.parent.parent.parent
    env_file = root / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


_load_env_file()


class Settings:
    """Application settings read from environment variables with sensible defaults."""

    PROJECT_NAME: str = "Sentinel — AI-Powered Security Video Investigation Platform"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"

    # Root paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Server binding
    HOST: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

    # CORS
    @property
    def CORS_ORIGINS(self) -> List[str]:
        default_origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://sentinel-frontend-production-8154.up.railway.app",
        ]
        raw = os.getenv("CORS_ORIGINS") or os.getenv("BACKEND_CORS_ORIGINS")
        if not raw:
            return default_origins
        raw = raw.strip()
        if raw == "*":
            return ["*"]
        if raw.startswith("[") and raw.endswith("]"):
            import json
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if item]
            except Exception:
                pass
        custom = [o.strip() for o in raw.split(",") if o.strip()]
        combined = list(default_origins)
        for c in custom:
            if c not in combined:
                combined.append(c)
        return combined

    # Storage Paths
    @property
    def STORAGE_BASE_DIR(self) -> Path:
        env_val = os.getenv("STORAGE_BASE_DIR", "./storage")
        p = Path(env_val)
        return p if p.is_absolute() else (self.PROJECT_ROOT / p).resolve()

    @property
    def STORAGE_DIR(self) -> Path:
        return self.STORAGE_BASE_DIR

    @property
    def STORAGE_UPLOADS_DIR(self) -> Path:
        env_val = os.getenv("STORAGE_UPLOADS_DIR", "./storage/uploads")
        p = Path(env_val)
        target = p if p.is_absolute() else (self.PROJECT_ROOT / p).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def STORAGE_EVIDENCE_DIR(self) -> Path:
        env_val = os.getenv("STORAGE_EVIDENCE_DIR", "./storage/evidence")
        p = Path(env_val)
        target = p if p.is_absolute() else (self.PROJECT_ROOT / p).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def STORAGE_REPORTS_DIR(self) -> Path:
        env_val = os.getenv("STORAGE_REPORTS_DIR", "./storage/reports")
        p = Path(env_val)
        target = p if p.is_absolute() else (self.PROJECT_ROOT / p).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def STORAGE_EVENTS_DIR(self) -> Path:
        """Directory for JSON detection event files (one per video)."""
        env_val = os.getenv("STORAGE_EVENTS_DIR", "./storage/events")
        p = Path(env_val)
        target = p if p.is_absolute() else (self.PROJECT_ROOT / p).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def STORAGE_PLAYBACK_DIR(self) -> Path:
        """Directory for browser-compatible converted playback video files."""
        env_val = os.getenv("STORAGE_PLAYBACK_DIR", "./storage/playback")
        p = Path(env_val)
        target = p if p.is_absolute() else (self.PROJECT_ROOT / p).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def STORAGE_EVIDENCE_PLAYBACK_DIR(self) -> Path:
        """Directory for browser-compatible converted evidence video clips."""
        env_val = os.getenv("STORAGE_EVIDENCE_PLAYBACK_DIR", "./storage/evidence_playback")
        p = Path(env_val)
        target = p if p.is_absolute() else (self.PROJECT_ROOT / p).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    # Video Upload Configuration
    ALLOWED_VIDEO_EXTENSIONS: Set[str] = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
    }

    ALLOWED_VIDEO_MIME_TYPES: Set[str] = {
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/avi",
        "video/msvideo",
        "video/x-matroska",
        "video/webm",
        "application/octet-stream",  # Fallback for some browsers, extension validation applies
    }

    # Large video upload configuration
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))

    @property
    def MAX_UPLOAD_SIZE_BYTES(self) -> int:
        env_val = os.getenv("MAX_UPLOAD_SIZE_BYTES")
        if env_val:
            return int(env_val)
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # -------------------------------------------------------------------------
    # Phase 3: Video Intelligence Pipeline Configuration
    # -------------------------------------------------------------------------

    # YOLO model to load. "yolov8n.pt" = YOLOv8 Nano (~6MB), lightweight for CPU.
    YOLO_MODEL_NAME: str = os.getenv("YOLO_MODEL_NAME", "yolov8n.pt")

    # Minimum confidence threshold for accepting a YOLO detection [0.0, 1.0].
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.25"))

    # Phase 20: Candidate generation threshold passed to model inference before Sentinel validation
    YOLO_CANDIDATE_CONFIDENCE: float = float(os.getenv("YOLO_CANDIDATE_CONFIDENCE", "0.20"))

    # Optional per-class minimum confidence overrides [0.0, 1.0].
    CLASS_CONFIDENCE_THRESHOLDS: Dict[str, float] = {}

    # Frame sampling rate: number of frames sampled per second of video.
    # 1.0 = 1 frame per second. Lower values = fewer frames = faster processing.
    VIDEO_SAMPLE_RATE_FPS: float = float(os.getenv("VIDEO_SAMPLE_RATE_FPS", "1.0"))

    # Surveillance-relevant COCO classes to report. Only classes supported
    # by the selected pretrained YOLO model are included.
    SURVEILLANCE_CLASSES: Set[str] = {
        "person", "bicycle", "car", "motorcycle", "bus", "truck", "boat", "train", "airplane",
        "backpack", "handbag", "suitcase", "bottle", "cell phone", "laptop", "mouse",
        "remote", "keyboard", "book", "clock", "vase", "scissors", "umbrella", "cup",
        "chair", "bench", "couch", "bed", "dining table", "traffic light", "stop sign", "fire hydrant",
        "general_object", "package", "box", "merchandise", "unknown_portable_object",
    }

    # -------------------------------------------------------------------------
    # Phase 7: LLM-Assisted Video Investigation Configuration
    # -------------------------------------------------------------------------
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # -------------------------------------------------------------------------
    # Phase 8.1: Security Behavior & Theft Detection Configuration
    # -------------------------------------------------------------------------
    THEFT_MIN_INTERACTION_SECONDS: float = float(os.getenv("THEFT_MIN_INTERACTION_SECONDS", "2.0"))
    THEFT_INTERACTION_MAX_DISTANCE: float = float(os.getenv("THEFT_INTERACTION_MAX_DISTANCE", "120.0"))
    THEFT_MIN_DEPARTURE_DISTANCE: float = float(os.getenv("THEFT_MIN_DEPARTURE_DISTANCE", "45.0"))
    THEFT_OBJECT_LOSS_WINDOW: float = float(os.getenv("THEFT_OBJECT_LOSS_WINDOW", "4.0"))
    THEFT_TARGET_CLASSES: Set[str] = {
        "backpack", "handbag", "suitcase", "laptop", "cell phone",
        "bottle", "umbrella", "bicycle", "box", "package", "book",
        "cup", "vase", "scissors", "clock", "remote", "merchandise", "general_object", "unknown_portable_object",
    }

    # -------------------------------------------------------------------------
    # Phase 15: Specialized Visual Detection Core Configuration
    # -------------------------------------------------------------------------
    SPECIALIZED_VISUAL_ENABLED: bool = os.getenv("SPECIALIZED_VISUAL_ENABLED", "1").strip().lower() in ("1", "true", "yes")
    FIRE_DETECTION_ENABLED: bool = os.getenv("FIRE_DETECTION_ENABLED", "1").strip().lower() in ("1", "true", "yes")
    SMOKE_DETECTION_ENABLED: bool = os.getenv("SMOKE_DETECTION_ENABLED", "1").strip().lower() in ("1", "true", "yes")
    WEAPON_DETECTION_ENABLED: bool = os.getenv("WEAPON_DETECTION_ENABLED", "1").strip().lower() in ("1", "true", "yes")
    # -------------------------------------------------------------------------
    # Phase 0.2: Generic Micro-Object & Interaction Recovery Configuration
    # -------------------------------------------------------------------------
    MICRO_OBJECT_RECOVERY_ENABLED: bool = os.getenv("MICRO_OBJECT_RECOVERY_ENABLED", "1").strip().lower() in ("1", "true", "yes")

    # -------------------------------------------------------------------------
    # Phase 20.3: Production Memory Hardening Configuration
    # -------------------------------------------------------------------------
    # Inference batch size for YOLO object detection.
    # Default = 4 (memory-safe for CPU container limits, down from 32).
    YOLO_BATCH_SIZE: int = int(os.getenv("YOLO_BATCH_SIZE", "4"))

    # Maximum memory budget (MB) for bounded frame caching.
    # Default = 64.0 MB (replaces uncalibrated 512.0 MB budget).
    FRAME_CACHE_MAX_MEMORY_MB: float = float(os.getenv("FRAME_CACHE_MAX_MEMORY_MB", "64.0"))

    # Concurrency limit for heavy video analysis per worker.
    # Default = 1 (enforces sequential heavy video processing to prevent OOM spikes).
    MAX_CONCURRENT_HEAVY_JOBS: int = int(os.getenv("MAX_CONCURRENT_HEAVY_JOBS", "1"))

    # PyTorch and OpenMP CPU thread limit.
    # Default = 2 (prevents thread-local memory explosion on multi-core host nodes).
    TORCH_NUM_THREADS: int = int(os.getenv("TORCH_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "2")))

    # -------------------------------------------------------------------------
    # Phase 21A: Advanced Tracking Configuration
    # -------------------------------------------------------------------------
    # Tracker selection: "bytetrack" (production default) or "botsort" (Phase 21A candidate).
    # BoT-SORT is NOT the production default — it remains a benchmark candidate.
    TRACKER_TYPE: str = os.getenv("TRACKER_TYPE", "bytetrack")

    # -------------------------------------------------------------------------
    # Phase 21D: Sampling-Aware Incident Correlation Configuration
    # -------------------------------------------------------------------------
    # Controls whether the incident correlation layer uses sampling-density-aware
    # temporal episode continuity (True) or legacy fixed-interval correlation (False).
    # Default = False (preserves exact 1.0 FPS production baseline behavior).
    SAMPLING_AWARE_CORRELATION_ENABLED: bool = os.getenv(
        "SAMPLING_AWARE_CORRELATION_ENABLED", "0"
    ).strip().lower() in ("1", "true", "yes")
    CORRELATION_TEMPORAL_COOLDOWN_SECONDS: float = float(
        os.getenv("CORRELATION_TEMPORAL_COOLDOWN_SECONDS", "5.0")
    )


settings = Settings()

