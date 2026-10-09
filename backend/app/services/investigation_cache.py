"""
Sentinel Derived Data Cache for High-Performance Investigation
Phase 4: Thread-safe, bounded, per-video caching of immutable derived analytics.

Caches computationally heavy derived intelligence:
- Reconciled canonical entities
- Derived color summaries
- Pre-calculated concurrency values
- Grouped events and timeline summaries

Cache keys MUST be strictly scoped by video_id and analysis/update token.
Invalidation occurs automatically on video reprocessing or explicit cache clearing.
Strict cross-video isolation is enforced.
"""

import threading
import time
from typing import Dict, Any, List, Optional, Tuple


class InvestigationDerivedDataCache:
    """In-memory bounded cache for video-derived investigation structures."""

    _lock = threading.Lock()
    _canonical_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
    _summary_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    _max_entries_per_cache: int = 100
    _default_ttl_seconds: float = 3600.0  # 1 hour

    @classmethod
    def get_canonical_entities(cls, video_id: str, token: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached reconciled canonical entities if token matches and not expired."""
        if not video_id:
            return None
        key = f"{video_id}:{token}"
        with cls._lock:
            entry = cls._canonical_cache.get(key)
            if entry is None:
                return None
            ts, data = entry
            if time.time() - ts > cls._default_ttl_seconds:
                cls._canonical_cache.pop(key, None)
                return None
            return [dict(item) for item in data]

    @classmethod
    def set_canonical_entities(cls, video_id: str, token: str, entities: List[Dict[str, Any]]) -> None:
        """Store reconciled canonical entities under video_id and token."""
        if not video_id:
            return
        key = f"{video_id}:{token}"
        with cls._lock:
            # Enforce capacity
            if len(cls._canonical_cache) >= cls._max_entries_per_cache:
                oldest_key = min(cls._canonical_cache.keys(), key=lambda k: cls._canonical_cache[k][0])
                cls._canonical_cache.pop(oldest_key, None)
            cls._canonical_cache[key] = (time.time(), [dict(item) for item in entities])

    @classmethod
    def get_derived_summary(cls, video_id: str, token: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached video-level summary data."""
        if not video_id:
            return None
        key = f"{video_id}:{token}"
        with cls._lock:
            entry = cls._summary_cache.get(key)
            if entry is None:
                return None
            ts, data = entry
            if time.time() - ts > cls._default_ttl_seconds:
                cls._summary_cache.pop(key, None)
                return None
            return dict(data)

    @classmethod
    def set_derived_summary(cls, video_id: str, token: str, summary: Dict[str, Any]) -> None:
        """Store video-level summary data."""
        if not video_id:
            return
        key = f"{video_id}:{token}"
        with cls._lock:
            if len(cls._summary_cache) >= cls._max_entries_per_cache:
                oldest_key = min(cls._summary_cache.keys(), key=lambda k: cls._summary_cache[k][0])
                cls._summary_cache.pop(oldest_key, None)
            cls._summary_cache[key] = (time.time(), dict(summary))

    @classmethod
    def invalidate_video(cls, video_id: str) -> None:
        """Evict all cached entries for a given video ID on reprocessing or update."""
        if not video_id:
            return
        prefix = f"{video_id}:"
        with cls._lock:
            c_keys = [k for k in cls._canonical_cache if k.startswith(prefix)]
            for k in c_keys:
                cls._canonical_cache.pop(k, None)
            s_keys = [k for k in cls._summary_cache if k.startswith(prefix)]
            for k in s_keys:
                cls._summary_cache.pop(k, None)

    @classmethod
    def clear(cls) -> None:
        """Evict all entries across all caches."""
        with cls._lock:
            cls._canonical_cache.clear()
            cls._summary_cache.clear()
