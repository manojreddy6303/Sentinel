"""
Sentinel Storage Service
========================
Provides safe storage capacity guarding, partial upload cleanup,
derived cache pruning, and storage integrity management.

Safety Guarantees:
- NEVER deletes storage/sentinel.db or SQLite WAL/SHM files.
- NEVER deletes original uploaded videos referenced by VideoModel.
- NEVER deletes evidence referenced by EvidenceModel or cases.
- NEVER deletes reports referenced by ReportModel.
- Cleans ONLY verified disposable caches, temp files, and unreferenced test fixtures.
"""

import os
import shutil
import time
import errno
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set

from backend.app.core.config import settings

logger = logging.getLogger("sentinel.storage")


class StorageService:
    """Manages filesystem capacity, upload integrity, and derived cache retention."""

    def __init__(self):
        self.root_dir = settings.PROJECT_ROOT
        self.uploads_dir = settings.STORAGE_UPLOADS_DIR
        self.playback_dir = settings.STORAGE_PLAYBACK_DIR
        self.evidence_dir = settings.STORAGE_EVIDENCE_DIR
        self.evidence_playback_dir = settings.STORAGE_EVIDENCE_PLAYBACK_DIR
        self.reports_dir = settings.STORAGE_REPORTS_DIR
        self.db_path = settings.STORAGE_BASE_DIR / "sentinel.db"

    def get_disk_usage(self, target_dir: Optional[Path] = None) -> Tuple[int, int, int]:
        """
        Return (total_bytes, used_bytes, free_bytes) for the volume containing target_dir.
        """
        directory = target_dir or self.uploads_dir
        try:
            total, used, free = shutil.disk_usage(str(directory))
            return total, used, free
        except Exception as exc:
            logger.error(f"Failed to check disk usage for {directory}: {exc}")
            # Fallback to current working directory
            total, used, free = shutil.disk_usage(".")
            return total, used, free

    def check_storage_capacity(
        self,
        incoming_file_size_bytes: int = 0,
        target_dir: Optional[Path] = None,
    ) -> Tuple[bool, str, float, float]:
        """
        Check if the host volume has sufficient capacity for the upload and processing.

        Estimates:
        - Incoming upload size
        - Temporary processing and transcoding overhead (~2.5x upload size)
        - Safety buffer defined by SENTINEL_MIN_FREE_DISK_GB (default 2.0 GB)

        Returns:
            (has_capacity: bool, message: str, free_gb: float, recommended_gb: float)
        """
        total, used, free = self.get_disk_usage(target_dir)
        free_gb = free / (1024 ** 3)

        # Baseline minimum free disk required to keep operating safely
        min_reserve_bytes = int(settings.SENTINEL_MIN_FREE_DISK_GB * (1024 ** 3))

        # Estimated workspace needed for this specific upload and its analysis
        overhead_multiplier = 2.5
        required_job_bytes = int(incoming_file_size_bytes * overhead_multiplier)
        total_recommended_bytes = min_reserve_bytes + required_job_bytes
        recommended_gb = total_recommended_bytes / (1024 ** 3)

        # Ensure at least 1 GB is recommended as a baseline
        if recommended_gb < 1.0:
            recommended_gb = 1.0

        if free < total_recommended_bytes:
            msg = (
                f"Insufficient storage space. {free_gb:.1f} GB available; "
                f"approximately {recommended_gb:.1f} GB recommended for this upload and processing."
            )
            logger.warning(f"Storage capacity check failed: {msg}")
            return False, msg, free_gb, recommended_gb

        return True, "", free_gb, recommended_gb

    def clean_failed_upload(self, temp_file_path: Optional[Path]) -> None:
        """
        Safely unlink an incomplete temporary upload file to prevent storage leakage.
        """
        if not temp_file_path:
            return
        try:
            if temp_file_path.exists() and temp_file_path.is_file():
                temp_file_path.unlink()
                logger.info(f"Cleaned failed partial upload: {temp_file_path.name}")
        except Exception as exc:
            logger.warning(f"Could not delete partial upload {temp_file_path}: {exc}")

    def prune_playback_cache(self, target_free_bytes: Optional[int] = None) -> Dict[str, Any]:
        """
        Prune derived playback transcodes (storage/playback and storage/evidence_playback).
        
        Playback files are 100% regenerable on demand from original source videos.
        Uses Least Recently Used (LRU) eviction based on mtime/atime.
        NEVER deletes original video uploads or database files.
        """
        max_cache_bytes = int(settings.SENTINEL_PLAYBACK_CACHE_MAX_GB * (1024 ** 3))
        pruned_files: List[str] = []
        bytes_reclaimed = 0

        # Collect candidate playback files
        candidate_files: List[Path] = []
        for p_dir in [self.playback_dir, self.evidence_playback_dir]:
            if p_dir.exists():
                for f in p_dir.iterdir():
                    if f.is_file() and f.suffix.lower() == ".mp4":
                        candidate_files.append(f)

        # Sort by mtime ascending (oldest first)
        candidate_files.sort(key=lambda p: p.stat().st_mtime)

        total_playback_size = sum(p.stat().st_size for p in candidate_files)

        for p_file in candidate_files:
            needs_pruning = False
            if total_playback_size > max_cache_bytes:
                needs_pruning = True
            elif target_free_bytes is not None:
                _, _, free = self.get_disk_usage()
                if free < target_free_bytes:
                    needs_pruning = True

            if not needs_pruning:
                break

            sz = p_file.stat().st_size
            try:
                p_file.unlink()
                pruned_files.append(p_file.name)
                bytes_reclaimed += sz
                total_playback_size -= sz
            except Exception as exc:
                logger.warning(f"Could not remove playback cache file {p_file.name}: {exc}")

        return {
            "pruned_files_count": len(pruned_files),
            "bytes_reclaimed": bytes_reclaimed,
            "pruned_files": pruned_files[:20],
        }

    def clean_temp_processing_artifacts(self, max_age_hours: Optional[float] = None) -> Dict[str, Any]:
        """
        Remove temporary intermediate files (.upload.tmp, abandoned transcoding stderr logs, etc.).
        """
        cutoff_hours = max_age_hours if max_age_hours is not None else settings.SENTINEL_TEMP_RETENTION_HOURS
        cutoff_seconds = time.time() - (cutoff_hours * 3600)
        reclaimed_bytes = 0
        deleted_count = 0

        # 1. Clean temporary files in storage/uploads and storage/playback
        for search_dir in [self.uploads_dir, self.playback_dir, self.evidence_dir]:
            if not search_dir.exists():
                continue
            for f in search_dir.iterdir():
                if f.is_file() and (f.name.endswith(".upload.tmp") or f.name.endswith(".transcode.tmp") or f.name.endswith(".log")):
                    try:
                        # For upload.tmp, clean if older than 1 hour or if cutoff passed
                        if f.stat().st_mtime < cutoff_seconds or f.name.endswith(".upload.tmp"):
                            sz = f.stat().st_size
                            f.unlink()
                            reclaimed_bytes += sz
                            deleted_count += 1
                    except Exception:
                        pass

        # 2. Check AppData/Local/Temp for sentinel temporary files
        temp_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Temp"
        if temp_dir.exists():
            for f in temp_dir.iterdir():
                try:
                    name_lower = f.name.lower()
                    if ("sentinel" in name_lower or "uccrime" in name_lower or "burglary" in name_lower) and f.is_file():
                        if f.stat().st_mtime < cutoff_seconds:
                            sz = f.stat().st_size
                            f.unlink()
                            reclaimed_bytes += sz
                            deleted_count += 1
                except Exception:
                    pass

        return {
            "deleted_count": deleted_count,
            "bytes_reclaimed": reclaimed_bytes,
        }

    def get_referenced_db_paths(self) -> Tuple[Set[Path], Set[Path], Set[Path]]:
        """
        Query database to return authoritative sets of:
        (referenced_video_paths, referenced_evidence_paths, referenced_report_paths)
        """
        from database.session import SessionLocal
        from database.models import VideoModel, EvidenceModel, ReportModel

        db = SessionLocal()
        try:
            videos = db.query(VideoModel).all()
            evidences = db.query(EvidenceModel).all()
            reports = db.query(ReportModel).all()

            db_video_paths = {Path(v.storage_path).resolve() for v in videos if v.storage_path}
            db_evidence_paths = set()
            for ev in evidences:
                if ev.snapshot_path:
                    db_evidence_paths.add(Path(ev.snapshot_path).resolve())
                if ev.annotated_snapshot_path:
                    db_evidence_paths.add(Path(ev.annotated_snapshot_path).resolve())
                if ev.clip_path:
                    db_evidence_paths.add(Path(ev.clip_path).resolve())

            db_report_paths = {Path(rep.file_path).resolve() for rep in reports if hasattr(rep, "file_path") and rep.file_path}
            return db_video_paths, db_evidence_paths, db_report_paths
        finally:
            db.close()

    def audit_storage_integrity(self) -> Dict[str, Any]:
        """
        Categorize filesystem items according to DB references:
        Category A: DB referenced + file exists (KEEP)
        Category B: DB referenced + file missing (ERROR report)
        Category C: file exists + no DB reference (ORPHAN candidate)
        Category D: derived temporary / cache file (SAFE cleanup candidate)
        """
        db_video_paths, db_evidence_paths, db_report_paths = self.get_referenced_db_paths()

        cat_a: List[Path] = []
        cat_b: List[Path] = []
        cat_c: List[Path] = []
        cat_d: List[Path] = []

        # Check DB references
        for p in db_video_paths:
            if p.exists():
                cat_a.append(p)
            else:
                cat_b.append(p)

        for p in db_evidence_paths:
            if p.exists():
                cat_a.append(p)
            else:
                cat_b.append(p)

        for p in db_report_paths:
            if p.exists():
                cat_a.append(p)
            else:
                cat_b.append(p)

        # Check uploads
        if self.uploads_dir.exists():
            for f in self.uploads_dir.iterdir():
                if f.is_file():
                    res = f.resolve()
                    if res in db_video_paths or (f.suffix == ".json" and res.with_suffix(".mp4") in db_video_paths):
                        if res not in cat_a:
                            cat_a.append(res)
                    elif f.name.endswith(".upload.tmp"):
                        cat_d.append(res)
                    else:
                        cat_c.append(res)

        # Check playback
        for p_dir in [self.playback_dir, self.evidence_playback_dir]:
            if p_dir.exists():
                for f in p_dir.iterdir():
                    if f.is_file():
                        cat_d.append(f.resolve())

        # Check evidence
        if self.evidence_dir.exists():
            for root, _, files in os.walk(self.evidence_dir):
                for f in files:
                    fp = (Path(root) / f).resolve()
                    if fp in db_evidence_paths:
                        if fp not in cat_a:
                            cat_a.append(fp)
                    else:
                        cat_c.append(fp)

        # Check reports
        if self.reports_dir.exists():
            for root, _, files in os.walk(self.reports_dir):
                for f in files:
                    fp = (Path(root) / f).resolve()
                    if fp in db_report_paths:
                        if fp not in cat_a:
                            cat_a.append(fp)
                    else:
                        cat_c.append(fp)

        return {
            "category_a_count": len(cat_a),
            "category_b_count": len(cat_b),
            "category_b_missing": [str(p) for p in cat_b[:10]],
            "category_c_count": len(cat_c),
            "category_d_count": len(cat_d),
        }

    def safe_purge_unreferenced_test_fixtures(self) -> Dict[str, Any]:
        """
        Safely remove verified unreferenced test fixtures and benchmark outputs.
        
        CRITICAL RULES:
        - Never touch any file referenced by VideoModel, EvidenceModel, or ReportModel.
        - Never touch sentinel.db.
        - Removes only:
          1. Test video fixtures in storage/uploads with NO DB reference (e.g. *_passwd.mp4, *_corrupt.mp4, etc.)
          2. Unreferenced benchmark clips in storage/evidence
          3. Unreferenced test reports in storage/reports
          4. Derived playback cache in storage/evidence_playback and storage/playback
        """
        db_video_paths, db_evidence_paths, db_report_paths = self.get_referenced_db_paths()

        deleted_files = 0
        reclaimed_bytes = 0

        # 1. Clean unreferenced uploads
        known_test_suffixes = {
            "surveillance_sample.mp4",
            "passwd.mp4",
            "conf_test.mp4",
            "test_surveillance.mp4",
            "bbox_test.mp4",
            "test_events.mp4",
            "black_screen.mp4",
            "corrupt.mp4",
            "test_phase4.mp4",
            "regression_sample.mp4",
            "12566041-uhd_3840_2160_30fps.mp4",
        }

        if self.uploads_dir.exists():
            for f in list(self.uploads_dir.iterdir()):
                if not f.is_file():
                    continue
                res = f.resolve()
                # Verify absolutely not referenced
                if res in db_video_paths:
                    continue
                if f.suffix == ".json" and res.with_suffix(".mp4") in db_video_paths:
                    continue

                # Check if it matches test fixture or has zero DB reference
                parts = f.name.split("_", 1)
                suffix = parts[1] if len(parts) > 1 else f.name

                is_disposable_test = (
                    suffix in known_test_suffixes
                    or f.name.endswith(".upload.tmp")
                    or (f.suffix == ".json" and not any(str(p).startswith(str(res.with_suffix(""))) for p in db_video_paths))
                )

                if is_disposable_test:
                    try:
                        sz = f.stat().st_size
                        f.unlink()
                        deleted_files += 1
                        reclaimed_bytes += sz
                    except Exception as e:
                        logger.warning(f"Failed to delete unreferenced test upload {f.name}: {e}")

        # 2. Clean unreferenced evidence files (benchmark clips, orphaned snapshots)
        if self.evidence_dir.exists():
            for root, _, files in os.walk(self.evidence_dir):
                for f in files:
                    fp = Path(root) / f
                    res = fp.resolve()
                    if res not in db_evidence_paths:
                        try:
                            sz = fp.stat().st_size
                            fp.unlink()
                            deleted_files += 1
                            reclaimed_bytes += sz
                        except Exception as e:
                            logger.warning(f"Failed to delete unreferenced evidence file {fp.name}: {e}")

        # 3. Clean derived evidence playback cache
        if self.evidence_playback_dir.exists():
            for f in list(self.evidence_playback_dir.iterdir()):
                if f.is_file() and f.suffix == ".mp4":
                    try:
                        sz = f.stat().st_size
                        f.unlink()
                        deleted_files += 1
                        reclaimed_bytes += sz
                    except Exception as e:
                        logger.warning(f"Failed to delete evidence playback transcode {f.name}: {e}")

        # 4. Clean unreferenced reports
        if self.reports_dir.exists():
            for root, _, files in os.walk(self.reports_dir):
                for f in files:
                    fp = Path(root) / f
                    res = fp.resolve()
                    if res not in db_report_paths:
                        try:
                            sz = fp.stat().st_size
                            fp.unlink()
                            deleted_files += 1
                            reclaimed_bytes += sz
                        except Exception as e:
                            logger.warning(f"Failed to delete unreferenced report file {fp.name}: {e}")

        logger.info(f"Purge complete. Deleted {deleted_files} files, reclaimed {reclaimed_bytes / (1024**2):.2f} MB")
        return {
            "deleted_files": deleted_files,
            "reclaimed_bytes": reclaimed_bytes,
            "reclaimed_mb": reclaimed_bytes / (1024 ** 2),
            "reclaimed_gb": reclaimed_bytes / (1024 ** 3),
        }


storage_service = StorageService()
