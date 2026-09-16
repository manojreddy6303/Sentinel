"""
scripts/purge_test_fixtures.py — Phase 19.2 DB Cleanup (v4 - FINAL)

Strategy (most conservative that achieves the cleanup goal):
  1. ALWAYS protect: any video_id referenced by CaseVideoModel
  2. ALWAYS protect: known real-world filenames (no UUID, not test-pattern)
  3. DELETE: pure-UUID filenames that are NOT case-linked
  4. DELETE: filenames matching test patterns (even if unique)
  5. For non-UUID, non-test filenames that are duplicated: keep only FIRST upload

Result: The DB will contain only case-linked videos + one canonical copy of
each non-test, non-UUID filename. This collapses 2208 -> ~50-80 videos.

Usage:
  python -m scripts.purge_test_fixtures          # dry-run
  python -m scripts.purge_test_fixtures --execute # live purge
"""
import sys
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.session import SessionLocal
from database.models import (
    VideoModel, EventModel, EvidenceModel, CorrelatedIncidentModel,
    ReportModel, SurveillanceSessionModel, CameraSourceModel,
    CaseModel, CaseVideoModel,
)

TEMPLATE_CASE_TITLES = {
    "Warehouse Perimeter Breach Investigation",
    "Vehicle Theft Case",
    "Crowd Loitering Review",
    "Multi-Video Investigation",
}

# Patterns that indicate a filename is a test fixture
TEST_ONLY_RE = re.compile(
    r"^(test_|mock_|stale_test|cctv_test|surveillance_phase|bbox_test|conf_test|"
    r"black_screen|test_phase|test_events|test_surveillance|test_reproc_|test_inv_|"
    r"vid_inv_|vid_rep_|vid_iso_|vid_filt_|vid_bwd_|vid_va_|vid_th_|vid_emp_|vid_dup_|"
    r"camera_north|camera_south|test_mc_|theft_test|vid_a_test|vid_b_test|"
    r"api_test_|security_cam_test|cctv_entrance_test|duplicate_test|"
    r"large_timeline|cam_a\.mp4|cam_b\.mp4|cam_c\.mp4|cam1_gate|cam2_lobby|"
    r"cam_mc_|cam_phase|test_case_|theft\.mp4$)",
    re.IGNORECASE,
)


def is_uuid_filename(filename: str) -> bool:
    """Return True if the filename is a raw UUID (all-hex + dashes, no meaningful name)."""
    base = re.sub(r'\.(mp4|avi|mkv|mov|webm)$', '', filename, flags=re.IGNORECASE)
    # UUID pattern: 8-4-4-4-12 hex chars with dashes
    return bool(re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', base))


def run_purge(dry_run: bool = True):
    db = SessionLocal()
    try:
        # ── 1. Gather all case-linked video IDs ────────────────────────────────
        case_linked_ids = {link.video_id for link in db.query(CaseVideoModel).all()}

        # ── 2. Load all videos sorted oldest-first ─────────────────────────────
        all_videos = db.query(VideoModel).order_by(VideoModel.uploaded_at.asc()).all()

        # ── 3. Track first-seen non-UUID, non-test filenames ──────────────────
        first_canonical: dict = {}  # filename -> video_id (first-seen canonical copy)
        for v in all_videos:
            n = v.original_filename
            if not is_uuid_filename(n) and not TEST_ONLY_RE.match(n):
                if n not in first_canonical:
                    first_canonical[n] = v.id

        # ── 4. Classify ────────────────────────────────────────────────────────
        to_keep: list = []
        to_delete: list = []

        for v in all_videos:
            n = v.original_filename

            # Rule 1: case-linked → always keep
            if v.id in case_linked_ids:
                to_keep.append(v)
                continue

            # Rule 2: pure UUID filename → delete (all from tests)
            if is_uuid_filename(n):
                to_delete.append(v)
                continue

            # Rule 3: test-pattern filename → delete
            if TEST_ONLY_RE.match(n):
                to_delete.append(v)
                continue

            # Rule 4: real filename but duplicate → keep only canonical
            if first_canonical.get(n) == v.id:
                to_keep.append(v)
            else:
                to_delete.append(v)

        # ── 5. Print manifest ──────────────────────────────────────────────────
        print("=" * 72)
        print("SENTINEL PHASE 19.2 — TEST FIXTURE PURGE MANIFEST (v4 FINAL)")
        print("=" * 72)
        print(f"Total videos in DB:  {len(all_videos)}")
        print(f"Videos to PRESERVE:  {len(to_keep)}")
        print(f"Videos to DELETE:    {len(to_delete)}")
        print()

        print("--- PRESERVED VIDEOS ---")
        for v in sorted(to_keep, key=lambda x: x.original_filename):
            ev_count = db.query(EventModel).filter(EventModel.video_id == v.id).count()
            label = " [CASE-LINKED]" if v.id in case_linked_ids else ""
            print(f"  KEEP  {v.original_filename[:62]:<64} ev={ev_count}{label}")

        print()
        t_ev = t_evid = t_inc = 0
        print("--- FIRST 30 TO DELETE ---")
        for v in to_delete[:30]:
            ev = db.query(EventModel).filter(EventModel.video_id == v.id).count()
            evid = db.query(EvidenceModel).filter(EvidenceModel.video_id == v.id).count()
            inc = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.video_id == v.id).count()
            t_ev += ev; t_evid += evid; t_inc += inc
            print(f"  DEL   {v.original_filename[:60]:<62} ev={ev} evid={evid} inc={inc}")

        remaining = to_delete[30:]
        if remaining:
            for v in remaining:
                t_ev += db.query(EventModel).filter(EventModel.video_id == v.id).count()
                t_evid += db.query(EvidenceModel).filter(EvidenceModel.video_id == v.id).count()
                t_inc += db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.video_id == v.id).count()
            print(f"  ... and {len(remaining)} more")

        print()
        print(f"Cascade children to remove: events={t_ev} evidence={t_evid} incidents={t_inc}")

        if dry_run:
            print()
            print("=" * 72)
            print("DRY RUN — no changes made. Re-run with --execute to apply.")
            print("=" * 72)
            return

        # ── 6. Execute ─────────────────────────────────────────────────────────
        print("\nEXECUTING...")
        deleted_v = 0
        for v in to_delete:
            db.delete(v)
            deleted_v += 1
        db.flush()

        # Orphaned test sessions
        test_sess_kw = ["test surveillance", "campus site alpha", "phase 19 multi-cam topology test"]
        del_sess = 0
        for s in db.query(SurveillanceSessionModel).all():
            cam_ct = db.query(CameraSourceModel).filter(CameraSourceModel.session_id == s.id).count()
            if cam_ct == 0 and any(kw in (s.name or "").lower() for kw in test_sess_kw):
                db.delete(s)
                del_sess += 1

        # Template cases with 0 links
        del_cases = 0
        for c in db.query(CaseModel).all():
            if (c.title or "") in TEMPLATE_CASE_TITLES:
                if db.query(CaseVideoModel).filter(CaseVideoModel.case_id == c.id).count() == 0:
                    db.delete(c)
                    del_cases += 1

        db.commit()

        print()
        print("=" * 72)
        print("PURGE COMPLETE")
        print(f"  Videos deleted:   {deleted_v}")
        print(f"  Sessions deleted: {del_sess}")
        print(f"  Cases deleted:    {del_cases}")
        print()
        vt = db.query(VideoModel).count()
        ev_v = db.query(EventModel).filter(EventModel.validation_status == "VALID").count()
        ev_r = db.query(EventModel).filter(EventModel.validation_status == "REJECTED").count()
        ev_u = db.query(EventModel).filter(EventModel.validation_status == "UNCERTAIN").count()
        ev_total = db.query(EventModel).count()
        print("FINAL COUNTS:")
        print(f"  Videos:     {vt}")
        print(f"  Cases:      {db.query(CaseModel).count()}")
        print(f"  Evidence:   {db.query(EvidenceModel).count()}")
        print(f"  Events:     {ev_total}  (VALID={ev_v} REJECTED={ev_r} UNCERTAIN={ev_u}  SUM={ev_v+ev_r+ev_u})")
        print(f"  Incidents:  {db.query(CorrelatedIncidentModel).count()}")
        print(f"  Reports:    {db.query(ReportModel).count()}")
        print(f"  Sessions:   {db.query(SurveillanceSessionModel).count()}")
        parity_ok = ev_total == (ev_v + ev_r + ev_u)
        print(f"  Parity:     {'OK' if parity_ok else 'BROKEN'}")
        print("=" * 72)

    finally:
        db.close()


if __name__ == "__main__":
    run_purge(dry_run="--execute" not in sys.argv)
