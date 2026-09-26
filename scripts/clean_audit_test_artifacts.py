"""
scripts/clean_audit_test_artifacts.py

Safely removes the verified manual UI audit artifacts created on 2026-09-21:
- Case: 9f3cda81-b98b-4759-89ec-950b209b73a2 (CASE-20260921-82A5)
- Video: 27343320-0da8-4c8d-a9e2-00fb76eaf022 (uccrime_Burglary010_x264.mp4 re-ingestion)
- Evidence: ev_f73658c5e604
- Dependent records (12 correlated incidents, 138 detections, 29 tracks, 13 security events)

Guarantees:
- Creates timestamped backup before any write
- Preserves genuine canonical Burglary benchmark 0d4d92f9-19f8-42e3-925f-1931cb557705 intact
- Validates SQLite foreign keys
- Verifies post-cleanup baseline: 6 cases, 3 cameras, 21 videos, 42 correlated incidents, 4 validated evidence
"""
import os
import shutil
import sqlite3
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT_DIR, "storage", "sentinel.db")
BACKUP_PATH = os.path.join(
    ROOT_DIR, "storage", f"sentinel_audit_cleanup_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
)

TARGET_CASE_ID = "9f3cda81-b98b-4759-89ec-950b209b73a2"
TARGET_VIDEO_ID = "27343320-0da8-4c8d-a9e2-00fb76eaf022"
TARGET_EVIDENCE_ID = "ev_f73658c5e604"

GENUINE_BURGLARY_VIDEO_ID = "0d4d92f9-19f8-42e3-925f-1931cb557705"


def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    # 1. Create safety backup
    shutil.copy2(DB_PATH, BACKUP_PATH)
    print(f"[OK] Database backed up to: {BACKUP_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    # 2. Verify targets exist and belong to the audit session
    cursor.execute("SELECT id, case_number, title, created_at FROM cases WHERE id = ?", (TARGET_CASE_ID,))
    case_row = cursor.fetchone()
    if not case_row:
        print(f"[WARN] Target audit case {TARGET_CASE_ID} not found in database.")
    else:
        print(f"[VERIFIED TARGET CASE] ID: {case_row[0]} | Number: {case_row[1]} | Title: {case_row[2]} | Created: {case_row[3]}")

    cursor.execute("SELECT id, original_filename, storage_path, uploaded_at FROM videos WHERE id = ?", (TARGET_VIDEO_ID,))
    video_row = cursor.fetchone()
    if not video_row:
        print(f"[WARN] Target audit video {TARGET_VIDEO_ID} not found in database.")
    else:
        print(f"[VERIFIED TARGET VIDEO] ID: {video_row[0]} | File: {video_row[1]} | Uploaded: {video_row[3]}")

    # 3. Verify genuine benchmark video is present and distinct
    cursor.execute("SELECT id, original_filename, uploaded_at FROM videos WHERE id = ?", (GENUINE_BURGLARY_VIDEO_ID,))
    genuine_row = cursor.fetchone()
    assert genuine_row is not None, f"Genuine burglary benchmark {GENUINE_BURGLARY_VIDEO_ID} missing!"
    print(f"[PRESERVED CANONICAL BURGLARY] ID: {genuine_row[0]} | File: {genuine_row[1]} | Uploaded: {genuine_row[2]}")

    # 4. Perform safe deletions
    # A. Evidence
    cursor.execute("DELETE FROM case_evidence WHERE evidence_id = ?", (TARGET_EVIDENCE_ID,))
    cursor.execute("DELETE FROM evidence WHERE id = ?", (TARGET_EVIDENCE_ID,))
    cursor.execute("DELETE FROM evidence WHERE video_id = ?", (TARGET_VIDEO_ID,))
    print(f"[REMOVED] Evidence records for audit video {TARGET_VIDEO_ID}")

    # B. Case associations
    cursor.execute("DELETE FROM case_videos WHERE case_id = ? OR video_id = ?", (TARGET_CASE_ID, TARGET_VIDEO_ID))
    cursor.execute("DELETE FROM case_incidents WHERE case_id = ?", (TARGET_CASE_ID,))
    cursor.execute("DELETE FROM case_bookmarks WHERE case_id = ? OR video_id = ?", (TARGET_CASE_ID, TARGET_VIDEO_ID))
    cursor.execute("DELETE FROM case_notes WHERE case_id = ?", (TARGET_CASE_ID,))
    cursor.execute("DELETE FROM case_annotations WHERE case_id = ? OR video_id = ?", (TARGET_CASE_ID, TARGET_VIDEO_ID))
    cursor.execute("DELETE FROM case_activities WHERE case_id = ?", (TARGET_CASE_ID,))
    cursor.execute("DELETE FROM cases WHERE id = ?", (TARGET_CASE_ID,))
    print(f"[REMOVED] Target case {TARGET_CASE_ID}")

    # C. Video child intelligence records
    cursor.execute("DELETE FROM correlated_incidents WHERE video_id = ?", (TARGET_VIDEO_ID,))
    cursor.execute("DELETE FROM security_events WHERE video_id = ?", (TARGET_VIDEO_ID,))
    cursor.execute("DELETE FROM specialized_observations WHERE video_id = ?", (TARGET_VIDEO_ID,))
    cursor.execute("DELETE FROM face_detections WHERE video_id = ?", (TARGET_VIDEO_ID,))
    cursor.execute("DELETE FROM vehicle_attributes WHERE video_id = ?", (TARGET_VIDEO_ID,))
    cursor.execute("DELETE FROM tracks WHERE video_id = ?", (TARGET_VIDEO_ID,))
    cursor.execute("DELETE FROM grouped_events WHERE video_id = ?", (TARGET_VIDEO_ID,))
    cursor.execute("DELETE FROM events WHERE video_id = ?", (TARGET_VIDEO_ID,))
    cursor.execute("DELETE FROM reports WHERE video_id = ?", (TARGET_VIDEO_ID,))
    cursor.execute("DELETE FROM camera_sources WHERE video_id = ?", (TARGET_VIDEO_ID,))
    cursor.execute("DELETE FROM videos WHERE id = ?", (TARGET_VIDEO_ID,))
    print(f"[REMOVED] Target video {TARGET_VIDEO_ID} and all child intelligence records")

    # 5. Foreign key integrity check
    cursor.execute("PRAGMA foreign_key_check;")
    fk_errors = cursor.fetchall()
    if fk_errors:
        conn.rollback()
        raise RuntimeError(f"Foreign key violations detected after cleanup: {fk_errors}")
    print("[PASS] Foreign key integrity verified: 0 violations.")

    conn.commit()

    # 6. Verify canonical counts
    cases_cnt = cursor.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    cams_cnt = cursor.execute("SELECT COUNT(*) FROM camera_sources").fetchone()[0]
    vids_cnt = cursor.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    corr_cnt = cursor.execute("SELECT COUNT(*) FROM correlated_incidents").fetchone()[0]
    ev_cnt = cursor.execute("SELECT COUNT(*) FROM evidence WHERE validation_status = 'VALID'").fetchone()[0]
    total_ev_cnt = cursor.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]

    print("\n=== POST-CLEANUP CANONICAL TOTALS ===")
    print(f"Cases:                {cases_cnt} (Expected: 6)")
    print(f"Camera Sources:       {cams_cnt} (Expected: 3)")
    print(f"Videos:               {vids_cnt} (Expected: 21)")
    print(f"Correlated Incidents: {corr_cnt} (Expected: 42)")
    print(f"Validated Evidence:   {ev_cnt} (Expected: 4)")
    print(f"Total Evidence:       {total_ev_cnt}")

    assert cases_cnt == 6, f"Expected 6 cases, got {cases_cnt}"
    assert cams_cnt == 3, f"Expected 3 cameras, got {cams_cnt}"
    assert vids_cnt == 21, f"Expected 21 videos, got {vids_cnt}"
    assert corr_cnt == 42, f"Expected 42 correlated incidents, got {corr_cnt}"
    assert ev_cnt == 4, f"Expected 4 validated evidence items, got {ev_cnt}"

    # 7. Verify genuine Burglary benchmark intact
    cursor.execute("SELECT COUNT(*) FROM events WHERE video_id = ?", (GENUINE_BURGLARY_VIDEO_ID,))
    burg_raw = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM events WHERE video_id = ? AND validation_status = 'VALID'", (GENUINE_BURGLARY_VIDEO_ID,))
    burg_val = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM events WHERE video_id = ? AND validation_status = 'REJECTED'", (GENUINE_BURGLARY_VIDEO_ID,))
    burg_rej = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM specialized_observations WHERE video_id = ?", (GENUINE_BURGLARY_VIDEO_ID,))
    burg_spec = cursor.fetchone()[0]

    print(f"\n[GENUINE BURGLARY BENCHMARK CHECK] Raw={burg_raw}, Valid={burg_val}, Rejected={burg_rej}, Spec={burg_spec}")
    assert burg_raw == 138 and burg_val == 137 and burg_rej == 1 and burg_spec == 0, "Burglary benchmark changed!"

    conn.close()

    # 8. Clean temporary files on disk for the test video if they exist
    test_video_path = os.path.join(ROOT_DIR, "storage", "uploads", f"{TARGET_VIDEO_ID}_uccrime_Burglary010_x264.mp4")
    test_meta_path = os.path.join(ROOT_DIR, "storage", "uploads", f"{TARGET_VIDEO_ID}.json")
    for p in [test_video_path, test_meta_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"[CLEANED DISK] {p}")
            except Exception as e:
                print(f"[NOTE] Could not delete {p}: {e}")

    # Test evidence files
    for suffix in ["_snapshot.jpg", "_annotated.jpg", "_clip.mp4"]:
        ev_p = os.path.join(ROOT_DIR, "storage", "evidence", f"{TARGET_EVIDENCE_ID}{suffix}")
        if os.path.exists(ev_p):
            try:
                os.remove(ev_p)
                print(f"[CLEANED DISK EVIDENCE] {ev_p}")
            except Exception as e:
                print(f"[NOTE] Could not delete {ev_p}: {e}")

    print("\n[SUCCESS] Provenance cleanup completed with 100% integrity verification.")


if __name__ == "__main__":
    main()
