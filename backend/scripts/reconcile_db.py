"""
backend/scripts/reconcile_db.py

Forensic database reconciliation and integrity cleanup script.
- Backs up storage/sentinel.db with timestamp.
- Safely removes verified synthetic/duplicate test artifacts (Video 087ac9e4..., Camera 1, Case CASE-20260915-611A).
- Cleans orphaned records with missing foreign key parents (grouped_events, case_activities, security_zones).
- Verifies PRAGMA foreign_key_check returns 0 violations.
- Verifies canonical counts:
    Cases: 6
    Cameras: 3
    Videos: 21
    Correlated Incidents: 42
    Evidence: 4 (all VALID)
"""
import os
import shutil
import sqlite3
from datetime import datetime

DB_PATH = "C:/Sentinel/storage/sentinel.db"
BACKUP_DIR = "C:/Sentinel/storage"


def main():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    # 1. Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"sentinel_audit_backup_{timestamp}.db")
    shutil.copy2(DB_PATH, backup_path)
    print(f"[1/6] Backup created: {backup_path}")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Enable foreign keys
    c.execute("PRAGMA foreign_keys = OFF;")  # Disable temporarily during cleanup

    # 2. Clean orphaned grouped_events (missing video_id)
    c.execute("DELETE FROM grouped_events WHERE video_id NOT IN (SELECT id FROM videos);")
    deleted_ge = c.rowcount
    print(f"[2/6] Cleaned {deleted_ge} orphaned grouped_events")

    # 3. Clean orphaned case_activities (missing case_id)
    c.execute("DELETE FROM case_activities WHERE case_id NOT IN (SELECT id FROM cases);")
    deleted_ca = c.rowcount
    print(f"[3/6] Cleaned {deleted_ca} orphaned case_activities")

    # 4. Clean orphaned security_zones (missing video_id)
    c.execute("DELETE FROM security_zones WHERE video_id NOT IN (SELECT id FROM videos);")
    deleted_sz = c.rowcount
    print(f"[4/6] Cleaned {deleted_sz} orphaned security_zones")

    # 5. Reconcile verified manual test artifacts from 2026-09-15 UI testing:
    # A) Test Camera '1' (eafb0d8d-9449-4756-afb0-cae9167e1f94)
    c.execute("DELETE FROM camera_sources WHERE id = 'eafb0d8d-9449-4756-afb0-cae9167e1f94';")
    del_cam = c.rowcount

    # B) Duplicate Video '087ac9e4-6e19-4d6e-a5a1-129e262a581f' and its associated records
    c.execute("DELETE FROM face_detections WHERE video_id = '087ac9e4-6e19-4d6e-a5a1-129e262a581f' OR video_id NOT IN (SELECT id FROM videos);")
    del_faces = c.rowcount
    c.execute("DELETE FROM vehicle_attributes WHERE video_id = '087ac9e4-6e19-4d6e-a5a1-129e262a581f' OR video_id NOT IN (SELECT id FROM videos);")
    c.execute("DELETE FROM reports WHERE video_id = '087ac9e4-6e19-4d6e-a5a1-129e262a581f' OR video_id NOT IN (SELECT id FROM videos);")
    c.execute("DELETE FROM evidence WHERE video_id = '087ac9e4-6e19-4d6e-a5a1-129e262a581f';")
    del_ev = c.rowcount
    c.execute("DELETE FROM correlated_incidents WHERE video_id = '087ac9e4-6e19-4d6e-a5a1-129e262a581f';")
    del_inc = c.rowcount
    c.execute("DELETE FROM events WHERE video_id = '087ac9e4-6e19-4d6e-a5a1-129e262a581f';")
    del_events = c.rowcount
    c.execute("DELETE FROM grouped_events WHERE video_id = '087ac9e4-6e19-4d6e-a5a1-129e262a581f';")
    del_gevents = c.rowcount
    c.execute("DELETE FROM tracks WHERE video_id = '087ac9e4-6e19-4d6e-a5a1-129e262a581f';")
    del_trks = c.rowcount
    c.execute("DELETE FROM security_events WHERE video_id = '087ac9e4-6e19-4d6e-a5a1-129e262a581f';")
    del_sec = c.rowcount
    c.execute("DELETE FROM videos WHERE id = '087ac9e4-6e19-4d6e-a5a1-129e262a581f';")
    del_vid = c.rowcount

    # C) Manual test case 'CASE-20260915-611A' (0 linked videos, 0 cameras, 0 incidents, 0 evidence)
    c.execute("DELETE FROM case_activities WHERE case_id IN (SELECT id FROM cases WHERE case_number = 'CASE-20260915-611A');")
    c.execute("DELETE FROM cases WHERE case_number = 'CASE-20260915-611A';")
    del_case = c.rowcount

    conn.commit()

    print(f"[5/6] Reconciled manual test artifacts:")
    print(f"      - Camera '1': {del_cam} removed")
    print(f"      - Video '087ac9e4...': {del_vid} removed")
    print(f"      - Duplicate Evidence: {del_ev} removed")
    print(f"      - Duplicate Incidents: {del_inc} removed")
    print(f"      - Duplicate Events/Tracks: {del_events + del_gevents + del_trks + del_sec} removed")
    print(f"      - Manual Test Case 'CASE-20260915-611A': {del_case} removed")

    # 6. Verify Foreign Key Integrity
    c.execute("PRAGMA foreign_keys = ON;")
    c.execute("PRAGMA foreign_key_check;")
    violations = c.fetchall()
    print(f"[6/6] PRAGMA foreign_key_check violations: {len(violations)}")
    if violations:
        for v in violations:
            print(f"      Violation: {v}")
    else:
        print("      PASS: Zero foreign key violations.")

    # Canonical Count Audit
    print("\n=== RECONCILED CANONICAL COUNTS ===")
    total_cases = c.execute("SELECT COUNT(*) FROM cases;").fetchone()[0]
    total_cams = c.execute("SELECT COUNT(*) FROM camera_sources;").fetchone()[0]
    total_vids = c.execute("SELECT COUNT(*) FROM videos;").fetchone()[0]
    total_incs = c.execute("SELECT COUNT(*) FROM correlated_incidents;").fetchone()[0]
    total_ev = c.execute("SELECT COUNT(*) FROM evidence;").fetchone()[0]
    val_ev = c.execute("SELECT COUNT(*) FROM evidence WHERE validation_status = 'VALID';").fetchone()[0]
    total_raw = c.execute("SELECT COUNT(*) FROM events;").fetchone()[0]
    total_valid = c.execute("SELECT COUNT(*) FROM events WHERE validation_status = 'VALID';").fetchone()[0]
    total_rej = c.execute("SELECT COUNT(*) FROM events WHERE validation_status = 'REJECTED';").fetchone()[0]
    total_unc = c.execute("SELECT COUNT(*) FROM events WHERE validation_status = 'UNCERTAIN';").fetchone()[0]

    print(f"  Cases: {total_cases} (Expected: 6)")
    print(f"  Camera Sources: {total_cams} (Expected: 3)")
    print(f"  Videos: {total_vids} (Expected: 21)")
    print(f"  Correlated Incidents: {total_incs} (Expected: 42)")
    print(f"  Evidence: {total_ev} (Validated: {val_ev}) (Expected: 4)")
    print(f"  Raw Observations: {total_raw} = Valid({total_valid}) + Rejected({total_rej}) + Uncertain({total_unc})")
    print(f"  Parity check: {total_raw == (total_valid + total_rej + total_unc)}")

    conn.close()


if __name__ == "__main__":
    main()
