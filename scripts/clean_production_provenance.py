"""
Sentinel Production Database Provenance Audit & Cleanup Script
Safely backs up storage/sentinel.db, cleans test duplicates, and verifies data integrity.
"""

import os
import shutil
import sqlite3
from datetime import datetime

DB_PATH = "storage/sentinel.db"
BACKUP_PATH = f"storage/sentinel_pre_cleanup_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

def run_cleanup():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        return

    # Step 1: Backup
    shutil.copy2(DB_PATH, BACKUP_PATH)
    print(f"[OK] Database backed up to {BACKUP_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Step 2: Audit Cases
    print("\n--- AUDITING CASES ---")
    cursor.execute("SELECT id, case_number, title, status, created_at FROM cases")
    cases = cursor.fetchall()
    print(f"Total Cases Found: {len(cases)}")
    for c in cases:
        print(f"  {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]}")

    # Identify synthetic/duplicate test cases with 0 associations
    synthetic_case_ids = [
        "89d08a12-3aba-4639-92e4-660f569981b9",  # duplicate 'theft' test case
        "24918de9-81f1-4f19-8694-6edbff6d35df",  # test case 'tt'
    ]

    for cid in synthetic_case_ids:
        cursor.execute("SELECT COUNT(*) FROM case_videos WHERE case_id = ?", (cid,))
        v_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM case_incidents WHERE case_id = ?", (cid,))
        i_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM case_evidence WHERE case_id = ?", (cid,))
        e_count = cursor.fetchone()[0]
        print(f"  Target synthetic case {cid}: {v_count} videos, {i_count} incidents, {e_count} evidence.")
        if v_count == 0 and i_count == 0 and e_count == 0:
            cursor.execute("DELETE FROM cases WHERE id = ?", (cid,))
            print(f"  [REMOVED] Synthetic case {cid}")

    # Step 3: Audit Evidence
    print("\n--- AUDITING EVIDENCE ---")
    cursor.execute("SELECT id, video_id, timestamp_seconds, object_class, snapshot_path, annotated_snapshot_path, clip_path FROM evidence")
    evidence_rows = cursor.fetchall()
    print(f"Total Evidence Rows: {len(evidence_rows)}")

    # Check for missing files on disk
    orphan_ids = []
    for row in evidence_rows:
        eid, vid, ts, cls, snap, ann, clip = row
        has_file = False
        for p in [snap, ann, clip]:
            if p and os.path.exists(p):
                has_file = True
                break
        if not has_file:
            print(f"  Orphan Evidence (No files on disk): {eid} (video: {vid}, ts: {ts})")
            orphan_ids.append(eid)

    for oid in orphan_ids:
        cursor.execute("DELETE FROM case_evidence WHERE evidence_id = ?", (oid,))
        cursor.execute("DELETE FROM evidence WHERE id = ?", (oid,))
        print(f"  [REMOVED] Orphan evidence record {oid}")

    # Deduplicate repeated evidence for Burglary video
    cursor.execute("""
        SELECT video_id, timestamp_seconds, object_class, COUNT(*), GROUP_CONCAT(id)
        FROM evidence
        GROUP BY video_id, ROUND(timestamp_seconds, 0), object_class
        HAVING COUNT(*) > 1
    """)
    duplicate_groups = cursor.fetchall()
    print(f"\nDuplicate Evidence Groups Found: {len(duplicate_groups)}")
    for grp in duplicate_groups:
        vid, ts, cls, cnt, ids_str = grp
        ids = ids_str.split(",")
        print(f"  Group: video={vid}, ts={ts}, class={cls}, count={cnt}, ids={ids}")
        # Keep the first ID with existing files on disk, remove the others
        kept_id = None
        for eid in ids:
            cursor.execute("SELECT snapshot_path, annotated_snapshot_path, clip_path FROM evidence WHERE id = ?", (eid,))
            paths = cursor.fetchone()
            if paths and any(p and os.path.exists(p) for p in paths):
                kept_id = eid
                break
        if not kept_id:
            kept_id = ids[0]

        for eid in ids:
            if eid != kept_id:
                cursor.execute("DELETE FROM case_evidence WHERE evidence_id = ?", (eid,))
                cursor.execute("DELETE FROM evidence WHERE id = ?", (eid,))
                print(f"    [DEDUPLICATED] Removed duplicate evidence {eid} (kept {kept_id})")

    conn.commit()

    # Step 4: Verify Final Counts
    print("\n--- POST-CLEANUP VERIFICATION ---")
    cursor.execute("SELECT COUNT(*) FROM cases")
    final_cases = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM evidence")
    final_evidence = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM videos")
    final_videos = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM camera_sources")
    final_cameras = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM correlated_incidents")
    final_incidents = cursor.fetchone()[0]

    print(f"Total Cases: {final_cases}")
    print(f"Total Evidence Records: {final_evidence}")
    print(f"Total Videos: {final_videos}")
    print(f"Total Cameras: {final_cameras}")
    print(f"Total Incidents: {final_incidents}")

    conn.close()
    print("\n[SUCCESS] Production Database Provenance Audit & Cleanup Complete.")

if __name__ == "__main__":
    run_cleanup()
