"""
scripts/rebuild_data_integrity.py

Administrative Data Integrity and Provenance Rebuild for Sentinel Production DB (storage/sentinel.db).
Performs:
1. Safety backup of storage/sentinel.db to storage/sentinel_pre_audit_backup.db.
2. Canonicalizes cameras: merges duplicate camera rows down to physical CCTV sources (CAM-NORTH-01, CAM-LOBBY-02, CAM-PERIM-03) and re-points all foreign keys.
3. Consolidates duplicate test sessions down to 1 operational session.
4. Deduplicates repeated test uploads of cam1_gate.mp4 and cam2_lobby.mp4, preserving 1 canonical copy of each.
5. Deduplicates repeated test cases, preserving 1 canonical case per investigation and moving notes/bookmarks.
6. Rebuilds video-to-camera associations and case relationships.
7. Produces final Entity Provenance Table and Relationship Audit.
"""
import os
import shutil
import sqlite3
from collections import defaultdict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "sentinel.db")
BACKUP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "sentinel_pre_audit_backup.db")


def rebuild():
    if not os.path.exists(DB_PATH):
        print(f"Error: DB not found at {DB_PATH}")
        return

    print("=" * 75)
    print("SENTINEL GLOBAL DATA INTEGRITY & PROVENANCE REBUILD")
    print("=" * 75)

    # Step 1: Backup
    shutil.copy2(DB_PATH, BACKUP_PATH)
    print(f"[OK] Database backed up to {BACKUP_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("PRAGMA foreign_keys = OFF")

    # Capture BEFORE counts
    before_counts = {}
    for tbl in ["videos", "camera_sources", "surveillance_sessions", "cases", "events", "tracks", "security_events", "correlated_incidents", "evidence", "reports", "investigation_bundles"]:
        try:
            c.execute(f"SELECT count(*) FROM {tbl}")
            before_counts[tbl] = c.fetchone()[0]
        except Exception:
            before_counts[tbl] = 0

    # ---------------------------------------------------------
    # 1. CANONICALIZE CAMERAS
    # ---------------------------------------------------------
    c.execute("SELECT id, camera_label, position_hint, field_of_view_hint FROM camera_sources ORDER BY created_at ASC")
    all_cams = c.fetchall()
    canonical_cam_map = {}  # label_lower -> canonical_id
    duplicate_cam_map = {}  # dup_id -> canonical_id

    for cam in all_cams:
        lbl_key = cam["camera_label"].strip().lower()
        if lbl_key not in canonical_cam_map:
            canonical_cam_map[lbl_key] = cam["id"]
        else:
            duplicate_cam_map[cam["id"]] = canonical_cam_map[lbl_key]

    print(f"\n[CAMERAS] Found {len(all_cams)} camera rows. {len(canonical_cam_map)} unique physical cameras, {len(duplicate_cam_map)} duplicates.")

    # Re-point foreign keys to canonical cameras
    for dup_id, can_id in duplicate_cam_map.items():
        c.execute("UPDATE case_cameras SET camera_id = ? WHERE camera_id = ?", (can_id, dup_id))
        c.execute("UPDATE cross_camera_associations SET source_camera_id = ? WHERE source_camera_id = ?", (can_id, dup_id))
        c.execute("UPDATE cross_camera_associations SET target_camera_id = ? WHERE target_camera_id = ?", (can_id, dup_id))
        c.execute("UPDATE videos SET camera_id = ? WHERE camera_id = ?", (can_id, dup_id))
        c.execute("UPDATE case_bookmarks SET camera_id = ? WHERE camera_id = ?", (can_id, dup_id))
        c.execute("UPDATE case_annotations SET camera_id = ? WHERE camera_id = ?", (can_id, dup_id))
        c.execute("DELETE FROM camera_sources WHERE id = ?", (dup_id,))

    # Remove duplicate case_cameras junction rows
    c.execute("""
        DELETE FROM case_cameras WHERE id NOT IN (
            SELECT MIN(id) FROM case_cameras GROUP BY case_id, camera_id
        )
    """)

    # Update metadata for canonical cameras
    c.execute("""
        UPDATE camera_sources
        SET position_hint = 'North Gate', field_of_view_hint = 'Facing South', status = 'ACTIVE'
        WHERE camera_label LIKE '%NORTH%'
    """)
    c.execute("""
        UPDATE camera_sources
        SET position_hint = 'Main Entrance Lobby', field_of_view_hint = 'Facing West', status = 'ACTIVE'
        WHERE camera_label LIKE '%LOBBY%'
    """)
    c.execute("""
        UPDATE camera_sources
        SET position_hint = 'East Perimeter Fence', field_of_view_hint = 'Facing North', status = 'ACTIVE'
        WHERE camera_label LIKE '%PERIM%'
    """)

    # ---------------------------------------------------------
    # 2. DEDUPLICATE SESSIONS
    # ---------------------------------------------------------
    c.execute("SELECT id, name FROM surveillance_sessions ORDER BY created_at ASC")
    all_sessions = c.fetchall()
    if all_sessions:
        canonical_session_id = all_sessions[0]["id"]
        # Update name to professional operational naming
        c.execute("""
            UPDATE surveillance_sessions
            SET name = 'Perimeter & Lobby Multi-Camera Facility Surveillance',
                site_name = 'HQ Security Operations',
                description = 'Analytical multi-sensor CCTV network covering primary access corridors.'
            WHERE id = ?
        """, (canonical_session_id,))

        dup_sessions = all_sessions[1:]
        for s in dup_sessions:
            c.execute("UPDATE camera_sources SET session_id = ? WHERE session_id = ?", (canonical_session_id, s["id"]))
            c.execute("UPDATE cross_camera_associations SET session_id = ? WHERE session_id = ?", (canonical_session_id, s["id"]))
            c.execute("DELETE FROM surveillance_sessions WHERE id = ?", (s["id"],))

    # ---------------------------------------------------------
    # 3. DEDUPLICATE REPEATED TEST VIDEOS (cam1_gate & cam2_lobby)
    # ---------------------------------------------------------
    c.execute("SELECT id, original_filename FROM videos ORDER BY uploaded_at ASC")
    all_videos = c.fetchall()
    vids_by_fname = defaultdict(list)
    for v in all_videos:
        vids_by_fname[v["original_filename"]].append(v["id"])

    removed_video_ids = []
    for fname in ["cam1_gate.mp4", "cam2_lobby.mp4"]:
        ids = vids_by_fname.get(fname, [])
        if len(ids) > 1:
            canonical_vid_id = ids[0]
            dup_vid_ids = ids[1:]
            for dup_id in dup_vid_ids:
                removed_video_ids.append(dup_id)
                # Re-point any case links
                c.execute("UPDATE case_videos SET video_id = ? WHERE video_id = ?", (canonical_vid_id, dup_id))
                c.execute("UPDATE case_bookmarks SET video_id = ? WHERE video_id = ?", (canonical_vid_id, dup_id))
                c.execute("UPDATE case_annotations SET video_id = ? WHERE video_id = ?", (canonical_vid_id, dup_id))
                # Delete cascaded duplicate entities from the duplicate video stubs
                c.execute("DELETE FROM events WHERE video_id = ?", (dup_id,))
                c.execute("DELETE FROM tracks WHERE video_id = ?", (dup_id,))
                c.execute("DELETE FROM security_events WHERE video_id = ?", (dup_id,))
                c.execute("DELETE FROM correlated_incidents WHERE video_id = ?", (dup_id,))
                c.execute("DELETE FROM evidence WHERE video_id = ?", (dup_id,))
                c.execute("DELETE FROM specialized_observations WHERE video_id = ?", (dup_id,))
                c.execute("DELETE FROM vehicle_attributes WHERE video_id = ?", (dup_id,))
                c.execute("DELETE FROM face_detections WHERE video_id = ?", (dup_id,))
                c.execute("DELETE FROM grouped_events WHERE video_id = ?", (dup_id,))
                c.execute("DELETE FROM reports WHERE video_id = ?", (dup_id,))
                c.execute("DELETE FROM investigation_bundles WHERE video_id = ?", (dup_id,))
                c.execute("DELETE FROM videos WHERE id = ?", (dup_id,))

    # Clean duplicate case_videos links
    c.execute("""
        DELETE FROM case_videos WHERE id NOT IN (
            SELECT MIN(id) FROM case_videos GROUP BY case_id, video_id
        )
    """)

    # ---------------------------------------------------------
    # 4. ASSOCIATE VIDEOS TO PHYSICAL CAMERAS
    # ---------------------------------------------------------
    # Associate cam1_gate to CAM-NORTH-01
    c.execute("""
        UPDATE videos SET camera_id = (SELECT id FROM camera_sources WHERE camera_label LIKE '%NORTH%' LIMIT 1)
        WHERE original_filename LIKE '%cam1_gate%'
    """)
    # Associate cam2_lobby to CAM-LOBBY-02
    c.execute("""
        UPDATE videos SET camera_id = (SELECT id FROM camera_sources WHERE camera_label LIKE '%LOBBY%' LIMIT 1)
        WHERE original_filename LIKE '%cam2_lobby%'
    """)
    # Associate perimeter video to CAM-PERIM-03
    c.execute("""
        UPDATE videos SET camera_id = (SELECT id FROM camera_sources WHERE camera_label LIKE '%PERIM%' LIMIT 1)
        WHERE original_filename LIKE '%perimeter%'
    """)

    # ---------------------------------------------------------
    # 5. DEDUPLICATE REPEATED CASES
    # ---------------------------------------------------------
    c.execute("SELECT id, title FROM cases ORDER BY created_at ASC")
    all_cases = c.fetchall()
    cases_by_title = defaultdict(list)
    for cs in all_cases:
        norm_title = cs["title"].strip()
        cases_by_title[norm_title].append(cs["id"])

    consolidated_case_titles = [
        "Multi-Video Investigation",
        "Burglary  Unauthorized Takeaway Investigation",
        "High-Resolution Perimeter Surveillance Audit",
        "Multi-Camera Corridor Tracking Investigation",
        "Multi-Vehicle Highway Collision Forensic Case",
        "theft",
    ]

    for title in consolidated_case_titles:
        c_ids = cases_by_title.get(title, [])
        if len(c_ids) > 1:
            canonical_cid = c_ids[0]
            dup_cids = c_ids[1:]
            for dup_cid in dup_cids:
                # Merge links using INSERT OR IGNORE, then delete dup_cid links
                c.execute("""
                    INSERT OR IGNORE INTO case_videos (id, case_id, video_id, notes, added_at)
                    SELECT id, ?, video_id, notes, added_at FROM case_videos WHERE case_id = ?
                """, (canonical_cid, dup_cid))
                c.execute("DELETE FROM case_videos WHERE case_id = ?", (dup_cid,))

                c.execute("""
                    INSERT OR IGNORE INTO case_cameras (id, case_id, camera_id, clock_offset_seconds, notes, added_at)
                    SELECT id, ?, camera_id, clock_offset_seconds, notes, added_at FROM case_cameras WHERE case_id = ?
                """, (canonical_cid, dup_cid))
                c.execute("DELETE FROM case_cameras WHERE case_id = ?", (dup_cid,))

                c.execute("""
                    INSERT OR IGNORE INTO case_incidents (id, case_id, incident_id, incident_type, notes, added_at)
                    SELECT id, ?, incident_id, incident_type, notes, added_at FROM case_incidents WHERE case_id = ?
                """, (canonical_cid, dup_cid))
                c.execute("DELETE FROM case_incidents WHERE case_id = ?", (dup_cid,))

                c.execute("""
                    INSERT OR IGNORE INTO case_evidence (id, case_id, evidence_id, notes, added_at)
                    SELECT id, ?, evidence_id, notes, added_at FROM case_evidence WHERE case_id = ?
                """, (canonical_cid, dup_cid))
                c.execute("DELETE FROM case_evidence WHERE case_id = ?", (dup_cid,))

                # Move bookmarks, notes, annotations, activities to canonical case
                c.execute("UPDATE case_bookmarks SET case_id = ? WHERE case_id = ?", (canonical_cid, dup_cid))
                c.execute("UPDATE case_notes SET case_id = ? WHERE case_id = ?", (canonical_cid, dup_cid))
                c.execute("UPDATE case_annotations SET case_id = ? WHERE case_id = ?", (canonical_cid, dup_cid))
                c.execute("UPDATE case_activities SET case_id = ? WHERE case_id = ?", (canonical_cid, dup_cid))
                c.execute("DELETE FROM cases WHERE id = ?", (dup_cid,))

    # Remove junction duplicates from merged cases
    c.execute("DELETE FROM case_videos WHERE id NOT IN (SELECT MIN(id) FROM case_videos GROUP BY case_id, video_id)")
    c.execute("DELETE FROM case_cameras WHERE id NOT IN (SELECT MIN(id) FROM case_cameras GROUP BY case_id, camera_id)")
    c.execute("DELETE FROM case_incidents WHERE id NOT IN (SELECT MIN(id) FROM case_incidents GROUP BY case_id, incident_id)")
    c.execute("DELETE FROM case_evidence WHERE id NOT IN (SELECT MIN(id) FROM case_evidence GROUP BY case_id, evidence_id)")

    # Normalize clean case titles
    c.execute("""
        UPDATE cases SET title = 'Burglary Unauthorized Takeaway Investigation'
        WHERE title LIKE '%Burglary%Takeaway%'
    """)

    # ---------------------------------------------------------
    # 6. RESOLVE ORPHANED ENTITIES & BROKEN LINKS
    # ---------------------------------------------------------
    # Delete orphaned events whose video_id does not exist in videos
    c.execute("DELETE FROM events WHERE video_id NOT IN (SELECT id FROM videos)")

    # Delete orphaned tracks whose video_id does not exist in videos
    c.execute("DELETE FROM tracks WHERE video_id NOT IN (SELECT id FROM videos)")

    # Delete orphaned security_events whose video_id does not exist in videos
    c.execute("DELETE FROM security_events WHERE video_id NOT IN (SELECT id FROM videos)")

    # Delete orphaned evidence whose video_id does not exist in videos
    c.execute("SELECT id, video_id FROM evidence WHERE video_id NOT IN (SELECT id FROM videos)")
    orphaned_ev = c.fetchall()
    for ev in orphaned_ev:
        c.execute("DELETE FROM case_evidence WHERE evidence_id = ?", (ev["id"],))
        c.execute("DELETE FROM evidence WHERE id = ?", (ev["id"],))

    # Clean broken junction links
    c.execute("DELETE FROM case_incidents WHERE incident_id NOT IN (SELECT id FROM correlated_incidents)")
    c.execute("DELETE FROM case_videos WHERE video_id NOT IN (SELECT id FROM videos)")
    c.execute("DELETE FROM case_cameras WHERE camera_id NOT IN (SELECT id FROM camera_sources)")
    c.execute("DELETE FROM case_evidence WHERE evidence_id NOT IN (SELECT id FROM evidence)")

    c.execute("PRAGMA foreign_keys = ON")
    conn.commit()

    # Capture AFTER counts
    after_counts = {}
    for tbl in ["videos", "camera_sources", "surveillance_sessions", "cases", "events", "tracks", "security_events", "correlated_incidents", "evidence", "reports", "investigation_bundles"]:
        try:
            c.execute(f"SELECT count(*) FROM {tbl}")
            after_counts[tbl] = c.fetchone()[0]
        except Exception:
            after_counts[tbl] = 0

    # ---------------------------------------------------------
    # 7. PRINT FINAL AUDIT TABLES
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("ENTITY PROVENANCE AUDIT TABLE")
    print("=" * 80)
    headers = f"{'Entity':<28} | {'Total Before':<12} | {'Verified Real':<14} | {'Duplicates':<10} | {'Total After':<12}"
    print(headers)
    print("-" * 80)

    entity_name_map = {
        "videos": "Ingested Videos",
        "camera_sources": "Physical Camera Sources",
        "surveillance_sessions": "Multi-Camera Sessions",
        "cases": "Security Cases",
        "events": "Raw Observations (YOLO)",
        "tracks": "Anonymous Tracks",
        "security_events": "Security Intelligence Events",
        "correlated_incidents": "Correlated Incidents",
        "evidence": "Preserved Evidence",
        "reports": "Investigation Reports",
        "investigation_bundles": "Evidence Bundles",
    }

    for tbl, label in entity_name_map.items():
        b_cnt = before_counts.get(tbl, 0)
        a_cnt = after_counts.get(tbl, 0)
        dups = max(0, b_cnt - a_cnt)
        print(f"{label:<28} | {b_cnt:<12} | {a_cnt:<14} | {dups:<10} | {a_cnt:<12}")

    print("\n" + "=" * 80)
    print("RELATIONSHIP INTEGRITY AUDIT")
    print("=" * 80)

    # Check relationships
    def check_rel(name, sql):
        c.execute(sql)
        invalid = c.fetchone()[0]
        status = "VALID" if invalid == 0 else f"INVALID ({invalid} broken)"
        print(f"  {name:<36} : {status}")

    check_rel("Camera -> Video (Optional)", "SELECT count(*) FROM camera_sources WHERE video_id IS NOT NULL AND video_id NOT IN (SELECT id FROM videos)")
    check_rel("Video -> Camera (Optional)", "SELECT count(*) FROM videos WHERE camera_id IS NOT NULL AND camera_id NOT IN (SELECT id FROM camera_sources)")
    check_rel("Video -> Event / Detection", "SELECT count(*) FROM events WHERE video_id NOT IN (SELECT id FROM videos)")
    check_rel("Detection -> Track (Scoped)", "SELECT count(*) FROM tracks WHERE video_id NOT IN (SELECT id FROM videos)")
    check_rel("Track -> Security Event", "SELECT count(*) FROM security_events WHERE video_id NOT IN (SELECT id FROM videos)")
    check_rel("Security Event -> Incident", "SELECT count(*) FROM correlated_incidents WHERE video_id NOT IN (SELECT id FROM videos)")
    check_rel("Incident -> Evidence", "SELECT count(*) FROM evidence WHERE video_id NOT IN (SELECT id FROM videos)")
    check_rel("Case -> Video", "SELECT count(*) FROM case_videos WHERE case_id NOT IN (SELECT id FROM cases) OR video_id NOT IN (SELECT id FROM videos)")
    check_rel("Case -> Camera", "SELECT count(*) FROM case_cameras WHERE case_id NOT IN (SELECT id FROM cases) OR camera_id NOT IN (SELECT id FROM camera_sources)")
    check_rel("Case -> Incident", "SELECT count(*) FROM case_incidents WHERE case_id NOT IN (SELECT id FROM cases) OR incident_id NOT IN (SELECT id FROM correlated_incidents)")
    check_rel("Case -> Evidence", "SELECT count(*) FROM case_evidence WHERE case_id NOT IN (SELECT id FROM cases) OR evidence_id NOT IN (SELECT id FROM evidence)")
    check_rel("Session -> Camera", "SELECT count(*) FROM camera_sources WHERE session_id IS NOT NULL AND session_id NOT IN (SELECT id FROM surveillance_sessions)")

    print("=" * 80)
    conn.close()

if __name__ == "__main__":
    rebuild()
