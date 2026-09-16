"""
scripts/audit_database.py

Dry-run provenance and relationship auditor for Sentinel production database (storage/sentinel.db).
Classifies every entity into:
- REAL USER DATA
- TEST/SYNTHETIC DATA
- DUPLICATE
- ORPHAN
- DERIVED VALID DATA
- UNKNOWN
"""
import sqlite3
import os
import json
from collections import defaultdict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "sentinel.db")

def run_audit():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    print("=" * 70)
    print("SENTINEL PRODUCTION DATABASE PROVENANCE & INTEGRITY AUDIT")
    print("Database:", DB_PATH)
    print("=" * 70)

    # 1. VIDEOS AUDIT
    c.execute("SELECT id, original_filename, storage_path, file_size_bytes, duration_seconds, fps, status, camera_id, uploaded_at FROM videos")
    videos = c.fetchall()
    video_map = {v["id"]: v for v in videos}
    filename_groups = defaultdict(list)
    for v in videos:
        filename_groups[v["original_filename"]].append(v)

    print(f"\n--- 1. VIDEOS (Total: {len(videos)}) ---")
    duplicate_video_ids = []
    canonical_videos = {}
    for fname, rows in sorted(filename_groups.items()):
        if len(rows) > 1:
            print(f"  DUPLICATE GROUP: '{fname}' count={len(rows)}")
            canonical_videos[fname] = rows[0]["id"]
            for r in rows[1:]:
                duplicate_video_ids.append(r["id"])
        else:
            canonical_videos[fname] = rows[0]["id"]
            print(f"  SINGLE: '{fname}' id={rows[0]['id']} duration={rows[0]['duration_seconds']}s status={rows[0]['status']}")

    # 2. CAMERAS AUDIT
    c.execute("SELECT id, camera_label, position_hint, field_of_view_hint, session_id, video_id, status FROM camera_sources")
    cams = c.fetchall()
    cam_label_groups = defaultdict(list)
    for cam in cams:
        cam_label_groups[cam["camera_label"]].append(cam)

    print(f"\n--- 2. CAMERAS (Total: {len(cams)}) ---")
    canonical_cameras = {}
    duplicate_cam_ids = []
    for label, rows in sorted(cam_label_groups.items()):
        canonical_cameras[label] = rows[0]["id"]
        print(f"  LABEL: '{label}' count={len(rows)} (Canonical ID: {rows[0]['id']})")
        for r in rows[1:]:
            duplicate_cam_ids.append(r["id"])

    # 3. SURVEILLANCE SESSIONS AUDIT
    c.execute("SELECT id, name, site_name, description, status, created_at FROM surveillance_sessions")
    sessions = c.fetchall()
    print(f"\n--- 3. SURVEILLANCE SESSIONS (Total: {len(sessions)}) ---")
    session_name_groups = defaultdict(list)
    for s in sessions:
        session_name_groups[s["name"]].append(s)
    duplicate_session_ids = []
    for name, rows in session_name_groups.items():
        print(f"  SESSION: '{name}' count={len(rows)}")
        if len(rows) > 1:
            for r in rows[1:]:
                duplicate_session_ids.append(r["id"])

    # 4. CASES AUDIT
    c.execute("SELECT id, case_number, title, status, priority, created_at FROM cases")
    cases = c.fetchall()
    print(f"\n--- 4. CASES (Total: {len(cases)}) ---")
    case_title_groups = defaultdict(list)
    for cs in cases:
        case_title_groups[cs["title"]].append(cs)
    duplicate_case_ids = []
    for title, rows in sorted(case_title_groups.items()):
        print(f"  CASE TITLE: '{title}' count={len(rows)}")
        if len(rows) > 1 and "Investigation" in title:
            for r in rows[1:]:
                duplicate_case_ids.append(r["id"])

    # 5. EVIDENCE AUDIT
    c.execute("SELECT id, video_id, source_video_name, timestamp_seconds, evidence_type, validation_status, object_class FROM evidence")
    evidence = c.fetchall()
    print(f"\n--- 5. EVIDENCE (Total: {len(evidence)}) ---")
    ev_by_video = defaultdict(list)
    ev_orphans = []
    for ev in evidence:
        if ev["video_id"] not in video_map:
            ev_orphans.append(ev)
        else:
            ev_by_video[ev["source_video_name"]].append(ev)

    for vname, rows in sorted(ev_by_video.items()):
        print(f"  EVIDENCE FOR '{vname}': {len(rows)} items")
    if ev_orphans:
        print(f"  ORPHANED EVIDENCE: {len(ev_orphans)} items")

    # 6. INCIDENTS AUDIT
    c.execute("SELECT id, video_id, incident_category, assessment_score, validation_decision FROM correlated_incidents")
    incidents = c.fetchall()
    print(f"\n--- 6. CORRELATED INCIDENTS (Total: {len(incidents)}) ---")
    inc_by_vid = defaultdict(list)
    inc_orphans = []
    for inc in incidents:
        if inc["video_id"] and inc["video_id"] not in video_map:
            inc_orphans.append(inc)
        else:
            inc_by_vid[inc["video_id"]].append(inc)

    for vid_id, rows in inc_by_vid.items():
        vname = video_map[vid_id]["original_filename"] if vid_id in video_map else "Unknown"
        print(f"  INCIDENTS FOR VIDEO '{vname}' ({vid_id}): {len(rows)} incidents")
    if inc_orphans:
        print(f"  ORPHANED INCIDENTS: {len(inc_orphans)} incidents")

    # 7. SPECIALIZED OBSERVATIONS AUDIT (FIRE / SMOKE / WEAPON)
    c.execute("SELECT id, video_id, detector_name, class_name, confidence, validation_status FROM specialized_observations")
    spec_obs = c.fetchall()
    print(f"\n--- 7. SPECIALIZED OBSERVATIONS (Total: {len(spec_obs)}) ---")
    spec_counts = defaultdict(int)
    for obs in spec_obs:
        spec_counts[(obs["detector_name"], obs["class_name"], obs["validation_status"])] += 1
    for (det, cls, val), cnt in sorted(spec_counts.items()):
        print(f"  DETECTOR: {det} | CLASS: {cls} | STATUS: {val} -> {cnt}")

    # 8. RAW DETECTION PARITY INVARIANT: RAW = VALID + REJECTED + UNCERTAIN
    print("\n--- 8. DETECTION PARITY INVARIANT CHECK ---")
    c.execute("SELECT video_id, validation_status, count(*) FROM events GROUP BY video_id, validation_status")
    det_stats = defaultdict(lambda: defaultdict(int))
    for vid, status, cnt in c.fetchall():
        det_stats[vid][status] = cnt

    c.execute("SELECT video_id, count(*) FROM events GROUP BY video_id")
    for vid, total in c.fetchall():
        vname = video_map[vid]["original_filename"] if vid in video_map else "Unknown"
        v_valid = det_stats[vid].get("VALID", 0)
        v_rejected = det_stats[vid].get("REJECTED", 0)
        v_uncertain = det_stats[vid].get("UNCERTAIN", 0)
        sum_components = v_valid + v_rejected + v_uncertain
        matches = (total == sum_components)
        status_str = "PASS" if matches else "FAIL"
        print(f"  [{status_str}] Video '{vname}': Total={total}, Valid={v_valid}, Rejected={v_rejected}, Uncertain={v_uncertain} (Sum={sum_components})")

    conn.close()

if __name__ == "__main__":
    run_audit()
