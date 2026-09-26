import sqlite3
import json

conn = sqlite3.connect('storage/sentinel.db')
c = conn.cursor()
vid = '0d4d92f9-19f8-42e3-925f-1931cb557705'

raw_dets = c.execute('SELECT count(*) FROM events WHERE video_id = ?', (vid,)).fetchone()[0]
valid_dets = c.execute("SELECT count(*) FROM events WHERE video_id = ? AND validation_status = 'VALID'", (vid,)).fetchone()[0]
rej_dets = c.execute("SELECT count(*) FROM events WHERE video_id = ? AND validation_status = 'REJECTED'", (vid,)).fetchone()[0]
unc_dets = c.execute("SELECT count(*) FROM events WHERE video_id = ? AND validation_status = 'UNCERTAIN'", (vid,)).fetchone()[0]
tracks_cnt = c.execute('SELECT count(*) FROM tracks WHERE video_id = ?', (vid,)).fetchone()[0]
inc_cnt = c.execute('SELECT count(*) FROM correlated_incidents WHERE video_id = ?', (vid,)).fetchone()[0]
ev_cnt = c.execute('SELECT count(*) FROM evidence WHERE video_id = ?', (vid,)).fetchone()[0]

theft_sec = c.execute("SELECT confidence, incident_metadata FROM security_events WHERE video_id = ? AND event_type = 'POTENTIAL_THEFT'", (vid,)).fetchone()
theft_corr = c.execute("SELECT assessment_score, evidence_strength, validation_decision FROM correlated_incidents WHERE video_id = ? AND incident_subcategory = 'potential_object_takeaway_pattern'", (vid,)).fetchone()

print(f'Raw detections: {raw_dets} (expected 138)')
print(f'Valid detections: {valid_dets} (expected 137)')
print(f'Rejected: {rej_dets} (expected 1)')
print(f'Uncertain: {unc_dets} (expected 0)')
print(f'Tracks: {tracks_cnt} (expected 29)')
print(f'Correlated incidents: {inc_cnt} (expected 12)')
print(f'Evidence artifacts: {ev_cnt} (expected 2)')
if theft_sec:
    meta = json.loads(theft_sec[1])
    print(f"Potential Theft (security_events): PRESENT, PatternEvidenceStrength={meta.get('pattern_evidence_strength')}, FinalAssessment={theft_sec[0]}, Decision={meta.get('validation_decision')}")
if theft_corr:
    print(f"Potential Theft (correlated_incidents): PRESENT, Score={theft_corr[0]}, EvidenceStrength={theft_corr[1]}, Decision={theft_corr[2]}")

# Check evidence playback files
for evid in ['ev_3983bb8f4de0', 'ev_f972d41d3a2d']:
    orig = f'storage/evidence/{evid}_clip.mp4'
    pb = f'storage/evidence_playback/{evid}_clip_playback.mp4'
    import os
    print(f'Evidence {evid}: orig_exists={os.path.exists(orig)}, pb_exists={os.path.exists(pb)}')
