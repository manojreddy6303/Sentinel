import sqlite3

con = sqlite3.connect('storage/sentinel.db')
c = con.cursor()
print("Incidents for 0d4d92f9-19f8-42e3-925f-1931cb557705 (CAM-NORTH-01):")
for r in c.execute('SELECT id, incident_category, start_time, end_time, assessment_score, validation_decision, storyline FROM correlated_incidents WHERE video_id="0d4d92f9-19f8-42e3-925f-1931cb557705"'):
    print(" ", r)

print("\nEvidence for 0d4d92f9-19f8-42e3-925f-1931cb557705:")
for r in c.execute('SELECT id, event_id, timestamp_seconds, evidence_type, object_class, confidence, snapshot_path, clip_path FROM evidence WHERE video_id="0d4d92f9-19f8-42e3-925f-1931cb557705"'):
    print(" ", r)

con.close()
