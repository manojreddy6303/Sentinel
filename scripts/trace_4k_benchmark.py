import sqlite3
import os
import json

conn = sqlite3.connect('storage/sentinel.db')
c = conn.cursor()

c.execute("SELECT * FROM videos WHERE original_filename = '12566041-uhd_3840_2160_30fps.mp4'")
cols = [desc[0] for desc in c.description]
rows = c.fetchall()

print("=" * 60)
print("4K BENCHMARK VIDEO TRACE")
print("=" * 60)
print(f"Matching video rows: {len(rows)}")
for r in rows:
    row_dict = dict(zip(cols, r))
    for k, v in row_dict.items():
        print(f"  {k:20}: {v}")

vid = rows[0][0]

# Check storage path
storage_path = rows[0][3]
print(f"\nStorage path exists: {os.path.exists(storage_path)} ({storage_path})")

# Check cases linked
c.execute("""
SELECT c.id, c.title, c.case_number 
FROM cases c 
JOIN case_videos cv ON c.id = cv.case_id 
WHERE cv.video_id = ?
""", (vid,))
linked_cases = c.fetchall()
print(f"Linked Cases: {linked_cases}")

# Check tracks
c.execute("SELECT COUNT(*) FROM tracks WHERE video_id = ?", (vid,))
tr_cnt = c.fetchone()[0]
# Check vehicle attributes
c.execute("SELECT COUNT(*) FROM vehicle_attributes WHERE video_id = ?", (vid,))
va_cnt = c.fetchone()[0]
# Check face detections
c.execute("SELECT COUNT(*) FROM face_detections WHERE video_id = ?", (vid,))
fd_cnt = c.fetchone()[0]
# Check specialized observations
c.execute("SELECT COUNT(*) FROM specialized_observations WHERE video_id = ?", (vid,))
so_cnt = c.fetchone()[0]
# Check correlated incidents
c.execute("SELECT COUNT(*) FROM correlated_incidents WHERE video_id = ?", (vid,))
ci_cnt = c.fetchone()[0]
# Check evidence
c.execute("SELECT COUNT(*) FROM evidence WHERE video_id = ?", (vid,))
ev_cnt = c.fetchone()[0]
# Check events (detections)
c.execute("SELECT COUNT(*), validation_status FROM events WHERE video_id = ? GROUP BY validation_status", (vid,))
ev_breakdown = c.fetchall()

print(f"\n--- Benchmark Metrics for Video {vid} ---")
print(f"Tracks:                  {tr_cnt} (Benchmark expectation: 40)")
print(f"Vehicle Attributes:      {va_cnt} (Benchmark expectation: 133)")
print(f"Face Detections (Anon):  {fd_cnt} (Benchmark expectation: 27)")
print(f"Specialized Obs:         {so_cnt} (Benchmark expectation: 0)")
print(f"Correlated Incidents:    {ci_cnt}")
print(f"Evidence Artifacts:      {ev_cnt}")
print(f"Events by Validation:    {ev_breakdown}")

# Check if there are any other videos matching 4k or 3840
c.execute("SELECT id, original_filename, uploaded_at FROM videos WHERE original_filename LIKE '%3840%' OR original_filename LIKE '%4k%' OR original_filename LIKE '%4K%'")
all_4k_vids = c.fetchall()
print(f"\nAll 3840/4K videos in database:")
for v in all_4k_vids:
    print(f"  ID: {v[0]} | Filename: {v[1]} | Uploaded: {v[2]}")

conn.close()
