import sqlite3

conn = sqlite3.connect('storage/sentinel.db')
cursor = conn.cursor()

print("=" * 60)
print("1. BURGLARY BENCHMARK (0d4d92f9-19f8-42e3-925f-1931cb557705)")
print("=" * 60)
vid_burglary = '0d4d92f9-19f8-42e3-925f-1931cb557705'
cursor.execute('SELECT original_filename, status FROM videos WHERE id = ?', (vid_burglary,))
row = cursor.fetchone()
print(f"Video: {row}")
assert row[0] == 'uccrime_Burglary010_x264.mp4'

cursor.execute('SELECT COUNT(*) FROM tracks WHERE video_id = ?', (vid_burglary,))
tr_count = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM specialized_observations WHERE video_id = ? AND class_name IN ("smoke", "fire", "weapon")', (vid_burglary,))
spec_obs = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM security_events WHERE video_id = ? AND event_type IN ("smoke_detected", "fire_detected", "weapon_detected")', (vid_burglary,))
spec_ev = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM correlated_incidents WHERE video_id = ?', (vid_burglary,))
inc_count = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM evidence WHERE video_id = ?', (vid_burglary,))
ev_count = cursor.fetchone()[0]

print(f"Burglary Tracks: {tr_count}")
print(f"Burglary Specialized Obs: {spec_obs}")
print(f"Burglary Specialized Events: {spec_ev}")
print(f"Burglary Correlated Incidents: {inc_count}")
print(f"Burglary Evidence: {ev_count}")

assert spec_obs == 0, f"Expected 0 specialized observations for burglary, got {spec_obs}"
assert spec_ev == 0, f"Expected 0 specialized events for burglary, got {spec_ev}"
assert inc_count == 12, f"Expected 12 correlated incidents, got {inc_count}"
assert ev_count == 2, f"Expected 2 evidence artifacts, got {ev_count}"

print("\n" + "=" * 60)
print("2. 4K UHD BENCHMARK (3426f64b-dd44-48a7-8e29-2c5f77b748bb)")
print("=" * 60)
vid_4k = '3426f64b-dd44-48a7-8e29-2c5f77b748bb'
cursor.execute('SELECT original_filename, status FROM videos WHERE id = ?', (vid_4k,))
row_4k = cursor.fetchone()
print(f"4K Video: {row_4k}")
assert row_4k[0] == '12566041-uhd_3840_2160_30fps.mp4'

cursor.execute('SELECT COUNT(*) FROM tracks WHERE video_id = ?', (vid_4k,))
tr_4k = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM vehicle_attributes WHERE video_id = ?', (vid_4k,))
attr_4k = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM face_detections WHERE video_id = ?', (vid_4k,))
faces_4k = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM specialized_observations WHERE video_id = ? AND class_name IN ("smoke", "fire", "weapon")', (vid_4k,))
spec_4k = cursor.fetchone()[0]

print(f"4K Tracks: {tr_4k} (expected 40)")
print(f"4K Vehicle Attributes: {attr_4k} (expected 133)")
print(f"4K Anonymous Face Regions: {faces_4k} (expected 27)")
print(f"4K Specialized (Smoke/Fire/Weapon): {spec_4k} (expected 0)")

assert tr_4k == 40, f"Expected 40 tracks, got {tr_4k}"
assert attr_4k == 133, f"Expected 133 vehicle attributes, got {attr_4k}"
assert faces_4k == 27, f"Expected 27 anonymous face regions, got {faces_4k}"
assert spec_4k == 0, f"Expected 0 specialized observations, got {spec_4k}"

print("\n" + "=" * 60)
print("3. HIGHWAY / TRAFFIC BENCHMARK (a94e46c6-3742-44c1-84b9-24f7be1a6a97)")
print("=" * 60)
vid_hw = 'a94e46c6-3742-44c1-84b9-24f7be1a6a97'
cursor.execute('SELECT original_filename, status FROM videos WHERE id = ?', (vid_hw,))
row_hw = cursor.fetchone()
print(f"Highway Video: {row_hw}")
assert row_hw[0] == '17.avi'

cursor.execute('SELECT COUNT(*) FROM tracks WHERE video_id = ?', (vid_hw,))
tr_hw = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM security_events WHERE video_id = ?', (vid_hw,))
ev_hw = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM specialized_observations WHERE video_id = ? AND class_name IN ("smoke", "fire", "weapon")', (vid_hw,))
spec_hw = cursor.fetchone()[0]

print(f"Highway Tracks: {tr_hw}")
print(f"Highway Security Events: {ev_hw}")
print(f"Highway Specialized (Smoke/Fire/Weapon): {spec_hw}")

assert tr_hw > 0, "Expected tracks for highway video"
assert spec_hw == 0, f"Expected 0 specialized observations for highway, got {spec_hw}"

conn.close()
print("\n" + "=" * 60)
print("ALL REAL-VIDEO BENCHMARK REGRESSION CHECKS PASSED PERFECTLY!")
print("=" * 60)
