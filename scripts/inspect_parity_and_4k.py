import sqlite3

conn = sqlite3.connect('storage/sentinel.db')
c = conn.cursor()

print("=" * 60)
print("TABLE ROW COUNTS:")
print("=" * 60)
for table in ['videos', 'events', 'grouped_events', 'evidence', 'tracks', 'vehicle_attributes', 'face_detections', 'security_zones', 'security_events', 'reports', 'specialized_observations', 'correlated_incidents', 'cases', 'camera_sources']:
    c.execute(f'SELECT COUNT(*) FROM {table}')
    cnt = c.fetchone()[0]
    print(f'{table:25}: {cnt}')

print("\n" + "=" * 60)
print("PLATFORM-WIDE EVENTS PARITY:")
print("=" * 60)
c.execute("SELECT COUNT(*) FROM events")
total_raw = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM events WHERE validation_status = 'VALID'")
total_val = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM events WHERE validation_status = 'REJECTED'")
total_rej = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM events WHERE validation_status = 'UNCERTAIN'")
total_unc = c.fetchone()[0]
print(f"Total RAW events:       {total_raw}")
print(f"Total VALID events:     {total_val}")
print(f"Total REJECTED events:  {total_rej}")
print(f"Total UNCERTAIN events: {total_unc}")
print(f"Sum (V + R + U):        {total_val + total_rej + total_unc}")
print(f"Formula: RAW({total_raw}) = VALID({total_val}) + REJECTED({total_rej}) + UNCERTAIN({total_unc})")
print(f"Platform-wide parity match: {total_raw == (total_val + total_rej + total_unc)}")

print("\n" + "=" * 60)
print("EVENTS PER VIDEO:")
print("=" * 60)
c.execute("""
SELECT v.id, v.original_filename, COUNT(e.id),
       SUM(CASE WHEN e.validation_status = 'VALID' THEN 1 ELSE 0 END),
       SUM(CASE WHEN e.validation_status = 'REJECTED' THEN 1 ELSE 0 END),
       SUM(CASE WHEN e.validation_status = 'UNCERTAIN' THEN 1 ELSE 0 END)
FROM videos v
LEFT JOIN events e ON v.id = e.video_id
GROUP BY v.id, v.original_filename
""")
for row in c.fetchall():
    print(f"Video {row[0][:12]}... | {row[1]:35} | Total={row[2]} | Valid={row[3]} | Rej={row[4]} | Unc={row[5]}")

print("\n" + "=" * 60)
print("CHECKING FOR 204 IN ANY COLUMN / AGGREGATE:")
print("=" * 60)
# Check if any combinations sum to 204 or 193
c.execute("SELECT validation_status, COUNT(*) FROM events GROUP BY validation_status")
print("Events by validation status:", c.fetchall())

# Check vehicle_attributes, face_detections, specialized_observations, tracks
c.execute("SELECT COUNT(*) FROM vehicle_attributes")
print("Vehicle attributes count:", c.fetchone()[0])
c.execute("SELECT COUNT(*) FROM face_detections")
print("Face detections count:", c.fetchone()[0])
c.execute("SELECT COUNT(*) FROM tracks")
print("Tracks count:", c.fetchone()[0])

conn.close()
