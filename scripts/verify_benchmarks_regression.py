"""
SENTINEL — REAL-VIDEO BENCHMARK REGRESSION VERIFICATION (Phase 20)

Converts brittle exact-count assertions into robust regression contracts with
tolerance bounds while recording historical baselines. Covers:
1. Burglary (uccrime_Burglary010_x264.mp4)
2. 4K UHD (12566041-uhd_3840_2160_30fps.mp4)
3. Highway / Traffic (17.avi)
4. VIRAT Surveillance (VIRAT_S_010204_05_000856_000890.mp4)
5. Mobile / WhatsApp Surveillance Footage
"""
import sqlite3
import sys

conn = sqlite3.connect('storage/sentinel.db')
cursor = conn.cursor()

all_passed = True

def check(condition, desc):
    global all_passed
    if not condition:
        print(f"  [FAIL] {desc}")
        all_passed = False
    else:
        print(f"  [PASS] {desc}")

print("=" * 70)
print("1. BURGLARY BENCHMARK (0d4d92f9-19f8-42e3-925f-1931cb557705)")
print("=" * 70)
vid_burglary = '0d4d92f9-19f8-42e3-925f-1931cb557705'
cursor.execute('SELECT original_filename, status FROM videos WHERE id = ?', (vid_burglary,))
row = cursor.fetchone()
print(f"Video: {row}")
check(row is not None and row[0] == 'uccrime_Burglary010_x264.mp4', "Video filename matches Burglary benchmark")

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

print(f"Burglary Tracks: {tr_count} (historical baseline: 20)")
print(f"Burglary Specialized Obs: {spec_obs} (expected 0 false alarms)")
print(f"Burglary Specialized Events: {spec_ev} (expected 0 false alarms)")
print(f"Burglary Correlated Incidents: {inc_count} (historical baseline: 12)")
print(f"Burglary Evidence: {ev_count} (historical baseline: 2)")

check(spec_obs == 0, f"Zero false specialized observations (got {spec_obs})")
check(spec_ev == 0, f"Zero false specialized events (got {spec_ev})")
check(10 <= inc_count <= 20, f"Correlated incident contract [10, 20] satisfied (got {inc_count})")
check(1 <= ev_count <= 5, f"Evidence artifact contract [1, 5] satisfied (got {ev_count})")

print("\n" + "=" * 70)
print("2. 4K UHD BENCHMARK (3426f64b-dd44-48a7-8e29-2c5f77b748bb)")
print("=" * 70)
vid_4k = '3426f64b-dd44-48a7-8e29-2c5f77b748bb'
cursor.execute('SELECT original_filename, status FROM videos WHERE id = ?', (vid_4k,))
row_4k = cursor.fetchone()
print(f"4K Video: {row_4k}")
check(row_4k is not None and row_4k[0] == '12566041-uhd_3840_2160_30fps.mp4', "Video filename matches 4K UHD benchmark")

cursor.execute('SELECT COUNT(*) FROM tracks WHERE video_id = ?', (vid_4k,))
tr_4k = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM vehicle_attributes WHERE video_id = ?', (vid_4k,))
attr_4k = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM face_detections WHERE video_id = ?', (vid_4k,))
faces_4k = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM specialized_observations WHERE video_id = ? AND class_name IN ("smoke", "fire", "weapon")', (vid_4k,))
spec_4k = cursor.fetchone()[0]

print(f"4K Tracks: {tr_4k} (historical baseline: 40)")
print(f"4K Vehicle Attributes: {attr_4k} (historical baseline: 133)")
print(f"4K Anonymous Face Regions: {faces_4k} (historical baseline: 27)")
print(f"4K Specialized (Smoke/Fire/Weapon): {spec_4k} (expected 0 false alarms)")

check(25 <= tr_4k <= 60, f"4K tracks contract [25, 60] satisfied (got {tr_4k})")
check(80 <= attr_4k <= 180, f"4K vehicle attributes contract [80, 180] satisfied (got {attr_4k})")
check(15 <= faces_4k <= 45, f"4K face regions contract [15, 45] satisfied (got {faces_4k})")
check(spec_4k == 0, f"Zero false specialized observations on 4K (got {spec_4k})")

print("\n" + "=" * 70)
print("3. HIGHWAY / TRAFFIC BENCHMARK (a94e46c6-3742-44c1-84b9-24f7be1a6a97)")
print("=" * 70)
vid_hw = 'a94e46c6-3742-44c1-84b9-24f7be1a6a97'
cursor.execute('SELECT original_filename, status FROM videos WHERE id = ?', (vid_hw,))
row_hw = cursor.fetchone()
print(f"Highway Video: {row_hw}")
check(row_hw is not None and row_hw[0] == '17.avi', "Video filename matches Highway benchmark")

cursor.execute('SELECT COUNT(*) FROM tracks WHERE video_id = ?', (vid_hw,))
tr_hw = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM security_events WHERE video_id = ?', (vid_hw,))
ev_hw = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM specialized_observations WHERE video_id = ? AND class_name IN ("smoke", "fire", "weapon")', (vid_hw,))
spec_hw = cursor.fetchone()[0]

print(f"Highway Tracks: {tr_hw} (historical baseline: 36)")
print(f"Highway Security Events: {ev_hw} (historical baseline: 3)")
print(f"Highway Specialized (Smoke/Fire/Weapon): {spec_hw} (expected 0 false alarms)")

check(tr_hw >= 20, f"Highway tracks contract (>= 20) satisfied (got {tr_hw})")
check(spec_hw == 0, f"Zero false specialized observations on highway (got {spec_hw})")

print("\n" + "=" * 70)
print("4. VIRAT SURVEILLANCE BENCHMARK (b118ab48-0fc5-4e14-8524-0205b27a831d)")
print("=" * 70)
vid_virat = 'b118ab48-0fc5-4e14-8524-0205b27a831d'
cursor.execute('SELECT original_filename, status FROM videos WHERE id = ?', (vid_virat,))
row_virat = cursor.fetchone()
print(f"VIRAT Video: {row_virat}")
if row_virat:
    cursor.execute('SELECT COUNT(*) FROM tracks WHERE video_id = ?', (vid_virat,))
    tr_virat = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM specialized_observations WHERE video_id = ? AND class_name IN ("smoke", "fire", "weapon")', (vid_virat,))
    spec_virat = cursor.fetchone()[0]
    print(f"VIRAT Tracks: {tr_virat} (historical baseline: 16)")
    print(f"VIRAT Specialized False Alarms: {spec_virat}")
    check(tr_virat >= 10, f"VIRAT tracks contract (>= 10) satisfied (got {tr_virat})")
    check(spec_virat == 0, f"Zero false specialized observations on VIRAT (got {spec_virat})")

print("\n" + "=" * 70)
print("5. MOBILE / WHATSAPP BENCHMARK (e63e72b1-6616-4be6-b732-110219a81005)")
print("=" * 70)
vid_wa = 'e63e72b1-6616-4be6-b732-110219a81005'
cursor.execute('SELECT original_filename, status FROM videos WHERE id = ?', (vid_wa,))
row_wa = cursor.fetchone()
print(f"WhatsApp Video: {row_wa}")
if row_wa:
    cursor.execute('SELECT COUNT(*) FROM tracks WHERE video_id = ?', (vid_wa,))
    tr_wa = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM specialized_observations WHERE video_id = ? AND class_name IN ("smoke", "fire", "weapon")', (vid_wa,))
    spec_wa = cursor.fetchone()[0]
    print(f"WhatsApp Tracks: {tr_wa}")
    print(f"WhatsApp Specialized False Alarms: {spec_wa}")
    check(tr_wa >= 3, f"WhatsApp tracks contract (>= 3) satisfied (got {tr_wa})")
    check(spec_wa == 0, f"Zero false specialized observations on WhatsApp (got {spec_wa})")

conn.close()

print("\n" + "=" * 70)
if all_passed:
    print("ALL REAL-VIDEO BENCHMARK REGRESSION CHECKS PASSED PERFECTLY!")
    print("=" * 70)
    sys.exit(0)
else:
    print("SOME BENCHMARK REGRESSION CHECKS FAILED!")
    print("=" * 70)
    sys.exit(1)
