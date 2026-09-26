import sqlite3

con = sqlite3.connect('storage/sentinel.db')
c = con.cursor()
videos = c.execute('SELECT count(*) FROM videos').fetchone()[0]
cameras = c.execute('SELECT count(*) FROM camera_sources').fetchone()[0]
cases = c.execute('SELECT count(*) FROM cases').fetchone()[0]
incidents = c.execute('SELECT count(*) FROM correlated_incidents').fetchone()[0]
evidence_valid = c.execute('SELECT count(*) FROM evidence WHERE validation_status="VALID"').fetchone()[0]
evidence_total = c.execute('SELECT count(*) FROM evidence').fetchone()[0]
cyber_events = c.execute('SELECT count(*) FROM cyber_security_events').fetchone()[0]
raw = c.execute('SELECT count(*) FROM events').fetchone()[0]
valid = c.execute('SELECT count(*) FROM events WHERE validation_status="VALID"').fetchone()[0]
rej = c.execute('SELECT count(*) FROM events WHERE validation_status="REJECTED"').fetchone()[0]
unc = c.execute('SELECT count(*) FROM events WHERE validation_status="UNCERTAIN"').fetchone()[0]

print("=" * 60)
print("SENTINEL CANONICAL PRODUCTION BASELINE INTEGRITY CHECK")
print("=" * 60)
print(f"VIDEOS: {videos} (Expected: 21)")
print(f"CAMERAS: {cameras} (Expected: 3)")
print(f"CASES: {cases} (Expected: 6)")
print(f"CORRELATED INCIDENTS: {incidents} (Expected: 42)")
print(f"VALIDATED EVIDENCE: {evidence_valid} (Expected: 4, Total: {evidence_total})")
print(f"CYBERSECURITY EVENTS: {cyber_events}")
print(f"DETECTION PARITY: RAW={raw} | VALID={valid} + REJECTED={rej} + UNCERTAIN={unc} = {valid+rej+unc}")

assert videos == 21, f"Expected 21 videos, got {videos}"
assert cameras == 3, f"Expected 3 cameras, got {cameras}"
assert cases == 6, f"Expected 6 cases, got {cases}"
assert incidents == 42, f"Expected 42 incidents, got {incidents}"
assert evidence_valid == 4, f"Expected 4 validated evidence, got {evidence_valid}"
assert raw == 1244, f"Expected 1244 raw events, got {raw}"
assert valid == 1062, f"Expected 1062 valid events, got {valid}"
assert rej == 129, f"Expected 129 rejected events, got {rej}"
assert unc == 53, f"Expected 53 uncertain events, got {unc}"
assert raw == (valid + rej + unc), "Parity invariant violated!"

print("=" * 60)
print("ALL BASELINE AND DETECTION PARITY CHECKS PASSED PERFECTLY!")
print("=" * 60)
con.close()
