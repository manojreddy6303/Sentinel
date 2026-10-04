"""
Sentinel Pre-Deployment User-Facing Smoke Test
Tests all 6 required user-facing scenarios against the running Sentinel backend & frontend.
"""
import urllib.request
import json
import sys

BASE_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://localhost:3000"

def get(endpoint):
    req = urllib.request.Request(f"{BASE_URL}{endpoint}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def post(endpoint, data):
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

results = {}

print("========================================================================")
print("SENTINEL PRE-DEPLOYMENT SMOKE TEST SUITE")
print("========================================================================")

# 0. Health checks
health = get("/api/health")
print(f"Backend Health: {health.get('status')} ({health.get('service')})")

with urllib.request.urlopen(FRONTEND_URL) as resp:
    print(f"Frontend HTTP Status: {resp.status}")

# 1. Canonical Burglary Video
print("\n--- 1. CANONICAL BURGLARY VIDEO (0d4d92f9-19f8-42e3-925f-1931cb557705) ---")
burglary_id = "0d4d92f9-19f8-42e3-925f-1931cb557705"
meta_b = get(f"/api/videos/{burglary_id}")
tracks_b = get(f"/api/videos/{burglary_id}/tracks")
incidents_b = get(f"/api/videos/{burglary_id}/correlated-incidents")
events_b = get(f"/api/videos/{burglary_id}/events")
evidence_b = get(f"/api/videos/{burglary_id}/evidence")

inc_list = incidents_b.get("correlated_incidents", [])
inc_subcats = [i.get("incident_subcategory", "") for i in inc_list]
event_types = [e.get("event_type", "") for e in events_b.get("events", [])]

print(f"Frames processed: {meta_b.get('frames_processed')}")
print(f"Tracks count: {len(tracks_b.get('tracks', []))}")
print(f"Correlated incidents: {len(inc_list)} -> subcategories: {set(inc_subcats)}")
print(f"Evidence items count: {len(evidence_b.get('evidence', []))}")

has_theft = any("takeaway" in s.lower() or "theft" in s.lower() for s in inc_subcats) or any("theft" in e.lower() for e in event_types)
has_false_fight = any("altercation" in s.lower() or "fight" in s.lower() for s in inc_subcats)
has_evidence = len(evidence_b.get("evidence", [])) > 0

results["BURGLARY THEFT"] = "PASS" if (has_theft and not has_false_fight and has_evidence) else "FAIL"
print(f"Burglary Verification: theft/takeaway={has_theft}, false_altercation={has_false_fight}, evidence={has_evidence} -> {results['BURGLARY THEFT']}")

# 2. Blue-dress / Small-object / WhatsApp Mobile Video
print("\n--- 2. SMALL-OBJECT / THEFT-LIKE VIDEO (1287e521-1f3a-46bb-8c95-c369f81926dc) ---")
small_obj_id = "1287e521-1f3a-46bb-8c95-c369f81926dc"
tracks_s = get(f"/api/videos/{small_obj_id}/tracks")
incidents_s = get(f"/api/videos/{small_obj_id}/correlated-incidents")
classes_s = {t.get("object_class") for t in tracks_s.get("tracks", [])}
inc_subcats_s = [i.get("incident_subcategory", "") for i in incidents_s.get("correlated_incidents", [])]
print(f"Tracked classes: {classes_s}")
print(f"Incidents: {inc_subcats_s}")
results["SMALL-OBJECT/THEFT VIDEO"] = "PASS" if len(tracks_s.get("tracks", [])) > 0 else "FAIL"
print(f"Small-object / Mobile Verification: -> {results['SMALL-OBJECT/THEFT VIDEO']}")

# 3. Normal / No-incident Video
print("\n--- 3. NORMAL / NO-INCIDENT VIDEO (VIRAT: b118ab48-0fc5-4e14-8524-0205b27a831d) ---")
normal_id = "b118ab48-0fc5-4e14-8524-0205b27a831d"
incidents_n = get(f"/api/videos/{normal_id}/correlated-incidents")
inc_n_subcats = [i.get("incident_subcategory", "") for i in incidents_n.get("correlated_incidents", [])]
print(f"Clean video incidents count: {len(inc_n_subcats)}")
no_fake_theft = not any("theft" in s.lower() or "takeaway" in s.lower() for s in inc_n_subcats)
no_fake_fire = not any("fire" in s.lower() for s in inc_n_subcats)
no_fake_alt = not any("altercation" in s.lower() or "fight" in s.lower() for s in inc_n_subcats)
results["NO-INCIDENT VIDEO"] = "PASS" if (no_fake_theft and no_fake_fire and no_fake_alt) else "FAIL"
print(f"No-incident Verification: no_theft={no_fake_theft}, no_fire={no_fake_fire}, no_alt={no_fake_alt} -> {results['NO-INCIDENT VIDEO']}")

# 4. Retail / Warm-color / False-fire Protection
print("\n--- 4. FALSE FIRE PROTECTION ---")
all_incidents = get("/api/incidents")
fire_incidents = [i for i in all_incidents.get("incidents", []) if "fire" in str(i.get("incident_subcategory", "")).lower() or "fire" in str(i.get("incident_category", "")).lower()]
print(f"Platform-wide false fire incidents on non-fire videos: {len(fire_incidents)}")
results["FALSE FIRE"] = "PASS" if len(fire_incidents) == 0 else "FAIL"
print(f"False Fire Protection: -> {results['FALSE FIRE']}")

# 5. 4K UHD Video
print("\n--- 5. 4K UHD / MOBILE VIDEO (3426f64b-dd44-48a7-8e29-2c5f77b748bb) ---")
uhd_id = "3426f64b-dd44-48a7-8e29-2c5f77b748bb"
meta_u = get(f"/api/videos/{uhd_id}")
tracks_u = get(f"/api/videos/{uhd_id}/tracks")
print(f"4K Video Status: {meta_u.get('status')}")
print(f"4K Tracks: {len(tracks_u.get('tracks', []))}")
results["4K/MOBILE"] = "PASS" if (meta_u.get("status") == "processed" and len(tracks_u.get("tracks", [])) > 0) else "FAIL"
print(f"4K / Mobile Verification: -> {results['4K/MOBILE']}")

# 6. Investigation UI Grounded Answers
print("\n--- 6. INVESTIGATION UI (POST /api/videos/{id}/ai-investigate) ---")
q1 = post(f"/api/videos/{burglary_id}/ai-investigate", {"query": "What happened?"})
print("Q: What happened?")
print(f"Mode: {q1.get('mode')}")
print(f"Answer:\n{q1.get('answer')}")

q2 = post(f"/api/videos/{burglary_id}/ai-investigate", {"query": "Was anything taken?"})
print("\nQ: Was anything taken?")
print(f"Mode: {q2.get('mode')}")
print(f"Answer:\n{q2.get('answer')}")

ans1 = q1.get("answer", "")
ans2 = q2.get("answer", "")
q1_valid = "theft" in ans1.lower() or "takeaway" in ans1.lower() or "tracks" in ans1.lower()
q2_valid = "theft" in ans2.lower() or "takeaway" in ans2.lower() or "taken" in ans2.lower() or "suitcase" in ans2.lower()
results["INVESTIGATION"] = "PASS" if (q1_valid and q2_valid) else "FAIL"
print(f"\nInvestigation Verification: q1_grounded={q1_valid}, q2_grounded={q2_valid} -> {results['INVESTIGATION']}")

print("\n========================================================================")
print("FINAL SMOKE TEST SUMMARY")
print("========================================================================")
all_pass = all(v == "PASS" for v in results.values())
results["SMOKE TEST"] = "PASS" if all_pass else "FAIL"

for k in ["SMOKE TEST", "BURGLARY THEFT", "SMALL-OBJECT/THEFT VIDEO", "NO-INCIDENT VIDEO", "FALSE FIRE", "4K/MOBILE", "INVESTIGATION"]:
    print(f"{k}: {results[k]}")
