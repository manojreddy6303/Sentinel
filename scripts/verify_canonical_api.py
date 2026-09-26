import urllib.request
import json
import sys

def get_json(url, data=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8') if data else None,
        headers={'Content-Type': 'application/json'} if data else {}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

base = 'http://127.0.0.1:8000'
vid_id = '0d4d92f9-19f8-42e3-925f-1931cb557705'

print("=== 1. SYSTEM HEALTH ===")
health = get_json(f'{base}/api/health')
print('Status:', health.get('status'))

print("\n=== 2. PLATFORM ANALYTICS ===")
analytics = get_json(f'{base}/api/analytics')
print(f'Total Videos (Platform-Wide): {analytics.get("total_videos")}')
print(f'Total Correlated Incidents (Platform-Wide): {analytics.get("total_correlated_incidents")}')
print(f'Total Evidence (Platform-Wide): {analytics.get("total_evidence")}')
print(f'Total Cases (Platform-Wide): {analytics.get("total_cases")}')

print("\n=== 3. CANONICAL BENCHMARK METADATA ===")
meta = get_json(f'{base}/api/videos/{vid_id}')
fname = meta.get("filename") or meta.get("original_filename")
print(f'Video: {fname} ({meta.get("video_id") or meta.get("id")})')
print(f'Duration: {meta.get("duration_seconds")}s | FPS: {meta.get("fps")} | Frames: {meta.get("frames_processed") or meta.get("frame_count")}')

print("\n=== 4. CANONICAL DETECTIONS & VALIDATION ===")
events = get_json(f'{base}/api/videos/{vid_id}/events?include_unvalidated=true')
ev_list = events.get('events', [])
valid_ev = [e for e in ev_list if e.get('validation_status') == 'VALID']
rej_ev = [e for e in ev_list if e.get('validation_status') == 'REJECTED']
unc_ev = [e for e in ev_list if e.get('validation_status') == 'UNCERTAIN']
print(f'Raw Detections: {len(ev_list)}')
print(f'Validated: {len(valid_ev)}')
print(f'Rejected: {len(rej_ev)}')
print(f'Uncertain: {len(unc_ev)}')
print(f'Sum Check (137 + 1 + 0 = 138): {len(valid_ev) + len(rej_ev) + len(unc_ev) == len(ev_list)}')

print("\n=== 5. ANONYMOUS TRACKS ===")
tracks_res = get_json(f'{base}/api/videos/{vid_id}/tracks')
tracks_list = tracks_res.get('tracks', [])
print(f'Anonymous Tracks: {len(tracks_list)}')

print("\n=== 6. CORRELATED INCIDENTS ===")
inc_res = get_json(f'{base}/api/videos/{vid_id}/correlated-incidents')
inc_list = inc_res.get('correlated_incidents', [])
print(f'Correlated Incidents: {len(inc_list)}')

print("\n=== 7. PRESERVED EVIDENCE ===")
ev_res = get_json(f'{base}/api/videos/{vid_id}/evidence')
evidence_list = ev_res.get('evidence', [])
print(f'Evidence Items Count: {len(evidence_list)}')
for e in evidence_list:
    eid = e.get("evidence_id") or e.get("id")
    ts = e.get("timestamp") or e.get("timestamp_seconds")
    print(f'  - ID: {eid} | Class: {e.get("object_class")} | Timestamp: {ts}s')

print("\n=== 8. POTENTIAL THEFT & SECURITY EVENTS ===")
sec_res = get_json(f'{base}/api/videos/{vid_id}/security-events')
sec_list = sec_res.get('events', [])
theft = next((s for s in sec_list if s.get('event_type') == 'POTENTIAL_THEFT'), None)
print(f'Security Events Total: {len(sec_list)}')
print(f'Potential Theft Present: {bool(theft)}')
if theft:
    meta_th = theft.get('incident_metadata', {})
    print(f'  Pattern Evidence Strength: {meta_th.get("pattern_evidence_strength")}')
    print(f'  Assessment Score: {meta_th.get("assessment_score")}')
    print(f'  Confidence: {theft.get("confidence")}')
    print(f'  Validation Decision: {meta_th.get("validation_decision")}')
    print(f'  Track IDs: {meta_th.get("track_ids")}')
    print(f'  Description: {theft.get("description")}')

print("\n=== 9. GROUNDED AI INVESTIGATION FALLBACK ===")
ai_res = get_json(
    f'{base}/api/videos/{vid_id}/ai-investigate',
    {'query': 'Summarize the investigation findings and preserved evidence'}
)
print(f'Mode: {ai_res.get("mode")}')
print(f'Is Supported: {ai_res.get("is_supported")}')
print(f'Sources Evidence Count: {len(ai_res.get("sources", {}).get("evidence", []))}')
for line in ai_res.get('answer', '').split('\n'):
    if any(k in line.lower() for k in ['evidence', 'theft', 'tracks', 'incidents', 'provider', 'preserved']):
        print(f'  {line}')
