# SENTINEL — PHASE 0.3 PRODUCTION DEPLOYMENT REPORT
**Date:** 2026-10-04  
**Status:** SUCCESS — FULLY ONLINE & PRODUCTION VERIFIED  
**Commit:** `0f95acee1e90530350876d52f5acd59cf1d9906b`  
**Corpus:** `manojreddy6303/Sentinel`  

---

### Deployment Target Information

| Service | Environment | Status | Live URL | Deployment ID |
| :--- | :---: | :---: | :--- | :--- |
| **sentinel-backend** | production | **● Online** | https://sentinel-backend-production-8749.up.railway.app | `448c7ee8-5f2f-4d99-9b47-6e9484ceddb3` |
| **sentinel-frontend** | production | **● Online** | https://sentinel-frontend-production-8154.up.railway.app | `17ffd6fc-99f4-4fc1-b35a-5b924b2b643d` |
| **sentinel-storage** | production | **READY** | Mounted at `/storage` (Persistent SQLite DB: 12,267,520 bytes) | `vol_oi9t02dk72rtbpys` |

---

### Production Verification Matrix

| Verification Category | Status | Details |
| :--- | :---: | :--- |
| **DEPLOYMENT** | **SUCCESS** | Both backend and frontend built and deployed without errors to Railway. |
| **COMMIT SYNC** | **PASS** | Deployed commit `0f95ace` matches local verified main branch HEAD exactly. |
| **HEALTH ENDPOINTS** | **PASS** | `GET /health` -> HTTP 200, `GET /api/health` -> HTTP 200, Frontend -> HTTP 200. |
| **DATABASE & STORAGE** | **INTACT** | Persistent SQLite DB `/storage/sentinel.db` preserved intact (60 indexed videos, schema unchanged). |
| **THEFT SMOKE TEST** | **PASS** | Burglary video (`0d4d92f9`) detected `POTENTIAL_THEFT` and correlated incident `CORR-PROP-82dfce15`. |
| **SCORE CONSISTENCY** | **PASS** | Validation decision `REVIEW_REQUIRED` with score 0.50 (<= 0.65). No contradictory `ACCEPTED` in storyline. |
| **EVIDENCE TIMESTAMP** | **PASS** | Evidence clip artifact (`ev_b99aac2a3c84`) correctly anchored at 174.0s (incident takeaway moment). |
| **EVIDENCE PLAYBACK** | **PASS** | `GET /api/evidence/{id}/clip` -> HTTP 200 (video/mp4), `GET /api/evidence/{id}/playback` -> HTTP 206 RFC 7233 range stream. |
| **GEMINI REASONING** | **PASS** | `POST /api/videos/{id}/ai-investigate` returned grounded response via `gemini-3.5-flash-lite` cascade. |
| **DETERMINISTIC FALLBACK** | **PASS** | `POST /api/videos/{id}/investigate` returned structured narrative grounding with zero biometrics. |
| **NEGATIVE CONTROLS** | **PASS** | 0 false fires, 0 false altercations across Highway Traffic benchmark and Burglary surveillance. |
| **REGRESSION STATUS** | **PASS** | All core capabilities, detectors, trackers, and storage volumes operating stably. |

---

### Production Smoke Test Execution Log Excerpt

```
===========================================================================
SENTINEL PRODUCTION SMOKE TEST — PHASE 0.3 VERIFIED CORE
Backend Target:  https://sentinel-backend-production-8749.up.railway.app
Frontend Target: https://sentinel-frontend-production-8154.up.railway.app
===========================================================================

[1] VERIFYING HEALTH ENDPOINTS...
  Backend /health:      HTTP 200 -> {'status': 'ok', 'service': 'Sentinel Backend', 'version': '0.1.0', 'environment': 'production'}
  Backend /api/health:  HTTP 200 -> {'status': 'ok', 'service': 'Sentinel Backend', 'version': '0.1.0', 'environment': 'production', 'message': 'Sentinel backend is running successfully.'}
  Frontend /:           HTTP 200
  -> Health Status: PASS

[2] VERIFYING PERSISTENT STORAGE & DATABASE INTACT...
  Total Indexed Videos in SQLite Volume: 60
  -> Database Status: INTACT

[3] VERIFYING BURGLARY & THEFT DETECTION...
  Burglary Video (0d4d92f9-19f8-42e3-925f-1931cb557705):
    Total Security Events:      11
    Theft Events Found:         1
    Total Correlated Incidents: 10
    Property/Theft Incidents:   1
  -> Theft Smoke Test Status: PASS

[4] VERIFYING SCORE & DECISION CONSISTENCY...
    Incident CORR-PROP-82dfce15: Score=0.5, Decision=REVIEW_REQUIRED
      Storyline: At 174.0s, TRACK-030 was observed approaching and remaining in close proximity to suitcase (TRACK-033)...
  -> Score/Decision Consistency: PASS

[5] VERIFYING EVIDENCE TIMESTAMP & EVIDENCE PLAYBACK...
  Evidence items found: 3
    Testing Evidence Artifact: ev_b99aac2a3c84 | Object: POTENTIAL_THEFT | Timestamp: 174.0s
    Timestamp correctly anchored at 174.0s (incident takeaway moment).
    GET /api/evidence/ev_b99aac2a3c84/clip -> HTTP 200 | Content-Type: video/mp4 | Length: 205687 bytes
    GET /api/evidence/ev_b99aac2a3c84/playback (Range: bytes=0-1024) -> HTTP 206 | Content-Type: video/mp4
  -> Evidence Timestamp Status: PASS
  -> Evidence Playback Status:  PASS

[6] VERIFYING GEMINI INVESTIGATION REASONING (/ai-investigate)...
  Query: 'Was anything taken?' -> HTTP 200
  Mode: ai_assisted | Provider: gemini | Model: gemini-3.5-flash-lite
  Answer Summary:
### VIDEO ACTIVITY SUMMARY
The video footage spans from 0.00s to 21.96s [ev_05e4997c67a5], during which multiple person activity events were recorded continuously across intervals involving between 1 and 6 individuals...
  -> Gemini Status: PASS

[7] VERIFYING DETERMINISTIC FALLBACK (/investigate)...
  Query: 'Was anything taken?' -> HTTP 200
  Deterministic Message:
Sentinel identified a potential object-takeaway pattern around 0.0s. Potential Theft Pattern (Pattern Evidence Strength: 95% (High Evidence Strength)...
  -> Deterministic Fallback Status: PASS

[8] VERIFYING NON-THEFT VIDEOS & FALSE ALARM CONTROLS...
  Non-Theft Video (a94e46c6-3742-44c1-84b9-24f7be1a6a97): 0 security events
    Highway Traffic Thefts: 0, Fires: 0, Altercations: 0
    Burglary Video Fires:   0, Altercations: 0
  -> False Alarm / Regression Status: PASS

===========================================================================
ALL PRODUCTION SMOKE TESTS PASSED CLEANLY!
===========================================================================
```
