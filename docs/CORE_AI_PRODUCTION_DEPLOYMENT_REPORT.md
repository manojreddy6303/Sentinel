# Sentinel Core AI Production Deployment Report
**Phase 0.2: Universal Micro-Object & Object-Interaction Recovery + Phase 0.1 Generalization**
**Date:** 2026-10-04  
**Deployment Target:** Railway Cloud Production  
**Status:** VERIFIED OPERATIONAL & COMPLIANT

---

## 1. Executive Summary

The verified Sentinel Core AI pipeline featuring universal micro-object recovery, interaction trajectory grouping, robust takeaway reasoning, and cross-video false alarm controls has been successfully built and deployed to Railway production.

Both `sentinel-backend` and `sentinel-frontend` services are running `Online` on Railway with commit `8708424e1ad58f2e476623dbd15ac924f521ab1e`. All health endpoints returned HTTP 200, the persistent volume storage remains intact with all historical video datasets preserved, and production smoke tests against the retail/blue-lady case validated end-to-end small portable object recovery, potential takeaway pattern detection, human review gating, and grounded AI investigation.

---

## 2. Deployment Architecture & Services

| Service | Railway Service ID | Production URL | Deployment ID | Status | Commit SHA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Backend** | `c843f7b0-85a3-4cf3-ae3e-981bab153714` | `https://sentinel-backend-production-8749.up.railway.app` | `5cec0813-9153-44d7-a7b8-74b56c845b17` | `SUCCESS` (Online) | `8708424e1ad58f2e476623dbd15ac924f521ab1e` |
| **Frontend** | `d098e80c-ab01-4cbb-863f-ba755e706d62` | `https://sentinel-frontend-production-8154.up.railway.app` | `fbe05a32-f427-48a7-b9fc-3357ee234168` | `SUCCESS` (Online) | `8708424e1ad58f2e476623dbd15ac924f521ab1e` |
| **Persistent Volume** | `sentinel-storage` | `/storage` (348.4 MB / 500 MB) | N/A | `READY` (Intact) | N/A |

### Health Check Verification
- `GET https://sentinel-backend-production-8749.up.railway.app/health` -> **200 OK**
  ```json
  {"status": "ok", "service": "Sentinel Backend", "version": "0.1.0", "environment": "production"}
  ```
- `GET https://sentinel-backend-production-8749.up.railway.app/api/health` -> **200 OK**
  ```json
  {"status": "ok", "service": "Sentinel Backend", "version": "0.1.0", "environment": "production", "message": "Sentinel API is healthy"}
  ```
- `GET https://sentinel-frontend-production-8154.up.railway.app` -> **200 OK** (HTML 200 with Next.js SPA dashboard)

---

## 3. Persistent Database Verification

- **Database Path:** `/storage/sentinel.db` (12,267,520 bytes)
- **Volume Mount:** `/storage` mounted from volume `sentinel-storage`
- **Total Registered Videos:** 58 videos preserved across all historical runs
- **Integrity Status:** INTACT — zero records lost, zero schema migrations broken

---

## 4. Production Smoke Test: Small-Object Theft & Takeaway Case

### Test Video Details
- **Video ID:** `5e68d2cc-b315-4534-af53-ec3be34ad076`
- **Original Filename:** `WhatsApp Video 2026-09-26 at 06.14.37.mp4`
- **Resolution:** 768x432 (retail store counter scenario with small portable object)

### Live Pipeline Execution
- **Pipeline Status:** Completed (`HTTP 200`)
- **Detections Count:** 43 detections generated across 22 sampled frames
- **Micro-Object Recovery:** Successfully localized and tracked unmodeled item as `unknown_portable_object` (`TRACK-003`, `TRACK-006`) during dwelling interval of person (`TRACK-001`).

### Live Incident Generation
- **Incident ID:** `FUSED-51420c47`
- **Incident Type:** `POTENTIAL_THEFT`
- **Object Class:** `unknown_portable_object`
- **Pattern Evidence Strength:** 95% (High Evidence Strength)
- **Validation Decision:** `REVIEW_REQUIRED` (`human_verification_required: true`)
- **Description:**  
  `"Potential Theft Pattern (Pattern Evidence Strength: 95% (High Evidence Strength) · Final Assessment: 95% (ACCEPTED) · Sensor-grounded pattern): Person [TRACK-001] approached unknown_portable_object [TRACK-003], remained in proximity for 18.0s, after which the object co-moved with departing person while person departed (193.1px displacement). (Fused across 2 observational segments)."`
- **Observable Signals:**
  - `Interaction Proximity: Person [TRACK-001] approached unknown_portable_object [TRACK-003] (dwell: 18.0s)`
  - `Takeaway Pattern: Pattern: co_movement_takeaway (person departure displacement: 193.1px)`
  - `Object Grounding: Target unknown_portable_object [TRACK-003] grounded at 1.0s`
  - `Takeaway Pattern: Pattern: object_disappearance (person departure displacement: 214.6px)`

### Live AI Investigation Endpoint
- **Endpoint:** `POST /api/videos/5e68d2cc-b315-4534-af53-ec3be34ad076/ai-investigate`
- **Query:** `"Was anything taken?"`
- **Live Response:**
  ```markdown
  ### VIDEO ACTIVITY SUMMARY
  - A potential theft/takeaway sequence was detected.
  - A person was observed interacting with an object/property.
  - The object interaction was followed by movement/removal consistent with takeaway behavior.
  - Relevant activity occurred around the detected incident interval [0.0s - 20.0s].
  - Preserved evidence is available for review.

  ### OBSERVED EVIDENCE
  - **Security Event:** Potential Theft Pattern (potential object-takeaway pattern) detected around 0.0s.
  - **Visual Telemetry:** Potential Theft Pattern (Pattern Evidence Strength: 95% (High Evidence Strength) ... Sensor-grounded pattern): Person [TRACK-001] approached unknown_portable_object [TRACK-003], remained in proximity for 18.0s, after which the object co-moved with departing person while person departed (193.1px displacement). (Fused across 2 observational segments).
  - **Preserved Records:** 2 forensic artifact(s) registered in vault.

  ### INTERPRETATION
  - Evidence is consistent with a potential theft / potential takeaway pattern.
  - Review recommended (Human verification required; Sentinel reports observational patterns and does not establish legal culpability).
  ```

---

## 5. Regression Controls & Safety

- **False Fire Regression:** 0 false fire incidents detected.
- **False Physical Altercation Regression:** 0 false altercation incidents detected.
- **Evidence Integrity:** Review gating strictly enforced with `REVIEW_REQUIRED` preserved.
- **Memory Safety:** Bounded frame cache, mini-batching, and heavy-processing semaphore operational.

---

## 6. Deployment Sign-off

| Check | Result |
| :--- | :--- |
| **Git Working Tree** | Clean, commit `8708424` pushed to `origin/main` |
| **Backend Deployment** | SUCCESS (Online at `https://sentinel-backend-production-8749.up.railway.app`) |
| **Frontend Deployment** | SUCCESS (Online at `https://sentinel-frontend-production-8154.up.railway.app`) |
| **Health Endpoints** | PASS (`/health` = 200, `/api/health` = 200, frontend = 200) |
| **Deployed Commit Match** | PASS (`8708424e1ad58f2e476623dbd15ac924f521ab1e` on both services) |
| **Database Persistence** | PASS (58 videos intact, `/storage/sentinel.db` preserved) |
| **Theft Smoke Test** | PASS (`unknown_portable_object` + `POTENTIAL_THEFT` + `REVIEW_REQUIRED`) |
| **AI Investigation** | PASS (Explains potential takeaway and points to visual evidence) |
| **Regression Controls** | PASS (0 false fire, 0 false altercation) |
