# SENTINEL — Final Hackathon Deployment Report
**Deployment Phase**: Final Hackathon Stabilization Production Deployment  
**Date**: October 2, 2026  
**Status**: SUCCESSFUL & LIVE ON RAILWAY  
**Environment**: Railway Production (`production`)

---

## 1. Deployment Metadata

| Metric / Attribute | Value |
|---|---|
| **Deployed Commit SHA** | `ae78b4138e0b06c94bdde83bb942f9be66826fca` (Short: `ae78b41`) |
| **Commit Message** | `fix(hackathon): deploy verified stabilization` |
| **Branch** | `main` → `origin/main` (`manojreddy6303/Sentinel`) |
| **Railway Project ID** | `d92c5f86-f996-4c20-8fb3-1db3423fa265` |
| **Railway Environment ID** | `6c9ff4b3-3fdd-4187-9b2c-296e0a410e91` (`production`) |
| **Railway Backend Service** | `sentinel-backend` (`c843f7b0-85a3-4cf3-ae3e-981bab153714`) |
| **Railway Deployment ID** | `2fb00abe-9e58-414e-bcea-adc131bbe9fb` |
| **Deployment Status** | `SUCCESS` (● Online) |
| **Backend Public URL** | `https://sentinel-backend-production-8749.up.railway.app` |
| **Frontend Public URL** | `https://sentinel-frontend-production-8154.up.railway.app` |

---

## 2. Production Baseline Invariants

All baseline constraints were preserved without regression or unauthorized modification:
- **Default Tracker**: `TRACKER_TYPE=bytetrack` (BoT-SORT candidate preserved as inactive research option)
- **Production Sampling**: `1.0 FPS` (Unchanged)
- **Correlation Mode**: `SAMPLING_AWARE_CORRELATION_ENABLED=false` (Production legacy correlation baseline active)
- **Memory Safety**: Phase 20.3.1 thread-bounding (`TORCH_NUM_THREADS=2`) and bounded playback transcode active
- **Database Schema**: No migration, no schema alteration, no historical record deletion

---

## 3. Health-Check & Service Status

| Endpoint | Method | Response Status | Verification Output | Status |
|---|:---:|:---:|---|:---:|
| `/health` | GET | `200 OK` | `{"status":"ok","service":"Sentinel Backend","version":"0.1.0","environment":"production"}` | ✅ PASS |
| `/api/health` | GET | `200 OK` | `{"status":"ok","service":"Sentinel Backend","version":"0.1.0","environment":"production","message":"Sentinel backend is running successfully."}` | ✅ PASS |
| Frontend URL | GET | `200 OK` | Next.js 16.3.4 (Turbopack) production web app loaded | ✅ PASS |

---

## 4. Database & Persistent Storage Preservation

The persistent SQLite database and media storage volume (`sentinel-storage`) mounted at `/storage` remain completely intact with zero data loss:

| Storage Entity | Pre-Deployment Count | Post-Deployment Count | Status |
|---|:---:|:---:|:---:|
| **Cases** | 6 | 6 | ✅ Intact (Zero data loss) |
| **Videos** | 50 | 50 | ✅ Intact |
| **Correlated Incidents** | 276 | 276 | ✅ Intact |
| **Evidence Records** | 29 | 29 | ✅ Intact |
| **Persistent Volume** | 210 MB / 500 MB | 210 MB / 500 MB | ✅ Mounted & Healthy (`/storage`) |
| **WAL Mode** | Valid | Valid | ✅ Intact |
| **Foreign-Key Violations** | 0 | 0 | ✅ Zero integrity violations |

---

## 5. Live Production Smoke-Test Results

Exercised directly against the live production endpoints:

1. **Evidence Clip Playback Fix (`ev_f32068b5bd6c`)**:
   - **Audit Condition**: Previously returned HTTP 500 because container headroom appeared to be ~0.4 MB due to Linux page-cache accounting.
   - **Post-Deploy Request**: `GET https://sentinel-backend-production-8749.up.railway.app/api/evidence/ev_f32068b5bd6c/playback` with `Range: bytes=0-1023`
   - **Response Code**: **HTTP 206 Partial Content**
   - **Headers**:
     - `Content-Type: video/mp4`
     - `Content-Range: bytes 0-1023/1266861`
   - **Result**: ✅ PASS — Evidence playback operational without HTTP 500.

2. **Natural-Language Visual Investigation Fix**:
   - **Query**: `"what is blue colour dress lady did"`
   - **Request**: `POST https://sentinel-backend-production-8749.up.railway.app/api/videos/{id}/ai-investigate`
   - **Response Code**: **HTTP 200 OK**
   - **Answer Content**:
     > *"Found 1 verified track(s) for person with blue clothing (TRACK-002). The individual was observed from 0.0s to 19.0s (19.0s duration, 20 detections)... (Note: Observational tracking only; zero personal identity attribution or biometric identification)."*
   - **Result**: ✅ PASS — Accurately parsed attire nouns, accumulated multi-frame clothing colors, and returned grounded non-biometric activity narrative even under external Gemini quota limits.

3. **Retail Merchandise False Fire Suppression**:
   - **Verification**: Verified zero false fire incidents or events created from warm retail packaging on store shelves. Genuine fire and theft/takeaway detection remain intact.
   - **Result**: ✅ PASS — Merchandise specular highlights cleanly suppressed.

4. **Container & Memory Stability**:
   - **OOM Events**: 0
   - **SIGKILL / Exit 137**: 0
   - **Unplanned Restarts**: 0
   - **Result**: ✅ PASS — Container remains `● Online` and completely stable.

---

## 6. Remaining Notes & Warnings

- **Railway Config Deprecation Notice**: Railway CLI logged a standard deprecation warning (`Config as Code is deprecated. Prefer Infrastructure as Code`). This does not affect live operations (supported until December 2026).
- **External Gemini Quotas**: When Google Gemini free tier rate limit (20 req/day) is exceeded, Sentinel automatically falls back to deterministic grounded query synthesis without any hallucination or crash.

---

HACKATHON STABILIZATION IS NOW LIVE ON RAILWAY.
