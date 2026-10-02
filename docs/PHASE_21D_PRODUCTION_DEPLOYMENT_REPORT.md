# SENTINEL — Phase 21D Production Deployment & Verification Report

**Date:** October 2, 2026  
**Status:** SUCCESSFUL & PRODUCTION-VERIFIED  
**Final State:** Production defaults strictly preserved (`TRACKER_TYPE=bytetrack`, `SAMPLING_AWARE_CORRELATION_ENABLED=false`, `1 FPS` sampling regime).

---

## 1. Deployment Metadata

| Property | Value |
|---|---|
| **Commit SHA** | `1669147b4fae0d16c5b9679f1db7eb904fec8b50` (Short: `1669147`) |
| **Commit Message** | `feat(correlation): add sampling-aware incident correlation` |
| **Branch** | `main` → `origin/main` |
| **Railway Project ID** | `d92c5f86-f996-4c20-8fb3-1db3423fa265` |
| **Railway Environment** | `production` (`6c9ff4b3-3fdd-4187-9b2c-296e0a410e91`) |
| **Railway Service** | `sentinel-backend` (`c843f7b0-85a3-4cf3-ae3e-981bab153714`) |
| **Railway Deployment ID** | `9ed4ef86-3d9d-4822-be6c-2ac4b0c5e0b5` |
| **Deployment Status** | `SUCCESS` |
| **Backend Public URL** | `https://sentinel-backend-production-8749.up.railway.app` |
| **Frontend Public URL** | `https://sentinel-frontend-production-8154.up.railway.app` / `https://sentinel-frontend-production.up.railway.app` |

---

## 2. Production Configuration Verification

The running production container configuration was verified against the application defaults:

| Configuration Key | Runtime Value | Invariant Check |
|---|---|:---:|
| `TRACKER_TYPE` | `bytetrack` | ✅ PASS (Production default preserved; BoT-SORT remains research candidate) |
| `SAMPLING_AWARE_CORRELATION_ENABLED` | `False` (`0`) | ✅ PASS (Production baseline preserved; legacy correlation active) |
| `CORRELATION_TEMPORAL_COOLDOWN_SECONDS` | `5.0` | ✅ PASS (Intended baseline cooldown available when enabled) |
| `FRAME_SAMPLING_RATE_FPS` | `1.0` | ✅ PASS (1.0 FPS production sampling regime unchanged) |
| `PLAYBACK_MEMORY_HARDENING` | Enabled | ✅ PASS (Phase 20.3.1 bounded transcode intact) |
| `TORCH_NUM_THREADS` | `2` | ✅ PASS (Phase 20.3 thread bounding intact) |

---

## 3. Health & Frontend Verification

| Endpoint | Method | Response Code | Details | Status |
|---|:---:|:---:|---|:---:|
| `/health` | GET | `200 OK` | `{"status":"ok","service":"Sentinel Backend","version":"0.1.0","environment":"production"}` | ✅ PASS |
| `/api/health` | GET | `200 OK` | `{"status":"ok","service":"Sentinel Backend","version":"0.1.0","environment":"production","message":"Sentinel backend is running successfully."}` | ✅ PASS |
| Frontend URL | GET / HEAD | `200 OK` | Next.js 16.3.4 (Turbopack) production shell served (7,178 bytes, Cache-Control valid) | ✅ PASS |

---

## 4. Database & Persistent Volume Integrity

The persistent SQLite database and media storage volume (`sentinel-storage`) mounted at `/storage` remain completely intact with zero data loss or migration drift:

| Storage Entity | Pre-Deployment Count | Post-Deployment Count | Verification Status |
|---|:---:|:---:|:---:|
| **Cases** | 6 | 6 | ✅ Intact (Zero data loss) |
| **Videos** | 48 | 49 | ✅ Intact (+1 verified live test video) |
| **Correlated Incidents** | 266 | 276 | ✅ Intact (+10 verified live test incidents) |
| **Evidence Records** | Preserved | Preserved | ✅ Intact |
| **Persistent Volume** | 210 MB / 500 MB | 210 MB / 500 MB | ✅ Mounted & Healthy (`/storage`) |
| **WAL Mode** | Valid | Valid | ✅ Intact |
| **Foreign-Key Violations** | 0 | 0 | ✅ Zero integrity violations |

---

## 5. Playback Verification (MP4 & AVI Hardening)

Both standard MP4 streaming and memory-bounded AVI playback conversion were exercised in production:

1. **MP4 Video Playback** (`748d1c03-5856-46cd-b697-6d79d48cc6e6`):
   - Request: `GET /api/videos/748d1c03-5856-46cd-b697-6d79d48cc6e6/playback` with `Range: bytes=0-1023`
   - Response: **HTTP 206 Partial Content**
   - Content-Type: `video/mp4`
   - Content-Range: `bytes 0-1023/5248590`
   - Result: ✅ PASS — Streaming byte-range support verified.
2. **Legacy AVI Playback** (`76445915-92da-442c-bf99-451522e586df`):
   - Request: `GET /api/videos/76445915-92da-442c-bf99-451522e586df/playback`
   - Response: **HTTP 200 OK**
   - Content-Type: `video/mp4`
   - Transcoded Bytes Delivered: `551,712 bytes`
   - Post-Transcode Health: Immediate **HTTP 200 OK** on `/api/health`
   - Result: ✅ PASS — Phase 20.3.1 memory-bounded transcode protections preserved.

---

## 6. Real-Video Processing Verification (Production API)

A controlled real surveillance video (`VIRAT_S_010204_05_000856_000890.mp4`, 5.25 MB) was uploaded and processed through the live production API:

- **Assigned Video ID**: `da169fc7-b79f-4f39-a0a1-f6a6fb8c62be`
- **Upload Latency**: 17.92s (`HTTP 201 Created`)
- **Processing Time**: 10.33s (`HTTP 200 OK`)
- **Sampling Verification**: 25 frames processed across 24.36s duration ($\approx 1.0\text{ FPS}$ production sampling confirmed)
- **Detections Produced**: 187 valid YOLO detections
- **ByteTrack Tracking**: 18 active tracks confirmed
- **Validation Engine**: 187 validated events, 0 uncertain
- **Correlated Incidents**: **10 correlated incidents**
  *(Matches the exact 1 FPS legacy production baseline for VIRAT, proving that legacy correlation remains active as the production default)*
- **Specialized Threat False Alarms**: 0 (Weapon/fire/smoke false alarms strictly zero)
- **Post-Processing Health Check**: **HTTP 200 OK** (Container remained stable and responsive)

---

## 7. Memory & Container Stability

- **OOM / Out-of-Memory Events**: **0**
- **SIGKILL / Exit 137 Events**: **0**
- **Railway Unplanned Restarts**: **0**
- **Monotonic Leakage**: None observed.
- **Container State**: `● Online` with healthy cgroup headroom.

---

## 8. Existing Feature Regression Verification

All major SENTINEL investigation and security subsystems were checked post-deployment:
- `GET /api/videos` → 200 OK (Lists all ingested media)
- `GET /api/videos/{id}/tracks` → 200 OK (Returns ByteTrack tracking telemetry)
- `GET /api/videos/{id}/events` → 200 OK (Returns validated spatial/temporal events)
- `GET /api/videos/{id}/correlated-incidents` → 200 OK (Returns incident storylines)
- `GET /api/videos/{id}/evidence` → 200 OK (Returns forensic snapshots)
- `GET /api/cases` → 200 OK (Preserves active investigative case files)
- `GET /api/analytics` → 200 OK (Returns system-wide security analytics)

---

## 9. Final Decision & Status

- **Status**: Complete & Verified in Production.
- **Production Baseline**: ByteTrack @ 1.0 FPS with Legacy Correlation remains the active default.
- **Forensic Sampling-Aware Feature**: Deployed safely behind additive configuration flags (`SAMPLING_AWARE_CORRELATION_ENABLED=False`), ready for optional or controlled forensic evaluation.

---
**PHASE 21D PRODUCTION DEPLOYMENT COMPLETE — PRODUCTION DEFAULTS PRESERVED.**
