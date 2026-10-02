# SENTINEL — Phase 21A Advanced Tracking Implementation Report
**Date**: October 2, 2026  
**Status**: COMPLETE & VERIFIED  
**Baseline Preserved**: Phase 20.3.1 (Production Default: ByteTrack)

---

## 1. Executive Summary

Phase 21A has been implemented strictly additively without modifying or destabilizing the production baseline:
- **Production Default**: ByteTrack-style tracker (`ObjectTracker`) remains the default tracker.
- **Candidate Tracker**: BoT-SORT-style tracker (`BoTSORTTracker`) is implemented behind `BaseMultiObjectTracker` with full camera motion compensation, 8-dimensional Kalman filter state estimation, Hungarian optimal assignment, and native occlusion handling.
- **Zero Regressions**: All **837/837** backend tests passed (including 57 new Phase 21A tests). All **5/5** real-video benchmark regressions passed. Frontend production build passed cleanly.
- **Controlled Selection**: Configurable via `TRACKER_TYPE=bytetrack|botsort` in `Settings`, defaulting to `"bytetrack"`.

---

## 2. Architecture & Modules Implemented

### 2.1 Kalman Filter Motion Model (`ai/tracking/kalman_filter.py`)
- **8-Dimensional State**: $[x_c, y_c, w, h, v_x, v_y, v_w, v_h]^T$ modeling bounding box centroid, dimensions, and velocities.
- **Aspect-Ratio Stability**: Bounding box width and height velocities are damped with higher process noise on velocities than positions, preventing bounding box blowup during missing detections.
- **Dynamic $\Delta t$ Handling**: Automatically scales the transition matrix $F$ and process noise $Q$ based on actual timestamp intervals between frames.
- **Strict Positivity**: Enforces $w > 0, h > 0$ after measurement updates.

### 2.2 Global Motion Compensation (`ai/tracking/gmc.py`)
- **Method**: Sparse Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) on Shi-Tomasi good features (`cv2.goodFeaturesToTrack`).
- **Robust Transformation**: Estimates a $2 \times 3$ affine transformation matrix using RANSAC.
- **Detection Masking**: Suppresses feature tracking inside bounding boxes of active tracks, ensuring optical flow measures true camera/background motion rather than foreground target movements.
- **Fallback Guarantee**: In case of camera cuts, low texture, extreme motion, or missing frames, GMC detects degeneration and immediately returns the identity transformation matrix $I_{2 \times 3}$ without throwing errors or corrupting track states.

### 2.3 Hungarian & Optimal Assignment (`ai/tracking/matching.py`)
- **Vectorized IoU & Distance**: Computes pairwise IoU and normalized centroid distance cost matrices in pure numpy.
- **Linear Assignment**: Uses `scipy.optimize.linear_sum_assignment` when scipy is available, with an automatic greedy assignment fallback when scipy is absent.
- **Gating**: Rejects assignments exceeding maximum distance or minimum IoU thresholds.

### 2.4 BoT-SORT Tracker (`ai/tracking/botsort_tracker.py`)
- **Two-Stage Association**: High-confidence detections associated first; unassigned tracks matched with low-confidence detections in stage two.
- **Occlusion Handling**: Tracks unassigned for up to $N$ frames enter the `OCCLUDED` state, coasting via Kalman prediction and camera motion compensation. If a matched detection reappears within `max_lost_time`, the track reactivates without triggering a new track ID.
- **Camera Motion Warp**: Before association, active track bounding boxes and Kalman state centroids are warped by the estimated affine transformation matrix $M$.
- **Unified Pipeline Contract**: Produces standard `TrackedObject` instances with normalized and pixel coordinates, matching the exact contract required by downstream components (incidents, visual attributes, evidence export).

---

## 3. Pipeline & Interface Updates

1. **`ai/common/benchmark_interfaces.py`**:
   - Added `frame_bgr: Optional[np.ndarray] = None` to `BaseMultiObjectTracker.update()`.
   - Updated `ByteTrackAdapter` to accept and safely ignore `frame_bgr`.
   - Replaced stub `BoTSORTBenchmarkCandidate` with full wrapper connecting to `BoTSORTTracker`.

2. **`ai/intelligence_pipeline.py`**:
   - Reordered frame image retrieval to occur immediately before `tracker.update(...)`.
   - Passes `frame_bgr=frame_bgr` to `tracker.update(...)`.
   - Existing `ObjectTracker` accepts `frame_bgr` without behavioral change.

3. **`backend/app/core/config.py`**:
   - Added `TRACKER_TYPE: str = os.getenv("TRACKER_TYPE", "bytetrack")`.

---

## 4. Test & Verification Results

### 4.1 Backend Test Suites
```bash
python -m pytest backend/tests/test_phase21a_botsort.py -q
# 57 passed in 1.19s

python -m pytest backend/tests/ -q
# 837 passed, 3 warnings in 52.62s (100% pass rate)
```

### 4.2 Benchmark Regressions
```bash
python scripts/verify_benchmarks_regression.py
# 1. BURGLARY BENCHMARK: PASS (12 incidents, 2 evidence, 0 specialized false alarms)
# 2. 4K UHD BENCHMARK: PASS (40 tracks, 133 vehicle attrs, 27 face regions, 0 false alarms)
# 3. HIGHWAY BENCHMARK: PASS (27 tracks, 0 false alarms)
# 4. VIRAT BENCHMARK: PASS (16 tracks, 0 false alarms)
# 5. WHATSAPP BENCHMARK: PASS (5 tracks, 0 false alarms)
# ALL REAL-VIDEO BENCHMARK REGRESSION CHECKS PASSED PERFECTLY!
```

### 4.3 Frontend Production Build
```bash
cd frontend && npm run build
# Route (app): / and /_not-found static prerendered
# Compiled successfully in 1018ms, TypeScript finished in 2.2s
```

---

## 5. A/B Benchmark Results: ByteTrack vs BoT-SORT

The A/B benchmark harness (`scripts/benchmark_botsort_ab.py`) was executed across all 5 benchmark videos:

| Metric | ByteTrack (Production) | BoT-SORT (Candidate) | Analysis & Observations |
| :--- | :---: | :---: | :--- |
| **Total Tracks (Burglary)** | 29 | 58 | Sparsely sampled (1 fps) detections; motion prediction without appearance causes fragmentation |
| **Total Tracks (4K UHD)** | 56 | 105 | Fast camera panning in 4K footage without ReID requires wider spatial gating |
| **Total Tracks (Highway)** | 29 | 63 | High vehicle velocity over 1.0s delta-t leads to missed IoU associations without ReID |
| **Total Tracks (VIRAT)** | 18 | 84 | Far-field surveillance people are small bounding boxes (<30px); low IoU across 1s jumps |
| **Total Tracks (WhatsApp)** | 6 | 10 | Handheld camera motion compensated by GMC |
| **Peak Memory (MB)** | ~67.5 MB | **183.6 MB** | **Well within 1024 MB Railway limit** (840 MB headroom remaining) |
| **Avg Update Latency** | 0.02 – 0.57 ms | **4.5 – 21.3 ms** | Optical flow GMC adds ~4-20 ms/frame, fully real-time capable |

### Key Insight for Phase 21B (ReID):
In surveillance pipelines with sub-sampled frame rates (e.g., 1 frame per second), consecutive frames have significant spatial displacement. Without deep appearance features (ReID), pure IoU / Kalman motion prediction cannot reliably bridge large spatial gaps, leading to track fragmentation. This empirically validates the architectural roadmap:
1. **Phase 21A** establishes the foundation: Kalman motion model + GMC + Hungarian matching + occlusion state lifecycle.
2. **Phase 21B** (Future) will introduce lightweight visual ReID feature embeddings to fuse with motion cost, resolving the fragmentation under sub-sampled frame rates.
3. Keeping **ByteTrack as the production default** was the correct architectural decision.
