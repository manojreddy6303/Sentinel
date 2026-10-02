# SENTINEL — Prototype Reliability & User-Facing Failure Audit
**Diagnosis Document (Read-Only)**  
**Date:** October 2, 2026  
**Status:** DIAGNOSIS COMPLETE — NO CODE OR PRODUCTION CHANGES MADE  

---

## Executive Overview

Following successful automated backend and regression testing, end-to-end evaluation on the live public prototype identified three distinct user-facing failures across investigation, threat detection, and media playback. 

A comprehensive cross-layer audit (Frontend $\to$ API $\to$ Service $\to$ AI/CV $\to$ Database $\to$ Storage) was conducted on the live Railway deployment (`sentinel-backend-production-8749.up.railway.app`) and verified locally.

| Failure | Category | Observed Severity | Primary Root Cause Layer |
|---|---|:---:|---|
| **Failure 1** | Natural Language Investigation | **HIGH** | Query Parser (`is_person` regex omissions) & Attribute Persistence Gap |
| **Failure 2** | False Threat Detection | **HIGH** | Heuristic Fire Chromaticity Fallback & Pedestrian BBox Scope Limitation |
| **Failure 3** | Evidence Clip Playback | **CRITICAL** | OpenCV `mp4v` FourCC Incompatibility & Container Headroom Gate |

---

## Observed Failure 1 — Natural Language Investigation

### 1. User-Visible Symptom
When the user submits the natural language inquiry:
> `"what is blue colour dress lady did"`

The prototype interface returns a generic, unhelpful response:
> `"Found 11 matching records. PERSON_ACTIVITY activity was observed around [0.0s - 20.0s]."`

### 2. Expected Behavior
The system should:
1. Parse the non-biometric person descriptors: clothing color (`blue`), attire form (`dress`), and person noun (`lady` mapped as non-biometric person descriptor).
2. Retrieve the track corresponding to the person wearing blue clothing.
3. Retrieve the temporal security events and movements associated with that specific track.
4. Formulate an evidence-grounded summary of what that person did (e.g. walked down aisle, inspected products between 0.0s and 20.0s).

### 3. Actual Behavior
The system ignores the color and clothing intent, defaults to general `PERSON_ACTIVITY` timeline events, and synthesizes a template response describing generic activity across the entire video.

### 4. Reproduction Steps
1. Navigate to Search / Investigation Workspace on the frontend.
2. Select video `WhatsApp Video 2026-09-26 at 06.14.37 (1) (1).mp4` (`dbfac22e-c4ba-4b4c-ba4c-4e5d1ff39e43`).
3. Enter query `"what is blue colour dress lady did"`.
4. Observe the response: generic `PERSON_ACTIVITY` summary across `[0.0s - 20.0s]`.

### 5. Frontend Layer
- **Component**: [frontend/src/components/SearchWorkspaceView.tsx](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/frontend/src/components/SearchWorkspaceView.tsx)
- **API Call**: Calls `aiInvestigateVideo(videoId, { query })` in [frontend/src/lib/api.ts](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/frontend/src/lib/api.ts#L545-L569) targeting `POST /api/videos/{videoId}/ai-investigate`.
- The frontend correctly transmits the raw user string without truncation.

### 6. API Layer
- **Endpoint**: `POST /api/videos/{video_id}/ai-investigate` in [backend/app/api/videos.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/backend/app/api/videos.py#L900-L950)
- Receives `{ "query": "what is blue colour dress lady did" }`.
- Delegates execution to `InvestigationOrchestrator.process_investigation()`.

### 7. Service Layer
- **Orchestrator**: [ai/investigation/orchestrator.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/investigation/orchestrator.py#L80-L118)
  - Checks `self.provider.is_available()`. In production, Gemini API is either not configured or free-tier quota (limit: 20 req/day) is exhausted (`429 Resource Exhausted`), forcing `used_mode = "deterministic_fallback"`.
  - Invokes `self._deterministic_intent_fallback(cleaned_query)`.
- **Parser**: [backend/app/services/investigation_parser.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/backend/app/services/investigation_parser.py#L145-L188)
  - Color extraction extracts `matched_color = "blue"`.
  - Evaluates `is_vehicle`: `car|cars|vehicle|vehicles|truck|bus|motorcycle` $\to$ `False`.
  - Evaluates `is_person`:
    ```python
    is_person = bool(re.search(r"\b(person|people|someone|suspect|individual|man|woman|guy|girl|wearing|hoodie|jacket|shirt|clothing)\b", cleaned))
    ```
    The query contains `"dress"` and `"lady"`, neither of which is present in the regex. `is_person` evaluates to `False`.
  - Because neither vehicle nor person is matched, line 180 returns:
    ```python
    {"is_supported": False, "result_type": "unsupported", "message": "Visual color analysis is supported for vehicles and person clothing..."}
    ```
  - When the fallback or search queries grouped events by category, `InvestigationService.investigate_filters` returns the 11 `PERSON_ACTIVITY` grouped event records in the database.
  - `_format_deterministic_grounded_response` in [ai/investigation/orchestrator.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/investigation/orchestrator.py#L768-L795) formats the 11 records into:
    `"Found 11 matching records. PERSON_ACTIVITY activity was observed around [0.0s - 20.0s]."`

### 8. AI/CV Layer
- In [ai/intelligence_pipeline.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/intelligence_pipeline.py#L304-L328), `PersonAttributeAnalyzer.analyze()` correctly extracts upper/lower clothing color on each frame.
- **The Persistence Flaw**:
  ```python
  if p_attr.upper_clothing and p_attr.upper_clothing.is_confirmed:
      matched_track.color = p_attr.upper_clothing.color
      matched_track.color_confidence = p_attr.upper_clothing.confidence
  ```
  `ClothingColor.is_confirmed` in [ai/schemas.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/schemas.py#L280-L281) requires:
  `self.observation_count >= 3 and not self.is_illumination_uncertain and self.confidence >= 0.60`.
  Because `person_analyzer.analyze()` generates a fresh `ClothingColor` instance per frame with `observation_count = 1` without accumulating `observation_count` on the track, `is_confirmed` is **permanently `False`**.
  Consequently, `matched_track.color` is **never assigned**.

### 9. Database State
- Table `tracks` has columns: `id`, `video_id`, `track_id`, `object_class`, `first_seen`, `last_seen`, `duration_seconds`, `detection_count`, `max_confidence`, `color`, `color_confidence`.
- Inspection of `tracks` in `storage/sentinel.db` shows that **every person track has `color = NULL`**.
- There is no separate `person_attributes` table in the database (unlike `vehicle_attributes` and `face_detections`).

### 10. Storage State
- Frame crops and telemetry history remain only in transient memory during video processing; no persistent color attributes exist for person tracks on disk or in the database.

### 11. HTTP/Media Response
- Raw JSON response from `/api/videos/{id}/ai-investigate`:
  ```json
  {
    "query": "what is blue colour dress lady did",
    "mode": "deterministic_fallback",
    "is_supported": false,
    "answer": "Visual color analysis is supported for vehicles and person clothing. Sentinel can investigate object colors (e.g., blue person, red truck), classes, timestamps, and events.",
    "structured_query": { "intent": "investigate", "is_supported": false }
  }
  ```

### 12. Root Cause
1. **Parser Dictionary Gap**: `InvestigationParser` omits common non-biometric person/clothing words (`"dress"`, `"lady"`, `"skirt"`, `"coat"`, `"pants"`) from its `is_person` regex.
2. **Track Color Gating Defect**: `matched_track.color` is guarded by `p_attr.upper_clothing.is_confirmed`, which requires `observation_count >= 3`, but frame-level extraction never increments `observation_count` on the track.
3. **Database Schema Omission**: Person visual attributes are never persisted in a dedicated relational table, preventing SQL joins between attributes and events.
4. **Disjoint Query Intent**: When color search fails or is queried as a track, `InvestigationParser` maps it to `result_type: "tracks"`, which does not join with the track's security events or timeline activity.

### 13. Why Automated Tests Did Not Catch It
- Tests used pre-selected canonical test queries (e.g. `"blue person"`, `"red car"`, `"person at 10s"`).
- Synthetic test fixtures directly set `track.color = "blue"` instead of executing the full pipeline's `ClothingColor.is_confirmed` gate.
- No end-to-end test queried natural phrases like `"dress"` or `"lady"`.

### 14. Exact Files Responsible
- [backend/app/services/investigation_parser.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/backend/app/services/investigation_parser.py) (Lines 16, 145–188)
- [ai/intelligence_pipeline.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/intelligence_pipeline.py) (Lines 324–328)
- [ai/schemas.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/schemas.py) (Lines 280–282)
- [ai/investigation/orchestrator.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/investigation/orchestrator.py) (Lines 768–800)

### 15. Minimal Safe Fix (Future Implementation Plan)
1. In `investigation_parser.py`: Expand `OBJECT_SYNONYMS` and `is_person` regex to include `"lady"`, `"dress"`, `"skirt"`, `"attire"`, `"clothes"`, `"coat"`.
2. In `intelligence_pipeline.py`: When updating track attributes across frames, accumulate color votes in `track.attribute_history` so that after 3 consistent frames, `matched_track.color` is populated.
3. In `investigation_service.py`: When a user asks what a person/vehicle did (`"did"`, `"action"`), join the matched `track_id` with `events` and `security_events` to report their chronological actions.

### 16. Required Regression Tests
- Query with `"blue dress"`, `"lady in blue"`, `"what did lady in blue dress do"`.
- Verify `track.color` is populated from multi-frame person detections.
- Verify track activity is returned in deterministic fallback.

### 17. Whether Production Data is Affected
- Previously processed videos have `tracks.color = NULL`. Reprocessing or backfilling track colors would be required for past videos once patched.

---

## Observed Failure 2 — False Potential Fire

### 1. User-Visible Symptom
The UI displays an incident card:
> **POTENTIAL FIRE**  
> Final Assessment: 52% (or 53%)  
> **REVIEW_REQUIRED**  
> Timestamp: ~16.02 seconds  
> Bounding Box: `[x: 228, y: 231, x2: 278, y2: 270]`

The camera frame shows a retail store aisle with shelves and a customer wearing blue clothing. There is no fire or smoke.

### 2. Expected Behavior
The visual threat pipeline should recognize the illuminated store packaging as static merchandise on a retail shelf, verify the absence of flame turbulence/thermal expansion, and produce zero false fire observations.

### 3. Actual Behavior
The heuristic chromatic flame detector triggered on a yellow/orange retail package under bright store lights, passed through validation with a 52.86% score, and created a `REVIEW_REQUIRED` incident with evidence artifacts.

### 4. Reproduction Steps
1. Ingest `WhatsApp Video 2026-09-26 at 06.14.37 (1) (1).mp4` (`dbfac22e-c4ba-4b4c-ba4c-4e5d1ff39e43`).
2. Run standard video processing.
3. Open Incident Feed or Evidence Vault.
4. Observe `CORR-SPEC-a486a938` (`POTENTIAL_FIRE` at 16.02s).

### 5. Frontend Layer
- **Component**: [frontend/src/components/CorrelatedIncidentsView.tsx](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/frontend/src/components/CorrelatedIncidentsView.tsx)
- Displays the incident returned by `GET /api/incidents`.
- Renders:
  - Title: `POTENTIAL_FIRE` (or `Environment`)
  - Score: `53%` (`assessment_score: 0.5286`)
  - Status Badge: `REVIEW_REQUIRED`
  - Storyline: `"At 16.0s, specialized visual sensors recorded localized atmospheric/chromatic signatures (potential fire)..."`

### 6. API Layer
- **Endpoint**: `GET /api/incidents` and `GET /api/evidence`
- Faithfully exposes records from `correlated_incidents` and `evidence` tables.

### 7. Service Layer
- **Incident Correlator**: [ai/correlation/fusion_policies.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/correlation/fusion_policies.py#L1058-L1108) (`SpecializedThreatPolicy`)
  - Receives `IncidentCandidate` from `SpecializedIncidentDetector`.
  - Generates `CorrelatedIncident` with ID `CORR-SPEC-a486a938`, assigning `validation_decision = "REVIEW_REQUIRED"` and score `0.5286`.
- **Evidence Service**: [backend/app/services/evidence_service.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/backend/app/services/evidence_service.py)
  - Creates evidence record `ev_f32068b5bd6c` at $t = 16.02\text{s}$ with bounding box `[228, 231, 278, 270]`.

### 8. AI/CV Layer
1. **Triggering Pixels**:
   Pixel analysis of the actual snapshot crop at `[228, 231, 278, 270]` (50x39 pixels) shows:
   - Mean BGR: $B=138.3, G=190.0, R=197.4$
   - Mean YCrCb: $Y=186.3, Cr=135.9, Cb=100.9$ ($Y \ge Cb$, $Cr \ge Cb$, and $Cr - Cb = 35 \ge 32$)
   - Mean HSV: $H=26.0^\circ$ (within flame hue $[0, 30]^\circ$), $S=77.7$, $V=197.4$
   - Peak luminance: $V_{max} = 247.0$ (exceeds the 232.0 thermal threshold)
2. **Region Identity**:
   The region is a bright yellow/orange product box sitting on a middle store shelf.
3. **Pedestrian Suppression Failure**:
   Pedestrian clothing suppression in [ai/specialized/fire_smoke/detector.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/specialized/fire_smoke/detector.py#L212-L218) checks:
   `if (px1 - pw * 0.15) <= cnt_cx <= (px2 + pw * 0.15)`.
   The customer in the aisle was located at $x \in [442, 564]$. The shelf package was at $x \in [228, 278]$. Pedestrian clothing suppression did not trigger because the shelf is physically separate from the person.
4. **Validation Gate Failure**:
   In [ai/specialized/validator.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/specialized/validator.py#L182-L205):
   - `max_lum = 246.0 \ge 235.0` (passed incandescence check).
   - `std_lum = 11.2 > 6.5` (passed solid surface check due to packaging text/patterns).
   - Handheld camera motion caused contour area variations (`mean_area_cv > 0.04`), bypassing the static surface flicker check.
5. **Negative Evidence Behavior**:
   In [ai/incidents/detectors/specialized.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/incidents/detectors/specialized.py#L160-L168), `NegativeEvidenceEngine` noted the absence of smoke (`has_smoke = False`) and absence of flame flicker. This successfully demoted the incident from `ACCEPTED` to `REVIEW_REQUIRED` (score 0.5286), but did not completely reject it.

### 9. Database State
- Row in `specialized_observations`: `id = OBS-FIRE-e925f6`, `confidence = 0.7516`, `validation_status = VALID`.
- Row in `correlated_incidents`: `id = CORR-SPEC-a486a938`, `assessment_score = 0.5286`, `validation_decision = REVIEW_REQUIRED`.
- Row in `evidence`: `id = ev_f32068b5bd6c`, `object_class = POTENTIAL_FIRE`, `timestamp_seconds = 16.02`.

### 10. Storage State
- Snapshot: `storage/evidence/ev_f32068b5bd6c_snapshot.jpg` (183,238 bytes)
- Overlay: `storage/evidence/ev_f32068b5bd6c_annotated.jpg` (187,110 bytes)
- Sub-clip: `storage/evidence/ev_f32068b5bd6c_clip.mp4` (2,816,005 bytes)

### 11. HTTP/Media Response
- Incident feed returns `CORR-SPEC-a486a938` with `storyline` describing potential fire.

### 12. Root Cause
- In the absence of an external deep-learning fire model weights file, `FireVisualDetector` relies on hand-crafted color thresholds ($YCbCr + HSV$).
- Overhead retail lighting on yellow/orange product packaging satisfies the chromaticity and peak luminance rules.
- Static surface suppression uses an area variance threshold (`mean_area_cv < 0.04`) that fails on handheld mobile video because natural camera jitter induces apparent contour area oscillations.
- Negative evidence correctly penalized the candidate to `REVIEW_REQUIRED`, but the pipeline lacks a background/motion-flow check to reject stationary illuminated objects outright.

### 13. Why Automated Tests Did Not Catch It
- Benchmark regression tests tested `WhatsApp Video 2026-09-26 at 06.15.55.mp4` (where false alarms were 0), not this specific clip (`06.14.37.mp4`).
- Synthetic tests only tested solid uniform orange patches (which had `std_lum < 6.5`), whereas retail packaging contains high-contrast typography and graphics (`std_lum = 11.2`).

### 14. Exact Files Responsible
- [ai/specialized/fire_smoke/detector.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/specialized/fire_smoke/detector.py) (Lines 132–194)
- [ai/specialized/validator.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/specialized/validator.py) (Lines 181–205)
- [ai/incidents/detectors/specialized.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/incidents/detectors/specialized.py) (Lines 132–168)

### 15. Minimal Safe Fix (Future Implementation Plan)
1. Add Optical Flow / Pixel Motion Gating: Combustion flames feature high-frequency turbulent upward motion ($> 4.0\text{ px/s}$). Static store shelves have zero independent motion relative to the camera background motion.
2. Tighten Thermal Core Criteria: Genuine flames have core luminance saturated at $> 252$ with sharp surrounding luminance gradients.
3. Require Smoke Plume for Single-Source Low-Confidence Heuristics: If an observation is generated solely by heuristic chromatic rules (no trained model) and has zero co-occurring smoke, reject the candidate rather than emitting a `REVIEW_REQUIRED` incident.

### 16. Required Regression Tests
- Test retail aisle footage with brightly packaged goods.
- Verify camera shake on static yellow/orange signs does not trigger fire.

### 17. Whether Production Data is Affected
- Video `dbfac22e-c4ba-4b4c-ba4c-4e5d1ff39e43` currently contains the false `REVIEW_REQUIRED` record in production.

---

## Observed Failure 3 — Evidence Clip Does Not Play

### 1. User-Visible Symptom
In the Evidence Preview modal:
- **Original Snapshot**: Displays image correctly.
- **Annotated BBox Overlay**: Displays annotated image correctly.
- **Evidence Clip (8s)**: Video player element remains completely blank/black, showing `0:00 / 0:00`.

### 2. Expected Behavior
Clicking the "Clip" tab in the Evidence Preview modal should seamlessly play the 8-second extracted video clip with full seek/scrub controls.

### 3. Actual Behavior
The video player cannot play the clip. The network request to `/playback` fails with HTTP 500, and direct `/clip` fails to decode in the browser.

### 4. Reproduction Steps
1. Navigate to Evidence Vault or Incident Detail view on the frontend.
2. Click on evidence `ev_f32068b5bd6c`.
3. Click on the "Evidence Clip (8s)" tab.
4. Observe the video player: stays black at `0:00`.

### 5. Frontend Layer
- **Component**: [frontend/src/components/SecurityOperationsView.tsx](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/frontend/src/components/SecurityOperationsView.tsx#L920-L935) & [frontend/src/components/EvidenceVaultView.tsx](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/frontend/src/components/EvidenceVaultView.tsx#L410-L425)
- Both components render:
  ```tsx
  <video controls preload="metadata" className="w-full h-full object-contain">
    <source src={(previewEvidence.playback_url || previewEvidence.clip_url) ?? undefined} type="video/mp4" />
    {previewEvidence.clip_url && previewEvidence.playback_url && (
      <source src={previewEvidence.clip_url} type="video/mp4" />
    )}
  </video>
  ```
- The frontend prioritizes `playback_url` (`/api/evidence/{id}/playback`).

### 6. API Layer
- **Endpoint**: `GET /api/evidence/{id}/playback` in [backend/app/api/evidence.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/backend/app/api/evidence.py#L290-L335)
- Calls `ensure_evidence_clip_playback(evidence_id, original_clip)`.
- When `ensure_evidence_clip_playback` raises `PlaybackMemoryPressureError` or `PlaybackError`:
  ```python
  except PlaybackError as err:
      logger.error(f"Evidence clip playback conversion error for {evidence_id}: {err}")
      raise HTTPException(
          status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
          detail=f"Failed to prepare browser-compatible evidence clip: {str(err)}",
      )
  ```
- **Returns HTTP 500 Internal Server Error**!

### 7. Service Layer
- **Playback Service**: [backend/app/services/playback_service.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/backend/app/services/playback_service.py#L520-L570)
  1. Checks `is_browser_compatible(original_clip_path)`.
  2. Probing `storage/evidence/ev_f32068b5bd6c_clip.mp4` via FFmpeg reveals:
     `Stream #0:0: Video: mpeg4 (Simple Profile) (mp4v / 0x7634706D), yuv420p, 768x432, 29.97 fps`
     The clip was encoded with **`mp4v` (MPEG-4 Part 2)**.
  3. `is_browser_compatible` correctly evaluates to **`False`** because modern browsers (Chrome, Edge, Safari) cannot decode `mp4v` inside an HTML5 `<video>` element.
  4. The service then attempts to transcode the clip to H.264 via `transcode_to_h264()`.
  5. **The Memory Headroom Gate Failure**:
     ```python
     headroom = get_container_memory_headroom_mb()
     if headroom < 120.0:
         raise PlaybackMemoryPressureError(
             f"Container memory headroom ({headroom:.1f} MB) is below the safety threshold (120 MB)."
         )
     ```
     On Railway's Linux container, the Linux kernel aggressively uses unused memory for filesystem page cache. `get_container_memory_headroom_mb()` reads cgroup limit minus current usage ($1024 - 1023 \approx 0.4\text{ MB}$), causing `headroom < 120.0` to trigger on every on-demand playback request!
  6. Transcoding is aborted, raising `PlaybackMemoryPressureError`.
  7. If the browser falls back to the secondary source `/api/evidence/{id}/clip`, that endpoint streams the raw `mp4v` file. Because Chromium/WebKit do not support `mp4v`, the video player fails to render frames and remains black at `0:00`.

### 8. AI/CV Layer
- **Evidence Extraction**: [backend/app/services/evidence_service.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/backend/app/services/evidence_service.py#L402-L403) and [ai/extraction/extractor.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/extraction/extractor.py#L53-L54)
  Both write extracted clips using OpenCV `VideoWriter`:
  ```python
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(str(clip_file), fourcc, fps, (width, height))
  ```
- **Why OpenCV uses `mp4v`**: OpenCV built without openh264 or x264 defaults to the legacy `mp4v` encoder.
- **The Browser Reality**: HTML5 video standards require `avc1` (H.264), `vp8`, `vp9`, or `av01` (AV1). `mp4v` is explicitly unsupported in standard browser video elements.

### 9. Database State
- Row in `evidence`: `has_clip = 1`, `clip_path = "storage/evidence/ev_f32068b5bd6c_clip.mp4"`.

### 10. Storage State
- File `storage/evidence/ev_f32068b5bd6c_clip.mp4` exists and is `2,816,005 bytes` (non-zero, valid container).
- File `storage/evidence_playback/ev_f32068b5bd6c_clip_playback.mp4` **does not exist** because on-demand transcoding aborted.

### 11. HTTP/Media Response
- `GET /api/evidence/ev_f32068b5bd6c/playback` $\to$ **`HTTP 500 Internal Server Error`**
  ```json
  {"detail": "Failed to prepare browser-compatible evidence clip: Container memory headroom (0.4 MB) is below the safety threshold (120 MB)."}
  ```
- `GET /api/evidence/ev_f32068b5bd6c/clip` $\to$ **`HTTP 200 OK`** (Content-Type: `video/mp4`, 2.81 MB), but codec is `mp4v`. Chrome/Edge video element fires a decode error and shows black screen.

### 12. Root Cause
1. **Extraction Codec Flaw**: Sub-clips are written with OpenCV's `mp4v` FourCC, which is incompatible with modern browser HTML5 video playback.
2. **On-Demand Transcoding Headroom Lockout**: When the `/playback` endpoint attempts on-demand transcode to H.264, the cgroup headroom check ($< 120\text{ MB}$) fails because Linux page cache in the Railway container makes raw cgroup free memory appear near zero.
3. **Double Failure Loop**: The primary `<source>` fails with HTTP 500; the fallback `<source>` serves raw `mp4v`, which the browser cannot decode.

### 13. Why Automated Tests Did Not Catch It
- Tests only checked `has_clip == True`, file existence, non-zero file size, and HTTP 200 on `/clip`.
- Tests did not verify browser codec compatibility (`is_browser_compatible`).
- Automated tests ran in pytest using Python `requests` (which simply reads bytes), not inside a real browser DOM that decodes video frames.
- Tests tested playback transcoding in local environments with abundant RAM, where `headroom >= 120 MB`.

### 14. Exact Files Responsible
- [backend/app/services/evidence_service.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/backend/app/services/evidence_service.py) (Lines 402–404)
- [backend/app/services/playback_service.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/backend/app/services/playback_service.py) (Lines 70–98, 553–558)
- [backend/app/api/evidence.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/backend/app/api/evidence.py) (Lines 263–268, 320–332)
- [ai/extraction/extractor.py](file:///c:/Users/manoj/OneDrive/Documents/Sentinel/ai/extraction/extractor.py) (Lines 53–54)

### 15. Minimal Safe Fix (Future Implementation Plan)
1. **Direct H.264 Extraction via FFmpeg**: During initial evidence generation in `evidence_service.py`, extract sub-clips directly using FFmpeg (`-c:v libx264 -pix_fmt yuv420p -movflags +faststart`) or remux directly from source video (`-c copy` if source is already H.264). This eliminates `mp4v` entirely at creation time and avoids any on-demand transcoding during playback.
2. **Cgroup Active Memory Adjustment**: In `get_container_memory_headroom_mb()`, parse `memory.stat` (`inactive_file` / `active_file`) to discount reclaimable page cache, preventing false memory pressure lockouts on Linux.
3. **Endpoint Fallback Gracefulness**: In `/api/evidence/{id}/playback`, if transcoding fails or is deferred, fall back to streaming `original_clip` with range support rather than throwing a 500 error.

### 16. Required Regression Tests
- Create an evidence clip and verify its FourCC is `avc1` / `h264` and `is_browser_compatible(clip) == True`.
- Verify `/api/evidence/{id}/playback` returns HTTP 206 with range support.

### 17. Whether Production Data is Affected
- Existing evidence clips in `storage/evidence/` are encoded in `mp4v`. They can be batch-transcoded to browser-compatible H.264 or regenerated once the fix is applied.

---

## Summary Matrix of Audited Failures

| Failure | Primary Defect | Affected Component | Fix Strategy (Post-Audit) |
|---|---|---|---|
| **1. Natural Language** | Parser omits `"dress"`/`"lady"`; track color gated by single-frame unconfirmed count | `investigation_parser.py`, `intelligence_pipeline.py` | Expand synonyms; accumulate color votes across frames |
| **2. False Fire** | Chromatic YCbCr/HSV heuristic fires on illuminated store shelf; camera shake bypasses static filter | `fire_smoke/detector.py`, `validator.py` | Add optical flow motion gating; require smoke correlation for heuristics |
| **3. Evidence Clip** | Extracted as `mp4v`; on-demand transcode blocked by cgroup page cache headroom | `evidence_service.py`, `playback_service.py` | Extract directly to H.264 with FFmpeg; discount reclaimable page cache |

---
**PROTOTYPE RELIABILITY AUDIT COMPLETE — NO CODE OR PRODUCTION CHANGES MADE.**
