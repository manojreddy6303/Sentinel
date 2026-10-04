# SENTINEL — CORE AI RELIABILITY & ARCHITECTURE AUDIT (PHASE 0)
**System Audit: Universal Video Intelligence Readiness & Root-Cause Analysis**
**Date:** October 2026  
**Status:** COMPLETE (Read-Only Audit Phase)

---

## 1. Executive Summary

A comprehensive architectural and algorithmic audit of the Sentinel video intelligence platform was conducted to identify why the pipeline functions well on specific benchmark videos (e.g., canonical burglary CCTV) but exhibits reliability degradation, missed small objects, missed theft/takeaway events, and inappropriate generic classifications (`PERSON_ACTIVITY`) on arbitrary real-world surveillance video.

The audit examined all layers: ingestion, metadata extraction, illumination/frame quality, YOLO detection, tiled multi-scale inference, geometric/temporal detection validation, ByteTrack multi-object tracking, visual attribute extraction, specialized visual models (fire/smoke/weapon/pose), incident candidate generation, negative evidence filtering, incident arbitration/fusion, evidence gating, and natural language investigation.

**Key Finding:** The failures to generalize across different video types are **NOT** due to missing models or tracking architectures, but are rooted in **hardcoded class filters, rigid pixel-area cutoff rules, disconnected object classes between detection/validation/tracking, and unhandled small-object temporal recovery**.

---

## 2. Actual System Data Flow

```text
VIDEO SOURCE (.mp4, .avi, .mov, etc.)
  │
  ▼
[VideoProcessor / MediaMetadataExtractor]
  ├── Native FPS, Frame Count, Duration, Container PTS, Aspect Ratio
  └── Frame Sampling (Fixed-Rate or Adaptive Motion-Burst)
  │
  ▼
[SceneConditionAnalyzer & AdaptiveLowLightEnhancer]
  ├── Illumination & Exposure Assessment (dark, overexposed, usable)
  └── Conditional Low-Light Contrast/Brightness Enhancement
  │
  ▼
[YOLODetector (YOLOv8n / PyTorch Safe-Load)]
  ├── InferenceResolutionPolicy: Dynamic imgsz (640 for SD/HD, 960 for 4K)
  └── Mini-Batch Frame Inference (batch size = 4)
  │
  ▼
[DetectionValidator & Policy]
  ├── Bounding Box Sanity (non-negative, finite coordinates)
  ├── Aspect Ratio & Area Fraction Gating
  └── Temporal Persistence Support Check across neighboring frames
  │
  ▼
[ObjectTracker (ByteTrack 2-Stage Association)]
  ├── High-Confidence Primary Matching + Low-Confidence Secondary Recovery
  ├── Linear Velocity Motion Extrapolation
  └── State: TENTATIVE -> VALIDATED -> LOST -> DELETED
  │
  ▼
[Visual Attributes & Specialized Telemetry]
  ├── PersonAttributeAnalyzer (Upper/Lower clothing color, face telemetry)
  ├── VehicleColorAnalyzer (Color clustering, orientation)
  └── SpecializedVisualDetector (Fire, Smoke, Weapon, Pose)
  │
  ▼
[IncidentIntelligenceEngine (Registry of 20+ Detectors)]
  ├── IncidentContextBuilder (Assembles tracks, detections, spatial telemetry)
  ├── Detector Execution: Theft, Altercation, Forced Movement, Vehicle, Crowd
  ├── NegativeEvidenceEngine (Counter-evidence refutation)
  └── IncidentCandidateValidator (ACCEPTED, REVIEW_REQUIRED, REJECTED)
  │
  ▼
[IncidentFusionEngine & AdvancedIncidentCorrelationEngine]
  ├── Competing Hypothesis Arbitration (Property vs Person, etc.)
  ├── Sampling-Aware Spatio-Temporal Clustering
  └── Storyline Graph Construction & Diagnostic Metrics
  │
  ▼
[EvidenceEligibilityGate & Persistence]
  ├── Canonical Video Clips & Keyframe JPEG extraction
  ├── Database Sync (VideoModel, EventModel, GroupedEventModel, EvidenceModel)
  └── H.264 Playback Transcoding
  │
  ▼
[InvestigationOrchestrator & Gemini / Deterministic Fallback]
  └── Grounded NL Query Answering via Preserved Structured Evidence
```

---

## 3. Subsystem Breakdown & Diagnostic Findings

### A. Detection Pipeline (`ai/detection/detector.py`, `ai/detection/tiled_detector.py`, `ai/detection/inference_policy.py`)
- **Implementation:** Pretrained `yolov8n.pt` loaded lazily with PyTorch 2.6 safe-weights compatibility. Batch inference of 4 frames.
- **Strengths:** Lightweight (~6MB model), memory-safe, runs on CPU without GPU requirements, handles bounded concurrency.
- **Weaknesses & Root Causes:**
  1. *Hardcoded `SURVEILLANCE_CLASSES` Filter:* Lines 28-32 in `detector.py` filter YOLO predictions to 15 hardcoded classes:
     `{"person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack", "handbag", "suitcase", "bottle", "cell phone", "chair", "bench", "traffic light", "stop sign"}`.
     Investigation-critical items such as `laptop`, `umbrella`, `book`, `clock`, `remote`, `package`, `box` are dropped at inference time even if YOLO detects them.
  2. *Tiled Detector Early-Exit Bypass:* In `tiled_detector.py` (line 178), `detect_tiled()` has:
     `if max(h, w) <= 720: return self.base_detector.detect(...)`
     This means for 320x240, 480p, and 720p videos, tiled small-object recovery is completely skipped! However, small handheld objects (phones, wallets, bottles) in low-resolution video occupy only 10-30 pixels and desperately need multi-scale or zoomed ROI inspection.
  3. *Static Input Scale for SD Video:* A 320x240 video is upscaled to 640x640 with letterboxing, where single-pixel interpolation artifacts can dilute small object features.

### B. Validation Pipeline (`ai/validation/validator.py`, `ai/validation/policy.py`)
- **Implementation:** Evaluates raw detections against geometric bounds, aspect ratios, confidence, and temporal support.
- **Strengths:** Robustly filters single-pixel sensor noise and degenerate boxes.
- **Weaknesses & Root Causes:**
  1. *Immediate Small-Object Rejection:* In `validator.py` (lines 201-220):
     `if effective_area_frac < rule.min_area_fraction:`
     `  elif obj_class in {"backpack", "suitcase", "handbag"}: ...`
     `  else: return ValidationResult(status=ValidationStatus.REJECTED)`
     Only backpacks, suitcases, and handbags are allowed to reach `UNCERTAIN` for temporal dwell. Any smaller item (e.g. `cell phone`, `bottle`, `general_object`) whose area fraction is under 0.0001 is immediately tagged `REJECTED`.
  2. *Exclusion from Tracking:* `intelligence_pipeline.py` (lines 265-268) explicitly filters:
     `valid_frame_dets = [d for d in frame_dets if str(d.get("validation_status", "VALID")).upper() != "REJECTED"]`
     Thus, any small object rejected by geometric cutoff never reaches the tracker at all!

### C. Tracking Pipeline (`ai/tracking/tracker.py`, `ai/tracking/botsort_tracker.py`)
- **Implementation:** ByteTrack two-stage spatial association with linear velocity motion prediction.
- **Strengths:** Clean anonymous tracking without identity persistence or biometrics; handles temporary occlusion.
- **Weaknesses & Root Causes:**
  1. *Severe `trackable_classes` Whitelist Restriction:* In `tracker.py` (lines 39-52):
     `self.trackable_classes = set(trackable_classes or ["person", "car", "bus", "truck", "motorcycle", "bicycle", "backpack", "suitcase", "handbag"])`
     Notice that `"cell phone"` and `"bottle"`, even if detected and validated, are **NOT in `trackable_classes`**! They are completely dropped from tracking, resulting in zero `TrackedObject` records.
  2. *Single-Frame Track Lifecycles:* Small objects with intermittent detector dropouts never achieve the 2-observation confirmation threshold (`t.detection_count >= 2`), causing them to be excluded from `validated_tracks`.

### D. Object Attribute & Color Pipeline (`ai/attributes/`)
- **Implementation:** HSV/Lab color histogram analysis across anatomical bounding box regions (upper clothing, lower clothing) and vehicle bodies.
- **Strengths:** Multi-frame aggregation with consensus weighting (`aggregate_track_clothing_color`); lighting-aware confidence reduction.
- **Weaknesses:** If tracking drops a person into multiple fragmented track IDs, color consensus must restart from scratch.

### E. Specialized Visual Detectors (`ai/specialized/`)
- **Implementation:** `FireVisualDetector` and `SmokeVisualDetector` with chromatic YCbCr/HSV rules, thermal core requirements, specular highlight rejection, and rigid background motion checks.
- **Strengths:** High immunity to false positives from red shirts, car taillights, and orange vests.
- **Weaknesses:** High thermal core threshold (232 luminance) requires sufficient exposure; very distant small flames under heavy compression may be missed.

### F. Theft & Takeaway Reasoning (`ai/incidents/detectors/theft_and_takeaway.py`, `ai/incidents/detectors/property/`)
- **Implementation:** Sequences requiring person approach, dwell in proximity, object disappearance or co-movement, and person departure.
- **Strengths:** Rigorous multi-stage physical reasoning preventing false alerts on simple walking or proximity.
- **Weaknesses & Root Causes:**
  1. *Missing Object Track Dependency:* The detector iterates over `non_person_tracks = [t for t in context.tracks if t.object_class in target_classes]`. Because `ObjectTracker` omitted small items (`cell phone`, `bottle`, `package`) from `trackable_classes`, `non_person_tracks` is frequently empty in small-theft scenarios.
  2. *Small-Theft Blindspot:* When a thief picks up a small unclassified or poorly-classified object, the pipeline produces no theft event because the object was never tracked as a named COCO object. The system lacks a generic object / interaction displacement recovery mechanism.
  3. *Rigid Proximity Distance:* `THEFT_INTERACTION_MAX_DISTANCE` (120px) scaled by `resolution_scale_factor` works for medium/large scenes, but in close-up or highly compressed footage, bounding box centroids may not fall within the expected Euclidean distance.

### G. Physical Altercation & Forced Movement Detectors (`ai/incidents/detectors/person/`)
- **Implementation:** Evaluates reciprocal motion, distance oscillations, and course deflections between pairs of persons.
- **Strengths:** Includes negative evidence for parallel walking, queues, and brief passing.
- **Weaknesses:** In crowded scenes or compressed WhatsApp video with camera shake, global camera motion can induce artificial relative bounding box oscillations, generating false `POTENTIAL_PHYSICAL_ALTERCATION` or `POTENTIAL_FORCED_MOVEMENT` if GMC is not factored in.

### H. Incident Classification & Evidence Gate (`ai/incidents/engine.py`, `ai/incidents/evidence_gate.py`, `ai/incidents/fusion.py`)
- **Implementation:** Modular execution of 20+ detectors, validation gating (`ACCEPTED`, `REVIEW_REQUIRED`, `REJECTED`), multi-hypothesis arbitration, and evidence candidate generation.
- **Strengths:** Strict adherence to uncertainty preservation (`REVIEW_REQUIRED` over false positive).
- **Weaknesses:**
  1. *Duplicated Code:* In `fusion.py`, `_arbitrate_competing_person_interactions` is defined twice (lines 154 and 357). The second definition silently overwrites the first in Python runtime.
  2. *Fallback to Generic `PERSON_ACTIVITY`:* When specialized or property detectors fail due to missing object tracks, the only remaining event is baseline activity density (`PERSON_ACTIVITY`), leaving investigators with unhelpful summaries.

### I. Investigation & Natural Language Querying (`ai/investigation/`, `backend/app/services/`)
- **Implementation:** Translates queries via Gemini Flash or deterministic SQL/filter fallback; checks strict ethical guardrails (no identity attribution).
- **Strengths:** Grounded strictly in database records; honest abstention when evidence is absent.
- **Weaknesses:** If the underlying pipeline missed the small-theft event or classified it only as `PERSON_ACTIVITY`, the investigation truthfully reports "No theft detected", correctly reflecting the database but failing the investigator's real-world need.

---

## 4. Benchmark-Specific vs Universal Assumptions

| Dimension | Benchmark Assumption (Burglary/Highway) | Real-World Universal Reality | Impact on Non-Benchmark Videos |
| :--- | :--- | :--- | :--- |
| **Resolution** | 320x240 (burglary) or 3840x2160 (4K street) | 360p, 480p, 720p, 1080p, portrait/vertical (9:16) | Pixel thresholds fail or distort bounding box aspect ratio rules |
| **Frame Rate** | Fixed 30 FPS or 1 FPS uniform sampling | Variable frame rate (VFR), 10-60 FPS, compressed WhatsApp | Monotonic PTS gaps cause false acceleration or dropped temporal links |
| **Object Scale** | Large humans (>100px) or visible bags (>50px) | Small handheld merchandise, phones, wallets (<30px) | Objects filtered out as "sensor noise" by validation policy |
| **Object Classes** | Standard COCO classes (person, car, backpack) | Unspecified retail goods, packages, tools, merchandise | System drops non-standard classes or lacks `GENERAL_OBJECT` representation |
| **Camera Stability**| Fixed CCTV mount | Handheld mobile, pan-tilt-zoom, vibration/jitter | Camera motion mistaken for local object kinematics |
| **Incident Sequence**| Person enters -> stands -> takes bag -> exits | Quick grab-and-go, concealed item, interaction without loss | Rigid multi-second dwell rules fail on fast takeaway |

---

## 5. Reusable Architectural Assets

The codebase already contains high-grade modular components that should be preserved and reinforced:
1. **`AdaptiveTemporalSamplingEngine` & `BoundedFrameCache`**: High-performance frame handling that respects container memory limits (1GB Railway deployment).
2. **`SceneConditionAnalyzer` & `AdaptiveLowLightEnhancer`**: Illumination-aware analytical proxy for dark footage.
3. **`NegativeEvidenceEngine`**: Comprehensive counter-evidence checks across all incident types.
4. **`AdvancedIncidentCorrelationEngine`**: Storyline and incident clustering with sampling awareness.
5. **`FrameQualityGate`**: Pre-flight rejection of corrupt, blurry, or near-uniform transition frames.
6. **`DetectorHealthRegistry`**: Provenance tracking distinguishing trained models, heuristics, and unavailable detectors.

---

## 6. Actionable Reliability Roadmap (Phase 0 -> Phase 1)

1. **Resolution-Aware & Class-Aware Object Detection:**
   - Expand `SURVEILLANCE_CLASSES` in `detector.py` and `config.py` to cover all surveillance-relevant COCO portable/investigative classes (`laptop`, `cell phone`, `bottle`, `umbrella`, `book`, etc.).
   - Support generic object representation (`general_object`) when evidence indicates a localized foreground entity that does not cleanly match a specific COCO class.
   - Refactor `tiled_detector.py` so that low-resolution frames with small-object risk undergo adaptive multi-scale or zoomed ROI inspection rather than early-exiting.
2. **Detection Validation Policy Tuning:**
   - Add explicit class rules for small portable objects (`cell phone`, `bottle`, `umbrella`, `laptop`, `general_object`) with realistic pixel dimensions (min 6-8px, min area 36-60px) rather than falling back to the 100px cutoff.
   - For candidate small objects near persons, route them to `UNCERTAIN` for temporal association rather than immediate rejection.
3. **Tracking Layer Generalization:**
   - Expand `trackable_classes` in `ObjectTracker` to include all portable objects and generic items.
   - Implement temporal recovery for weak/flickering small objects across neighboring frames.
4. **Generalized Theft & Takeaway Engine:**
   - Support takeaway detection where the target is a tracked small or general object.
   - Detect person-object interaction patterns (approach -> contact/overlap -> disappearance or co-movement departure) even when the exact object class is uncertain.
5. **Camera Motion Separation:**
   - Ensure Global Motion Compensation (GMC) informs relative velocity in pairwise altercation and forced-movement detectors to prevent handheld camera shake from triggering false altercation alerts.
6. **Code Cleanup:**
   - Remove the duplicate method definition in `ai/incidents/fusion.py`.
