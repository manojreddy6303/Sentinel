# Sentinel Specialized Visual Detection Core (Phase 15)

## 1. Architectural Overview

The Sentinel Specialized Visual Detection Core provides a dedicated, extensible visual intelligence layer for visually distinctive security-relevant phenomena that generic YOLO object detection and heuristic incident logic cannot reliably infer.

```
VIDEO FRAME
    ↓
SPECIALIZED VISUAL DETECTOR (Failure-Isolated)
    ↓
RAW SPECIALIZED OBSERVATION (SpecializedObservation)
    ↓
TEMPORAL VALIDATION (SpecializedTemporalTracker)
    ↓
NEGATIVE EVIDENCE ENGINE (SpecializedNegativeEvidenceEngine)
    ↓
SCENE CONTEXT & CAMERA STABILITY
    ↓
INCIDENT CANDIDATE (SpecializedFireIncidentDetector / Smoke / Weapon)
    ↓
CANDIDATE VALIDATION (IncidentValidator: ACCEPTED / REVIEW_REQUIRED / REJECTED)
    ↓
INCIDENT FUSION (IncidentFusionEngine: POTENTIAL_FIRE_SMOKE Cross-Modal Fusion)
    ↓
EVIDENCE GATE (EvidenceGate / EvidenceCandidate with Full Provenance)
    ↓
SECURITY EVENT (SecurityEventModel with Observational Nomenclature)
    ↓
TIMELINE / INVESTIGATION / DOSSIER REPORTING
```

---

## 2. Core Components & Modular Abstraction

All specialized visual detection logic resides under `ai/specialized/`:

| Module | Responsibility |
| :--- | :--- |
| `ai/specialized/schemas.py` | Data contracts (`SpecializedObservation`, `SpecializedTemporalTrack`, `SpecializedModelInfo`, `SpecializedDetectorStatus`, `SpecializedValidationStatus`) |
| `ai/specialized/base.py` | `BaseSpecializedDetector` abstract class enforcing failure isolation (`safe_detect`), error catching, and lifecycle contracts |
| `ai/specialized/registry.py` | `SpecializedDetectorRegistry` singleton managing active detectors with failure isolation during batch execution |
| `ai/specialized/temporal.py` | `SpecializedTemporalTracker` tracking multi-frame persistence, aspect ratio evolution, and spatial centroid/IoU continuity |
| `ai/specialized/negative_evidence.py` | `SpecializedNegativeEvidenceEngine` evaluating static orange surfaces, flame flicker, vehicle glare, global atmospheric haze, optical limits, and boundary clipping |
| `ai/specialized/fire_smoke/detector.py` | `FireVisualDetector` and `SmokeVisualDetector` with YOLO weight support and chromaticity fallback |
| `ai/specialized/weapon/detector.py` | `WeaponVisualDetector` foundation (`NOT_CONFIGURED` when weights are unconfigured; zero fabrication) |
| `ai/specialized/pose/detector.py` | `PoseActionDetector` foundation reusing Phase 12 `PoseFeatureEngine` |

---

## 3. Failure Isolation & Model Management

### Failure Isolation Guarantee
Every specialized detector inherits from `BaseSpecializedDetector`. Detections execute through `safe_detect(frame, timestamp, **kwargs)` inside a top-level try/except block.
- If a detector encounters a missing model, corrupted frame, division-by-zero, or CUDA failure, it returns an empty observation list.
- An individual failure **NEVER crashes** the central intelligence pipeline or prevents other detectors from executing.

### Model Lifecycle States
Detectors report one of five lifecycle states:
- `AVAILABLE`: Weights loaded and hardware verified.
- `CONFIGURED`: Configuration specified, ready for lazy loading.
- `NOT_CONFIGURED`: No weights path specified in settings; runs non-fabricating heuristic baseline or remains inactive.
- `UNAVAILABLE`: Runtime environment or weights file unresolvable.
- `FAILED`: Incurred unrecoverable runtime initialization error.

### Runtime Configuration
All specialized models are controlled via environment settings in `backend/app/core/config.py`:
- `SPECIALIZED_VISUAL_ENABLED` (default: `True`)
- `FIRE_DETECTION_ENABLED` (default: `True`)
- `SMOKE_DETECTION_ENABLED` (default: `True`)
- `WEAPON_DETECTION_ENABLED` (default: `False` - foundation only)
- `POSE_DETECTION_ENABLED` (default: `False` - foundation only)
- `FIRE_MODEL_PATH` (default: `""`)
- `SMOKE_MODEL_PATH` (default: `""`)
- `WEAPON_MODEL_PATH` (default: `""`)

**Zero Silent Downloads**: Large weights are never downloaded silently at runtime.

---

## 4. Fire Visual Detection

### Methodology
1. **Model-Based Detection**: If custom YOLO weights are provided via `FIRE_MODEL_PATH`, the detector executes tensor inference for flame classes.
2. **Forensic Chromaticity Fallback**: When weights are unconfigured, the detector uses an HSV/YCbCr color space segmenter targeting high-intensity flame cores (`V > 200`, `S > 90`, `Y > Cr > Cb`).
3. **Temporal Flicker Analysis**: Real flames exhibit high-frequency perimeter variance. Static orange objects (safety vests, traffic cones, orange signs) have variance below 0.05 and are penalized or rejected by negative evidence.
4. **Temporal Persistence**: Detections must persist across multiple frames (`min_observations >= 3`, span >= 0.5s) to graduate from `RAW` to `VALIDATED` and spawn `IncidentCandidate(event_type="POTENTIAL_FIRE")`.

---

## 5. Smoke Visual Detection

### Methodology
1. **Model-Based Detection**: If custom YOLO weights are provided via `SMOKE_MODEL_PATH`, the detector performs tensor inference for smoke/plume classes.
2. **Forensic Low-Saturation Plume Fallback**: When weights are unconfigured, regions of low color saturation (`S < 60`) and elevated luminance (`100 < L < 220`) are evaluated for texture irregularity and expansion.
3. **Negative Evidence Engine**:
   - **Global Atmospheric Haze/Fog**: If low-saturation regions span >40% of the camera frame, global fog is flagged and candidate confidence is sharply reduced.
   - **Exhaust & Steam**: Transient, non-persisting plumes are rejected before reaching incident thresholds.
   - **Single-Frame Artifacts**: Temporal validation requires persistent tracking across time.

---

## 6. Fire + Smoke Cross-Modal Fusion

When both fire and smoke visual detections occur in the same spatio-temporal vicinity (temporal distance <= 3.0s and spatial bounding box IoU >= 0.05 or normalized centroid distance <= 0.25):
- `IncidentFusionEngine` fuses them into a single high-priority `POTENTIAL_FIRE_SMOKE` security event.
- The fused event retains all individual supporting signals (`SupportingSignal` for fire and smoke observations).
- Neither detector automatically "proves" the other; they remain independent evidence streams combined at the fusion gate.

---

## 7. Weapon / Suspicious Object Foundation

### Strict Safety & Forensic Integrity
- **Zero Fabrication**: When `WEAPON_MODEL_PATH` is empty, `WeaponVisualDetector` reports `NOT_CONFIGURED` and produces **zero** detections.
- **Review-Required Terminology**: Any model-backed detection produces `POTENTIAL_WEAPON_VISUAL` with `human_verification_required = True`. Sentinel never issues definitive criminal conclusions.
- **Resolution Penalties**: Bounding boxes under 32x32 pixels receive negative evidence penalties for optical resolution limits.

---

## 8. Pose & Action Foundation

- Reuses the existing Phase 12 `PoseFeatureEngine`.
- Extracts body postures (`standing`, `crouching`, `lying_down`, `raised_arm`).
- Treats pose purely as **supporting evidence** for interpersonal or movement incidents (e.g., fall down, loitering, physical altercation). Never classifies a pose alone as an assault or fight.

---

## 9. Database & Evidence Provenance

Specialized visual observations are persisted to SQLite table `specialized_observations` via `SpecializedObservationModel`:
- `id`: Unique observation identifier (`OBS-FIRE-...`, `OBS-SMOKE-...`).
- `video_id`: UUID foreign key.
- `event_id`: Fused security event reference (nullable).
- `detector_name`: Name of detector (`fire_visual_detector`, etc.).
- `detector_version`: SemVer string (`1.0.0`).
- `class_name`: Visual class (`fire`, `smoke`, `handgun`, `rifle`, etc.).
- `timestamp_seconds`: Video timestamp in seconds.
- `confidence`: Machine detector score (0.0 - 1.0).
- `evidence_strength`: Calibrated forensic strength score.
- `validation_status`: `RAW`, `VALIDATED`, `UNCERTAIN`, or `REJECTED`.
- `bounding_box`: Normalised / pixel coordinates JSON.
- `metrics`: Textural, chromatic, and temporal track metrics JSON.
- `created_at`: UTC timestamp.

All accepted incidents retain full snapshot provenance through Sentinel's `EvidenceGate`.

---

## 10. Natural Language Investigation Queries

The deterministic investigation engine (`backend/app/services/investigation_parser.py` & `investigation_service.py`) supports queries targeting specialized visual phenomena:
- *"show fire events"*
- *"find possible smoke"*
- *"show potential weapon detections"*
- *"find suspicious visual objects"*
- *"show fire between 30 and 60 seconds"*
- *"show all specialized visual events"*
- *"show smoke evidence with high evidence strength"*

Results are strictly grounded in stored machine observations and formulated with neutral observational language.

---

## 11. Reporting & Intelligence Dossiers

Executive intelligence dossiers include a dedicated **Specialized Visual Phenomena** audit section:
- Summarizes detected fire, smoke, and weapon visual candidates.
- Formulates observational findings:
  - *"Potential fire visual evidence was detected in surveillance footage at 00:14 (duration: 3.2s, evidence strength: 0.82). Human verification required."*
  - *"Potential smoke-like visual evidence was detected at 00:15 (duration: 2.1s, evidence strength: 0.74). Human verification required."*
- Cross-references captured evidence frame snapshots in the Evidence Vault.

---

## 12. Frontend Specialized Visual Intelligence Tab

The web UI includes a dedicated **Specialized Visual** tab featuring:
- **Category Filters**: `All`, `Fire`, `Smoke`, `Weapon/Object`, `Pose/Action`.
- **Status Badges**: Highlighting `ACCEPTED`, `REVIEW REQUIRED`, and `REJECTED` states.
- **Evidence Provenance**: Timestamp seek buttons, confidence vs. evidence strength meters, model source badges, and direct links to Evidence Vault snapshots.

---

## 13. Limitations & Disclaimers

> [!CAUTION]
> **Probabilistic Visual Intelligence Disclaimer**
> Specialized visual detection is probabilistic and derived from machine vision models and heuristic texture/chromatic analysis. It must **never** be interpreted as guaranteed ground truth or conclusive legal/forensic proof. All specialized visual events require qualified human review.
