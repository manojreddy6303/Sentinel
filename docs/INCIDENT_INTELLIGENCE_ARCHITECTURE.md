# Sentinel Phase 10: Universal Incident Intelligence Engine Architecture

## 1. Executive Summary & Objective

The **Universal Incident Intelligence Engine** transforms Sentinel from an object-detection and tracking platform into a universal, modular security-incident intelligence platform.

Sentinel strictly enforces the principle:
> **Object detection is not incident intelligence.**
> YOLO detects physical entities (a person, a car, a backpack). An *incident* is a spatiotemporally verified pattern of physical interaction, motion anomaly, or perimeter violation supported by observable signals.

This document serves as the global architecture specification and plug-in developer guide for Sentinel's incident detection ecosystem.

---

## 2. Core Architectural Philosophy

### 2.1 The 7-Stage Intelligence Hierarchy
```
STAGE 1: OBJECT DETECTION (YOLO)
           ↓
STAGE 2: SPATIO-TEMPORAL VALIDATION (VALID / UNCERTAIN / REJECTED)
           ↓
STAGE 3: MULTI-FRAME TRACKING (Trajectories, IDs, Color, Anonymous Regions)
           ↓
STAGE 4: MOTION & SPATIAL & TEMPORAL TELEMETRY (Pixel-space kinematics, relationships)
           ↓
STAGE 5: INCIDENT DETECTORS (Modular plug-in candidates)
           ↓
STAGE 6: INCIDENT FUSION & DEDUPLICATION (Clustering & multi-signal synthesis)
           ↓
STAGE 7: SECURITY EVENT PERSISTENCE & EVIDENCE RECONCILIATION (Database, Dossier, Vault)
```

### 2.2 Invariant Rules
1. **Zero Criminal/Identity Attribution**: Use strictly observational naming (`POTENTIAL_THEFT`, `POTENTIAL_INTRUSION`, `POTENTIAL_VEHICLE_COLLISION`). Never declare "Theft committed" or "Subject is a criminal".
2. **Failure Isolation**: A crash or exception in one detector must never crash the pipeline or affect other detectors.
3. **Calibrated Evidence Strength**: Confidence scores represent observable evidence strength (capped at 95%), accompanied by `"Human verification required"`.
4. **Calibrated vs Normalized Kinematics**: Unless camera calibration matrices are present, velocities and accelerations are computed in pixel-space or normalized coordinates without making false claims of real-world km/h.
5. **Context-Aware Semantics**: Moving vehicles on a normal road must not trigger false prolonged presence alerts simply because they remain visible in traffic.

---

## 3. Data Contracts (`ai/incidents/schemas.py`)

### 3.1 `IncidentCandidate`
The standardized structure emitted by all detectors:

| Field | Type | Description |
|---|---|---|
| `incident_id` | `str` | Unique incident identifier (`INC-xxxxxxxx`, `THEFT-xxxxxxxx`, etc.) |
| `video_id` | `str` | Video identifier |
| `event_type` | `str` | Standardized event type (`POTENTIAL_THEFT`, `POTENTIAL_INTRUSION`, etc.) |
| `category` | `IncidentCategory` | Enum: `vehicle`, `person`, `property`, `crowd`, `zone`, `multi_signal`, `general` |
| `start_time` | `float` | Onset timestamp in video seconds |
| `end_time` | `float` | Termination timestamp in video seconds |
| `duration` | `float` | Duration in seconds |
| `severity` | `str` | `LOW`, `NORMAL`, or `HIGH` |
| `confidence` | `float` | Calibrated evidence strength (0.10 to 0.95) |
| `track_ids` | `List[str]` | Involved track identifiers |
| `object_classes` | `List[str]` | Involved object classes |
| `source_detection_ids` | `List[str]` | IDs of raw supporting detections |
| `supporting_signals` | `List[SupportingSignal]` | Structured physical observations |
| `spatial_context` | `Optional[SpatialContext]` | Bounding coordinates, zone name, centroid |
| `temporal_context` | `Optional[TemporalContext]` | Time intervals, persistence, onset |
| `explanation` | `str` | Grounded observational explanation |
| `evidence_candidates` | `List[EvidenceCandidate]` | Suggested timestamps/bounding boxes for evidence capture |
| `validation_status` | `str` | `VALID`, `UNCERTAIN`, or `DISMISSED` |
| `human_verification_required` | `bool` | Invariant flag (`True`) |
| `detector_name` | `str` | Name of emitting detector |
| `detector_version` | `str` | Version of detector |

### 3.2 `IncidentContext`
Structured context provided to every detector's `analyze(context)` method:
- `video_id`: Video ID
- `fps`, `duration_seconds`, `sample_rate_fps`: Stream parameters
- `validated_detections`: Only high-confidence, verified visual detections
- `tracks`: Multi-frame `TrackedObject` records with verified trajectories
- `vehicle_attributes`: Color classifications and visual metrics
- `face_detections`: Anonymous visual region bounding boxes
- `zones`: User-defined polygon boundaries (`ZoneDefinition`)
- `track_motions`: Precomputed frame-by-frame kinematics (`TrackMotion`)
- `motion_summaries`: Lifetime motion statistics per track
- `scene_density`: Overall activity distribution and density metrics

---

## 4. Reusable Intelligence Engines

### 4.1 Universal Motion Engine (`ai/incidents/motion.py`)
Computes kinematic telemetry without requiring external models:
- **Displacement**: Net euclidean distance from track origin:
  $$\Delta d = \sqrt{(x_t - x_0)^2 + (y_t - y_0)^2}$$
- **Distance Traveled**: Cumulative path length:
  $$\sum \sqrt{(\Delta x)^2 + (\Delta y)^2}$$
- **Instantaneous Direction**: Angle in radians and compass degrees ($0^\circ$ to $360^\circ$).
- **Velocity Estimate**: $\Delta d / \Delta t$ (pixels/sec).
- **Acceleration Estimate**: $\Delta v / \Delta t$ (pixels/sec$^2$).
- **Stationary vs Moving**: Determines whether an entity is resting or localized ($< 12\text{ px/s}$).
- **Path Consistency**: Ratio of net displacement to cumulative distance ($\in [0.0, 1.0]$).
- **Inter-Track Relative Motion**: Approach rate ($-\Delta \text{dist} / \Delta t$) and separation rate.

### 4.2 Spatial Relationship Engine (`ai/incidents/spatial.py`)
Provides geometric reasoning:
- `iou(box1, box2)`: Intersection over Union.
- `bbox_overlap(box1, box2)`: Boolean area intersection.
- `centroid_distance(p1, p2)`: Euclidean distance between centers.
- `relative_orientation(p1, p2)`: Compass heading (north, southwest, etc.).
- `point_in_polygon(point, polygon)`: Ray-casting algorithm for zone boundaries.
- `track_zone_dwell_analysis(track, zone)`: Full lifecycle analysis (entered, exited, entry_time, exit_time, dwell_seconds).
- `check_trajectory_intersection(track_a, track_b)`: 2D segment crossing detection.

### 4.3 Temporal Analysis Engine (`ai/incidents/temporal.py`)
Provides time-series reasoning:
- `is_temporally_overlapping(t1, t2)`: Window overlap evaluation.
- `temporal_intersection(t1, t2)`: Intersection window.
- `detect_disappearance(target, reference)`: Target track cessation while reference persists.
- `detect_sudden_deceleration(motions)`: Sudden velocity drops/braking/impacts.
- `calculate_persistence(timestamps)`: Regularity score across observation windows.

### 4.4 Incident Scorer (`ai/incidents/scoring.py`)
Calibrates multi-signal scores into standardized evidence tiers:
- Base confidence from visual detections.
- Independent physical signal count bonus ($+0.04$ per distinct signal, up to $+0.20$).
- Track validation and multi-frame depth bonus ($+0.05$).
- Duration persistence bonus ($+0.04$ to $+0.08$).
- Capped at $0.95$ with explicit notice: `Observable Pattern Score: X% (Tier) — Human verification required`.

### 4.5 Incident Fusion Engine (`ai/incidents/fusion.py`)
Prevents event flooding:
- Clusters incident candidates sharing event types, temporal overlap (within tolerance $\le 3.0\text{s}$), and track identity or spatial proximity ($\le 120\text{ px}$).
- Unions supporting signals and evidence candidates without duplication.
- Harmonizes severity (highest severity governs the cluster).

---

## 5. Detector Registry & Failure Isolation (`ai/incidents/registry.py`)

The registry executes all registered detectors in a fault-tolerant loop:

```python
for name, detector in self._detectors.items():
    if not detector.enabled:
        continue
    try:
        candidates = detector.analyze(context)
        all_candidates.extend(candidates)
    except Exception as exc:
        # Isolated! Logged and recorded in diagnostics. Other detectors proceed normally.
        logger.error(f"Detector failure isolation: '{name}' raised {exc}")
        diagnostics["failed_detectors"] += 1
```

---

## 6. How to Add a New Incident Detector (Plug-in Guide)

To add any future incident detector (e.g., `FireDetector`, `FallDetector`, `AltercationDetector`):

### Step 1: Create the detector file in `ai/incidents/detectors/`
```python
from typing import List
from ai.incidents.base_detector import BaseIncidentDetector
from ai.incidents.schemas import IncidentCandidate, IncidentContext, IncidentCategory, SupportingSignal

class FallDetector(BaseIncidentDetector):
    detector_name = "fall_detector"
    detector_version = "1.0.0"
    category = IncidentCategory.PERSON

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates = []
        for track in context.get_tracks_by_class("person"):
            # Use Universal Motion Engine & Spatial Engine
            motions = context.track_motions.get(track.track_id, [])
            # e.g., analyze aspect ratio change from vertical to horizontal + sudden drop
            ...
            if fall_detected:
                cand = self.build_candidate(...)
                candidates.append(cand)
        return candidates
```

### Step 2: Register in `ai/incidents/detectors/__init__.py`
```python
def register_standard_detectors(registry):
    ...
    registry.register(FallDetector())
```

### Step 3: Zero other code modifications needed!
The new detector automatically participates in:
- Context delivery
- Motion & spatial telemetry
- Safe isolated execution
- Candidate fusion
- Database persistence via `SecurityEventModel`
- Evidence candidate generation
- Forensic Dossier reports

---

## 7. Backward Compatibility Guarantee

All `IncidentCandidate` objects provide `.to_security_event()`, ensuring 100% interoperability with Sentinel's existing:
- Database tables (`security_events`)
- Timeline event viewer
- Evidence Vault snapshot and clip extraction
- Report generation (`Incident Dossier`)
- Investigation services (`Ask Sentinel` / Gemini grounded query engine)
