# Sentinel — Crowd, Density & Zone Intelligence Core (Phase 14)

## Overview & Ethical Principles

> **Core Ethical Mandate:**
> Crowd and zone intelligence describes observable density, occupancy, movement, and spatial patterns. It does not establish threat, intent, criminal activity, or group identity.

Sentinel strictly decouples physical observations from moral and criminal assumptions:
- A crowd is never assumed to be dangerous or hostile.
- High pedestrian density is an observational metric, not evidence of misconduct.
- Presence within a restricted zone is an occupancy observation, not proof of criminal intent.
- Groups and individuals remain strictly anonymous. No facial recognition, demographic profiling, or biometric tracking is permitted.

---

## Architecture Flow

```
RAW VIDEO
  │
  ▼
DETECTION & VALIDATION (YOLO + Multi-frame Confidence Verification)
  │
  ▼
ANONYMOUS MULTI-OBJECT TRACKING (ByteTrack / Kalman)
  │
  ▼
SCENE CONTEXT & BASELINE INFERENCE (Pedestrian Flow, Roadway, Prior Densities)
  │
  ▼
SECURITY ZONE GEOMETRY (Defined Boundary Polygons, Hysteresis Tracking)
  │
  ▼
DENSITY & CLUSTERING ENGINE (Euclidean Adjacency, Area Occupancy, Inter-Distance)
  │
  ▼
QUEUE & FLOW ENGINE (Linear Regression, Collinearity, Directional Alignment)
  │
  ▼
TEMPORAL BASELINE COMPARISON (Moving Mean, Ratio to Learned Baseline)
  │
  ▼
CROWD & ZONE CANDIDATE GENERATION (Surge, Density, Dispersal, Restricted Crowding)
  │
  ▼
NEGATIVE EVIDENCE ENGINE (Queue Refutation, Normal Pace, Boundary Jitter Counter-Evidence)
  │
  ▼
GLOBAL VALIDATOR (ACCEPTED, REVIEW_REQUIRED, REJECTED)
  │
  ▼
INCIDENT FUSION (Arbitration of Overlapping Hypotheses, Deduplication)
  │
  ▼
EVIDENCE GATE (Grounded Snapshots, Timeline Intervals, Metadata Provenance)
  │
  ▼
SECURITY EVENT / TIMELINE / INVESTIGATION / PDF DOSSIER
```

---

## Core Components

### 1. Crowd Density Engine (`ai/incidents/detectors/crowd/density_engine.py`)
- **Track-Grounded Calculations:** Operates strictly on verified, multi-frame anonymous person tracks (`TrackedObject.is_validated == True`). Prevents detection double-counting.
- **Class Separation:** Pedestrians, vehicles, and belongings are tracked in separate populations. Vehicles are never counted as crowd members.
- **Anonymous Spatial Clustering:** Implements connected-component graph grouping based on pairwise Euclidean distance thresholds ($d \le 120\text{px}$). Groups are anonymous clusters without personal identities.
- **Windowed Density Metrics:** Computes moving window metrics:
  - Active person count
  - Unique track count
  - Cluster count and sizes
  - Average and minimum inter-person distances
  - Image-plane occupied area ($\text{px}^2$)
  - Rate of change ($\Delta N / \Delta t$)
  - Density ratio relative to scene baseline

### 2. Perspective Normalization & Scene Baseline
- In standard CCTV setups lacking 3D metric camera calibration, density is explicitly reported as an **image-plane relative density estimate** rather than an unsupported metric ($\text{persons/m}^2$).
- The engine integrates with `SceneContextData` and `NormalBehaviorBaseline`:
  - `average_density_per_second`
  - `dominant_heading_degrees`
  - `average_velocity`
- High density events trigger only when observed density significantly exceeds local baseline ($\ge 1.8\times$).

### 3. Queue & Orderly Flow Engine (`ai/incidents/detectors/crowd/queue_flow_engine.py`)
- Computes principal axis linear regression and perpendicular scatter across active pedestrians.
- Verifies elongation: queues must exhibit longitudinal span $\ge 2.2\times$ transverse scatter.
- Evaluates directional heading coherence (circular variance $\ge 0.70$) and advance speed ($\le 22\text{px/s}$).
- Acts as **critical negative evidence**: an orderly waiting line or structured commute stream is refuted from being classified as a crowd surge.

### 4. Crowd Surge Detector (`ai/incidents/detectors/crowd/crowd_surge.py`)
- Conservative candidate requiring all of the following independent signals:
  1. Multi-person presence ($\ge 4$ persons).
  2. Directional heading coherence ($\ge 0.70$).
  3. Abnormal group velocity ($> 24\text{px/s}$).
  4. Spatial cluster compression ($\le 85\text{px}$ inter-person distance).
  5. Refutation of orderly queue structure via Queue & Flow Engine.
- Outputs observational label: `POTENTIAL_CROWD_SURGE`. Never infers panic or aggression.

### 5. Crowd Dispersal Detector (`ai/incidents/detectors/crowd/crowd_dispersal.py`)
- Detects rapid fragmentation or departure of a previously concentrated crowd cluster.
- Evaluates temporal cluster decay ($\ge 3$ persons departing or cluster fragmenting over $< 3.0\text{s}$).
- Outputs observational label: `POTENTIAL_CROWD_DISPERSAL`.

### 6. Unusual Crowd Movement Detector (`ai/incidents/detectors/crowd/unusual_crowd_movement.py`)
- Evaluates cohesive group transit deviating materially ($> 75^\circ$) from the dominant scene baseline flow angle.
- Outputs observational label: `POTENTIAL_UNUSUAL_CROWD_MOVEMENT`.

### 7. Zone Occupancy & Activity Detector (`ai/incidents/detectors/crowd/zone_occupancy.py`)
- Reuses existing `SecurityZone` polygon definitions.
- Tracks entry, dwell, and continuous occupancy with **boundary hysteresis** to eliminate tracker jitter oscillations.
- Supports multi-occupant presence in restricted zones: `POTENTIAL_RESTRICTED_ZONE_CROWDING`.
- Differentiates brief transit crossings ($< 2.5\text{s}$) from sustained group presence.

---

## Negative Evidence & Severe Contradictions

The global `NegativeEvidenceEngine` and `IncidentCandidateValidator` enforce counter-evidence refutations:
- **Orderly Queue Flow:** Refutes `POTENTIAL_CROWD_SURGE`.
- **Steady Scene Baseline:** Refutes abnormal crowd movement if within normal velocity variance.
- **Transient Boundary Crossing:** Refutes `POTENTIAL_RESTRICTED_ZONE_CROWDING` if dwell is under minimum threshold.
- **Boundary Jitter / Low Hit Count:** Suppresses oscillations at zone edges.

When severe counter-evidence is present, candidates are downgraded (`ACCEPTED` $\to$ `REVIEW_REQUIRED` or `REJECTED`).

---

## Incident Fusion & Arbitration

`IncidentFusionEngine` arbitrates overlapping crowd hypotheses in the same temporal window:
```
Hierarchy:
1. POTENTIAL_CROWD_SURGE (Priority 5)
2. POTENTIAL_RESTRICTED_ZONE_CROWDING (Priority 4)
3. POTENTIAL_UNUSUAL_CROWD_MOVEMENT (Priority 3)
4. POTENTIAL_CROWD_DISPERSAL (Priority 3)
5. POTENTIAL_UNUSUAL_ZONE_ACTIVITY (Priority 2)
6. HIGH_PEDESTRIAN_DENSITY (Priority 1)
7. CROWD_DENSITY_INCREASE (Priority 1)
8. ZONE_OCCUPANCY_OBSERVATION (Priority 0)
```
- Subsumes co-occurring interpretations (e.g. surge + high density) into a singular primary event.
- Retains alternate hypotheses in `incident_metadata["alternate_hypotheses"]`.
- Merges supporting signals without duplicate amplification.

---

## Natural-Language Investigation Queries

Natural language investigation maps investigator queries directly into database filter operations:
- `"show crowded areas"`, `"high pedestrian density"` $\to$ `HIGH_PEDESTRIAN_DENSITY`
- `"find crowd surges"`, `"crowd surge"` $\to$ `POTENTIAL_CROWD_SURGE`
- `"show crowd dispersal"` $\to$ `POTENTIAL_CROWD_DISPERSAL`
- `"show unusual crowd movement"` $\to$ `POTENTIAL_UNUSUAL_CROWD_MOVEMENT`
- `"show restricted zone occupancy"`, `"crowding in the entrance gate"` $\to$ `POTENTIAL_RESTRICTED_ZONE_CROWDING`
- `"which zones had the highest occupancy?"` $\to$ `ZONE_OCCUPANCY_OBSERVATION`

---

## Known Limitations

1. **2D Perspective & Optical Compression:** In oblique CCTV views, distant individuals appear artificially close together due to foreshortening. True metric calibration requires camera intrinsic and ground-plane homography.
2. **Foreground Occlusions:** Large physical barriers (pillars, vehicles) can segment continuous crowds into artificial separate clusters.
3. **Severe Perspective Scaling:** Extremely high camera angles compress vertical movement; low camera angles suffer from mutual pedestrian occlusion.
4. **Adaptive Frame Sampling:** Sampling every 1.0s provides high temporal efficiency for security surveillance, but sub-second micro-surges require full-frame processing.
