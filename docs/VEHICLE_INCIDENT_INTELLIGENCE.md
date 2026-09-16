# Sentinel Phase 11: Vehicle Incident Intelligence Architecture

> **Guiding Principle**: *"Normal traffic behavior is not an incident."*
> Sentinel operates on the strict tenet of `CORRECT + UNCERTAIN / ABSTAIN` over `INCORRECT + CONFIDENT`. No incident candidate is generated solely from bounding box proximity, trajectory crossings, or isolated velocity changes.

---

## 1. System Overview

Phase 11 establishes a modular, generalized Vehicle Incident Intelligence layer built directly on top of the Phase 10 Universal Incident Intelligence and Phase 10-R Global Reliability Architecture.

The architecture follows a strict multi-tier pipeline:
```
DETECTION (YOLOv8 + Bounding Boxes)
    ↓
DETECTION VALIDATION (VALID / UNCERTAIN / REJECTED)
    ↓
TRACKING (ByteTrack / Track Quality Verification)
    ↓
MOTION EXTRACTION (Velocity, Acceleration, Headings)
    ↓
SPATIAL RELATIONSHIPS (Perspective Normalization, IoU, Enclosing Proximity)
    ↓
VEHICLE INTERACTION MODEL (Pairwise Metrics, Approach Rates, Divergence)
    ↓
SCENE CONTEXT ENGINE (Roadway Inferences, Dominant Traffic Flow Angle)
    ↓
NEGATIVE EVIDENCE ENGINE (Counter-Signal Verification)
    ↓
INCIDENT CANDIDATE GENERATION (Two-Stage Local Evaluation)
    ↓
INCIDENT CANDIDATE VALIDATOR (Global Gate, Refutations)
    ↓
EVIDENCE GATE & ADAPTIVE SAMPLING (Nyquist Verification, Burst Triggering)
    ↓
INCIDENT FUSION (Temporal/Spatial Merging)
    ↓
SECURITY INTELLIGENCE & AUDIT TRAIL (Persistence, Ask Sentinel, Dossier)
```

---

## 2. Vehicle Detector Architecture

All vehicle incident detectors inherit from `BaseIncidentDetector` and reside within `ai/incidents/detectors/vehicle/`:

| Detector | Event Type | Primary Positive Signals | Key Negative Signals / Refutations |
| :--- | :--- | :--- | :--- |
| **`VehicleCollisionDetector`** | `POTENTIAL_VEHICLE_COLLISION` | Verified physical contact (IoU > 0.08), rapid mutual deceleration, abnormal trajectory deflection, post-contact rest. | Zero physical overlap (clearance > 0), parallel lane transit, normal overtaking, continued transit flow. |
| **`NearCollisionDetector`** | `POTENTIAL_NEAR_COLLISION` | Extremely close approach, rapid mutual approach rate, evasive lateral swerve, continued transit without contact. | Physical contact (promotes to collision review), normal passing clearance, stable lane separation. |
| **`SuddenStopDetector`** | `POTENTIAL_SUDDEN_VEHICLE_STOP` | Abrupt velocity reduction (>70% drop to near-zero), isolated to target vehicle, sustained stoppage. | Synchronized slowdown across neighboring vehicles, traffic queue formation, traffic light/intersection approach. |
| **`WrongWayVehicleDetector`** | `POTENTIAL_WRONG_WAY_VEHICLE` | Persistent movement opposing inferred dominant flow (>135° angle difference), high flow confidence. | Ambiguous flow direction, turning maneuvers, ramp/intersection context, insufficient duration. |
| **`UnusualTrajectoryDetector`** | `POTENTIAL_UNUSUAL_VEHICLE_TRAJECTORY` | High lateral oscillation / weaving (repeated heading shifts), severe abrupt heading deviation (>65°). | Smooth continuous lane changes, turning at roadway curves, normal alignment. |
| **`StationaryVehicleDetector`** | `POTENTIAL_STATIONARY_VEHICLE` | Prolonged stationary dwell (>4s) in active roadway corridor, isolated stoppage with moving traffic around. | General traffic congestion, designated parking context, synchronized traffic halts. |

---

## 3. Vehicle Interaction Model (`VehicleInteractionModel`)

Rather than duplicating pairwise geometry inside individual detectors, `ai/incidents/detectors/vehicle/interaction.py` encapsulates pairwise vehicle calculations:
- **Perspective-Normalized Distance**: Normalizes pixel distance by the geometric mean diagonal of the two vehicles:
  $$\text{norm\_dist} = \frac{d}{\sqrt{\text{diag}_A \cdot \text{diag}_B}}$$
- **Relative Distance & Delta**: Computes current distance and displacement change over observation windows.
- **Approach & Separation Rates**: Quantifies convergence speed (pixels/sec or normalized units/sec).
- **Relative Heading & Divergence**: Computes angular differential between velocity vectors in degrees $[0^\circ, 180^\circ]$.
- **Physical Contact & Overlap**: Checks intersection-over-union (IoU) and bounding box overlap.
- **Pre/Post-Event Velocities**: Evaluates kinematic discontinuities across interaction windows.

---

## 4. Camera Perspective Normalization

CCTV cameras exhibit significant perspective distortions:
- Vehicles near the camera appear large and move large pixel distances per second.
- Vehicles far along the horizon appear small and move tiny pixel distances per second.
- A fixed 50-pixel distance threshold is mathematically invalid across varying depths.

**Normalization Approach**:
1. Centroid distances are normalized against observed vehicle bounding-box scales ($scale = \sqrt{w \cdot h}$).
2. Proximity checks test whether distance is within fractional vehicle body lengths ($\le 0.45 \times \text{average dimension}$).
3. Physical contact checks strictly test 2D bounding-box overlap and verified intersection.

---

## 5. Adaptive Temporal Sampling (`AdaptiveTemporalSamplingEngine`)

Standard surveillance video processing may analyze frames at lower rates (e.g., 1–5 FPS) to conserve compute:
- A collision or near-miss interaction occurring across 200–500 milliseconds cannot be definitively proven or refuted at 1 FPS.
- The `AdaptiveTemporalSamplingEngine` inspects kinematics and triggers high-rate burst analysis (5, 10, 15, or 30 FPS) when suspicious rapid motion or close approach is observed.
- **Nyquist Temporal Evidence**: If the source video stream or processing context lacks high-rate frames (effective sampling interval > 0.4s), the incident is tagged with `TEMPORAL_EVIDENCE_LIMITED` and automatically downgraded to `ValidationDecision.REVIEW_REQUIRED` or `ABSTAIN`.

---

## 6. Negative Evidence Engine

Every candidate undergoes negative counter-evidence evaluation before entering global validation:
1. **Collision Refutation**:
   - `zero_physical_overlap`: Bounding boxes maintained measurable positive clearance.
   - `parallel_lane_following`: Vehicles traveled in parallel trajectories with stable lateral offset.
   - `continued_normal_transit`: Both vehicles maintained forward transit velocity without post-event stoppage.
2. **Sudden Stop Refutation**:
   - `synchronized_traffic_slowdown`: Neighboring vehicles simultaneously reduced velocity (traffic wave / signal stop).
   - `traffic_queue_present`: Multiple stopped vehicles queued ahead in the same corridor.
3. **Wrong-Way Refutation**:
   - `ambiguous_roadway_flow`: Insufficient observed vehicle trajectories to establish ground truth flow direction.
   - `turning_maneuver_present`: Trajectory aligns with intersection or curve negotiation rather than direct opposing flow.
4. **Stationary Refutation**:
   - `general_traffic_congestion`: Multiple vehicles stationary across adjacent lanes.
   - `parking_lane_context`: Stationary vehicle positioned along roadway periphery.

---

## 7. Incident Candidate Validation & Abstention

`IncidentCandidateValidator` acts as the impartial global gatekeeper:
- Evaluates track reliability: One-frame tracks or tracks under 0.25s duration are flagged or rejected.
- Evaluates negative counter-signals: If any severe negative evidence signal has confidence $\ge 0.70$, the candidate is marked `ValidationDecision.REJECTED`.
- If evidence is partial or ambiguous, the candidate is marked `ValidationDecision.REVIEW_REQUIRED` (Human Verification Required).
- Purely speculative or contradictory claims are rejected (`ABSTAIN`).

---

## 8. Incident Fusion & Evidence Grounding

- **Fusion**: `IncidentFusionEngine` groups overlapping candidates within 2.5 seconds sharing track IDs and event categories, preventing alert spam while preserving all supporting observations.
- **Evidence Vault**: Grounded forensic clips and snapshots are extracted with 2.0-second pre-event and post-event temporal padding to clearly display vehicle approach, the interaction window, and subsequent behavior.

---

## 9. Security Intelligence UI, Reports, and Ask Sentinel Compatibility

- **Consistent Truth**: All components (`SecurityEventModel`, `IncidentScorer`, `InvestigationService`, `InvestigationOrchestrator`, and PDF Dossier reports) read from the same validated database records.
- **Honest Wording**: UI cards and Ask Sentinel display calibrated labels (*"Evidence Strength: Moderate"*, *"Human verification required"*, *"Initial Observation"*), never fabricated percentage certainties.
- **Zero-Incident Honesty**: When no verified vehicle incidents exist in the video, Sentinel responds definitively:
  > *"No reliable vehicle incident was detected in the available Sentinel data."*
