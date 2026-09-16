# Sentinel — Global Intelligence Reliability Architecture (Phase 10-R)
## Universal False-Positive Prevention, Candidate Validation & Observational Integrity

---

### Executive Summary

In automated computer vision surveillance, raw mathematical signals (bounding-box proximity, trajectory crossing, velocity vectors) frequently combine into false-positive security incidents when physical context and counter-evidence are ignored.

Phase 10-R establishes a **Global Intelligence Reliability Architecture** across Sentinel's entire AI pipeline, enforcing the governing principle:

$$\textbf{CORRECT + UNCERTAIN} \quad \succ \quad \textbf{INCORRECT + CONFIDENT}$$

Sentinel guarantees:
1. **Zero Hardcoding**: All reliability, contextual, and negative-evidence mechanisms are mathematically generalized and scene-derived. No video IDs, filenames, or coordinate constants exist anywhere in detection logic.
2. **Mandatory Candidate Validation**: Every incident candidate emitted by any detector must pass through the universal `IncidentCandidateValidator` across 12 rigorous evaluation dimensions.
3. **Negative Evidence Engine**: Active challenge of incident hypotheses via physical counter-evidence (e.g. continuous transit flow, zero physical bounding-box overlap, parallel lane following, object retention at rest).
4. **Explicit Abstention**: When evidence is ambiguous or refuted by counter-evidence, the system purposefully **abstains** (`NO_INCIDENT_DETECTED` or `REVIEW_REQUIRED`) rather than forcing a low-grade anomaly into a false high-confidence alarm.
5. **Evidence Eligibility Gate**: Unvalidated or refuted candidates are strictly gated from generating automatic snapshots or clips in the Evidence Vault.

---

### Core Philosophy: Why Object Detection $\ne$ Incident Detection

A common pitfall in AI surveillance systems is treating object detection as incident detection:
- **Object Detection** answers: *"Is there an object of class $C$ at coordinate $(x, y)$ in frame $t$?"*
- **Incident Detection** answers: *"Did an unpermitted physical event $E$ occur given multi-second scene geometry, temporal continuity, object agency, spatial constraints, and the absence of contradictory physical counter-evidence?"*

#### The Highway Collision Case Study
In highway CCTV footage (`livevid_crash0.mp4`), vehicles traveling in adjacent lanes frequently:
1. Pass near each other ($< 80\text{px}$ centroid distance) due to perspective foreshortening.
2. Approach each other at high relative speeds ($> 30\text{px/s}$) during overtaking maneuvers.
3. Show 2D trajectory lines that intersect on screen due to camera vanishing points.

A naive heuristic combining proximity + approach rate + 2D intersection concluded `POTENTIAL_VEHICLE_COLLISION` with 95% confidence on normal flowing traffic.

**Why this was semantically invalid:**
- **Zero Physical Overlap**: The 3D vehicles (and their 2D bounding boxes) maintained clearance at all times ($\text{IoU} = 0.0$).
- **Continued Normal Transit**: Neither vehicle experienced velocity discontinuity, spin, deflection, or post-impact stoppage. Both continued traveling down the roadway at highway speed.
- **Parallel Lane Following**: Both trajectory heading vectors aligned with standard multi-lane traffic flow.

Under Phase 10-R, Sentinel’s `NegativeEvidenceEngine` detects these physical counter-signals and `VehicleCollisionDetector` **abstains** from generating an alarm, correctly returning `NO RELIABLE COLLISION DETECTED`.

---

### Conceptual Architecture Pipeline

Every frame sequence traverses the non-bypassable intelligence pipeline:

```
                  RAW CCTV VIDEO
                        ↓
                 OBJECT DETECTION
                        ↓
               DETECTION VALIDATION
            (VALID / UNCERTAIN / REJECTED)
                        ↓
              STABLE OBJECT TRACKING
                        ↓
          MOTION & SPATIAL TELEMETRY
                        ↓
                 SCENE CONTEXT
          (Roadway, Perimeter, Crowd, etc.)
                        ↓
               NORMAL BASELINE
                        ↓
             INCIDENT CANDIDATE
                        ↓
          NEGATIVE EVIDENCE ENGINE
          (Physical Counter-Evidence)
                        ↓
        INCIDENT CANDIDATE VALIDATOR
      (12-Dimension Holistic Challenge)
                        ↓
                DECISION MATRIX
    [ACCEPTED]   [REVIEW_REQUIRED]   [REJECTED]
         │               │                │
         ▼               ▼                ▼
   HIGH STRENGTH   HUMAN REVIEW       SUPPRESSED
         │               │
         └───────┬───────┘
                 ↓
          INCIDENT FUSION
          (Deduplication & Merge)
                 ↓
      EVIDENCE ELIGIBILITY GATE
                 ↓
       SECURITY EVENTS / VAULT
```

---

### 12-Dimension Candidate Validator

Every detector candidate is evaluated by `IncidentCandidateValidator`:

| # | Dimension | Evaluation Method | Failure Impact |
|---|-----------|-------------------|----------------|
| 1 | Source Detection Validity | All source detections must be `VALID` or `UNCERTAIN`; `REJECTED` detections are prohibited. | Downgrades to `REJECTED` |
| 2 | Track Validity & Stability | Multi-frame track continuity; rejects 1-observation transient spikes claiming sustained events. | Downgrades to `REVIEW_REQUIRED` |
| 3 | Timestamp Validity | Monotonic positive timestamps ($t_{start} \ge 0, t_{end} \ge t_{start}$). | Rejects candidate |
| 4 | Bounding-Box Plausibility | Coordinates within camera bounds with positive non-zero area. | Rejects candidate |
| 5 | Spatial Consistency | Spatial coordinates align with track centroids and zone polygons. | Downgrades to `REVIEW_REQUIRED` |
| 6 | Temporal Consistency | Event duration conforms to semantic event category (e.g. theft $\ge 2.0\text{s}$, loitering $\ge 4.0\text{s}$). | Downgrades to `REVIEW_REQUIRED` |
| 7 | Motion Consistency | Observed velocity and acceleration agree with physical event physics. | Penalizes score |
| 8 | Scene Compatibility | Behavioral priors for scene type (roadway, pedestrian walkway, perimeter). | Suppresses false loitering/collision on highways |
| 9 | Supporting Signal Count | Requires at least 2 independent positive observable signals for high confidence. | Limits to `REVIEW_REQUIRED` |
| 10 | Negative Evidence Check | Counter-evidence strength penalty ($Score = Base - \sum 0.35 \times Neg$). | If severe counter-evidence: `REJECTED` |
| 11 | Duplicate Risk | Temporal/spatial overlap deduplication against active events. | Merges supporting signals |
| 12 | Evidence Grounding | Verified frame crops and track presence in target temporal window. | Suppresses ungrounded evidence generation |

---

### Negative Evidence Engine

The `NegativeEvidenceEngine` formalizes physical disproof:

1. **Vehicle Collision Counter-Evidence**:
   - `Negative: Zero Physical Contact`: Bounding boxes maintained physical clearance ($0.0\%$ overlap) throughout convergence.
   - `Negative: Continued Normal Transit`: Both vehicles continued uninterrupted motion at speed post-convergence with zero post-impact stoppage.
   - `Negative: Parallel Lane Following`: Trajectory heading vectors are parallel ($\le 25^\circ$ alignment), indicating multi-lane passage.

2. **Theft & Takeaway Counter-Evidence**:
   - `Negative: Object Remains Present`: Object track remains detectable at original coordinates with $< 25\text{px}$ displacement after person departs.

3. **Prolonged Presence / Loitering Counter-Evidence**:
   - `Negative: Continuous Transit Flow`: Subject exhibited high average velocity ($> 20\text{px/s}$) and linear displacement ($> 100\text{px}$), characteristic of normal passage.

4. **Zone Intrusion Counter-Evidence**:
   - `Negative: Unconfigured Zone`: Zone lacks valid boundary coordinates or is disabled.
   - `Negative: Unverified Track`: Subject track has insufficient multi-frame confirmation.

---

### Calibrated Scoring & Semantic Terminology

Sentinel enforces observational honesty:
- Heuristic scores represent **Internal Evidence Strength**, NOT Bayesian posterior probabilities of criminal guilt or mechanical collision.
- The UI and reports display:
  - `Evidence Strength: High (0.85)`
  - `Potential Incident (Review Required)`
  - `Human Verification Required`
- Disallowed misleading terminology:
  - ~~"Confirmed Crime"~~
  - ~~"95% Probability of Theft"~~
  - ~~"Guaranteed Collision"~~

---

### Evidence Eligibility Gate

Automatic preservation into the Evidence Vault requires:
1. `candidate.validation_decision != REJECTED`.
2. Evidence timestamp strictly within $[t_{start} - 1.0\text{s}, t_{end} + 1.0\text{s}]$.
3. Target track ID exists in verified context tracks.
4. Bounding box coordinates non-empty and grounded in source frame observations.

If grounding fails, evidence is stripped and marked `Evidence unavailable / review required`.
