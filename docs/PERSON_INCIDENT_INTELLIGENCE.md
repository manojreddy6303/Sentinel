# Sentinel Phase 12: Person Incident Intelligence Architecture

> **Core Ethical & Safety Declaration**:
> *"Person incident intelligence identifies potential observable patterns and does not establish intent, identity, or criminal responsibility."*
> Sentinel operates on the strict tenet of `CORRECT + UNCERTAIN / ABSTAIN` over `INCORRECT + CONFIDENT`. Visual analysis is conducted strictly on anonymous tracked subjects (`TRACK-001`, `TRACK-014`). Zero facial recognition, zero identity mapping, and zero intent/criminal attribution.

---

## 1. System Overview

Phase 12 establishes a robust, generalized, and modular Person Incident Intelligence layer integrated directly into Sentinel's Universal Incident Intelligence Core (Phase 10) and Global Reliability Architecture (Phase 10-R).

The pipeline executes as follows:
```
YOLO DETECTION (COCO Person Class)
    ↓
DETECTION VALIDATION (VALID / UNCERTAIN / REJECTED)
    ↓
ANONYMOUS MULTI-FRAME TRACKING (ByteTrack Track Quality Gate)
    ↓
PERSON MOTION & POSE FEATURES (Aspect Ratio Dynamics, Descent Velocity, Normalized Speed, Reciprocal Oscillations)
    ↓
SCENE CONTEXT ENGINE (Pedestrian Density, Normal Scene Flow)
    ↓
NEGATIVE EVIDENCE ENGINE (Upright Walking Resumption, Queueing, Abreast Transit Refutations)
    ↓
INCIDENT CANDIDATE GENERATION (8 Modular Person Detectors)
    ↓
INCIDENT CANDIDATE VALIDATOR (Global Gate, Severe Refutations)
    ↓
INCIDENT FUSION (Multi-Detector Clustering)
    ↓
EVIDENCE GATE (Forensic Snapshot/Clip Binding with Padding)
    ↓
SECURITY INTELLIGENCE, ASK SENTINEL & AUDIT REPORTS (Persistent Truth)
```

---

## 2. Supported Person Incident Classes

| Incident Class | Event Type | Primary Positive Signals | Counter-Evidence / Suppressions | Verification Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **Potential Person Fall** | `POTENTIAL_PERSON_FALL` | Rapid downward vertical displacement ($\ge 30$px/s), abrupt aspect-ratio inversion ($<0.65 \to \ge 0.85$), post-descent velocity drop. | Resumed upright walking transit, seated bench/chair context, upright posture maintained. | Mandatory Human Verification (`REVIEW_REQUIRED`) |
| **Potential Person Down** | `POTENTIAL_PERSON_DOWN` | Extended low-mobility dwell in horizontal/ground geometry ($\ge 3.0$s, $w/h \ge 0.75$), minimal centroid displacement. | Standing still upright ($w/h \approx 0.35$), resumed walking transit. | Mandatory Human Verification (`REVIEW_REQUIRED`) |
| **Potential Panic / Running** | `POTENTIAL_PANIC_RUNNING` | Sustained high normalized speed ($> 3.0$ body-lengths/s), sudden acceleration burst, multi-person concurrent rapid dispersion. | Consistent scene-wide pedestrian pace, expected recreational activity. | Observational Finding |
| **Unusual Rapid Movement** | `UNUSUAL_RAPID_PERSON_MOVEMENT` | Disproportionate velocity disparity relative to other pedestrians in the same scene ($> 2.2\times$ scene mean). | High-traffic synchronized transit flow, all pedestrians moving briskly. | Observational Finding |
| **Potential Physical Altercation** | `POTENTIAL_PHYSICAL_ALTERCATION` | Sustained close proximity ($\le 1.2$ body widths), repeated rapid reciprocal motion oscillations, opposing heading vectors. | Parallel side-by-side walking, pedestrian queue formation, normal low-mobility conversation. | Strictly `REVIEW_REQUIRED` (Never definitive "fight") |
| **Potential Forced Movement** | `POTENTIAL_FORCED_MOVEMENT` | Persistent close spatial contact ($< 1.0$ body width) with synchronized constrained trajectory deflections. | Normal pair walking abreast, pedestrian queueing. | Strictly `REVIEW_REQUIRED` (Never "kidnapping" or "abduction") |
| **Person Following** | `PERSON_FOLLOWING` | Pairwise lagged trajectory correlation (Person B traverses path of Person A with temporal lag $\Delta t \in [0.5, 3.0]$s, persistent spatial alignment). | Pedestrian queue formation, shared narrow pedestrian corridor, pairs walking abreast. | Observational Finding |
| **Coordinated Movement** | `COORDINATED_PERSON_MOVEMENT` | Synchronized heading vectors ($\le 25^\circ$ angular alignment), synchronized velocities, and persistent spatial group cohesion across 2+ individuals. | Independent crossing paths, random crowd dispersal. | Observational Finding (Never "conspiracy" or "criminal group") |

---

## 3. Person Motion Feature Engine (`PersonMotionFeatureEngine`)

Located in [`ai/incidents/detectors/person/motion_features.py`](file:///C:/Sentinel/ai/incidents/detectors/person/motion_features.py), this engine encapsulates reusable kinematic features:
1. **Aspect-Ratio Dynamics**:
   - Upright walking posture: $w/h \approx 0.30 - 0.55$.
   - Horizontal/fallen posture: $w/h \ge 0.75 - 2.0$.
   - Transition detection: monitors vertical-to-horizontal collapse within short temporal windows ($\le 2.5$s).
2. **Downward Vertical Velocity**:
   - Measures rate of centroid downward displacement in image space ($\Delta cy / \Delta t$).
3. **Scale-Normalized Velocity**:
   - Velocity is expressed in **body-lengths per second** ($\text{vel} / \text{avg\_height}$), ensuring scale invariance across near-camera and far-camera subjects.
4. **Lagged Trajectory Correlation**:
   - Evaluates whether Person B passes through historical coordinates of Person A with a time delay $\Delta t \in [0.5, 3.0]$s.
5. **Reciprocal Agitation Score**:
   - Quantifies opposing motion vectors and rapid distance expansions/contractions between two interacting individuals.

---

## 4. Modular Pose Feature Layer (`PoseFeatureEngine`)

Located in [`ai/incidents/detectors/person/pose_features.py`](file:///C:/Sentinel/ai/incidents/detectors/person/pose_features.py):
- **Design Principle**: YOLO bounding boxes alone cannot guarantee human joint articulation certainty.
- **Safety Guarantee**: Pose estimation is configurable (`SENTINEL_ENABLE_POSE_ESTIMATION=0` by default). The system **never** triggers unexpected runtime downloads or crashes if models are absent.
- **Graceful Fallback**: When pose estimation is inactive or weights are not found locally, Sentinel explicitly records `pose_available: False` and adds observational notes to the incident candidate, honestly labeling evidence strength as `Moderate / Review Required` rather than fabricating synthetic keypoints.

---

## 5. Negative Evidence & Abstention

The [`NegativeEvidenceEngine`](file:///C:/Sentinel/ai/incidents/negative_evidence.py) actively evaluates counter-signals:
- **Fall Refutation**: If a person descends but resumes upright walking transit within 2 seconds, the candidate is suppressed.
- **Person Down Refutation**: If a stationary person maintains an upright aspect ratio ($w/h \approx 0.35$), they are identified as standing still, not down.
- **Altercation Refutation**: If two individuals in close proximity walk in parallel heading vectors ($< 20^\circ$ difference), physical altercation is refuted.
- **Following Refutation**: If two individuals are stationary in a queue, following is refuted.

Candidates with severe counter-evidence are marked `ValidationDecision.REJECTED` by the [`IncidentCandidateValidator`](file:///C:/Sentinel/ai/incidents/validator.py).

---

## 6. Known Limitations

1. **Camera Occlusion & Crowding**: In extremely dense crowds, bounding box fragmentation can mask posture transitions. Tracks under 0.8s duration abstain from fall/down assertions.
2. **Sitting on Low Furniture**: Transitioning from standing to sitting on low seats may resemble aspect-ratio changes. Sentinel mitigates this via environmental context and requires human review.
3. **Recreational Sports**: People jogging in parks move at running speeds; Sentinel uses local scene baselines to differentiate isolated panic running from general activity.
