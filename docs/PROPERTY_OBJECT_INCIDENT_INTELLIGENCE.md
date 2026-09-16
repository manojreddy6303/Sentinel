# Sentinel — Property & Object Incident Intelligence Core (Phase 13)

---

### 1. Executive Summary

Phase 13 introduces the **Property & Object Incident Intelligence Core** to Sentinel (`C:\Sentinel`), extending the Universal Incident Intelligence Engine (Phase 10), Global Reliability Architecture (Phase 10-R), Vehicle Intelligence (Phase 11), and Person Intelligence (Phase 12).

The core provides deterministic, observationally grounded analysis of physical objects, portable belongings, and stationary fixtures over time.

$$\textbf{CORRECT + UNCERTAIN} \quad \succ \quad \textbf{INCORRECT + CONFIDENT}$$

> [!IMPORTANT]
> **Strict Observational Safety Mandate**:
> *"Property and object intelligence identifies potential observable object/state changes and does not establish theft, criminal intent, or criminal responsibility."*
> No legal conclusions, criminal accusations, or intent attributions are ever generated. All ambiguous interactions require human operator verification.

---

### 2. Architecture Overview

Property intelligence integrates directly into Sentinel's universal incident pipeline:

```
YOLO Detections (Multi-Frame)
       │
       ▼
Detection Validation & Filter
       │
       ▼
Multi-Object & Person Tracking (TrackModel)
       │
       ├──► Camera Stability Engine (Global scene translation / jitter detection)
       │
       ├──► Object Temporal State Machine (DETECTED ➔ STATIONARY ➔ ASSOCIATED ➔ LEFT_BEHIND / MOVING / DISPLACED)
       │
       └──► Object-Person Association Layer (Approach vector, contact proximity, departure displacement, co-movement)
               │
               ▼
       Property Incident Detectors (8 Observational Classes)
               │
               ▼
       Negative Evidence Engine (Refutations & Counter-Signals)
               │
               ▼
       Incident Candidate Validator (ACCEPT / REVIEW_REQUIRED / REJECT)
               │
               ▼
       Incident Fusion Engine (Cross-Detector Arbitration & Hierarchy)
               │
               ▼
       Evidence Gate ➔ SecurityEventModel (Database, Timeline, Dossier Reports, UI)
```

---

### 3. Supported Observational Incident Classes (8 Classes)

| Incident Class | Key Observable Physical Signatures | Negative Evidence Refutations | Default Abstention / Review |
|---|---|---|---|
| `POTENTIAL_ABANDONED_OBJECT` | Portable belonging remains stationary and unattended for sustained duration; no tracked person in proximity | Tracked person in immediate vicinity; continuous displacement (being carried) | If person nearby $\rightarrow$ Refuted; if ambiguous $\rightarrow$ `REVIEW_REQUIRED` |
| `POTENTIAL_OBJECT_LEFT_BEHIND` | Person interaction $\rightarrow$ person departs ($> 65\text{px}$) $\rightarrow$ object persists stationary at initial coordinates | Person returns to vicinity; person never established physical interaction | Requires temporal departure chain; defaults to `REVIEW_REQUIRED` |
| `POTENTIAL_OBJECT_PICKUP` | Previously stationary object is approached by person $\rightarrow$ physical contact $\rightarrow$ object initiates motion concurrently with person | Object never moved; person walked past without dwell or contact | Ambiguous pickup moment $\rightarrow$ `REVIEW_REQUIRED` |
| `POTENTIAL_THEFT` / `POTENTIAL_OBJECT_TAKEAWAY` | Person approaches object $\rightarrow$ contact dwell $\rightarrow$ object disappears or co-moves $\rightarrow$ person departs | Object remains present at original coordinates; person leaves empty-handed | Grounded forensic evidence required; defaults to `REVIEW_REQUIRED` |
| `POTENTIAL_OBJECT_DISPLACEMENT` | Object relocates from initial stable coordinates to a distinct subsequent stable position ($> 40\text{px}$ and $> 0.8\times$ body scale) | Global camera motion/pan; displacement below scale noise threshold | Sub-scale movement $\rightarrow$ `REJECT`; camera motion $\rightarrow$ `REJECT` |
| `POTENTIAL_PROPERTY_TAMPERING` | Sustained physical contact ($> 3.0\text{s}$) with stationary fixture/barrier accompanied by observable coordinate disturbance ($> 12\text{px}$) | Transient passing proximity; zero property coordinate alteration | No physical state change $\rightarrow$ Refuted; defaults to `REVIEW_REQUIRED` |
| `POTENTIAL_RESTRICTED_OBJECT_MOVEMENT` | Portable object transits into or dwells inside a configured `SecurityZone` polygon | Object trajectory strictly exterior to zone | Requires zone polygon intersection; defaults to `REVIEW_REQUIRED` |
| `POTENTIAL_OBJECT_REMOVAL` | Object stably tracked for $\ge 3.0\text{s}$ ceases detection inside camera FOV without frame boundary transit | Frame boundary exit ($x, y$ within margin); concurrent track occlusion | Boundary exit $\rightarrow$ `REJECT`; occlusion $\rightarrow$ `REVIEW_REQUIRED` |

---

### 4. Reusable Engines & Models

#### A. Temporal Object State Machine (`ObjectStateMachine`)
Tracks state transitions over time:
- `DETECTED`: Initial observation.
- `STATIONARY`: Centroid velocity $< 12\text{px/s}$, displacement $< 25\text{px}$.
- `ASSOCIATED_WITH_PERSON`: Tracked person in interaction proximity ($\le 120\text{px}$) with temporal overlap.
- `MOVING`: Centroid velocity $> 15\text{px/s}$.
- `DISPLACED`: Settled at a distinct coordinate ($\Delta d \ge 45\text{px}$).
- `LEFT_BEHIND`: Person was associated, departed ($> 65\text{px}$), object remains stationary.
- `REMOVED`: Object ceased detection inside camera FOV without occlusion or boundary exit.
- `LOST_TRACK`: Boundary transit or severe occlusion leading to track loss.

#### B. Object-Person Association Layer (`ObjectPersonAssociationEngine`)
- Evaluates spatial proximity, duration of contact, approach vector ($(\vec{x}_{\text{obj}} - \vec{x}_{\text{person}}) \cdot \vec{v}_{\text{person}} > 0$), departure vector, and co-movement trajectory correlation.
- Prevents spurious associations with passersby who do not pause or physically interact with the object.

#### C. Camera Stability & Scene Motion Protection (`CameraStabilityEngine`)
- Evaluates global displacement vectors across background and static tracks.
- If coherent scene-wide motion is detected (camera pan, tilt, or shake), object displacement detections are suppressed to prevent camera-induced false alarms.

---

### 5. Incident Fusion Arbitration

To prevent multiple overlapping alerts on the same object (e.g. pickup + takeaway + displacement), `IncidentFusionEngine._arbitrate_competing_property_interactions` enforces a strict specificity hierarchy:

$$\text{POTENTIAL\_THEFT} \succ \text{POTENTIAL\_OBJECT\_REMOVAL} \succ \text{POTENTIAL\_OBJECT\_LEFT\_BEHIND} \succ \text{POTENTIAL\_OBJECT\_PICKUP} \succ \text{POTENTIAL\_OBJECT\_DISPLACEMENT} \succ \text{POTENTIAL\_ABANDONED\_OBJECT}$$

The most specific and complete hypothesis is emitted as primary, while alternative interpretations are preserved in `incident_metadata["alternate_hypotheses"]`.

---

### 6. Known Limitations

1. **Occlusion by Larger Vehicles/Objects**:
   - Small portable belongings (phones, bottles) occluded behind a vehicle or person may temporarily vanish. The system treats these as `REVIEW_REQUIRED` or `LOST_TRACK` rather than confirmed removals.
2. **Monocular 2D Proximity**:
   - In single-camera views without 3D depth calibration, a person walking in front of a distant object can appear close in 2D pixel space. Temporal contact duration and approach/departure vectors are required to mitigate this.
3. **Frame-Rate & Rapid Pickups**:
   - Fast takeaway interactions occurring in sub-second intervals may lack intermediate co-movement frames. In such cases, the system documents sampling limitations and flags the event for human verification.
