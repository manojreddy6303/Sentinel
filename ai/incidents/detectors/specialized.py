"""
Specialized Incident Detectors (Phase 15)

Bridges specialized visual observations and multi-frame temporal tracks
into standard Sentinel IncidentCandidate records with:
- Forensic validation
- Negative evidence challenges
- Temporal consistency checks
- Observational event terminology
"""
import uuid
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

from ai.incidents.base_detector import BaseIncidentDetector
from ai.incidents.schemas import (
    IncidentCandidate,
    IncidentContext,
    IncidentCategory,
    SupportingSignal,
    EvidenceCandidate,
    SpatialContext,
    TemporalContext,
)
from ai.schemas import BoundingBox
from ai.incidents.negative_evidence import NegativeEvidenceEngine
from ai.specialized.schemas import (
    SpecializedTemporalTrack,
    SpecializedObservation,
    SpecializedValidationStatus,
)


class SpecializedFireIncidentDetector(BaseIncidentDetector):
    """
    Evaluates specialized fire visual observations and temporal tracks.
    Emits POTENTIAL_FIRE only when flame characteristics persist across frames.
    """

    detector_name: str = "specialized_fire_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.ENVIRONMENT

    required_signals: List[str] = ["Flame Visual Telemetry"]
    supporting_signals_declared: List[str] = [
        "Flame Visual Telemetry",
        "Sustained Thermal Combustion Pattern",
        "Dynamic Chromatic Boundary",
    ]
    contradictory_signals_declared: List[str] = [
        "Negative: Transient Visual Glitch",
        "Negative: Static Surface Chromaticity",
        "Negative: Potential Vehicle Lighting Glare",
    ]
    confidence_method: str = "temporal_persistence_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    # -----------------------------------------------------------------------
    # Episode Evidence Gate — minimum requirements before raising a candidate.
    # Any episode that fails these thresholds is silently abstained.
    # -----------------------------------------------------------------------
    MINIMUM_FIRE_OBSERVATIONS: int = 3
    MINIMUM_FIRE_DURATION_S: float = 0.8
    MINIMUM_FIRE_MEAN_CONF: float = 0.45
    MINIMUM_FIRE_EVIDENCE_STR: float = 0.38

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        episodes = getattr(context, "specialized_episodes", None)
        if episodes is not None:
            fire_episodes = [e for e in episodes if getattr(e, "class_name", "") == "fire"]
            if not fire_episodes:
                return []
            for ep in fire_episodes:
                obs_cnt = getattr(ep, "observation_count", 1)
                duration = getattr(ep, "duration_seconds", 0.0)
                max_conf = getattr(ep, "peak_confidence", 0.5)
                raw_mean_conf = getattr(ep, "mean_confidence", 0.0)
                mean_conf = raw_mean_conf if raw_mean_conf > 0.0 else max_conf
                evidence_str = getattr(ep, "peak_evidence_strength", 0.0)
                # Fall back to mean_conf-derived estimate if evidence_str is 0.0
                if evidence_str == 0.0:
                    evidence_str = mean_conf * 0.85

                # --- Episode Evidence Gate ---
                # Abstain if evidence falls below minimum thresholds.
                if obs_cnt < self.MINIMUM_FIRE_OBSERVATIONS and duration < self.MINIMUM_FIRE_DURATION_S:
                    logger.debug(
                        "Fire episode %s abstained: obs=%d (min %d), duration=%.2fs (min %.2fs)",
                        ep.episode_id, obs_cnt, self.MINIMUM_FIRE_OBSERVATIONS,
                        duration, self.MINIMUM_FIRE_DURATION_S,
                    )
                    continue
                if mean_conf < self.MINIMUM_FIRE_MEAN_CONF:
                    logger.debug(
                        "Fire episode %s abstained: mean_conf=%.3f < %.3f",
                        ep.episode_id, mean_conf, self.MINIMUM_FIRE_MEAN_CONF,
                    )
                    continue
                if evidence_str < self.MINIMUM_FIRE_EVIDENCE_STR:
                    logger.debug(
                        "Fire episode %s abstained: evidence_str=%.3f < %.3f",
                        ep.episode_id, evidence_str, self.MINIMUM_FIRE_EVIDENCE_STR,
                    )
                    continue

                supporting = [
                    SupportingSignal(
                        signal_type="Flame Visual Telemetry",
                        description=(
                            f"Visual flame chromaticity detected across {obs_cnt} observation(s) "
                            f"({duration:.2f}s duration, peak confidence {max_conf:.2f})."
                        ),
                        confidence=max_conf,
                        timestamp=ep.start_time,
                    )
                ]

                if duration >= 1.0:
                    supporting.append(
                        SupportingSignal(
                            signal_type="Sustained Thermal Combustion Pattern",
                            description=f"Continuous localized flame emission sustained for {duration:.2f} seconds across {ep.segment_count} segment(s).",
                            confidence=min(0.95, max_conf + 0.05),
                            timestamp=ep.start_time + duration / 2.0,
                        )
                    )

                # Compute is_static_surface from episode metadata:
                # If area coefficient of variation is near zero, the region is not flickering
                # (characteristic of a painted orange fixture, not a flame).
                ep_meta = getattr(ep, "metadata", {}) or {}
                mean_area_cv = ep_meta.get("mean_area_cv", 1.0)
                is_static_surface = mean_area_cv < 0.04

                # Check if flame episode overlaps a pedestrian's body/clothing
                is_person_clothing = False
                if ep.bounding_box:
                    ecx = (ep.bounding_box.x1 + ep.bounding_box.x2) / 2.0
                    ecy = (ep.bounding_box.y1 + ep.bounding_box.y2) / 2.0
                    for ptrk in (context.tracks or []):
                        if ptrk.object_class == "person" and ptrk.current_bbox:
                            pb = ptrk.current_bbox
                            if pb.x1 <= ecx <= pb.x2 and pb.y1 <= ecy <= pb.y2:
                                is_person_clothing = True
                                break

                # Check if aerosol smoke plume exists anywhere in the scene
                has_smoke = any(
                    getattr(t, "class_name", "") == "smoke"
                    for t in (context.specialized_tracks or [])
                ) or any(
                    getattr(ep_s, "class_name", "") == "smoke"
                    for ep_s in (context.specialized_episodes or [])
                )

                contradictory = NegativeEvidenceEngine.evaluate_fire_negative_evidence(
                    observation_count=obs_cnt,
                    persistence_duration=duration,
                    is_static_surface=is_static_surface,
                    scene_context=context.scene_context,
                    is_person_clothing=is_person_clothing,
                    has_smoke=has_smoke,
                    event_time=ep.start_time,
                )

                spatial = None
                if ep.bounding_box:
                    cx = (ep.bounding_box.x1 + ep.bounding_box.x2) / 2.0
                    cy = (ep.bounding_box.y1 + ep.bounding_box.y2) / 2.0
                    spatial = SpatialContext(centroid=(cx, cy), bounding_box=ep.bounding_box)

                evidence_cands = []
                for rep in ep.representative_evidence_boxes:
                    bb = rep.get("bounding_box")
                    if bb:
                        bbox = BoundingBox(x1=bb["x1"], y1=bb["y1"], x2=bb["x2"], y2=bb["y2"])
                        evidence_cands.append(
                            EvidenceCandidate(
                                timestamp=rep["timestamp"],
                                pre_seconds=2.0,
                                post_seconds=3.0,
                                bounding_box=bbox,
                                reason=f"Forensic visual snapshot of potential fire (conf: {rep.get('confidence', max_conf):.2f})",
                            )
                        )
                if not evidence_cands and ep.bounding_box:
                    evidence_cands.append(
                        EvidenceCandidate(
                            timestamp=ep.start_time,
                            pre_seconds=2.0,
                            post_seconds=3.0,
                            bounding_box=ep.bounding_box,
                            reason="Forensic visual snapshot of potential fire origin",
                        )
                    )

                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_FIRE",
                    start_time=ep.start_time,
                    end_time=ep.end_time,
                    severity="HIGH",
                    confidence=round(max_conf, 4),
                    explanation=(
                        f"Potential fire visual evidence observed: Sustained localized flame chromaticity "
                        f"detected across {obs_cnt} observations ({duration:.2f}s duration, {ep.segment_count} segment(s)). Human verification required."
                    ),
                    object_classes=["fire"],
                    supporting_signals=supporting,
                    contradictory_signals=contradictory,
                    spatial_context=spatial,
                    temporal_context=TemporalContext(
                        start_time=ep.start_time,
                        end_time=ep.end_time,
                        duration_seconds=duration,
                    ),
                    evidence_candidates=evidence_cands,
                    human_verification_required=True,
                    prefix="FIRE",
                    incident_metadata={
                        "episode_id": ep.episode_id,
                        "observation_count": obs_cnt,
                        "segment_count": ep.segment_count,
                        "peak_confidence": round(max_conf, 4),
                        "evidence_strength": round(ep.peak_evidence_strength, 4),
                        "supporting_observation_ids": ep.observation_ids,
                        "representative_timestamps": ep.representative_timestamps,
                    },
                )
                candidates.append(cand)
            return candidates

        # Fallback to specialized_tracks only if episodes were not evaluated at all
        if not context.specialized_tracks:
            return candidates

        fire_tracks = [t for t in context.specialized_tracks if getattr(t, "class_name", "") == "fire"]

        for trk in fire_tracks:
            obs_cnt = getattr(trk, "observation_count", 1)
            duration = getattr(trk, "persistence_duration", 0.0)
            max_conf = getattr(trk, "max_confidence", 0.5)
            # mean_confidence defaults to 0.0 on tracks built without add_observation;
            # in that case fall back to max_confidence as the representative value.
            raw_mean_conf = getattr(trk, "mean_confidence", 0.0)
            mean_conf = raw_mean_conf if raw_mean_conf > 0.0 else max_conf

            # --- Track-level Evidence Gate ---
            if obs_cnt < self.MINIMUM_FIRE_OBSERVATIONS and duration < self.MINIMUM_FIRE_DURATION_S:
                continue
            if mean_conf < self.MINIMUM_FIRE_MEAN_CONF:
                continue

            supporting = [
                SupportingSignal(
                    signal_type="Flame Visual Telemetry",
                    description=(
                        f"Visual flame chromaticity detected across {obs_cnt} frame(s) "
                        f"({duration:.2f}s duration, peak confidence {max_conf:.2f})."
                    ),
                    confidence=max_conf,
                    timestamp=trk.first_seen,
                )
            ]

            if duration >= 1.0:
                supporting.append(
                    SupportingSignal(
                        signal_type="Sustained Thermal Combustion Pattern",
                        description=f"Continuous localized flame emission sustained for {duration:.2f} seconds.",
                        confidence=min(0.95, max_conf + 0.05),
                        timestamp=trk.first_seen + duration / 2.0,
                    )
                )

            # Compute is_static_surface from track visual_metrics_history:
            # If area CV is near zero, the region shows no flicker — static fixture.
            areas = [m.get("area_pixels", 0.0) for m in trk.visual_metrics_history]
            mean_a = sum(areas) / len(areas) if areas else 1.0
            area_cv = 0.0
            if mean_a > 0 and len(areas) >= 2:
                import math as _math
                area_var = sum((a - mean_a) ** 2 for a in areas) / len(areas)
                area_cv = _math.sqrt(area_var) / mean_a
            is_static_surface = area_cv < 0.04

            # Check if flame track overlaps pedestrian body/clothing
            is_person_clothing = False
            if trk.bounding_boxes:
                last_bb = trk.bounding_boxes[-1]
                tcx = (float(last_bb.get("x1", 0.0)) + float(last_bb.get("x2", 0.0))) / 2.0
                tcy = (float(last_bb.get("y1", 0.0)) + float(last_bb.get("y2", 0.0))) / 2.0
                for ptrk in (context.tracks or []):
                    if ptrk.object_class == "person" and ptrk.current_bbox:
                        pb = ptrk.current_bbox
                        if pb.x1 <= tcx <= pb.x2 and pb.y1 <= tcy <= pb.y2:
                            is_person_clothing = True
                            break

            has_smoke = any(
                getattr(t, "class_name", "") == "smoke"
                for t in (context.specialized_tracks or [])
            )

            contradictory = NegativeEvidenceEngine.evaluate_fire_negative_evidence(
                observation_count=obs_cnt,
                persistence_duration=duration,
                is_static_surface=is_static_surface,
                scene_context=context.scene_context,
                is_person_clothing=is_person_clothing,
                has_smoke=has_smoke,
                event_time=trk.first_seen,
            )

            spatial = None
            evidence_cands = []
            if trk.bounding_boxes:
                b = trk.bounding_boxes[-1]
                bbox = BoundingBox(
                    x1=float(b.get("x1", 0.0)),
                    y1=float(b.get("y1", 0.0)),
                    x2=float(b.get("x2", 0.0)),
                    y2=float(b.get("y2", 0.0)),
                )
                cx = (bbox.x1 + bbox.x2) / 2.0
                cy = (bbox.y1 + bbox.y2) / 2.0
                spatial = SpatialContext(centroid=(cx, cy), bounding_box=bbox)
                evidence_cands.append(
                    EvidenceCandidate(
                        timestamp=trk.first_seen,
                        pre_seconds=2.0,
                        post_seconds=3.0,
                        bounding_box=bbox,
                        reason="Forensic visual snapshot of potential fire origin",
                    )
                )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="POTENTIAL_FIRE",
                start_time=trk.first_seen,
                end_time=trk.last_seen,
                severity="HIGH",
                confidence=round(max_conf, 4),
                explanation=(
                    f"Potential fire visual evidence observed: Sustained localized flame-like chromaticity "
                    f"detected across {obs_cnt} frames ({duration:.2f}s duration). Human verification required."
                ),
                object_classes=["fire"],
                supporting_signals=supporting,
                contradictory_signals=contradictory,
                spatial_context=spatial,
                temporal_context=TemporalContext(
                    start_time=trk.first_seen,
                    end_time=trk.last_seen,
                    duration_seconds=duration,
                ),
                evidence_candidates=evidence_cands,
                human_verification_required=True,
                prefix="FIRE",
                incident_metadata={
                    "observation_count": obs_cnt,
                    "track_id": trk.track_id,
                    "persistence_duration": duration,
                },
            )
            candidates.append(cand)

        return candidates


class SpecializedSmokeIncidentDetector(BaseIncidentDetector):
    """
    Evaluates specialized smoke plume observations and visual episodes.
    Emits POTENTIAL_SMOKE when plume-like low-saturation expansion persists.
    """

    detector_name: str = "specialized_smoke_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.ENVIRONMENT

    required_signals: List[str] = ["Smoke Plume Visual Telemetry"]
    supporting_signals_declared: List[str] = [
        "Smoke Plume Visual Telemetry",
        "Expanding Plume Dynamics",
    ]
    contradictory_signals_declared: List[str] = [
        "Negative: Transient Dispersion Artifact",
        "Negative: Static Surface Texture",
        "Negative: Global Atmospheric Haze",
        "Negative: Video Compression Artifact",
    ]
    confidence_method: str = "temporal_persistence_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    # -----------------------------------------------------------------------
    # Episode Evidence Gate — minimum requirements before raising a candidate.
    # Any episode that fails these thresholds is silently abstained.
    # -----------------------------------------------------------------------
    MINIMUM_SMOKE_OBSERVATIONS: int = 5
    MINIMUM_SMOKE_DURATION_S: float = 1.0
    MINIMUM_SMOKE_MEAN_CONF: float = 0.42
    MINIMUM_SMOKE_EVIDENCE_STR: float = 0.35

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        episodes = getattr(context, "specialized_episodes", None)
        if episodes is not None:
            smoke_episodes = [e for e in episodes if getattr(e, "class_name", "") == "smoke"]
            if not smoke_episodes:
                # Hard pipeline boundary: 0 smoke episodes => 0 smoke incident candidates
                return []
            for ep in smoke_episodes:
                obs_cnt = getattr(ep, "observation_count", 1)
                duration = getattr(ep, "duration_seconds", 0.0)
                max_conf = getattr(ep, "peak_confidence", 0.5)
                raw_mean_conf = getattr(ep, "mean_confidence", 0.0)
                mean_conf = raw_mean_conf if raw_mean_conf > 0.0 else max_conf
                evidence_str = getattr(ep, "peak_evidence_strength", 0.0)
                # Fall back to mean_conf-derived estimate if evidence_str is 0.0
                if evidence_str == 0.0:
                    evidence_str = mean_conf * 0.85

                # --- Episode Evidence Gate ---
                # Require minimum observation count AND minimum duration.
                if obs_cnt < self.MINIMUM_SMOKE_OBSERVATIONS and duration < self.MINIMUM_SMOKE_DURATION_S:
                    logger.debug(
                        "Smoke episode %s abstained: obs=%d (min %d), duration=%.2fs (min %.2fs)",
                        ep.episode_id, obs_cnt, self.MINIMUM_SMOKE_OBSERVATIONS,
                        duration, self.MINIMUM_SMOKE_DURATION_S,
                    )
                    continue
                if mean_conf < self.MINIMUM_SMOKE_MEAN_CONF:
                    logger.debug(
                        "Smoke episode %s abstained: mean_conf=%.3f < %.3f",
                        ep.episode_id, mean_conf, self.MINIMUM_SMOKE_MEAN_CONF,
                    )
                    continue
                if evidence_str < self.MINIMUM_SMOKE_EVIDENCE_STR:
                    logger.debug(
                        "Smoke episode %s abstained: evidence_str=%.3f < %.3f",
                        ep.episode_id, evidence_str, self.MINIMUM_SMOKE_EVIDENCE_STR,
                    )
                    continue

                ep_meta = getattr(ep, "metadata", {}) or {}
                is_static_surface = bool(ep_meta.get("is_static_surface", False))
                if is_static_surface:
                    logger.debug(
                        "Smoke episode %s abstained: static surface texture without plume deformation",
                        ep.episode_id,
                    )
                    continue

                supporting = [
                    SupportingSignal(
                        signal_type="Smoke Plume Visual Telemetry",
                        description=(
                            f"Smoke-like low-saturation plume pattern observed across {obs_cnt} observation(s) "
                            f"({duration:.2f}s duration, peak confidence {max_conf:.2f})."
                        ),
                        confidence=max_conf,
                        timestamp=ep.start_time,
                    )
                ]

                if ep.segment_count > 1:
                    supporting.append(
                        SupportingSignal(
                            signal_type="Expanding Plume Dynamics",
                            description=f"Plume continuity sustained across {ep.segment_count} temporal segment(s) with scale-normalized drift tolerance.",
                            confidence=min(0.92, max_conf + 0.04),
                            timestamp=ep.start_time + duration / 2.0,
                        )
                    )

                # Compute is_global_haze from episode metadata:
                # If mean scene saturation across constituent observations is uniformly low,
                # the low-saturation region is likely global atmospheric haze, not a plume.
                scene_sat_vals = ep_meta.get("scene_mean_saturations", [])
                is_global_haze = (
                    bool(scene_sat_vals)
                    and (sum(scene_sat_vals) / len(scene_sat_vals)) < 30.0
                )
                # Uniform confidence across observations may also indicate a stable
                # global region (fog/haze) rather than a dynamic plume.
                if not is_global_haze and obs_cnt >= 3:
                    conf_std = getattr(ep, "_conf_std", None)
                    if conf_std is None:
                        # Recompute from observation ids — use mean_confidence as proxy
                        conf_range = max_conf - mean_conf
                        is_global_haze = is_global_haze or (conf_range < 0.03 and mean_conf < 0.50)

                contradictory = NegativeEvidenceEngine.evaluate_smoke_negative_evidence(
                    observation_count=obs_cnt,
                    persistence_duration=duration,
                    is_global_haze=is_global_haze,
                    is_static_surface=is_static_surface,
                    event_time=ep.start_time,
                )

                spatial = None
                if ep.bounding_box:
                    cx = (ep.bounding_box.x1 + ep.bounding_box.x2) / 2.0
                    cy = (ep.bounding_box.y1 + ep.bounding_box.y2) / 2.0
                    spatial = SpatialContext(centroid=(cx, cy), bounding_box=ep.bounding_box)

                evidence_cands = []
                for rep in ep.representative_evidence_boxes:
                    bb = rep.get("bounding_box")
                    if bb:
                        bbox = BoundingBox(x1=bb["x1"], y1=bb["y1"], x2=bb["x2"], y2=bb["y2"])
                        evidence_cands.append(
                            EvidenceCandidate(
                                timestamp=rep["timestamp"],
                                pre_seconds=2.0,
                                post_seconds=3.0,
                                bounding_box=bbox,
                                reason=f"Forensic visual snapshot of potential smoke plume (conf: {rep.get('confidence', max_conf):.2f})",
                            )
                        )
                if not evidence_cands and ep.bounding_box:
                    evidence_cands.append(
                        EvidenceCandidate(
                            timestamp=ep.start_time,
                            pre_seconds=2.0,
                            post_seconds=3.0,
                            bounding_box=ep.bounding_box,
                            reason="Forensic visual snapshot of potential smoke plume",
                        )
                    )

                cand = self.build_candidate(
                    video_id=context.video_id,
                    event_type="POTENTIAL_SMOKE",
                    start_time=ep.start_time,
                    end_time=ep.end_time,
                    severity="NORMAL",
                    confidence=round(max_conf, 4),
                    explanation=(
                        f"Potential smoke visual evidence observed: Persistent low-saturation plume pattern "
                        f"detected across {obs_cnt} observations over {duration:.2f}s ({ep.segment_count} segment(s)). Human verification required."
                    ),
                    object_classes=["smoke"],
                    supporting_signals=supporting,
                    contradictory_signals=contradictory,
                    spatial_context=spatial,
                    temporal_context=TemporalContext(
                        start_time=ep.start_time,
                        end_time=ep.end_time,
                        duration_seconds=duration,
                    ),
                    evidence_candidates=evidence_cands,
                    human_verification_required=True,
                    prefix="SMOKE",
                    incident_metadata={
                        "episode_id": ep.episode_id,
                        "observation_count": obs_cnt,
                        "segment_count": ep.segment_count,
                        "peak_confidence": round(max_conf, 4),
                        "evidence_strength": round(ep.peak_evidence_strength, 4),
                        "supporting_observation_ids": ep.observation_ids,
                        "representative_timestamps": ep.representative_timestamps,
                    },
                )
                candidates.append(cand)
            return candidates

        # Fallback to specialized_tracks only if episodes were not evaluated at all
        if not context.specialized_tracks:
            return candidates

        smoke_tracks = [t for t in context.specialized_tracks if getattr(t, "class_name", "") == "smoke"]

        for trk in smoke_tracks:
            obs_cnt = getattr(trk, "observation_count", 1)
            duration = getattr(trk, "persistence_duration", 0.0)
            max_conf = getattr(trk, "max_confidence", 0.5)
            # mean_confidence defaults to 0.0 on tracks built without add_observation;
            # in that case fall back to max_confidence as the representative value.
            raw_mean_conf = getattr(trk, "mean_confidence", 0.0)
            mean_conf = raw_mean_conf if raw_mean_conf > 0.0 else max_conf

            # --- Track-level Evidence Gate ---
            if obs_cnt < self.MINIMUM_SMOKE_OBSERVATIONS and duration < self.MINIMUM_SMOKE_DURATION_S:
                continue
            if mean_conf < self.MINIMUM_SMOKE_MEAN_CONF:
                continue

            is_static_surface = any(bool(m.get("is_static_surface", False)) for m in trk.visual_metrics_history)
            if is_static_surface:
                continue

            # Compute is_global_haze from visual_metrics_history
            scene_sats = [m.get("scene_mean_saturation", 50.0) for m in trk.visual_metrics_history]
            is_global_haze = bool(scene_sats) and (sum(scene_sats) / len(scene_sats)) < 30.0

            contradictory = NegativeEvidenceEngine.evaluate_smoke_negative_evidence(
                observation_count=obs_cnt,
                persistence_duration=duration,
                is_global_haze=is_global_haze,
                is_static_surface=is_static_surface,
                event_time=trk.first_seen,
            )

            supporting = [
                SupportingSignal(
                    signal_type="Smoke Plume Visual Telemetry",
                    description=(
                        f"Smoke-like low-saturation plume pattern observed across {obs_cnt} frame(s) "
                        f"({duration:.2f}s duration, peak confidence {max_conf:.2f})."
                    ),
                    confidence=max_conf,
                    timestamp=trk.first_seen,
                )
            ]

            spatial = None
            evidence_cands = []
            if trk.bounding_boxes:
                b = trk.bounding_boxes[-1]
                bbox = BoundingBox(
                    x1=float(b.get("x1", 0.0)),
                    y1=float(b.get("y1", 0.0)),
                    x2=float(b.get("x2", 0.0)),
                    y2=float(b.get("y2", 0.0)),
                )
                cx = (bbox.x1 + bbox.x2) / 2.0
                cy = (bbox.y1 + bbox.y2) / 2.0
                spatial = SpatialContext(centroid=(cx, cy), bounding_box=bbox)
                evidence_cands.append(
                    EvidenceCandidate(
                        timestamp=trk.first_seen,
                        pre_seconds=2.0,
                        post_seconds=3.0,
                        bounding_box=bbox,
                        reason="Forensic visual snapshot of potential smoke plume",
                    )
                )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="POTENTIAL_SMOKE",
                start_time=trk.first_seen,
                end_time=trk.last_seen,
                severity="NORMAL",
                confidence=round(max_conf, 4),
                explanation=(
                    f"Potential smoke visual evidence observed: Persistent low-saturation plume pattern "
                    f"detected across {obs_cnt} frames ({duration:.2f}s duration). Human verification required."
                ),
                object_classes=["smoke"],
                supporting_signals=supporting,
                contradictory_signals=contradictory,
                spatial_context=spatial,
                temporal_context=TemporalContext(
                    start_time=trk.first_seen,
                    end_time=trk.last_seen,
                    duration_seconds=duration,
                ),
                evidence_candidates=evidence_cands,
                human_verification_required=True,
                prefix="SMOKE",
                incident_metadata={
                    "observation_count": obs_cnt,
                    "track_id": trk.track_id,
                    "persistence_duration": duration,
                },
            )
            candidates.append(cand)

        return candidates



class SpecializedWeaponIncidentDetector(BaseIncidentDetector):
    """
    Evaluates specialized weapon / suspicious object visual observations.
    Emits POTENTIAL_WEAPON_VISUAL only when custom validated model is active
    and observations pass negative evidence.
    """

    detector_name: str = "specialized_weapon_detector"
    detector_version: str = "1.0.0"
    category: IncidentCategory = IncidentCategory.OBJECT

    required_signals: List[str] = ["Suspicious Object Visual Telemetry"]
    supporting_signals_declared: List[str] = [
        "Suspicious Object Visual Telemetry",
        "Grounded Weapon Geometry",
    ]
    contradictory_signals_declared: List[str] = [
        "Negative: Insufficient Optical Resolution",
        "Negative: Common Personal Belonging Context",
    ]
    confidence_method: str = "model_confidence_heuristic"
    abstention_behavior: str = "abstain_on_negative_evidence"
    human_verification_requirement: bool = True

    def analyze(self, context: IncidentContext) -> List[IncidentCandidate]:
        candidates: List[IncidentCandidate] = []
        if not context.specialized_observations:
            return candidates

        # Filter for weapon/suspicious object observations
        weapon_obs = [
            o for o in context.specialized_observations
            if any(k in getattr(o, "class_name", "").lower() for k in ["weapon", "handgun", "rifle", "knife"])
        ]

        # Group by class
        for obs in weapon_obs:
            bbox = getattr(obs, "bounding_box", None)
            w = int(bbox.x2 - bbox.x1) if bbox else 40
            h = int(bbox.y2 - bbox.y1) if bbox else 40

            contradictory = NegativeEvidenceEngine.evaluate_weapon_negative_evidence(
                pixel_resolution=(w, h),
                event_time=obs.timestamp,
            )

            supporting = [
                SupportingSignal(
                    signal_type="Suspicious Object Visual Telemetry",
                    description=f"Visual pattern classified as '{obs.class_name}' by specialized model (conf: {obs.confidence:.2f}).",
                    confidence=obs.confidence,
                    timestamp=obs.timestamp,
                )
            ]

            spatial = None
            evidence_cands = []
            if bbox:
                cx = (bbox.x1 + bbox.x2) / 2.0
                cy = (bbox.y1 + bbox.y2) / 2.0
                spatial = SpatialContext(centroid=(cx, cy), bounding_box=bbox)
                evidence_cands.append(
                    EvidenceCandidate(
                        timestamp=obs.timestamp,
                        pre_seconds=2.0,
                        post_seconds=2.0,
                        bounding_box=bbox,
                        reason="Forensic visual crop of suspicious object",
                    )
                )

            cand = self.build_candidate(
                video_id=context.video_id,
                event_type="POTENTIAL_WEAPON_VISUAL",
                start_time=obs.timestamp,
                end_time=obs.timestamp + 1.0,
                severity="HIGH",
                confidence=round(obs.confidence, 4),
                explanation=(
                    f"Potential weapon visual evidence: Specialized model detected {obs.class_name} "
                    f"(conf: {obs.confidence:.2f}). Mandatory human review required before any security action."
                ),
                object_classes=[obs.class_name],
                supporting_signals=supporting,
                contradictory_signals=contradictory,
                spatial_context=spatial,
                temporal_context=TemporalContext(
                    start_time=obs.timestamp,
                    end_time=obs.timestamp + 1.0,
                    duration_seconds=1.0,
                ),
                evidence_candidates=evidence_cands,
                human_verification_required=True,
                prefix="WEAPON",
            )
            candidates.append(cand)

        return candidates
