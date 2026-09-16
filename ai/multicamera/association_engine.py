"""
ai/multicamera/association_engine.py — Phase 18

CrossCameraAssociationEngine: matches tracks across camera pairs within a session.

Algorithm:
  For each ordered (source_camera, target_camera) pair:
    For each (source_track, target_track) of the same object class:
      1. Check adjacency hint between cameras
      2. AttributeMatcher → attribute_match_score + attribute evidence
      3. TemporalCompatibilityReasoner → temporal_plausibility_score + temporal evidence
      4. combined_confidence = w_attr * attribute_score + w_temporal * temporal_score
      5. If combined_confidence < 0.40 → abstain (correct abstention)
      6. Else → create CrossCameraHypothesis with appropriate AssociationType

De-duplication: only the highest-confidence hypothesis is kept for each
(source_track, target_track) pair.

PRIVACY: No biometric signals. Class-match gate enforced in AttributeMatcher.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Tuple, Optional

from ai.multicamera.schemas import (
    CameraObservation,
    CrossCameraHypothesis,
    AssociationEvidence,
    AssociationType,
    AnalystVerdict,
)
from ai.multicamera.attribute_matcher import AttributeMatcher
from ai.multicamera.temporal_reasoner import TemporalCompatibilityReasoner

logger = logging.getLogger(__name__)

# Scoring weights
_W_ATTRIBUTE: float = 0.55
_W_TEMPORAL: float = 0.45

# Confidence threshold below which no association is created
_ABSTENTION_THRESHOLD: float = 0.40


class CrossCameraAssociationEngine:
    """
    Generates anonymous cross-camera track association hypotheses for a session.

    Accepts a list of (CameraObservation) records for each camera pair and
    produces CrossCameraHypothesis objects above the abstention threshold.
    """

    def __init__(
        self,
        attribute_matcher: Optional[AttributeMatcher] = None,
        temporal_reasoner: Optional[TemporalCompatibilityReasoner] = None,
        abstention_threshold: float = _ABSTENTION_THRESHOLD,
        weight_attribute: float = _W_ATTRIBUTE,
        weight_temporal: float = _W_TEMPORAL,
    ):
        self.matcher = attribute_matcher or AttributeMatcher()
        self.reasoner = temporal_reasoner or TemporalCompatibilityReasoner()
        self.abstention_threshold = abstention_threshold
        self.w_attribute = weight_attribute
        self.w_temporal = weight_temporal

    def run(
        self,
        session_id: str,
        camera_observations: Dict[str, List[CameraObservation]],
    ) -> List[CrossCameraHypothesis]:
        """
        Main entrypoint.

        Args:
            session_id: SurveillanceSession ID
            camera_observations: {camera_id: [CameraObservation, ...], ...}

        Returns:
            List of CrossCameraHypothesis with confidence >= abstention_threshold,
            deduplicated and sorted by confidence descending.
        """
        if len(camera_observations) < 2:
            logger.debug("Association engine: need ≥ 2 cameras, got %d — returning empty", len(camera_observations))
            return []

        camera_ids = list(camera_observations.keys())
        # Deduplicated results keyed by (source_camera_id, source_track_id, target_camera_id, target_track_id)
        best: Dict[Tuple[str, str, str, str], CrossCameraHypothesis] = {}

        # Iterate ordered pairs (source, target) — both directions
        for i, src_cam_id in enumerate(camera_ids):
            for j, tgt_cam_id in enumerate(camera_ids):
                if i == j:
                    continue  # same camera, skip

                src_obs_list = camera_observations[src_cam_id]
                tgt_obs_list = camera_observations[tgt_cam_id]
                if not src_obs_list or not tgt_obs_list:
                    continue

                # Determine adjacency between these two cameras
                src_sample = src_obs_list[0]
                tgt_sample = tgt_obs_list[0]
                cameras_adjacent = self._are_adjacent(src_sample, tgt_sample)

                for src_obs in src_obs_list:
                    for tgt_obs in tgt_obs_list:
                        hyp = self._evaluate_pair(
                            session_id=session_id,
                            source=src_obs,
                            target=tgt_obs,
                            cameras_adjacent=cameras_adjacent,
                        )
                        if hyp is None:
                            continue

                        key = (
                            hyp.source_camera_id,
                            hyp.source_track_id,
                            hyp.target_camera_id,
                            hyp.target_track_id,
                        )
                        if key not in best or hyp.confidence > best[key].confidence:
                            best[key] = hyp

        results = sorted(best.values(), key=lambda h: h.confidence, reverse=True)
        logger.info(
            "Association engine: session=%s produced %d hypotheses from %d camera(s)",
            session_id, len(results), len(camera_ids),
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _are_adjacent(
        self,
        src: CameraObservation,
        tgt: CameraObservation,
    ) -> bool:
        """
        Returns True if the target camera's label appears in source's adjacency
        hints (or vice-versa).
        """
        if tgt.camera_label in src.adjacent_camera_labels:
            return True
        if src.camera_label in tgt.adjacent_camera_labels:
            return True
        return False

    def _evaluate_pair(
        self,
        session_id: str,
        source: CameraObservation,
        target: CameraObservation,
        cameras_adjacent: bool,
    ) -> Optional[CrossCameraHypothesis]:
        """
        Evaluate one (source_track, target_track) pair.
        Returns CrossCameraHypothesis if confidence >= threshold, else None.
        """
        # 1. Attribute similarity
        attr_score, attr_evidence = self.matcher.compute(source, target)

        # AttributeMatcher already gates on class mismatch (returns 0.0, [])
        if attr_score == 0.0 and not attr_evidence:
            return None  # different object classes — hard abstention

        # 2. Temporal plausibility
        temp_score, gap_seconds, temp_evidence = self.reasoner.compute(
            source_last_seen=source.last_seen,
            target_first_seen=target.first_seen,
            cameras_are_adjacent=cameras_adjacent,
        )

        # Hard veto: negative gap means target appeared BEFORE source left frame.
        # This is a physical impossibility — no attribute match can override it.
        if gap_seconds is not None and gap_seconds < 0.0:
            return None

        # 3. Combined confidence
        confidence = (
            self.w_attribute * attr_score
            + self.w_temporal * temp_score
        )
        confidence = max(0.0, min(1.0, confidence))

        # 4. Abstain below threshold
        if confidence < self.abstention_threshold:
            return None

        # 5. Determine trajectory compatibility score (from attribute evidence)
        traj_score = next(
            (e.score for e in attr_evidence if e.signal_type == "trajectory_direction_compatibility"),
            0.0,
        )

        # 6. Adjacency bonus evidence
        all_evidence = list(attr_evidence) + [temp_evidence]
        if cameras_adjacent:
            all_evidence.append(AssociationEvidence(
                signal_type="adjacency_hint",
                description=(
                    f"Camera '{source.camera_label}' and '{target.camera_label}' "
                    f"are marked as physically adjacent"
                ),
                score=0.80,
                metadata={
                    "source_label": source.camera_label,
                    "target_label": target.camera_label,
                },
            ))

        # 7. Classify
        hyp = CrossCameraHypothesis(
            session_id=session_id,
            source_camera_id=source.camera_id,
            source_video_id=source.video_id,
            source_track_id=source.track_id,
            source_last_seen=source.last_seen,
            target_camera_id=target.camera_id,
            target_video_id=target.video_id,
            target_track_id=target.track_id,
            target_first_seen=target.first_seen,
            confidence=confidence,
            attribute_match_score=attr_score,
            trajectory_compatibility_score=traj_score,
            temporal_gap_seconds=gap_seconds,
            temporal_plausibility_score=temp_score,
            evidence_basis=all_evidence,
            analyst_review_required=True,
            analyst_verdict=AnalystVerdict.PENDING,
        )
        hyp.association_type = hyp.classify_type()
        return hyp
