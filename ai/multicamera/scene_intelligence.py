"""
ai/multicamera/scene_intelligence.py — Phase 18

MultiCameraSceneIntelligence: higher-level analysis across a surveillance session.

Provides:
  - build_movement_narrative: chronological cross-camera movement sequence
  - detect_scene_anomalies: patterns only visible across multiple cameras
  - summarize_session: cross-camera summary dict for API responses

All outputs are observational and hypothesis-framed. No identity assertions.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ai.multicamera.schemas import CrossCameraHypothesis, AssociationType, AnalystVerdict

logger = logging.getLogger(__name__)


@dataclass
class MovementSegment:
    """One appearance of an anonymous subject cluster in one camera."""
    camera_id: str
    camera_label: str
    video_id: str
    track_id: str
    first_seen: float
    last_seen: float
    duration_seconds: float
    object_class: str
    confidence: Optional[float] = None     # from TrackModel.max_confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "camera_label": self.camera_label,
            "video_id": self.video_id,
            "track_id": self.track_id,
            "first_seen": round(self.first_seen, 4),
            "last_seen": round(self.last_seen, 4),
            "duration_seconds": round(self.duration_seconds, 4),
            "object_class": self.object_class,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
        }


@dataclass
class CrossCameraMovementNarrative:
    """
    Cross-camera chronological movement trace for one anonymous subject cluster.

    A 'subject cluster' is a group of tracks linked by confirmed/probable
    cross-camera associations. No identity is inferred.
    """
    cluster_key: str                            # e.g. "cluster_0" — anonymous
    object_class: str
    segments: List[MovementSegment] = field(default_factory=list)
    association_ids: List[str] = field(default_factory=list)  # hypothesis IDs
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    note: str = "Anonymous cross-camera movement trace — hypothesis only"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_key": self.cluster_key,
            "object_class": self.object_class,
            "segments": [s.to_dict() for s in self.segments],
            "association_ids": self.association_ids,
            "min_confidence": round(self.min_confidence, 4),
            "max_confidence": round(self.max_confidence, 4),
            "cameras_visited": list({s.camera_label for s in self.segments}),
            "total_span_seconds": (
                max(s.last_seen for s in self.segments)
                - min(s.first_seen for s in self.segments)
            ) if self.segments else 0.0,
            "note": self.note,
        }


@dataclass
class SceneAnomaly:
    """
    An anomaly only detectable by cross-camera reasoning.
    Example: track exits Camera A toward Camera B but never appears in Camera B.
    """
    anomaly_type: str
    description: str
    severity: str                   # LOW, MEDIUM, HIGH
    involved_cameras: List[str] = field(default_factory=list)
    involved_tracks: List[str] = field(default_factory=list)
    confidence: float = 0.50
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type,
            "description": self.description,
            "severity": self.severity,
            "involved_cameras": self.involved_cameras,
            "involved_tracks": self.involved_tracks,
            "confidence": round(self.confidence, 4),
            "metadata": self.metadata,
        }


class MultiCameraSceneIntelligence:
    """
    Synthesises higher-level scene intelligence from cross-camera hypotheses.

    All analysis is hypothesis-framed, privacy-safe, and observational only.
    """

    def build_movement_narrative(
        self,
        hypotheses: List[CrossCameraHypothesis],
        camera_observations: Dict[str, Any],  # {camera_id: List[CameraObservation]}
    ) -> List[CrossCameraMovementNarrative]:
        """
        Build anonymous cross-camera movement sequences by clustering
        linked tracks via confirmed/probable hypotheses.

        Uses Union-Find to cluster (camera_id, track_id) nodes linked by
        SAME_OBJECT or PROBABLE_SAME hypotheses (excluding REJECTED).
        """
        # Collect all (camera_id, track_id) nodes
        all_nodes = set()
        for cam_id, obs_list in camera_observations.items():
            for obs in obs_list:
                all_nodes.add((cam_id, obs.track_id))

        # Union-Find
        parent: Dict[tuple, tuple] = {n: n for n in all_nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for hyp in hypotheses:
            if hyp.analyst_verdict == AnalystVerdict.REJECTED:
                continue
            if hyp.association_type not in (AssociationType.SAME_OBJECT, AssociationType.PROBABLE_SAME):
                continue
            src_node = (hyp.source_camera_id, hyp.source_track_id)
            tgt_node = (hyp.target_camera_id, hyp.target_track_id)
            if src_node in parent and tgt_node in parent:
                union(src_node, tgt_node)

        # Group nodes by cluster root
        clusters: Dict[tuple, List[tuple]] = {}
        for node in all_nodes:
            root = find(node)
            clusters.setdefault(root, []).append(node)

        # Build observations lookup: (camera_id, track_id) → CameraObservation
        obs_lookup: Dict[tuple, Any] = {}
        for cam_id, obs_list in camera_observations.items():
            for obs in obs_list:
                obs_lookup[(cam_id, obs.track_id)] = obs

        # Build hypotheses lookup by (src_cam, src_track)
        hyp_by_src: Dict[tuple, List[CrossCameraHypothesis]] = {}
        for hyp in hypotheses:
            k = (hyp.source_camera_id, hyp.source_track_id)
            hyp_by_src.setdefault(k, []).append(hyp)

        narratives: List[CrossCameraMovementNarrative] = []
        for cluster_idx, (root, nodes) in enumerate(clusters.items()):
            if len(nodes) < 2:
                continue  # single-camera track, not a cross-camera narrative

            # Collect observations for all cluster nodes
            segments: List[MovementSegment] = []
            for node in nodes:
                obs = obs_lookup.get(node)
                if obs is None:
                    continue
                segments.append(MovementSegment(
                    camera_id=obs.camera_id,
                    camera_label=obs.camera_label,
                    video_id=obs.video_id,
                    track_id=obs.track_id,
                    first_seen=obs.first_seen,
                    last_seen=obs.last_seen,
                    duration_seconds=obs.duration_seconds,
                    object_class=obs.object_class,
                    confidence=obs.max_confidence,
                ))

            # Sort by first_seen
            segments.sort(key=lambda s: s.first_seen)

            # Collect related hypothesis IDs
            related_hyp_ids = [
                hyp.hypothesis_id
                for hyp in hypotheses
                if (hyp.source_camera_id, hyp.source_track_id) in nodes
                or (hyp.target_camera_id, hyp.target_track_id) in nodes
            ]
            related_hyps = [h for h in hypotheses if h.hypothesis_id in related_hyp_ids]
            confidences = [h.confidence for h in related_hyps]

            object_class = segments[0].object_class if segments else "unknown"
            narrative = CrossCameraMovementNarrative(
                cluster_key=f"cluster_{cluster_idx:03d}",
                object_class=object_class,
                segments=segments,
                association_ids=related_hyp_ids,
                min_confidence=min(confidences) if confidences else 0.0,
                max_confidence=max(confidences) if confidences else 0.0,
            )
            narratives.append(narrative)

        return narratives

    def detect_scene_anomalies(
        self,
        hypotheses: List[CrossCameraHypothesis],
        camera_observations: Dict[str, Any],
        adjacency_map: Dict[str, List[str]],  # {camera_label: [adjacent_labels]}
    ) -> List[SceneAnomaly]:
        """
        Detect anomalies only visible with cross-camera reasoning.

        Current detectors:
          - MISSING_CONTINUATION: track exits toward adjacent camera but
            never appears in it (and no lower-confidence hypothesis exists)
        """
        anomalies: List[SceneAnomaly] = []

        # Build confirmed hypothesis targets: {(tgt_cam_id, tgt_track_id)} set
        matched_targets = set()
        for hyp in hypotheses:
            if hyp.analyst_verdict != AnalystVerdict.REJECTED:
                matched_targets.add((hyp.target_camera_id, hyp.target_track_id))

        # Build camera label → camera_id map
        label_to_id: Dict[str, str] = {}
        id_to_label: Dict[str, str] = {}
        for cam_id, obs_list in camera_observations.items():
            if obs_list:
                lbl = obs_list[0].camera_label
                label_to_id[lbl] = cam_id
                id_to_label[cam_id] = lbl

        # For each track that exited with a known exit direction, check adjacency
        for cam_id, obs_list in camera_observations.items():
            cam_label = id_to_label.get(cam_id, cam_id)
            adjacent_labels = adjacency_map.get(cam_label, [])

            for obs in obs_list:
                if obs.exit_direction_degrees is None:
                    continue
                if not adjacent_labels:
                    continue

                # Check if this track appears in any adjacent camera
                appeared_in_adjacent = any(
                    (label_to_id.get(adj_label, ""), obs.track_id) in matched_targets
                    or any(
                        hyp.source_track_id == obs.track_id and hyp.source_camera_id == cam_id
                        and id_to_label.get(hyp.target_camera_id, "") == adj_label
                        for hyp in hypotheses
                        if hyp.analyst_verdict != AnalystVerdict.REJECTED
                    )
                    for adj_label in adjacent_labels
                )

                if not appeared_in_adjacent:
                    anomalies.append(SceneAnomaly(
                        anomaly_type="MISSING_CONTINUATION",
                        description=(
                            f"Track '{obs.track_id}' ({obs.object_class}) exited "
                            f"camera '{cam_label}' but was not detected in "
                            f"adjacent camera(s): {adjacent_labels}. "
                            f"Possible exit via unmonitored area or missed detection."
                        ),
                        severity="MEDIUM",
                        involved_cameras=[cam_label] + adjacent_labels,
                        involved_tracks=[obs.track_id],
                        confidence=0.60,
                        metadata={
                            "exit_direction_degrees": obs.exit_direction_degrees,
                            "last_seen": obs.last_seen,
                            "adjacent_cameras": adjacent_labels,
                        },
                    ))

        return anomalies

    def summarize_session(
        self,
        session_id: str,
        hypotheses: List[CrossCameraHypothesis],
        camera_observations: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce a compact session-level summary for API responses.
        """
        total_tracks = sum(len(obs_list) for obs_list in camera_observations.values())
        pending = [h for h in hypotheses if h.analyst_verdict == AnalystVerdict.PENDING]
        confirmed = [h for h in hypotheses if h.analyst_verdict == AnalystVerdict.CONFIRMED]
        rejected = [h for h in hypotheses if h.analyst_verdict == AnalystVerdict.REJECTED]

        by_type: Dict[str, int] = {}
        for hyp in hypotheses:
            by_type[hyp.association_type.value] = by_type.get(hyp.association_type.value, 0) + 1

        avg_conf = (
            sum(h.confidence for h in hypotheses) / len(hypotheses)
            if hypotheses else 0.0
        )

        return {
            "session_id": session_id,
            "camera_count": len(camera_observations),
            "total_tracks_analyzed": total_tracks,
            "total_associations": len(hypotheses),
            "associations_by_type": by_type,
            "pending_review": len(pending),
            "confirmed": len(confirmed),
            "rejected": len(rejected),
            "average_confidence": round(avg_conf, 4),
            "privacy_note": (
                "All associations are anonymous observable-attribute hypotheses. "
                "No identity claims are made. Analyst review required."
            ),
        }
