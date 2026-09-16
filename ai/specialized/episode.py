"""
Specialized Visual Episode Aggregation Engine (Phase 15.2)

Provides generic multi-signal aggregation for specialized visual phenomena
(smoke, fire, weapons, pose, and future specialized detectors).

Semantic Hierarchy:
  RAW OBSERVATION (SpecializedObservation)
      ↓
  TEMPORAL SEGMENT (SpecializedTemporalTrack)
      ↓
  VISUAL EPISODE (SpecializedVisualEpisode)
      ↓
  INCIDENT (IncidentCandidate -> SecurityEventModel)

Decouples localized frame tracking from continuous physical visual phenomena:
- Supports deformable phenomena (expanding/contracting plumes, centroid drift, thermal cores)
- Provides bounded temporal coasting across detector dropouts without premature termination
- Enforces strict scale-normalized spatial separation so distinct distant plumes remain separate
- Generates deterministic, video-scoped episode identifiers
"""
from dataclasses import dataclass, field
import math
from typing import List, Dict, Any, Optional, Tuple

from ai.schemas import BoundingBox
from ai.specialized.schemas import SpecializedObservation, SpecializedTemporalTrack
from ai.specialized.validator import SpecializedValidationStatus
from ai.common.numeric import ensure_finite, clamp_finite


@dataclass
class SpecializedVisualEpisode:
    """
    Higher-level continuous visual phenomenon grouping one or more temporal segments.
    Provides the authoritative basis for security incident candidate generation.
    """
    episode_id: str
    video_id: str
    class_name: str
    detector_name: str
    start_time: float
    end_time: float
    duration_seconds: float
    observation_count: int
    segment_count: int
    segments: List[SpecializedTemporalTrack] = field(default_factory=list)
    observation_ids: List[str] = field(default_factory=list)
    peak_confidence: float = 0.0
    mean_confidence: float = 0.0
    peak_evidence_strength: float = 0.0
    bounding_box: Optional[BoundingBox] = None
    spatial_envelope: Dict[str, float] = field(default_factory=dict)
    representative_timestamps: List[float] = field(default_factory=list)
    representative_evidence_boxes: List[Dict[str, Any]] = field(default_factory=list)
    validation_status: str = "VALID"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> str:
        return self.class_name

    @property
    def start_timestamp(self) -> float:
        return self.start_time

    @property
    def end_timestamp(self) -> float:
        return self.end_time

    @property
    def supporting_observation_ids(self) -> List[str]:
        return self.observation_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "video_id": self.video_id,
            "class_name": self.class_name,
            "detector_name": self.detector_name,
            "start_time": round(ensure_finite(self.start_time, 0.0), 4),
            "end_time": round(ensure_finite(self.end_time, 0.0), 4),
            "duration_seconds": round(ensure_finite(self.duration_seconds, 0.0), 4),
            "observation_count": self.observation_count,
            "segment_count": self.segment_count,
            "peak_confidence": round(ensure_finite(self.peak_confidence, 0.0), 4),
            "mean_confidence": round(ensure_finite(self.mean_confidence, 0.0), 4),
            "peak_evidence_strength": round(ensure_finite(self.peak_evidence_strength, 0.0), 4),
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "spatial_envelope": self.spatial_envelope,
            "representative_timestamps": [round(t, 2) for t in self.representative_timestamps],
            "representative_evidence_boxes": self.representative_evidence_boxes,
            "validation_status": self.validation_status,
            "metadata": self.metadata,
        }


class SpecializedEpisodeAssociationPolicy:
    """Base interface for detector-specific episode association rules."""

    def is_compatible(
        self,
        episode: SpecializedVisualEpisode,
        candidate_segment: SpecializedTemporalTrack,
    ) -> bool:
        raise NotImplementedError


class SmokeEpisodePolicy(SpecializedEpisodeAssociationPolicy):
    """
    Association policy tailored for deformable smoke plumes:
    - Allows bounded temporal coasting across short detector drops (up to 4.5s)
    - Tolerates scale-normalized centroid drift and plume expansion
    - Rejects distant spatial plumes (e.g. left side vs right side of frame)
    """

    def __init__(self, max_temporal_gap: float = 4.5, max_scale_dist_ratio: float = 1.4):
        self.max_temporal_gap = max_temporal_gap
        self.max_scale_dist_ratio = max_scale_dist_ratio

    def is_compatible(
        self,
        episode: SpecializedVisualEpisode,
        candidate_segment: SpecializedTemporalTrack,
    ) -> bool:
        if episode.class_name != candidate_segment.class_name:
            return False

        # 1. Temporal gap evaluation
        # Candidate segment must start after or near episode start
        time_gap = candidate_segment.first_seen - episode.end_time
        if time_gap < -0.5:
            # Overlapping or preceding in time
            overlap = not (candidate_segment.last_seen < episode.start_time - 0.5)
            if not overlap:
                return False
        elif time_gap > self.max_temporal_gap:
            return False

        # 2. Spatial continuity evaluation
        ep_env = episode.spatial_envelope
        cand_boxes = candidate_segment.bounding_boxes
        if not ep_env or not cand_boxes:
            # Fallback to temporal continuity if boxes missing
            return True

        cand_box = cand_boxes[0]
        cx_c = (cand_box.get("x1", 0.0) + cand_box.get("x2", 0.0)) / 2.0
        cy_c = (cand_box.get("y1", 0.0) + cand_box.get("y2", 0.0)) / 2.0
        cand_w = max(1.0, cand_box.get("x2", 0.0) - cand_box.get("x1", 0.0))
        cand_h = max(1.0, cand_box.get("y2", 0.0) - cand_box.get("y1", 0.0))
        cand_diag = math.hypot(cand_w, cand_h)

        ep_cx = (ep_env.get("x1", 0.0) + ep_env.get("x2", 0.0)) / 2.0
        ep_cy = (ep_env.get("y1", 0.0) + ep_env.get("y2", 0.0)) / 2.0
        ep_w = max(1.0, ep_env.get("x2", 0.0) - ep_env.get("x1", 0.0))
        ep_h = max(1.0, ep_env.get("y2", 0.0) - ep_env.get("y1", 0.0))
        ep_diag = math.hypot(ep_w, ep_h)

        dist = math.hypot(cx_c - ep_cx, cy_c - ep_cy)

        # Scale reference: maximum characteristic dimension of either plume or baseline 60px
        scale_ref = max(ep_diag, cand_diag, 60.0)
        norm_dist = dist / scale_ref

        # Check direct bounding box overlap or expansion
        ix1 = max(ep_env.get("x1", 0.0), cand_box.get("x1", 0.0))
        iy1 = max(ep_env.get("y1", 0.0), cand_box.get("y1", 0.0))
        ix2 = min(ep_env.get("x2", 0.0), cand_box.get("x2", 0.0))
        iy2 = min(ep_env.get("y2", 0.0), cand_box.get("y2", 0.0))
        has_overlap = (ix2 > ix1) and (iy2 > iy1)

        if has_overlap:
            return True

        # Distance threshold: plume expansion and drift allowed within scale ratio
        if norm_dist <= self.max_scale_dist_ratio:
            return True

        # Margin expansion check: plume edge proximity within 40% of plume dimension
        margin_x = max(20.0, ep_w * 0.40)
        margin_y = max(20.0, ep_h * 0.40)
        near_boundary = (
            cand_box.get("x2", 0.0) >= ep_env.get("x1", 0.0) - margin_x and
            cand_box.get("x1", 0.0) <= ep_env.get("x2", 0.0) + margin_x and
            cand_box.get("y2", 0.0) >= ep_env.get("y1", 0.0) - margin_y and
            cand_box.get("y1", 0.0) <= ep_env.get("y2", 0.0) + margin_y
        )
        return near_boundary


class FireEpisodePolicy(SpecializedEpisodeAssociationPolicy):
    """
    Association policy tailored for localized fire / flame evidence:
    - Bounded temporal gap (up to 3.5s)
    - Thermal core proximity (norm_dist <= 1.0)
    """

    def __init__(self, max_temporal_gap: float = 3.5, max_scale_dist_ratio: float = 1.0):
        self.max_temporal_gap = max_temporal_gap
        self.max_scale_dist_ratio = max_scale_dist_ratio

    def is_compatible(
        self,
        episode: SpecializedVisualEpisode,
        candidate_segment: SpecializedTemporalTrack,
    ) -> bool:
        if episode.class_name != candidate_segment.class_name:
            return False

        time_gap = candidate_segment.first_seen - episode.end_time
        if time_gap < -0.5:
            overlap = not (candidate_segment.last_seen < episode.start_time - 0.5)
            if not overlap:
                return False
        elif time_gap > self.max_temporal_gap:
            return False

        ep_env = episode.spatial_envelope
        cand_boxes = candidate_segment.bounding_boxes
        if not ep_env or not cand_boxes:
            return True

        cand_box = cand_boxes[0]
        cx_c = (cand_box.get("x1", 0.0) + cand_box.get("x2", 0.0)) / 2.0
        cy_c = (cand_box.get("y1", 0.0) + cand_box.get("y2", 0.0)) / 2.0
        ep_cx = (ep_env.get("x1", 0.0) + ep_env.get("x2", 0.0)) / 2.0
        ep_cy = (ep_env.get("y1", 0.0) + ep_env.get("y2", 0.0)) / 2.0

        dist = math.hypot(cx_c - ep_cx, cy_c - ep_cy)
        cand_diag = math.hypot(
            cand_box.get("x2", 0.0) - cand_box.get("x1", 0.0),
            cand_box.get("y2", 0.0) - cand_box.get("y1", 0.0)
        )
        ep_diag = math.hypot(
            ep_env.get("x2", 0.0) - ep_env.get("x1", 0.0),
            ep_env.get("y2", 0.0) - ep_env.get("y1", 0.0)
        )
        scale_ref = max(ep_diag, cand_diag, 50.0)
        return (dist / scale_ref) <= self.max_scale_dist_ratio


class DefaultEpisodePolicy(SpecializedEpisodeAssociationPolicy):
    """Fallback policy for rigid/localized objects (weapons, posture anomalies)."""

    def is_compatible(
        self,
        episode: SpecializedVisualEpisode,
        candidate_segment: SpecializedTemporalTrack,
    ) -> bool:
        if episode.class_name != candidate_segment.class_name:
            return False
        time_gap = candidate_segment.first_seen - episode.end_time
        return 0.0 <= time_gap <= 2.0


class SpecializedVisualEpisodeAggregator:
    """
    Generic aggregator organizing frame-level specialized observations and
    temporal segments into cohesive visual episodes.
    """

    def __init__(self, policies: Optional[Dict[str, SpecializedEpisodeAssociationPolicy]] = None):
        self.policies: Dict[str, SpecializedEpisodeAssociationPolicy] = policies or {
            "smoke": SmokeEpisodePolicy(),
            "plume": SmokeEpisodePolicy(),
            "fire": FireEpisodePolicy(),
            "flame": FireEpisodePolicy(),
        }
        self.default_policy = DefaultEpisodePolicy()

    def get_policy(self, class_name: str) -> SpecializedEpisodeAssociationPolicy:
        cls_lower = (class_name or "").lower()
        for key, pol in self.policies.items():
            if key in cls_lower:
                return pol
        return self.default_policy

    def aggregate(
        self,
        video_id: str,
        observations: List[SpecializedObservation],
        temporal_tracks: Optional[List[SpecializedTemporalTrack]] = None,
    ) -> List[SpecializedVisualEpisode]:
        """
        Aggregate observations and tracks into visual episodes for a given video.
        Assigns deterministic episode IDs and tags constituent observations.
        """
        if observations:
            observations = [
                o for o in observations
                if not (
                    getattr(o, "validation_status", None) == SpecializedValidationStatus.REJECTED
                    or "REJECTED" in str(getattr(o, "validation_status", ""))
                )
            ]
        if not observations and not temporal_tracks:
            return []

        # If temporal_tracks not provided, cluster observations into tracks first
        if not temporal_tracks and observations:
            import uuid
            from ai.specialized.temporal import SpecializedTemporalTracker
            tracker = SpecializedTemporalTracker()
            by_ts: Dict[float, List[SpecializedObservation]] = {}
            for obs in sorted(observations, key=lambda o: getattr(o, "timestamp", 0.0)):
                ts = getattr(obs, "timestamp", 0.0)
                by_ts.setdefault(ts, []).append(obs)
            for ts in sorted(by_ts.keys()):
                tracker.update(by_ts[ts])
                tracker.prune_stale(ts)
            temporal_tracks = tracker.finalize()
            if not temporal_tracks:
                temporal_tracks = []
                for obs in observations:
                    c_name = getattr(obs, "class_name", "UNKNOWN")
                    trk_id = f"SPEC-{c_name[:4].upper()}-{uuid.uuid4().hex[:6]}"
                    trk = SpecializedTemporalTrack(
                        track_id=trk_id,
                        class_name=c_name,
                        detector_name=getattr(obs, "detector_name", "SpecializedVisualDetector"),
                        first_seen=getattr(obs, "timestamp", 0.0),
                        last_seen=getattr(obs, "timestamp", 0.0),
                        observation_count=1,
                        max_confidence=getattr(obs, "confidence", 0.0),
                        mean_confidence=getattr(obs, "confidence", 0.0),
                        confidence_history=[getattr(obs, "confidence", 0.0)],
                        bounding_boxes=[obs.bounding_box.to_dict()] if getattr(obs, "bounding_box", None) else [],
                        visual_metrics_history=[obs.visual_metrics] if getattr(obs, "visual_metrics", None) else [],
                    )
                    temporal_tracks.append(trk)

        # Ensure video isolation
        v_id = video_id or "default"

        # 1. Sort segments chronologically
        sorted_tracks = sorted(temporal_tracks, key=lambda t: t.first_seen)

        # 2. Cluster segments into episodes using detector-specific policies
        episodes_by_class: Dict[str, List[SpecializedVisualEpisode]] = {}
        episode_counter = 1

        for trk in sorted_tracks:
            cls_name = trk.class_name
            policy = self.get_policy(cls_name)

            if cls_name not in episodes_by_class:
                episodes_by_class[cls_name] = []

            # Check if track attaches to any active episode of the same class
            matched_episode = None
            for ep in episodes_by_class[cls_name]:
                if policy.is_compatible(ep, trk):
                    matched_episode = ep
                    break

            if matched_episode is not None:
                # Merge into existing episode
                self._append_segment_to_episode(matched_episode, trk)
            else:
                # Create a new visual episode
                prefix = cls_name[:4].upper()
                v_short = v_id.replace("-", "")[:8]
                ep_id = f"EP-{prefix}-{v_short}-{episode_counter:03d}"
                episode_counter += 1

                new_ep = self._create_episode_from_segment(ep_id, v_id, trk)
                episodes_by_class[cls_name].append(new_ep)

        # Flatten all created episodes
        all_episodes: List[SpecializedVisualEpisode] = []
        for eps in episodes_by_class.values():
            all_episodes.extend(eps)

        # 3. Associate raw observations with their matching episode
        self._associate_observations_with_episodes(all_episodes, observations)

        # 4. Finalize episode statistics, envelope, and representative timestamps
        for ep in all_episodes:
            self._finalize_episode(ep, observations)

        # Sort chronologically by start_time
        all_episodes.sort(key=lambda ep: ep.start_time)
        return all_episodes

    def _create_episode_from_segment(
        self,
        episode_id: str,
        video_id: str,
        trk: SpecializedTemporalTrack,
    ) -> SpecializedVisualEpisode:
        envelope = self._compute_initial_envelope(trk.bounding_boxes)
        ep = SpecializedVisualEpisode(
            episode_id=episode_id,
            video_id=video_id,
            class_name=trk.class_name,
            detector_name=trk.detector_name,
            start_time=trk.first_seen,
            end_time=trk.last_seen,
            duration_seconds=trk.persistence_duration,
            observation_count=trk.observation_count,
            segment_count=1,
            segments=[trk],
            observation_ids=[],
            peak_confidence=trk.max_confidence,
            mean_confidence=trk.mean_confidence,
            peak_evidence_strength=trk.max_confidence * 0.95,
            spatial_envelope=envelope,
            representative_timestamps=[trk.first_seen, trk.last_seen],
        )
        setattr(trk, "episode_id", episode_id)
        return ep

    def _append_segment_to_episode(
        self,
        episode: SpecializedVisualEpisode,
        trk: SpecializedTemporalTrack,
    ) -> None:
        episode.segments.append(trk)
        episode.segment_count += 1
        episode.start_time = min(episode.start_time, trk.first_seen)
        episode.end_time = max(episode.end_time, trk.last_seen)
        episode.duration_seconds = max(0.0, episode.end_time - episode.start_time)
        episode.observation_count += trk.observation_count
        episode.peak_confidence = max(episode.peak_confidence, trk.max_confidence)

        # Expand spatial envelope
        if trk.bounding_boxes:
            cand_env = self._compute_initial_envelope(trk.bounding_boxes)
            if cand_env:
                if not episode.spatial_envelope:
                    episode.spatial_envelope = cand_env
                else:
                    episode.spatial_envelope = {
                        "x1": min(episode.spatial_envelope["x1"], cand_env["x1"]),
                        "y1": min(episode.spatial_envelope["y1"], cand_env["y1"]),
                        "x2": max(episode.spatial_envelope["x2"], cand_env["x2"]),
                        "y2": max(episode.spatial_envelope["y2"], cand_env["y2"]),
                    }

        setattr(trk, "episode_id", episode.episode_id)

    def _associate_observations_with_episodes(
        self,
        episodes: List[SpecializedVisualEpisode],
        observations: List[SpecializedObservation],
    ) -> None:
        """Link every raw observation to its corresponding episode."""
        for obs in observations:
            obs_cls = (obs.class_name or "").lower()
            matching_ep = None
            best_spatial_score = -1.0

            for ep in episodes:
                if ep.class_name.lower() != obs_cls:
                    continue
                # Time tolerance: within episode interval or slight tolerance
                if (ep.start_time - 1.0) <= obs.timestamp <= (ep.end_time + 1.0):
                    # Compute spatial affinity
                    score = self._compute_spatial_affinity(ep.spatial_envelope, obs.bounding_box)
                    if score > best_spatial_score:
                        best_spatial_score = score
                        matching_ep = ep

            if matching_ep is not None:
                setattr(obs, "episode_id", matching_ep.episode_id)
                if obs.observation_id not in matching_ep.observation_ids:
                    matching_ep.observation_ids.append(obs.observation_id)
                matching_ep.peak_confidence = max(matching_ep.peak_confidence, obs.confidence)
                matching_ep.peak_evidence_strength = max(
                    matching_ep.peak_evidence_strength, obs.evidence_strength
                )

    def _finalize_episode(
        self,
        episode: SpecializedVisualEpisode,
        all_observations: List[SpecializedObservation],
    ) -> None:
        """Compute final metrics, envelope bounding box, and representative evidence."""
        # Find constituent observations
        constituent_obs = [
            o for o in all_observations
            if getattr(o, "episode_id", None) == episode.episode_id
        ]
        if constituent_obs:
            episode.observation_count = len(constituent_obs)
            episode.start_time = min(o.timestamp for o in constituent_obs)
            episode.end_time = max(o.timestamp for o in constituent_obs)
            episode.duration_seconds = max(0.0, round(episode.end_time - episode.start_time, 2))
            confs = [o.confidence for o in constituent_obs]
            episode.mean_confidence = sum(confs) / len(confs)
            episode.peak_confidence = max(confs)
            episode.peak_evidence_strength = max(o.evidence_strength for o in constituent_obs)

            # Representative timestamps: onset, peak confidence, midpoint, latest
            sorted_obs = sorted(constituent_obs, key=lambda o: o.timestamp)
            onset_ts = sorted_obs[0].timestamp
            latest_ts = sorted_obs[-1].timestamp
            peak_obs = max(sorted_obs, key=lambda o: o.confidence)
            mid_obs = sorted_obs[len(sorted_obs) // 2]

            rep_times = sorted(list({
                round(onset_ts, 2),
                round(peak_obs.timestamp, 2),
                round(mid_obs.timestamp, 2),
                round(latest_ts, 2),
            }))
            episode.representative_timestamps = rep_times

            # Representative evidence bounding boxes
            rep_boxes = []
            for t in rep_times:
                matching = next((o for o in sorted_obs if abs(o.timestamp - t) <= 0.1 and o.bounding_box), None)
                if matching and matching.bounding_box:
                    rep_boxes.append({
                        "timestamp": round(matching.timestamp, 2),
                        "bounding_box": matching.bounding_box.to_dict(),
                        "confidence": round(matching.confidence, 4),
                    })
            episode.representative_evidence_boxes = rep_boxes

            # Evaluate volumetric plume deformation & centroid movement
            areas = []
            for o in constituent_obs:
                m = getattr(o, "visual_metrics", {}) or {}
                if "area_pixels" in m:
                    areas.append(float(m["area_pixels"]))
                elif o.bounding_box:
                    areas.append(float(o.bounding_box.area))
            if len(areas) >= 2:
                mean_a = sum(areas) / len(areas)
                if mean_a > 0.0:
                    area_var = sum((a - mean_a) ** 2 for a in areas) / len(areas)
                    area_cv = math.sqrt(area_var) / mean_a
                    episode.metadata["mean_area_cv"] = round(area_cv, 4)

            if len(sorted_obs) >= 2 and sorted_obs[0].bounding_box and sorted_obs[-1].bounding_box:
                c_first = sorted_obs[0].bounding_box.centroid
                c_last = sorted_obs[-1].bounding_box.centroid
                c_shift = math.hypot(c_last[0] - c_first[0], c_last[1] - c_first[1])
                episode.metadata["centroid_displacement"] = round(c_shift, 2)
                has_obs_static = any(getattr(o, "visual_metrics", {}).get("is_static_surface", False) for o in constituent_obs)
                if has_obs_static or (c_shift < 3.0 and episode.metadata.get("mean_area_cv", 1.0) < 0.01 and len(constituent_obs) >= 15):
                    episode.metadata["is_static_surface"] = True

        # Spatial envelope to BoundingBox
        if episode.spatial_envelope:
            try:
                episode.bounding_box = BoundingBox(
                    x1=round(episode.spatial_envelope["x1"], 2),
                    y1=round(episode.spatial_envelope["y1"], 2),
                    x2=round(episode.spatial_envelope["x2"], 2),
                    y2=round(episode.spatial_envelope["y2"], 2),
                )
            except Exception:
                pass

    @staticmethod
    def _compute_initial_envelope(boxes: List[Dict[str, Any]]) -> Dict[str, float]:
        if not boxes:
            return {}
        return {
            "x1": min(float(b.get("x1", 0.0)) for b in boxes),
            "y1": min(float(b.get("y1", 0.0)) for b in boxes),
            "x2": max(float(b.get("x2", 0.0)) for b in boxes),
            "y2": max(float(b.get("y2", 0.0)) for b in boxes),
        }

    @staticmethod
    def _compute_spatial_affinity(
        env: Dict[str, float],
        box: Optional[BoundingBox],
    ) -> float:
        if not env or not box:
            return 0.5
        ix1 = max(env.get("x1", 0.0), box.x1)
        iy1 = max(env.get("y1", 0.0), box.y1)
        ix2 = min(env.get("x2", 0.0), box.x2)
        iy2 = min(env.get("y2", 0.0), box.y2)
        if ix2 > ix1 and iy2 > iy1:
            inter = (ix2 - ix1) * (iy2 - iy1)
            box_area = box.area
            return inter / max(1.0, box_area)
        # Distance score fallback
        cx_env = (env.get("x1", 0.0) + env.get("x2", 0.0)) / 2.0
        cy_env = (env.get("y1", 0.0) + env.get("y2", 0.0)) / 2.0
        cx_box, cy_box = box.centroid
        dist = math.hypot(cx_box - cx_env, cy_box - cy_env)
        return max(0.0, 1.0 - (dist / 1000.0))
