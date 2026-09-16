"""
Temporal Object State Machine (Phase 13)

Models the temporal life-cycle and observational states of physical objects and portable belongings.
Provides a deterministic state transition pipeline:
DETECTED -> STATIONARY -> ASSOCIATED_WITH_PERSON -> MOVING -> DISPLACED -> LEFT_BEHIND -> REMOVED / LOST_TRACK
"""
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import List, Dict, Any, Optional, Tuple

from ai.schemas import BoundingBox, TrackedObject
from ai.incidents.schemas import IncidentContext


class ObjectTemporalState(str, Enum):
    DETECTED = "DETECTED"
    STATIONARY = "STATIONARY"
    ASSOCIATED_WITH_PERSON = "ASSOCIATED_WITH_PERSON"
    MOVING = "MOVING"
    DISPLACED = "DISPLACED"
    LEFT_BEHIND = "LEFT_BEHIND"
    REMOVED = "REMOVED"
    LOST_TRACK = "LOST_TRACK"


@dataclass
class ObjectStateTransition:
    from_state: ObjectTemporalState
    to_state: ObjectTemporalState
    timestamp: float
    trigger_track_id: Optional[str] = None
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": round(self.timestamp, 2),
            "trigger_track_id": self.trigger_track_id,
            "details": self.details,
        }


@dataclass
class ObjectStateRecord:
    track_id: str
    object_class: str
    video_id: str
    current_state: ObjectTemporalState
    initial_centroid: Optional[Tuple[float, float]]
    latest_centroid: Optional[Tuple[float, float]]
    net_displacement: float
    stationary_duration: float
    associated_person_ids: List[str] = field(default_factory=list)
    transitions: List[ObjectStateTransition] = field(default_factory=list)
    is_edge_clipped: bool = False
    is_occluded: bool = False
    track_quality: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "object_class": self.object_class,
            "video_id": self.video_id,
            "current_state": self.current_state.value,
            "initial_centroid": [round(c, 2) for c in self.initial_centroid] if self.initial_centroid else None,
            "latest_centroid": [round(c, 2) for c in self.latest_centroid] if self.latest_centroid else None,
            "net_displacement": round(self.net_displacement, 2),
            "stationary_duration": round(self.stationary_duration, 2),
            "associated_person_ids": self.associated_person_ids,
            "transitions": [t.to_dict() for t in self.transitions],
            "is_edge_clipped": self.is_edge_clipped,
            "is_occluded": self.is_occluded,
            "track_quality": round(self.track_quality, 2),
        }


class ObjectStateMachine:
    """
    Evaluates an object track over time against contextual person tracks and frame boundaries
    to derive a deterministic temporal state history.
    """

    def __init__(
        self,
        stationary_vel_threshold: float = 12.0,
        displacement_threshold: float = 45.0,
        proximity_association_distance: float = 120.0,
        edge_margin_px: float = 25.0,
    ):
        self.stationary_vel_threshold = stationary_vel_threshold
        self.displacement_threshold = displacement_threshold
        self.proximity_association_distance = proximity_association_distance
        self.edge_margin_px = edge_margin_px

    def is_near_boundary(self, bbox: Optional[BoundingBox], frame_w: Optional[float] = None, frame_h: Optional[float] = None) -> bool:
        if not bbox:
            return False
        return bbox.is_edge_clipped(frame_width=frame_w, frame_height=frame_h, margin=self.edge_margin_px)

    def analyze_track_state(
        self,
        obj_track: TrackedObject,
        person_tracks: List[TrackedObject],
        context: Optional[IncidentContext] = None,
        frame_w: Optional[float] = None,
        frame_h: Optional[float] = None,
    ) -> ObjectStateRecord:
        """
        Processes an object's trajectory and spatial-temporal interactions into an ObjectStateRecord.
        """
        if context and context.video_metadata:
            meta_w = context.video_metadata.get("width")
            meta_h = context.video_metadata.get("height")
            if meta_w and meta_w > 0:
                frame_w = float(meta_w)
            if meta_h and meta_h > 0:
                frame_h = float(meta_h)
        video_id = context.video_id if context else "unknown"
        if not obj_track.trajectory:
            return ObjectStateRecord(
                track_id=obj_track.track_id,
                object_class=obj_track.object_class,
                video_id=video_id,
                current_state=ObjectTemporalState.DETECTED,
                initial_centroid=None,
                latest_centroid=None,
                net_displacement=0.0,
                stationary_duration=0.0,
            )

        initial_centroid = (obj_track.trajectory[0][1], obj_track.trajectory[0][2])
        latest_centroid = (obj_track.trajectory[-1][1], obj_track.trajectory[-1][2])
        net_disp = math.hypot(latest_centroid[0] - initial_centroid[0], latest_centroid[1] - initial_centroid[1])

        transitions: List[ObjectStateTransition] = []
        current_state = ObjectTemporalState.DETECTED
        transitions.append(ObjectStateTransition(
            from_state=ObjectTemporalState.DETECTED,
            to_state=ObjectTemporalState.DETECTED,
            timestamp=obj_track.first_seen,
            details=f"Object {obj_track.object_class} initially detected",
        ))

        # Check motion summary
        summary = (context.get_motion_summary(obj_track.track_id) or {}) if context else {}
        avg_vel = summary.get("avg_velocity", 0.0)
        is_stationary = (avg_vel < self.stationary_vel_threshold and net_disp < self.displacement_threshold)
        stationary_duration = obj_track.duration_seconds if is_stationary else 0.0

        if is_stationary:
            prev = current_state
            current_state = ObjectTemporalState.STATIONARY
            transitions.append(ObjectStateTransition(
                from_state=prev,
                to_state=ObjectTemporalState.STATIONARY,
                timestamp=obj_track.first_seen,
                details=f"Object remained localized (avg vel: {avg_vel:.1f}px/s, disp: {net_disp:.1f}px)",
            ))

        # Evaluate Person Associations over time
        associated_persons = set()
        person_proximity_intervals: Dict[str, List[Tuple[float, float]]] = {}

        for p in person_tracks:
            if not p.trajectory:
                continue
            prox_times = []
            for p_pt in p.trajectory:
                t, px, py = p_pt[0], p_pt[1], p_pt[2]
                # Find closest obj point in time
                closest_obj_pt = min(obj_track.trajectory, key=lambda opt: abs(opt[0] - t))
                if abs(closest_obj_pt[0] - t) <= 1.0:
                    d = math.hypot(px - closest_obj_pt[1], py - closest_obj_pt[2])
                    if d <= self.proximity_association_distance:
                        prox_times.append(t)
            if len(prox_times) >= 2:
                associated_persons.add(p.track_id)
                person_proximity_intervals[p.track_id] = [(min(prox_times), max(prox_times))]

        if associated_persons:
            prev = current_state
            current_state = ObjectTemporalState.ASSOCIATED_WITH_PERSON
            transitions.append(ObjectStateTransition(
                from_state=prev,
                to_state=ObjectTemporalState.ASSOCIATED_WITH_PERSON,
                timestamp=obj_track.first_seen,
                trigger_track_id=list(associated_persons)[0],
                details=f"Associated with person(s): {', '.join(sorted(associated_persons))}",
            ))

        # Check if displaced or moving
        if net_disp >= self.displacement_threshold:
            prev = current_state
            if avg_vel > self.stationary_vel_threshold:
                current_state = ObjectTemporalState.MOVING
                transitions.append(ObjectStateTransition(
                    from_state=prev,
                    to_state=ObjectTemporalState.MOVING,
                    timestamp=obj_track.last_seen,
                    details=f"Object in motion (disp: {net_disp:.1f}px, vel: {avg_vel:.1f}px/s)",
                ))
            else:
                current_state = ObjectTemporalState.DISPLACED
                transitions.append(ObjectStateTransition(
                    from_state=prev,
                    to_state=ObjectTemporalState.DISPLACED,
                    timestamp=obj_track.last_seen,
                    details=f"Object displaced to new location (disp: {net_disp:.1f}px)",
                ))

        # Check if left behind (associated person departed, object remains stationary)
        if associated_persons and is_stationary:
            # Check if any associated person departed
            for pid in associated_persons:
                p_track = next((p for p in person_tracks if p.track_id == pid), None)
                if p_track and p_track.trajectory:
                    p_end = p_track.trajectory[-1]
                    d_depart = math.hypot(p_end[1] - latest_centroid[0], p_end[2] - latest_centroid[1])
                    if d_depart >= self.proximity_association_distance:
                        prev = current_state
                        current_state = ObjectTemporalState.LEFT_BEHIND
                        transitions.append(ObjectStateTransition(
                            from_state=prev,
                            to_state=ObjectTemporalState.LEFT_BEHIND,
                            timestamp=p_end[0],
                            trigger_track_id=pid,
                            details=f"Associated person [{pid}] departed ({d_depart:.1f}px); object remained stationary",
                        ))
                        break

        # Check edge clipping
        last_bbox = obj_track.current_bbox
        is_edge = self.is_near_boundary(last_bbox, frame_w, frame_h)

        # Track quality score
        det_count = len(obj_track.history_bboxes) if obj_track.history_bboxes else len(obj_track.trajectory)
        expected_dets = max(1.0, obj_track.duration_seconds * 2.0)
        track_quality = min(1.0, det_count / expected_dets)

        return ObjectStateRecord(
            track_id=obj_track.track_id,
            object_class=obj_track.object_class,
            video_id=video_id,
            current_state=current_state,
            initial_centroid=initial_centroid,
            latest_centroid=latest_centroid,
            net_displacement=net_disp,
            stationary_duration=stationary_duration,
            associated_person_ids=sorted(list(associated_persons)),
            transitions=transitions,
            is_edge_clipped=is_edge,
            is_occluded=False,
            track_quality=track_quality,
        )
