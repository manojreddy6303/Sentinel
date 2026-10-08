"""
Investigation Service Module for Sentinel

Handles structured execution of parsed queries against the relational database:
- Translates parsed filters into database operations using SQLAlchemy models
- Supports detections listing, grouped events listing, and exact record counts
- Ensures results are chronologically ordered and grounded exclusively in actual DB records
- Rejects hallucinations and invalid queries
"""

import json
import logging
import math
import re
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import func
from database.session import SessionLocal
from database.models import (
    EventModel,
    GroupedEventModel,
    VideoModel,
    TrackModel,
    VehicleAttributeModel,
    FaceDetectionModel,
    SecurityEventModel,
    SpecializedObservationModel,
    CorrelatedIncidentModel,
    EvidenceModel,
)
from backend.app.services.investigation_parser import InvestigationParser

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle", "bicycle"]


class InvestigationService:
    """Executes natural-language queries against Sentinel's detection and event database."""

    @staticmethod
    def _compute_bbox_iou(box1: Any, box2: Any) -> float:
        """Compute Intersection over Union between two bounding boxes (dict or list/tuple)."""
        if not box1 or not box2:
            return 0.0
        if isinstance(box1, str):
            try:
                box1 = json.loads(box1)
            except Exception:
                return 0.0
        if isinstance(box2, str):
            try:
                box2 = json.loads(box2)
            except Exception:
                return 0.0

        if isinstance(box1, dict):
            b1 = [box1.get("x1", 0.0), box1.get("y1", 0.0), box1.get("x2", 0.0), box1.get("y2", 0.0)]
        elif isinstance(box1, (list, tuple)) and len(box1) >= 4:
            b1 = [float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])]
        else:
            return 0.0

        if isinstance(box2, dict):
            b2 = [box2.get("x1", 0.0), box2.get("y1", 0.0), box2.get("x2", 0.0), box2.get("y2", 0.0)]
        elif isinstance(box2, (list, tuple)) and len(box2) >= 4:
            b2 = [float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])]
        else:
            return 0.0

        inter_x1 = max(b1[0], b2[0])
        inter_y1 = max(b1[1], b2[1])
        inter_x2 = min(b1[2], b2[2])
        inter_y2 = min(b1[3], b2[3])

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        denom = area1 + area2 - inter_area
        return (inter_area / denom) if denom > 0.0 else 0.0

    @staticmethod
    def _get_bbox_centroid(box: Any) -> Optional[Tuple[float, float]]:
        """Get (cx, cy) from bounding box."""
        if not box:
            return None
        if isinstance(box, str):
            try:
                box = json.loads(box)
            except Exception:
                return None
        if isinstance(box, dict):
            return ((box.get("x1", 0.0) + box.get("x2", 0.0)) / 2.0, (box.get("y1", 0.0) + box.get("y2", 0.0)) / 2.0)
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)
        return None

    def _are_tracks_same_entity(self, t1: Dict[str, Any], t2: Dict[str, Any]) -> bool:
        """
        Generic resolution criteria to determine if two track records represent the same physical entity.
        Zero hardcoding: applies uniformly to any video, track IDs, colors, or timestamps.
        """
        # 0. Strict Video Isolation: tracks from different videos can NEVER union
        vid1 = t1.get("video_id")
        vid2 = t2.get("video_id")
        if vid1 and vid2 and vid1 != vid2:
            return False

        # 1. Same object_class (person with person)
        cls1 = str(t1.get("object_class") or "").lower()
        cls2 = str(t2.get("object_class") or "").lower()
        if cls1 != cls2:
            return False

        # 2. Attribute consistency check: conflicting strong attributes should NOT be merged
        # unless spatial overlap strongly proves identity (IoU > 0.60)
        col1 = (t1.get("color") or "").lower()
        col2 = (t2.get("color") or "").lower()
        conf1 = float(t1.get("color_confidence") or 0.0)
        conf2 = float(t2.get("color_confidence") or 0.0)
        if col1 and col2 and col1 != col2 and conf1 >= 0.50 and conf2 >= 0.50:
            dark_set = {"black", "grey", "silver", "blue"}
            # Conflicting distinct chromatic colors (e.g. orange vs blue, or red vs green) block merging
            if not (col1 in dark_set and col2 in dark_set):
                bbox_iou = self._compute_bbox_iou(t1.get("current_bbox"), t2.get("current_bbox"))
                if bbox_iou < 0.60:
                    return False

        start1, end1 = float(t1.get("first_seen") or 0.0), float(t1.get("last_seen") or 0.0)
        end2, start2 = float(t2.get("last_seen") or 0.0), float(t2.get("first_seen") or 0.0)
        dur1 = max(float(t1.get("duration_seconds") or (end1 - start1)), 0.01)
        dur2 = max(float(t2.get("duration_seconds") or (end2 - start2)), 0.01)

        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_dur = overlap_end - overlap_start
        min_dur = min(dur1, dur2)

        traj1 = t1.get("trajectory") or []
        if isinstance(traj1, str):
            try:
                traj1 = json.loads(traj1)
            except Exception:
                traj1 = []
        traj2 = t2.get("trajectory") or []
        if isinstance(traj2, str):
            try:
                traj2 = json.loads(traj2)
            except Exception:
                traj2 = []

        bb1 = t1.get("current_bbox") or {}
        if isinstance(bb1, str):
            try:
                bb1 = json.loads(bb1)
            except Exception:
                bb1 = {}
        bb2 = t2.get("current_bbox") or {}
        if isinstance(bb2, str):
            try:
                bb2 = json.loads(bb2)
            except Exception:
                bb2 = {}

        # 3. For overlapping frames:
        if overlap_dur > 0:
            # Contemporaneous trajectories: check spatial distance during matching timestamps
            dists = []
            for p1 in traj1:
                t_ts1 = p1[0] if isinstance(p1, (list, tuple)) and len(p1) > 0 else None
                if t_ts1 is None:
                    continue
                for p2 in traj2:
                    t_ts2 = p2[0] if isinstance(p2, (list, tuple)) and len(p2) > 0 else None
                    if t_ts2 is not None and abs(t_ts1 - t_ts2) <= 0.35:
                        dists.append(math.hypot(p1[1] - p2[1], p1[2] - p2[2]))

            if dists:
                avg_dist = sum(dists) / len(dists)
                min_dist = min(dists)
                max_dist = max(dists)
                # If contemporaneous distance exceeds 38px at any time, they are distinct physical actors
                if max_dist > 38.0 or avg_dist > 32.0:
                    return False
                if avg_dist < 32.0 and min_dist < 30.0:
                    return True

            # Fallback to current_bbox if no overlapping trajectory timestamps
            iou = self._compute_bbox_iou(bb1, bb2)
            c1 = self._get_bbox_centroid(bb1)
            c2 = self._get_bbox_centroid(bb2)
            c_dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1]) if c1 and c2 else 999.0

            if iou >= 0.30 and c_dist <= 35.0:
                return True
            return False

        # 4. For non-overlapping frames:
        gap = start2 - end1 if start2 >= end1 else (start1 - end2 if start1 >= end2 else -1.0)
        if gap >= 0.0:
            earlier_traj = traj1 if end1 <= start2 else traj2
            later_traj = traj2 if end1 <= start2 else traj1

            exit_pt = earlier_traj[-1] if earlier_traj and isinstance(earlier_traj[-1], (list, tuple)) and len(earlier_traj[-1]) >= 3 else None
            entry_pt = later_traj[0] if later_traj and isinstance(later_traj[0], (list, tuple)) and len(later_traj[0]) >= 3 else None
            if exit_pt and entry_pt:
                boundary_dist = math.hypot(exit_pt[1] - entry_pt[1], exit_pt[2] - entry_pt[2])
            else:
                c1 = self._get_bbox_centroid(bb1)
                c2 = self._get_bbox_centroid(bb2)
                boundary_dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1]) if c1 and c2 else 999.0

            c1 = self._get_bbox_centroid(bb1)
            c2 = self._get_bbox_centroid(bb2)
            bbox_dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1]) if c1 and c2 else 999.0
            iou = self._compute_bbox_iou(bb1, bb2)
            vel = boundary_dist / max(gap, 0.1)

            # Short gap (<= 2.0s):
            if gap <= 2.0:
                if (boundary_dist <= 25.0 or (boundary_dist < 45.0 and vel < 40.0)) and (bbox_dist <= 40.0 or boundary_dist <= 25.0):
                    return True
                if iou > 0.35:
                    return True

            # Moderate occlusion gap (2.0 < gap <= 16.0s):
            # Scored physical continuation based on boundary proximity and feasible indoor movement speed:
            elif gap <= 16.0:
                # 1. Localized continuation (e.g. lingering/bending behind ATM, safe, counter, door)
                if (boundary_dist <= 25.0 or (iou >= 0.35 and bbox_dist <= 32.0)) and vel < 6.0:
                    return True
                # 2. Moving human continuation with realistic walking speed
                if boundary_dist <= 35.0 and vel <= 8.0:
                    return True
                # 3. Trajectory continuity for walking actor across room
                if boundary_dist <= 50.0 and 1.0 <= vel <= 10.0 and (gap <= 8.0 or boundary_dist <= 30.0):
                    return True

        return False

    def _reconcile_canonical_entities(self, tracks: List[Any], video_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Clusters duplicate or fragmented track records representing the same physical entity
        into unified canonical entities with consolidated attributes and activity summaries.
        Strictly scoped to a single video_id: cross-video tracks are never merged.
        """
        items: List[Dict[str, Any]] = []
        for t in tracks:
            t_vid = getattr(t, "video_id", None) or (t.get("video_id") if isinstance(t, dict) else None)
            if video_id is not None and t_vid is not None and t_vid != video_id:
                continue

            if hasattr(t, "track_id"):
                dur = getattr(t, "duration_seconds", None)
                if dur is None:
                    fs = float(getattr(t, "first_seen", 0.0) or 0.0)
                    ls = float(getattr(t, "last_seen", 0.0) or 0.0)
                    dur = max(0.0, ls - fs)
                dets = getattr(t, "detection_count", None)
                if dets is None:
                    hist = getattr(t, "history_bboxes", [])
                    dets = len(hist) if hist else 1
                max_c = getattr(t, "max_confidence", getattr(t, "confidence", 0.0))
                traj_val = getattr(t, "trajectory", None)
                if isinstance(traj_val, str):
                    try:
                        traj_val = json.loads(traj_val)
                    except Exception:
                        traj_val = []
                c_bbox = getattr(t, "current_bbox", {})
                if isinstance(c_bbox, str):
                    try:
                        c_bbox = json.loads(c_bbox)
                    except Exception:
                        c_bbox = {}
                elif hasattr(c_bbox, "to_dict"):
                    c_bbox = c_bbox.to_dict()
                items.append({
                    "video_id": t_vid or video_id,
                    "track_id": t.track_id,
                    "object_class": t.object_class,
                    "first_seen": float(t.first_seen or 0.0),
                    "last_seen": float(t.last_seen or 0.0),
                    "duration_seconds": float(dur or 0.0),
                    "detection_count": int(dets or 1),
                    "max_confidence": float(max_c or 0.0),
                    "color": getattr(t, "color", None),
                    "color_confidence": float(getattr(t, "color_confidence", 0.0) or 0.0) if getattr(t, "color_confidence", None) else 0.0,
                    "current_bbox": c_bbox,
                    "trajectory": traj_val or [],
                    "active": bool(getattr(t, "active", False)),
                    "security_events": getattr(t, "security_events", []),
                    "activity_summary": getattr(t, "activity_summary", ""),
                })
            elif isinstance(t, dict):
                traj_val = t.get("trajectory")
                if isinstance(traj_val, str):
                    try:
                        traj_val = json.loads(traj_val)
                    except Exception:
                        traj_val = []
                c_bbox = t.get("current_bbox")
                if isinstance(c_bbox, str):
                    try:
                        c_bbox = json.loads(c_bbox)
                    except Exception:
                        c_bbox = {}
                elif hasattr(c_bbox, "to_dict"):
                    c_bbox = c_bbox.to_dict()
                items.append({
                    "video_id": t_vid or video_id,
                    "track_id": t.get("track_id", "UNKNOWN"),
                    "object_class": t.get("object_class", "unknown"),
                    "first_seen": float(t.get("first_seen", 0.0)),
                    "last_seen": float(t.get("last_seen", 0.0)),
                    "duration_seconds": float(t.get("duration_seconds", 0.0)),
                    "detection_count": int(t.get("detection_count", 1)),
                    "max_confidence": float(t.get("max_confidence", 0.0)),
                    "color": t.get("color"),
                    "color_confidence": float(t.get("color_confidence") or 0.0),
                    "current_bbox": c_bbox,
                    "trajectory": traj_val or [],
                    "active": bool(t.get("active", False)),
                    "security_events": t.get("security_events", []),
                    "activity_summary": t.get("activity_summary", ""),
                })

        n = len(items)
        if n == 0:
            return []

        # Generic detection of static false-person artifacts (posters, mannequins, transient background noise)
        def _is_static_false_artifact(t: Dict[str, Any]) -> bool:
            if t.get("object_class") != "person":
                return False
            dur = float(t.get("duration_seconds", 0.0))
            dets = int(t.get("detection_count", 1))
            conf = float(t.get("max_confidence", 0.0))
            traj = t.get("trajectory", [])

            # Isolated transient noise (1-2 detections, short duration, weak confidence)
            if dets <= 2 and dur <= 1.5 and conf < 0.75:
                return True

            # Static background / poster artifact (intermittent low-density detections staying on wall/poster)
            if len(traj) >= 2 and dur <= 8.0:
                traj_dur = float(traj[-1][0] - traj[0][0]) if len(traj) >= 2 else dur
                disp = math.hypot(traj[-1][1] - traj[0][1], traj[-1][2] - traj[0][2])
                spd = disp / max(traj_dur, 0.1)
                # Net displacement < 15px with low speed
                if disp < 15.0 and spd < 1.5 and conf < 0.85:
                    return True
                # Intermittent low-density detection with high ratio of points staying clustered on poster
                if len(traj) >= 4 and dur >= 3.0 and (dets / dur) <= 1.1 and conf < 0.82:
                    p0 = traj[0]
                    ratio_near = sum(1 for p in traj if math.hypot(p[1] - p0[1], p[2] - p0[2]) < 20.0) / len(traj)
                    if ratio_near >= 0.65:
                        return True

            return False

        # Clean leading static latch from tracklets that idled on background features before moving (e.g. TRACK-030)
        for t in items:
            traj = t.get("trajectory", [])
            if len(traj) >= 4 and t.get("duration_seconds", 0.0) >= 5.0:
                p0 = traj[0]
                static_idx = 0
                for k in range(1, len(traj)):
                    if math.hypot(traj[k][1] - p0[1], traj[k][2] - p0[2]) < 3.0:
                        static_idx = k
                    else:
                        break
                # If at least 3 initial points had near-zero drift (< 3.0px) and subsequent points move
                if static_idx >= 3 and static_idx < len(traj) - 1:
                    move_disp = math.hypot(traj[-1][1] - traj[static_idx][1], traj[-1][2] - traj[static_idx][2])
                    if move_disp > 15.0:
                        t["clean_trajectory"] = traj[static_idx:]
                        t["clean_first_seen"] = float(traj[static_idx][0])
                    else:
                        t["clean_trajectory"] = traj
                        t["clean_first_seen"] = t["first_seen"]
            else:
                t["clean_trajectory"] = traj
                t["clean_first_seen"] = t["first_seen"]

        from collections import defaultdict

        # Compute dynamic max_concurrent_verified_people from validated detections per frame
        max_concurrent_people = 1
        if video_id:
            try:
                from database.session import SessionLocal
                from database.models import EventModel
                db_concur = SessionLocal()
                try:
                    ev_rows = (
                        db_concur.query(EventModel)
                        .filter(EventModel.video_id == video_id, EventModel.object_class == "person", EventModel.validation_status == "VALID")
                        .all()
                    )
                    if ev_rows:
                        by_ts = defaultdict(list)
                        for ev in ev_rows:
                            by_ts[round(ev.timestamp_seconds, 1)].append(ev)
                        for ts, ev_list in by_ts.items():
                            distinct_evs = []
                            for ev in ev_list:
                                b1 = (ev.bbox_x1, ev.bbox_y1, ev.bbox_x2, ev.bbox_y2)
                                area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
                                is_dup = False
                                for dev in distinct_evs:
                                    b2 = (dev.bbox_x1, dev.bbox_y1, dev.bbox_x2, dev.bbox_y2)
                                    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
                                    iw = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                                    ih = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                                    inter = iw * ih
                                    min_a = min(area1, area2) if min(area1, area2) > 0 else 1.0
                                    iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0.0
                                    ioa = inter / min_a
                                    c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
                                    c2 = ((b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0)
                                    cdist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                                    if iou >= 0.35 or ioa >= 0.50 or cdist < 18.0:
                                        is_dup = True
                                        break
                                if not is_dup:
                                    distinct_evs.append(ev)
                            if len(distinct_evs) > max_concurrent_people:
                                max_concurrent_people = len(distinct_evs)
                finally:
                    db_concur.close()
            except Exception as c_err:
                logger.debug(f"Concurrency calculation from DB skipped: {c_err}")

        # Also verify concurrency directly across items' trajectories
        by_ts_items = defaultdict(list)
        for it in items:
            if it.get("object_class") == "person" and not _is_static_false_artifact(it):
                for p in it.get("clean_trajectory", it.get("trajectory", [])):
                    if isinstance(p, (list, tuple)) and len(p) >= 3:
                        by_ts_items[round(p[0], 1)].append((p[1], p[2], it.get("track_id")))
        for ts, pts in by_ts_items.items():
            distinct_pts = []
            for pt in pts:
                if not any(math.hypot(pt[0] - d[0], pt[1] - d[1]) < 20.0 for d in distinct_pts):
                    distinct_pts.append(pt)
            if len(distinct_pts) > max_concurrent_people:
                max_concurrent_people = len(distinct_pts)

        # Build hard Cannot-Link constraints
        cannot_link = set()
        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = items[i], items[j]
                vid1, vid2 = t1.get("video_id"), t2.get("video_id")
                if vid1 and vid2 and vid1 != vid2:
                    cannot_link.add((i, j))
                    cannot_link.add((j, i))
                    continue
                # Exclude static false artifacts from linking with valid persons
                if _is_static_false_artifact(t1) or _is_static_false_artifact(t2):
                    cannot_link.add((i, j))
                    cannot_link.add((j, i))
                    continue
                if t1.get("object_class") != "person" or t2.get("object_class") != "person":
                    continue

                s1 = t1.get("clean_first_seen", t1["first_seen"])
                e1 = t1["last_seen"]
                s2 = t2.get("clean_first_seen", t2["first_seen"])
                e2 = t2["last_seen"]
                overlap = min(e1, e2) - max(s1, s2)

                traj1_map = {round(p[0], 1): (p[1], p[2]) for p in t1.get("clean_trajectory", t1.get("trajectory", [])) if isinstance(p, (list, tuple)) and len(p) >= 3}
                traj2_map = {round(p[0], 1): (p[1], p[2]) for p in t2.get("clean_trajectory", t2.get("trajectory", [])) if isinstance(p, (list, tuple)) and len(p) >= 3}
                shared_ts = set(traj1_map.keys()).intersection(traj2_map.keys())

                # Any shared timestamp where positions are spatially distinct (> 20px)
                separated_simultaneous = False
                for ts in shared_ts:
                    p1, p2 = traj1_map[ts], traj2_map[ts]
                    if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) > 20.0:
                        separated_simultaneous = True
                        break
                if separated_simultaneous:
                    cannot_link.add((i, j))
                    cannot_link.add((j, i))
                    continue

                # Contemporaneous temporal overlap > 0.0s
                if overlap > 0.0:
                    t1_pts = t1.get("clean_trajectory", t1.get("trajectory", []))
                    t2_pts = t2.get("clean_trajectory", t2.get("trajectory", []))
                    if t1_pts and t2_pts and isinstance(t1_pts[0], (list, tuple)) and isinstance(t2_pts[0], (list, tuple)):
                        p1 = t1_pts[0]
                        p2 = t2_pts[0]
                        if math.hypot(p1[1] - p2[1], p1[2] - p2[2]) > 25.0:
                            cannot_link.add((i, j))
                            cannot_link.add((j, i))
                            continue

                # Spatial infeasibility / teleportation
                gap = s2 - e1 if s2 >= e1 else (s1 - e2 if s1 >= e2 else -1.0)
                if gap >= 0.0:
                    earlier = t1 if e1 <= s2 else t2
                    later = t2 if e1 <= s2 else t1
                    e_traj = earlier.get("clean_trajectory", earlier.get("trajectory", []))
                    l_traj = later.get("clean_trajectory", later.get("trajectory", []))
                    if e_traj and l_traj and isinstance(e_traj[-1], (list, tuple)) and isinstance(l_traj[0], (list, tuple)):
                        p_ex = e_traj[-1]
                        p_en = l_traj[0]
                        d = math.hypot(p_ex[1] - p_en[1], p_ex[2] - p_en[2])
                        vel = d / max(gap, 1.0)
                        if (gap <= 2.0 and d > 50.0) or (gap <= 5.0 and vel > 25.0) or d > 80.0:
                            cannot_link.add((i, j))
                            cannot_link.add((j, i))
                            continue

        def _eval_continuation(t1: Dict[str, Any], t2: Dict[str, Any]) -> Tuple[float, bool]:
            s1, e1 = t1.get("clean_first_seen", t1["first_seen"]), t1["last_seen"]
            s2, e2 = t2.get("clean_first_seen", t2["first_seen"]), t2["last_seen"]
            if s2 < e1:
                t1, t2 = t2, t1
                s1, e1 = t1.get("clean_first_seen", t1["first_seen"]), t1["last_seen"]
                s2, e2 = t2.get("clean_first_seen", t2["first_seen"]), t2["last_seen"]
            gap = s2 - e1
            t1_pts = t1.get("clean_trajectory", t1.get("trajectory", []))
            t2_pts = t2.get("clean_trajectory", t2.get("trajectory", []))
            if gap < 0:
                if abs(gap) <= 1.0 and t1_pts and t2_pts and isinstance(t1_pts[-1], (list, tuple)) and isinstance(t2_pts[0], (list, tuple)):
                    d = math.hypot(t1_pts[-1][1] - t2_pts[0][1], t1_pts[-1][2] - t2_pts[0][2])
                    if d <= 20.0:
                        return d, True
                return 999.0, False
            if gap > 10.0 or not t1_pts or not t2_pts:
                return 999.0, False
            if not (isinstance(t1_pts[-1], (list, tuple)) and isinstance(t2_pts[0], (list, tuple))):
                return 999.0, False
            p_ex = t1_pts[-1]
            p_en = t2_pts[0]
            d = math.hypot(p_ex[1] - p_en[1], p_ex[2] - p_en[2])
            vel = d / max(gap, 1.0)
            if gap <= 2.5 and d <= 30.0 and vel <= 25.0:
                return d + gap * 1.5, True
            if gap <= 6.0 and d <= 35.0 and vel <= 8.0:
                return d + gap * 1.5, True
            if gap <= 10.0 and d <= 30.0 and vel <= 5.0:
                return d + gap * 1.5, True
            return 999.0, False

        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in cannot_link:
                    continue
                vid_i = items[i].get("video_id")
                vid_j = items[j].get("video_id")
                if vid_i and vid_j and vid_i != vid_j:
                    continue
                sc, ok = _eval_continuation(items[i], items[j])
                if ok:
                    candidates.append((sc, i, j))
                elif self._are_tracks_same_entity(items[i], items[j]):
                    candidates.append((25.0, i, j))
        candidates.sort(key=lambda x: x[0])

        clusters: List[set] = [{i} for i in range(n)]
        for sc, i, j in candidates:
            c_i = next((c for c in clusters if i in c), None)
            c_j = next((c for c in clusters if j in c), None)
            if c_i is not None and c_j is not None and c_i is not c_j:
                # 1. Cross-member cannot-link conflict check
                if any((u, v) in cannot_link for u in c_i for v in c_j):
                    continue
                # 2. Internal conflict check
                union_list = list(c_i.union(c_j))
                has_internal = False
                for u_idx in range(len(union_list)):
                    for v_idx in range(u_idx + 1, len(union_list)):
                        if (union_list[u_idx], union_list[v_idx]) in cannot_link:
                            has_internal = True
                            break
                    if has_internal:
                        break
                if has_internal:
                    continue

                # 3. Concurrency invariant check:
                # Never merge if remaining verified person clusters would drop below max_concurrent_people
                rem_person_clusters = sum(
                    1 for cl in clusters
                    if cl is not c_i and cl is not c_j and any(not _is_static_false_artifact(items[k]) and items[k].get("object_class") == "person" for k in cl)
                ) + 1
                if rem_person_clusters < max_concurrent_people:
                    continue

                clusters.remove(c_i)
                clusters.remove(c_j)
                clusters.append(c_i.union(c_j))

        canonical_entities: List[Dict[str, Any]] = []
        for cluster_members_indices in clusters:
            members = [items[idx] for idx in cluster_members_indices]
            primary = max(members, key=lambda m: (m["duration_seconds"], m["max_confidence"]))

            # Weighted consensus color across all member tracks (conf * detection_count)
            color_weights: Dict[str, float] = {}
            for m in members:
                c_val = m.get("color")
                if c_val and c_val not in ("unknown", "uncertain"):
                    w = float(m.get("color_confidence") or 0.5) * max(1, int(m.get("detection_count") or 1))
                    color_weights[c_val] = color_weights.get(c_val, 0.0) + w

            if color_weights:
                top_c, top_w = max(color_weights.items(), key=lambda kv: kv[1])
                tot_w = sum(color_weights.values())
                canonical_color = top_c
                canonical_color_conf = round(min(0.95, (top_w / tot_w) * 0.90 + 0.05), 4)
            else:
                canonical_color = None
                canonical_color_conf = None

            min_first = min(m["first_seen"] for m in members)
            max_last = max(m["last_seen"] for m in members)
            total_dets = sum(m["detection_count"] for m in members)
            max_conf = max(m["max_confidence"] for m in members)

            latest_track = max(members, key=lambda m: m["last_seen"])

            # Merge security events without duplicate event_type & timestamp
            combined_sec = []
            seen_sec_keys = set()
            for m in members:
                for se in m.get("security_events", []):
                    key = (se.get("event_type"), round(se.get("timestamp", 0.0), 1))
                    if key not in seen_sec_keys:
                        seen_sec_keys.add(key)
                        combined_sec.append(se)

            # Best activity summary
            act_summary = primary.get("activity_summary") or ""
            for m in members:
                if m.get("activity_summary") and ("theft" in m["activity_summary"].lower() or "fall" in m["activity_summary"].lower()):
                    act_summary = m["activity_summary"]
                    break

            canonical_entities.append({
                "video_id": primary.get("video_id") or video_id,
                "track_id": primary["track_id"],
                "canonical_id": primary["track_id"],
                "member_track_ids": sorted(list(set(m["track_id"] for m in members))),
                "object_class": primary["object_class"],
                "first_seen": round(min_first, 2),
                "last_seen": round(max_last, 2),
                "duration_seconds": round(max(max_last - min_first, primary["duration_seconds"]), 2),
                "detection_count": total_dets,
                "max_confidence": round(max_conf, 4),
                "color": canonical_color,
                "color_confidence": canonical_color_conf,
                "current_bbox": latest_track.get("current_bbox"),
                "active": any(m.get("active") for m in members),
                "security_events": combined_sec,
                "activity_summary": act_summary,
            })

        # Quality gating:
        # Distinguish raw detection observations from verified searchable physical entities.
        verified_canonical_entities: List[Dict[str, Any]] = []
        for ce in canonical_entities:
            total_dets = ce.get("detection_count", 1)
            dur = ce.get("duration_seconds", 0.0)
            max_c = ce.get("max_confidence", 0.0)
            bbox = ce.get("current_bbox") or {}
            x1 = bbox.get("x1", 10.0)
            y1 = bbox.get("y1", 10.0)
            x2 = bbox.get("x2", 100.0)
            y2 = bbox.get("y2", 100.0)
            obj_cls = str(ce.get("object_class") or "").lower()

            # Check if isolated 1-frame transient artifact
            if total_dets <= 1 and dur < 0.2:
                is_boundary = (x1 <= 4.0 or y1 <= 4.0 or x2 >= 316.0 or y2 >= 236.0 or x2 >= 238.0)
                if max_c < 0.70 or is_boundary:
                    continue

            # Check if transient boundary noise cluster (few detections touching boundary with weak confidence)
            if total_dets <= 3 and dur <= 10.0 and max_c < 0.45:
                is_boundary = (x1 <= 4.0 or y1 <= 4.0 or x2 >= 316.0 or y2 >= 236.0 or x2 >= 238.0)
                if is_boundary:
                    continue

            # Generic quality gating for static false-person artifacts (posters/mannequins)
            if obj_cls == "person":
                # Find member items for this cluster
                cluster_items = [it for it in items if it.get("track_id") in ce.get("member_track_ids", [])]
                if cluster_items and all(_is_static_false_artifact(it) for it in cluster_items):
                    continue
                if total_dets <= 2 and dur <= 1.5 and max_c < 0.75:
                    continue

            verified_canonical_entities.append(ce)

        verified_canonical_entities.sort(key=lambda ce: (ce["first_seen"], ce["track_id"]))
        return verified_canonical_entities


    def __init__(self):
        self.parser = InvestigationParser()

    def investigate(self, video_id: str, query_text: str) -> Dict[str, Any]:
        """
        Parse and execute an investigation query for a given video.

        Returns structured response with count, interpreted filters, and records.
        """
        if not query_text or not query_text.strip():
            return {
                "query": query_text,
                "is_supported": False,
                "message": "Query cannot be empty. Please enter an investigation question.",
                "interpreted_filters": {},
                "result_type": "error",
                "count": 0,
                "results": [],
            }

        parsed = self.parser.parse_query(query_text)
        if not parsed["is_supported"]:
            return {
                "query": query_text,
                "is_supported": False,
                "message": parsed.get("message", "This investigation query is not currently supported."),
                "interpreted_filters": {},
                "result_type": "unsupported",
                "count": 0,
                "results": [],
            }

        filters = parsed["interpreted_filters"]
        result_type = parsed["result_type"]
        return self.investigate_filters(video_id, filters, result_type, query_text)

    def investigate_filters(
        self,
        video_id: str,
        filters: Dict[str, Any],
        result_type: str = "detections",
        query_text: str = "",
    ) -> Dict[str, Any]:
        """
        Directly execute structured filters against Sentinel database records.
        Preserves deterministic database execution for LLM orchestrator queries.
        """
        db = SessionLocal()
        try:
            # 1. Check if video exists
            video_exists = db.query(VideoModel).filter(VideoModel.id == video_id).first()
            # If not in DB table, verify if events exist with this video_id
            if not video_exists:
                events_count = db.query(EventModel).filter(EventModel.video_id == video_id, EventModel.validation_status == "VALID").count()
                if events_count == 0:
                    return {
                        "query": query_text,
                        "is_supported": True,
                        "message": f"Video with ID '{video_id}' has not been processed or does not exist.",
                        "interpreted_filters": filters,
                        "result_type": result_type,
                        "count": 0,
                        "results": [],
                    }

            # 2. Execute query based on result_type
            if result_type == "events":
                return self._query_grouped_events(db, video_id, query_text, filters)
            elif result_type == "count":
                return self._query_counts(db, video_id, query_text, filters)
            elif result_type == "vehicle_attributes":
                return self._query_vehicle_attributes(db, video_id, query_text, filters)
            elif result_type == "tracks":
                return self._query_tracks(db, video_id, query_text, filters)
            elif result_type == "faces":
                return self._query_faces(db, video_id, query_text, filters)
            elif result_type == "security_events":
                return self._query_security_events(db, video_id, query_text, filters)
            elif result_type == "specialized":
                return self._query_specialized_observations(db, video_id, query_text, filters)
            elif result_type == "correlated_incidents":
                return self._query_correlated_incidents(db, video_id, query_text, filters)
            elif result_type == "evidence":
                return self._query_evidence(db, video_id, query_text, filters)
            else:
                return self._query_detections(db, video_id, query_text, filters)
        except Exception as exc:
            logger.error(f"Investigation query failed for video {video_id}: {exc}")
            raise
        finally:
            db.close()

    def _query_detections(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch matching validated detections from DB."""
        q = db.query(EventModel).filter(EventModel.video_id == video_id, EventModel.validation_status == "VALID")

        obj_class = filters.get("object_class")
        if obj_class == "vehicle_group":
            q = q.filter(EventModel.object_class.in_(VEHICLE_CLASSES))
        elif obj_class:
            q = q.filter(EventModel.object_class.ilike(obj_class))

        if filters.get("start_time") is not None:
            q = q.filter(EventModel.timestamp_seconds >= filters["start_time"])
        if filters.get("end_time") is not None:
            q = q.filter(EventModel.timestamp_seconds <= filters["end_time"])
        if filters.get("min_confidence") is not None:
            q = q.filter(EventModel.confidence >= filters["min_confidence"])
        if filters.get("max_confidence") is not None:
            q = q.filter(EventModel.confidence <= filters["max_confidence"])

        rows = q.order_by(EventModel.timestamp_seconds.asc()).all()

        results = [
            {
                "event_id": r.id,
                "video_id": r.video_id,
                "timestamp": round(r.timestamp_seconds, 2),
                "object_class": r.object_class,
                "confidence": round(r.confidence, 4),
                "frame_number": r.frame_number,
                "bounding_box": {
                    "x1": round(r.bbox_x1, 1),
                    "y1": round(r.bbox_y1, 1),
                    "x2": round(r.bbox_x2, 1),
                    "y2": round(r.bbox_y2, 1),
                },
            }
            for r in rows
        ]

        if results:
            msg = f"Found {len(results)} matching detections."
        elif obj_class and obj_class != "vehicle_group":
            msg = f"No validated {obj_class} detections were found. No matching detections were found."
        else:
            msg = "No matching detections were found."

        return {
            "query": query_text,
            "is_supported": True,
            "result_type": "detections",
            "interpreted_filters": filters,
            "count": len(results),
            "message": msg,
            "results": results,
        }

    def _query_grouped_events(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch matching grouped timeline events from DB."""
        q = db.query(GroupedEventModel).filter(GroupedEventModel.video_id == video_id)

        if filters.get("start_time") is not None:
            q = q.filter(GroupedEventModel.end_time >= filters["start_time"])
        if filters.get("end_time") is not None:
            q = q.filter(GroupedEventModel.start_time <= filters["end_time"])
        if filters.get("min_confidence") is not None:
            q = q.filter(GroupedEventModel.max_confidence >= filters["min_confidence"])

        rows = q.order_by(GroupedEventModel.start_time.asc()).all()

        results = []
        obj_class = filters.get("object_class")
        for r in rows:
            objs = r.objects_summary or []
            if obj_class == "vehicle_group":
                if not any(o.get("class", "").lower() in VEHICLE_CLASSES for o in objs):
                    continue
            elif obj_class:
                if not any(o.get("class", "").lower() == obj_class.lower() for o in objs):
                    continue

            results.append(
                {
                    "event_id": r.id,
                    "video_id": r.video_id,
                    "event_type": r.event_type,
                    "start_time": round(r.start_time, 2),
                    "end_time": round(r.end_time, 2),
                    "duration_seconds": r.duration_seconds,
                    "objects": objs,
                    "total_detections": r.total_detections,
                    "max_confidence": round(r.max_confidence, 4),
                    "priority": r.priority,
                }
            )

        msg = f"Found {len(results)} timeline events." if results else "No matching events were found."
        return {
            "query": query_text,
            "is_supported": True,
            "result_type": "events",
            "interpreted_filters": filters,
            "count": len(results),
            "message": msg,
            "results": results,
        }

    def _query_counts(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate exact detection observations and unique canonical track counts with strict semantic distinction."""
        detection_res = self._query_detections(db, video_id, query_text, filters)
        det_count = detection_res["count"]
        obj_name = filters.get("object_class") or "object"
        is_unique = filters.get("is_unique", False)

        all_video_tracks = (
            db.query(TrackModel)
            .filter(TrackModel.video_id == video_id)
            .order_by(TrackModel.first_seen.asc())
            .all()
        )
        canonical_entities = self._reconcile_canonical_entities(all_video_tracks, video_id=video_id)

        # Filter canonical entities by object class
        if obj_name == "vehicle_group":
            filtered_canonical = [ce for ce in canonical_entities if ce.get("object_class") in VEHICLE_CLASSES]
            raw_track_count = db.query(TrackModel).filter(TrackModel.video_id == video_id, TrackModel.object_class.in_(VEHICLE_CLASSES)).count()
        elif filters.get("object_class"):
            filtered_canonical = [ce for ce in canonical_entities if ce.get("object_class", "").lower() == obj_name.lower()]
            raw_track_count = db.query(TrackModel).filter(TrackModel.video_id == video_id, TrackModel.object_class.ilike(obj_name)).count()
        else:
            filtered_canonical = canonical_entities
            raw_track_count = len(all_video_tracks)

        target_color = filters.get("color")
        if target_color:
            filtered_canonical = [ce for ce in filtered_canonical if (ce.get("color") or "").lower() == target_color.lower()]

        canonical_count = len(filtered_canonical)
        disclaimer = " (Note: Sentinel reports detection observations and does not attribute unique personal identities)."

        count_mode = filters.get("count_mode")
        is_person_target = bool(filters.get("object_class") in ("person", "people", "persons") or obj_name.lower() in ("person", "people", "persons"))
        is_track_query = (count_mode == "tracklets") or bool(re.search(r"\btracks?\b", (query_text or "").lower()))
        is_raw_det_query = (count_mode == "raw_detections") or bool(re.search(r"\b(detection\s+records?|detection\s+observations?|person\s+detections?)\b", (query_text or "").lower()))
        is_ambiguous_detected = (count_mode == "ambiguous_detected") or (is_person_target and bool(re.search(r"\b(were\s+detected|was\s+detected|how\s+many.*detected)\b", (query_text or "").lower())))

        plural_ent = "people" if is_person_target else ("entities" if canonical_count != 1 else "entity")

        if not is_person_target:
            if is_track_query:
                final_count = raw_track_count
                plural_trk = "tracks" if raw_track_count != 1 else "track"
                msg = f"{raw_track_count} {plural_trk} recorded in the video ({raw_track_count} tracks representing {canonical_count} distinct physical {plural_ent}).{disclaimer}"
            else:
                final_count = det_count
                msg = f"Detected {det_count} total {obj_name} detection records ({det_count} validated {obj_name} detection observations across {raw_track_count} tracks).{disclaimer}"
        else:
            plural_ent = "people"
            if is_track_query:
                final_count = raw_track_count
                plural_trk = "tracks" if raw_track_count != 1 else "track"
                msg = f"{raw_track_count} {plural_trk} recorded in the video ({raw_track_count} tracks representing {canonical_count} distinct physical {plural_ent}).{disclaimer}"
            elif is_raw_det_query:
                final_count = det_count
                plural_obs = "observations" if det_count != 1 else "observation"
                msg = f"Detected {det_count} total person detection records ({det_count} validated person detection observations across {raw_track_count} anonymous tracks).{disclaimer}"
            elif is_ambiguous_detected:
                final_count = canonical_count
                msg = f"{canonical_count} distinct physical {plural_ent} were identified from {det_count} validated {obj_name} detection observations across {raw_track_count} anonymous tracks.{disclaimer}"
            elif is_unique:
                final_count = canonical_count
                msg = f"{canonical_count} anonymous person tracks.{disclaimer}"
            else:
                final_count = canonical_count
                if target_color:
                    msg = f"{canonical_count} distinct physical {plural_ent} wearing {target_color} clothing identified across {raw_track_count} tracks ({det_count} validated detection observations).{disclaimer}"
                else:
                    msg = f"{canonical_count} distinct physical {plural_ent} identified across {raw_track_count} tracks ({det_count} validated detection observations).{disclaimer}"

        effective_mode = count_mode or ("tracklets" if is_track_query else ("raw_detections" if is_raw_det_query else ("ambiguous_detected" if is_ambiguous_detected else "canonical_people")))

        return {
            "query": query_text,
            "is_supported": True,
            "result_type": "count",
            "interpreted_filters": filters,
            "count": final_count,
            "canonical_entity_count": canonical_count,
            "canonical_entities": filtered_canonical,
            "detection_observations": det_count,
            "track_count": raw_track_count,
            "count_mode": effective_mode,
            "message": msg,
            "results": detection_res["results"][:10],  # preview top 10
        }

    def _query_vehicle_attributes(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch matching vehicle visual color attribute records from DB."""
        # Color queries are only supported when actual vehicle color-analysis records exist
        total_attrs = db.query(VehicleAttributeModel).filter(VehicleAttributeModel.video_id == video_id).count()
        if total_attrs == 0:
            return {
                "query": query_text,
                "is_supported": False,
                "message": "Vehicle color analysis records do not exist for this video. Color attributes have not been analyzed or are unavailable.",
                "interpreted_filters": filters,
                "result_type": "unsupported",
                "count": 0,
                "results": [],
            }

        q = db.query(VehicleAttributeModel).filter(VehicleAttributeModel.video_id == video_id)

        target_color = filters.get("color")
        if target_color:
            q = q.filter(VehicleAttributeModel.color.ilike(target_color))

        obj_class = filters.get("object_class")
        if obj_class and obj_class != "vehicle_group":
            q = q.filter(VehicleAttributeModel.object_class.ilike(obj_class))

        if filters.get("start_time") is not None:
            q = q.filter(VehicleAttributeModel.timestamp_seconds >= filters["start_time"])
        if filters.get("end_time") is not None:
            q = q.filter(VehicleAttributeModel.timestamp_seconds <= filters["end_time"])

        rows = q.order_by(VehicleAttributeModel.timestamp_seconds.asc()).all()
        results = [
            {
                "id": r.id,
                "video_id": r.video_id,
                "track_id": r.track_id,
                "object_class": r.object_class,
                "color": r.color,
                "confidence": round(r.confidence, 4),
                "timestamp": round(r.timestamp_seconds, 2),
                "bounding_box": r.bounding_box,
            }
            for r in rows
        ]

        if results:
            msg = f"Found {len(results)} verified vehicle records matching visual color '{target_color}'."
        else:
            msg = f"No vehicles with verified visual color '{target_color}' were found in database records."

        return {
            "query": query_text,
            "is_supported": True,
            "result_type": "vehicle_attributes",
            "interpreted_filters": filters,
            "count": len(results),
            "message": msg,
            "results": results,
        }

    def _query_tracks(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch matching object tracking records with canonical entity reconciliation."""
        all_video_tracks = (
            db.query(TrackModel)
            .filter(TrackModel.video_id == video_id)
            .order_by(TrackModel.first_seen.asc())
            .all()
        )
        all_video_sec_events = (
            db.query(SecurityEventModel)
            .filter(SecurityEventModel.video_id == video_id)
            .order_by(SecurityEventModel.timestamp_seconds.asc())
            .all()
        )

        all_track_items = []
        for r in all_video_tracks:
            primary_sec_events = []
            bystander_sec_events = []
            for se in all_video_sec_events:
                primary_tids = set()
                if se.track_id:
                    primary_tids.add(se.track_id)
                meta = se.incident_metadata or {}
                if isinstance(meta, dict) and "primary_tracks" in meta and isinstance(meta["primary_tracks"], list):
                    primary_tids.update(meta["primary_tracks"])
                s_ctx = meta.get("spatial_context") if isinstance(meta, dict) else None
                if isinstance(s_ctx, dict) and "metadata" in s_ctx:
                    p_tid = s_ctx["metadata"].get("person_track_id")
                    if p_tid:
                        primary_tids.add(p_tid)
                bbox = se.bounding_box or {}
                if isinstance(bbox, dict) and bbox.get("person_track_id"):
                    primary_tids.add(bbox.get("person_track_id"))

                secondary_tids = set()
                if isinstance(meta, dict):
                    if "secondary_tracks" in meta and isinstance(meta["secondary_tracks"], list):
                        secondary_tids.update(meta["secondary_tracks"])
                    if "track_ids" in meta and isinstance(meta["track_ids"], list):
                        for tid in meta["track_ids"]:
                            if tid not in primary_tids:
                                secondary_tids.add(tid)

                if r.track_id in primary_tids:
                    primary_sec_events.append(se)
                elif r.track_id in secondary_tids:
                    bystander_sec_events.append(se)

            sec_event_list = [
                {
                    "event_type": se.event_type,
                    "timestamp": round(se.timestamp_seconds, 2),
                    "severity": se.severity,
                    "description": se.description,
                    "role": "primary",
                }
                for se in primary_sec_events
            ] + [
                {
                    "event_type": se.event_type,
                    "timestamp": round(se.timestamp_seconds, 2),
                    "severity": se.severity,
                    "description": se.description,
                    "role": "bystander",
                }
                for se in bystander_sec_events
            ]

            # Factual observational movement and activity summary (zero biometric attribution)
            dur_str = f"{round(r.duration_seconds, 1)}s"
            theft_ev = next((se for se in primary_sec_events if se.event_type in ("POTENTIAL_THEFT", "POTENTIAL_OBJECT_TAKEAWAY")), None)
            fall_ev = next((se for se in primary_sec_events if se.event_type in ("POTENTIAL_PERSON_FALL", "POTENTIAL_PERSON_DOWN")), None)
            loit_ev = next((se for se in primary_sec_events if se.event_type == "PROLONGED_PRESENCE"), None)
            color_desc = f"with {r.color} clothing " if r.color else ""

            if theft_ev:
                meta = theft_ev.incident_metadata or {}
                s_ctx = meta.get("spatial_context") if isinstance(meta, dict) else None
                o_tid = ""
                o_cls = "portable object"
                if isinstance(s_ctx, dict) and "metadata" in s_ctx:
                    o_tid = s_ctx["metadata"].get("object_track_id", "")
                    o_cls_raw = s_ctx["metadata"].get("object_class")
                    if o_cls_raw and o_cls_raw != "person":
                        o_cls = o_cls_raw.replace("_", " ")
                if not o_tid and isinstance(theft_ev.bounding_box, dict):
                    o_tid = theft_ev.bounding_box.get("object_track_id", "")
                obj_str = f"{o_cls} ({o_tid})" if o_tid else f"{o_cls}"
                act_summary = (
                    f"The person {color_desc}({r.track_id}) was observed interacting with a {obj_str}. "
                    f"The object subsequently moved with the departing person. "
                    f"This produced a potential takeaway/theft pattern. Human review is required."
                )
            elif fall_ev:
                act_summary = (
                    f"The person {color_desc}({r.track_id}) exhibited rapid downward descent and aspect-ratio change around {fall_ev.timestamp_seconds:.1f}s. "
                    f"Potential person fall pattern. Human review is required."
                )
            elif loit_ev:
                act_summary = (
                    f"The individual {color_desc}({r.track_id}) remained stationary in a localized area for {r.duration_seconds:.1f}s. "
                    f"Prolonged presence pattern observed."
                )
            elif bystander_sec_events:
                ev_names = ", ".join(sorted(set(se.event_type for se in bystander_sec_events)))
                act_summary = (
                    f"The person {color_desc}({r.track_id}) was observed moving through the scene from {round(r.first_seen, 1)}s to {round(r.last_seen, 1)}s "
                    f"across {r.detection_count} detections (duration: {dur_str}). "
                    f"No security infractions or suspicious activity patterns were recorded for this track "
                    f"(present as a bystander during nearby scene activity: {ev_names})."
                )
            elif primary_sec_events:
                ev_names = ", ".join(sorted(set(se["event_type"] for se in sec_event_list if se["role"] == "primary")))
                act_summary = (
                    f"Continuous visual presence observed from {round(r.first_seen, 1)}s to {round(r.last_seen, 1)}s "
                    f"across {r.detection_count} detections (duration: {dur_str}). Associated security event(s): {ev_names}."
                )
            else:
                act_summary = (
                    f"The person {color_desc}({r.track_id}) was observed moving through the scene from {round(r.first_seen, 1)}s to {round(r.last_seen, 1)}s "
                    f"across {r.detection_count} detections (duration: {dur_str}). "
                    f"No security infractions, loitering, or suspicious activity patterns were recorded for this track."
                )

            all_track_items.append({
                "track_id": r.track_id,
                "object_class": r.object_class,
                "first_seen": round(r.first_seen, 2),
                "last_seen": round(r.last_seen, 2),
                "duration_seconds": round(r.duration_seconds, 2),
                "detection_count": r.detection_count,
                "max_confidence": round(r.max_confidence, 4),
                "color": r.color,
                "color_confidence": round(r.color_confidence, 4) if r.color_confidence else None,
                "current_bbox": r.current_bbox,
                "trajectory": r.trajectory or [],
                "active": bool(r.active),
                "security_events": sec_event_list,
                "activity_summary": act_summary,
            })

        canonical_entities = self._reconcile_canonical_entities(all_track_items, video_id=video_id)

        # Filter canonical entities based on query filters
        results = list(canonical_entities)

        target_track_id = filters.get("track_id")
        if target_track_id:
            results = [
                ce for ce in results
                if target_track_id.upper() == ce["canonical_id"].upper()
                or target_track_id.upper() in [tid.upper() for tid in ce.get("member_track_ids", [])]
            ]

        obj_class = filters.get("object_class")
        if obj_class:
            if obj_class == "vehicle_group":
                results = [ce for ce in results if ce.get("object_class") in VEHICLE_CLASSES]
            else:
                results = [ce for ce in results if ce.get("object_class", "").lower() == obj_class.lower()]

        target_color = filters.get("color")
        if target_color:
            results = [ce for ce in results if (ce.get("color") or "").lower() == target_color.lower()]

        if target_color and obj_class:
            if results:
                first = results[0]
                msg = (
                    f"Found {len(results)} verified {obj_class} track(s) matching visual color '{target_color}' ({first['track_id']}). "
                    f"{first['activity_summary']} "
                    "(Note: Track IDs represent consistent visual objects in this video only; zero personal identity attribution)."
                )
            else:
                msg = (
                    f"No {obj_class} tracks with verified visual color '{target_color}' were found in database records. "
                    "Sentinel only establishes clothing attributes when consistent multi-frame evidence exists."
                )
        elif results:
            first = results[0]
            msg = (
                f"Found {len(results)} multi-frame tracked objects. "
                f"First track ({first['track_id']}): {first['activity_summary']} "
                "(Note: Track IDs represent consistent visual objects in this video only; zero personal identity attribution)."
            )
        else:
            msg = "No multi-frame tracked objects were found in database records."

        return {
            "query": query_text,
            "is_supported": True,
            "result_type": "tracks",
            "interpreted_filters": filters,
            "count": len(results),
            "canonical_entity_count": len(results),
            "canonical_entities": results,
            "raw_track_count": len(all_video_tracks),
            "message": msg,
            "results": results,
        }

    def _query_faces(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fetch anonymous face visual region detections from DB.
        STRICT SAFETY: Zero facial recognition, zero identity mapping.
        """
        q = db.query(FaceDetectionModel).filter(FaceDetectionModel.video_id == video_id)

        if filters.get("start_time") is not None:
            q = q.filter(FaceDetectionModel.timestamp_seconds >= filters["start_time"])
        if filters.get("end_time") is not None:
            q = q.filter(FaceDetectionModel.timestamp_seconds <= filters["end_time"])

        rows = q.order_by(FaceDetectionModel.timestamp_seconds.asc()).all()
        results = [
            {
                "id": r.id,
                "video_id": r.video_id,
                "track_id": r.track_id,
                "timestamp": round(r.timestamp_seconds, 2),
                "confidence": round(r.confidence, 4),
                "bounding_box": {
                    "x1": round(r.bbox_x1, 1),
                    "y1": round(r.bbox_y1, 1),
                    "x2": round(r.bbox_x2, 1),
                    "y2": round(r.bbox_y2, 1),
                },
            }
            for r in rows
        ]

        msg = (
            f"Found {len(results)} face visual region detections. "
            "STRICT OBSERVATIONAL NOTICE: Visual regions only; facial recognition, biometric identity inference, and database matching are strictly disabled."
        )

        return {
            "query": query_text,
            "is_supported": True,
            "result_type": "faces",
            "interpreted_filters": filters,
            "count": len(results),
            "message": msg,
            "results": results,
        }

    def _query_evidence(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch matching preserved forensic evidence records from DB."""
        q = db.query(EvidenceModel).filter(
            EvidenceModel.video_id == video_id,
            EvidenceModel.validation_status == "VALID",
        )
        if filters.get("start_time") is not None:
            q = q.filter(EvidenceModel.timestamp_seconds >= filters["start_time"])
        if filters.get("end_time") is not None:
            q = q.filter(EvidenceModel.timestamp_seconds <= filters["end_time"])

        rows = q.order_by(EvidenceModel.timestamp_seconds.asc()).all()
        results = [
            {
                "evidence_id": r.id,
                "video_id": r.video_id,
                "event_id": r.event_id,
                "timestamp": round(r.timestamp_seconds, 2),
                "evidence_type": r.evidence_type,
                "object_class": r.object_class,
                "confidence": round(r.confidence, 4) if r.confidence else None,
                "has_snapshot": bool(r.snapshot_path),
                "has_annotated": bool(r.annotated_snapshot_path),
                "has_clip": bool(r.clip_path),
                "description": f"{r.evidence_type} evidence preserved around {r.timestamp_seconds:.1f}s for {r.object_class or 'incident'}.",
            }
            for r in rows
        ]
        msg = (
            f"Found {len(results)} verified forensic evidence records registered in Sentinel vault."
            if results
            else "No preserved forensic evidence items were found matching this inquiry."
        )
        return {
            "query": query_text,
            "is_supported": True,
            "result_type": "evidence",
            "interpreted_filters": filters,
            "count": len(results),
            "message": msg,
            "results": results,
        }

    def _query_security_events(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch structured security intelligence events from DB."""
        q = db.query(SecurityEventModel).filter(SecurityEventModel.video_id == video_id)

        target_type = filters.get("event_type")
        target_category = filters.get("category")

        VEHICLE_EVENT_TYPES = [
            "POTENTIAL_VEHICLE_COLLISION",
            "POTENTIAL_NEAR_COLLISION",
            "POTENTIAL_SUDDEN_VEHICLE_STOP",
            "POTENTIAL_WRONG_WAY_VEHICLE",
            "POTENTIAL_UNUSUAL_VEHICLE_TRAJECTORY",
            "POTENTIAL_STATIONARY_VEHICLE",
        ]

        PERSON_EVENT_TYPES = [
            "POTENTIAL_PERSON_FALL",
            "POTENTIAL_PERSON_DOWN",
            "POTENTIAL_PANIC_RUNNING",
            "UNUSUAL_RAPID_PERSON_MOVEMENT",
            "POTENTIAL_PHYSICAL_ALTERCATION",
            "POTENTIAL_FORCED_MOVEMENT",
            "PERSON_FOLLOWING",
            "COORDINATED_PERSON_MOVEMENT",
        ]

        PROPERTY_EVENT_TYPES = [
            "POTENTIAL_ABANDONED_OBJECT",
            "POTENTIAL_OBJECT_LEFT_BEHIND",
            "POTENTIAL_OBJECT_PICKUP",
            "POTENTIAL_THEFT",
            "POTENTIAL_OBJECT_TAKEAWAY",
            "POTENTIAL_OBJECT_DISPLACEMENT",
            "POTENTIAL_PROPERTY_TAMPERING",
            "POTENTIAL_RESTRICTED_OBJECT_MOVEMENT",
            "POTENTIAL_OBJECT_REMOVAL",
        ]

        CROWD_EVENT_TYPES = [
            "HIGH_PEDESTRIAN_DENSITY",
            "CROWD_DENSITY_INCREASE",
            "POTENTIAL_CROWD_SURGE",
            "POTENTIAL_CROWD_DISPERSAL",
            "POTENTIAL_UNUSUAL_CROWD_MOVEMENT",
            "POTENTIAL_RESTRICTED_ZONE_CROWDING",
            "POTENTIAL_UNUSUAL_ZONE_ACTIVITY",
            "ZONE_OCCUPANCY_OBSERVATION",
        ]

        SPECIALIZED_EVENT_TYPES = [
            "POTENTIAL_FIRE",
            "POTENTIAL_SMOKE",
            "POTENTIAL_FIRE_SMOKE",
            "POTENTIAL_WEAPON_VISUAL",
        ]

        if target_type:
            q = q.filter(SecurityEventModel.event_type == target_type)
        elif target_category == "vehicle":
            q = q.filter(SecurityEventModel.event_type.in_(VEHICLE_EVENT_TYPES))
        elif target_category == "person":
            q = q.filter(SecurityEventModel.event_type.in_(PERSON_EVENT_TYPES))
        elif target_category == "property":
            q = q.filter(SecurityEventModel.event_type.in_(PROPERTY_EVENT_TYPES))
        elif target_category in ("crowd", "zone"):
            q = q.filter(SecurityEventModel.event_type.in_(CROWD_EVENT_TYPES))
        elif target_category in ("environment", "specialized"):
            q = q.filter(SecurityEventModel.event_type.in_(SPECIALIZED_EVENT_TYPES))
        elif target_category == "object":
            q = q.filter(SecurityEventModel.event_type.in_(PROPERTY_EVENT_TYPES + ["POTENTIAL_WEAPON_VISUAL"]))

        if filters.get("start_time") is not None:
            q = q.filter(
                (SecurityEventModel.timestamp_seconds + func.coalesce(SecurityEventModel.duration_seconds, 0.0) >= filters["start_time"] - 5.0)
            )
        if filters.get("end_time") is not None:
            q = q.filter(SecurityEventModel.timestamp_seconds <= filters["end_time"] + 5.0)
        if filters.get("min_confidence") is not None:
            q = q.filter(SecurityEventModel.confidence >= filters["min_confidence"])
        if filters.get("max_confidence") is not None:
            q = q.filter(SecurityEventModel.confidence <= filters["max_confidence"])

        rows = q.order_by(SecurityEventModel.timestamp_seconds.asc()).all()

        if not rows and target_type in SPECIALIZED_EVENT_TYPES:
            spec_cls = target_type.replace("POTENTIAL_", "").lower()
            return self._query_specialized_observations(db, video_id, query_text, {
                "class_name": spec_cls,
                "start_time": filters.get("start_time"),
                "end_time": filters.get("end_time"),
                "min_confidence": filters.get("min_confidence"),
            })

        results = [
            {
                "id": r.id,
                "video_id": r.video_id,
                "event_type": r.event_type,
                "timestamp": round(r.timestamp_seconds, 2),
                "duration_seconds": round(r.duration_seconds, 2) if r.duration_seconds else 0.0,
                "track_id": r.track_id,
                "object_class": r.object_class,
                "severity": r.severity,
                "confidence": round(r.confidence, 4),
                "zone_name": r.zone_name,
                "description": r.description,
                "observable_signals": r.observable_signals or {},
                "bounding_box": r.bounding_box,
                "evidence_id": r.evidence_id,
                "validation_decision": (
                    (r.incident_metadata or {}).get("validation_decision")
                    if isinstance(r.incident_metadata, dict) and (r.incident_metadata or {}).get("validation_decision")
                    else ("REVIEW_REQUIRED" if r.human_verification_required else "ACCEPTED")
                ),
                "human_verification_required": bool(r.human_verification_required),
            }
            for r in rows
        ]

        is_specialized_query = (
            target_category in ("environment", "specialized")
            or (target_type and target_type in SPECIALIZED_EVENT_TYPES)
        )

        is_vehicle_query = (
            target_category == "vehicle"
            or (target_type and (
                "VEHICLE" in target_type
                or target_type in ("POTENTIAL_NEAR_COLLISION", "POTENTIAL_SUDDEN_VEHICLE_STOP", "POTENTIAL_UNUSUAL_VEHICLE_TRAJECTORY")
            ))
        )

        is_person_query = (
            target_category == "person"
            or (target_type and target_type in PERSON_EVENT_TYPES)
        )

        is_property_query = (
            target_category in ("property", "object")
            or (target_type and target_type in PROPERTY_EVENT_TYPES and target_type != "POTENTIAL_THEFT")
        )

        is_crowd_query = (
            target_category in ("crowd", "zone")
            or (target_type and target_type in CROWD_EVENT_TYPES)
        )

        if is_specialized_query:
            if results:
                msg = (
                    f"Found {len(results)} specialized visual incident observation(s) matching '{query_text}'. "
                    "Observational machine evidence only; mandatory human verification required."
                )
            else:
                msg = f"No specialized visual events were detected matching '{query_text}' in available evidence."
        elif is_crowd_query:
            if results:
                msg = (
                    f"Found {len(results)} crowd & zone intelligence observation(s) in Sentinel database records. "
                    "(Observational density & spatial analytics only; zero inference of threat, intent, or criminality)."
                )
            else:
                msg = "No crowd, density, or zone activity events were detected in the available visual evidence."
        elif is_person_query:
            if results:
                msg = f"Found {len(results)} verified person incident event(s) in Sentinel database records."
            else:
                msg = "No reliable person incident was detected in the available Sentinel data."
        elif is_vehicle_query:
            if results:
                msg = f"Found {len(results)} verified vehicle incident event(s) in Sentinel database records."
            else:
                msg = "No reliable vehicle incident was detected in the available Sentinel data."
        elif is_property_query:
            if results:
                msg = f"Found {len(results)} verified property / object incident event(s) in Sentinel database records."
            else:
                msg = "No reliable property or object incidents were detected in the available Sentinel data."
        elif target_type == "POTENTIAL_THEFT":
            if results:
                first_ev = results[0]
                msg = (
                    f"Sentinel identified a potential object-takeaway pattern around {first_ev['timestamp']:.1f}s. "
                    f"{first_ev['description']} Review the linked evidence."
                )
            else:
                msg = "No potential theft pattern was detected in the available visual evidence."
        elif target_type:
            ev_label = target_type.replace("_", " ").lower()
            if results:
                msg = f"Found {len(results)} verified {ev_label} events in database records."
            else:
                msg = f"No {ev_label} events were detected in the available visual evidence."
        else:
            if results:
                msg = f"Found {len(results)} security intelligence events available for review."
            else:
                msg = "No security events or anomalies were detected in the available visual evidence."

        return {
            "query": query_text,
            "is_supported": True,
            "message": msg,
            "interpreted_filters": filters,
            "result_type": "security_events",
            "count": len(results),
            "results": results,
        }

    def _query_specialized_observations(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch matching specialized visual observations from SpecializedObservationModel."""
        q = db.query(SpecializedObservationModel).filter(SpecializedObservationModel.video_id == video_id)

        cls_name = filters.get("object_class") or filters.get("class_name")
        if cls_name:
            q = q.filter(SpecializedObservationModel.class_name.ilike(f"%{cls_name}%"))

        if filters.get("start_time") is not None:
            q = q.filter(SpecializedObservationModel.timestamp_seconds >= filters["start_time"])
        if filters.get("end_time") is not None:
            q = q.filter(SpecializedObservationModel.timestamp_seconds <= filters["end_time"])
        if filters.get("min_confidence") is not None:
            q = q.filter(SpecializedObservationModel.confidence >= filters["min_confidence"])

        rows = q.order_by(SpecializedObservationModel.timestamp_seconds.asc()).all()
        results = [
            {
                "id": r.id,
                "video_id": r.video_id,
                "event_id": r.event_id,
                "detector_name": r.detector_name,
                "detector_version": r.detector_version,
                "class_name": r.class_name,
                "object_class": r.class_name,
                "timestamp": round(r.timestamp_seconds, 2),
                "confidence": round(r.confidence, 4),
                "evidence_strength": round(r.evidence_strength, 4),
                "validation_status": r.validation_status,
                "bounding_box": r.bounding_box,
                "metrics": r.metrics,
            }
            for r in rows
        ]
        target_name = cls_name or "specialized"
        msg = f"Identified {len(results)} specialized visual observation(s). Potential {target_name} visual evidence was detected in surveillance footage; mandatory human verification required."
        return {
            "query": query_text,
            "is_supported": True,
            "message": msg,
            "interpreted_filters": filters,
            "result_type": "specialized",
            "count": len(results),
            "results": results,
        }

    def _query_correlated_incidents(
        self, db, video_id: str, query_text: str, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch matching correlated incidents from CorrelatedIncidentModel."""
        q = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.video_id == video_id)

        category = filters.get("category")
        if category:
            q = q.filter(CorrelatedIncidentModel.incident_category == category)

        val_decision = filters.get("validation_decision")
        if val_decision:
            q = q.filter(CorrelatedIncidentModel.validation_decision == val_decision)

        rows = q.order_by(CorrelatedIncidentModel.start_time.asc()).all()

        object_pair = filters.get("object_pair")
        filtered_rows = []
        for r in rows:
            classes = [c.lower() for c in (r.involved_object_classes or [])]
            if object_pair:
                p1, p2 = object_pair[0].lower(), object_pair[1].lower()
                has_p1 = any(p1 in c for c in classes)
                has_p2 = any(p2 in c for c in classes) or (p2 == "object" and any(c not in ["person", "car", "bus", "truck", "motorcycle"] for c in classes))
                if p2 == "vehicle":
                    has_p2 = any(c in VEHICLE_CLASSES or "vehicle" in c for c in classes)
                if not (has_p1 and has_p2):
                    continue
            filtered_rows.append(r)

        results = [
            {
                "incident_id": r.id,
                "video_id": r.video_id,
                "incident_category": r.incident_category,
                "incident_subcategory": r.incident_subcategory,
                "start_time": round(r.start_time, 2),
                "end_time": round(r.end_time, 2),
                "duration": round(r.duration, 2),
                "primary_track_ids": r.primary_track_ids or [],
                "supporting_track_ids": r.supporting_track_ids or [],
                "involved_object_classes": r.involved_object_classes or [],
                "assessment_score": round(r.assessment_score, 4),
                "evidence_strength": round(r.evidence_strength, 4),
                "reliability_rating": r.reliability_rating,
                "validation_decision": r.validation_decision,
                "storyline": r.storyline or "",
                "evidence_ids": r.evidence_ids or [],
                "source_candidate_ids": r.source_candidate_ids or [],
                "provenance": r.provenance or {},
            }
            for r in filtered_rows
        ]

        query_focus = filters.get("query_focus")
        if query_focus == "pre_collision_sequence":
            msg = f"Retrieved {len(results)} collision sequence storyline(s) showing trajectory convergence and contact."
        elif query_focus == "theft_storyline":
            msg = f"Retrieved {len(results)} coherent object takeaway storyline(s) with supporting approach, dwell, and departure provenance."
        elif query_focus == "evidence_support":
            msg = f"Identified {len(results)} correlated incident(s) with canonical supporting evidence links and track provenance."
        elif query_focus == "duplicate_fusion":
            msg = f"Identified {len(results)} correlated incident(s) formed by fusing multiple overlapping or adjacent candidate signals."
        elif object_pair:
            msg = f"Identified {len(results)} correlated incident(s) involving {' and '.join(object_pair)} interaction."
        elif val_decision:
            msg = f"Found {len(results)} incident(s) categorized as {val_decision}."
        else:
            msg = f"Identified {len(results)} canonical correlated incident(s) in video intelligence records."

        return {
            "query": query_text,
            "is_supported": True,
            "message": msg,
            "interpreted_filters": filters,
            "result_type": "correlated_incidents",
            "count": len(results),
            "results": results,
        }

    # -----------------------------------------------------------------------
    # Phase 17: Structured query entry point
    # -----------------------------------------------------------------------

    def investigate_structured(
        self,
        video_id: str,
        query_text: str = "",
        video_duration_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Phase 17: Parse a natural-language query into an InvestigationQuery,
        then execute it via InvestigationQueryExecutor, returning a full
        InvestigationResult serialized to dict.

        Falls back to investigate_filters if executor fails.
        """
        from backend.app.services.investigation_query import InvestigationQuery
        from backend.app.services.investigation_query_executor import InvestigationQueryExecutor

        # Build typed query via extended parser
        try:
            iq = InvestigationParser.parse_investigation_query(
                user_query=query_text,
                video_id=video_id,
                video_duration_seconds=video_duration_seconds,
            )
        except Exception as parse_err:
            logger.warning(f"Phase 17 parser failed: {parse_err}. Falling back to Phase 5A.")
            return self.investigate(video_id=video_id, query_text=query_text)

        try:
            executor = InvestigationQueryExecutor()
            result = executor.execute(iq)
            result_dict = result.to_dict()
            result_dict["query_text"] = query_text
            return result_dict
        except Exception as exec_err:
            logger.warning(f"Phase 17 executor failed: {exec_err}. Falling back to Phase 5A.")
            return self.investigate(video_id=video_id, query_text=query_text)

    def investigate_track(
        self,
        video_id: str,
        track_id: str,
    ) -> Dict[str, Any]:
        """
        Phase 17: Full investigation of a single anonymous track.

        Returns:
        - Track metadata (first_seen, last_seen, object_class, color)
        - Associated security events
        - Associated correlated incidents
        - Linked evidence
        - Object lifecycle summary (DETECTED → appearances → LAST_SEEN)
        """
        db = SessionLocal()
        try:
            # Verify track belongs to this video (critical isolation check)
            track = (
                db.query(TrackModel)
                .filter(
                    TrackModel.video_id == video_id,
                    TrackModel.track_id == track_id,
                )
                .first()
            )
            if not track:
                return {
                    "video_id": video_id,
                    "track_id": track_id,
                    "is_supported": False,
                    "message": f"Track '{track_id}' not found in video '{video_id}'.",
                    "result_type": "track_investigation",
                    "results": {},
                }

            # Security events for this track
            sec_events = (
                db.query(SecurityEventModel)
                .filter(
                    SecurityEventModel.video_id == video_id,
                    SecurityEventModel.track_id == track_id,
                )
                .order_by(SecurityEventModel.timestamp_seconds.asc())
                .all()
            )

            # Correlated incidents referencing this track
            all_incidents = (
                db.query(CorrelatedIncidentModel)
                .filter(CorrelatedIncidentModel.video_id == video_id)
                .all()
            )
            related_incidents = [
                r for r in all_incidents
                if track_id in (r.primary_track_ids or [])
                or track_id in (r.supporting_track_ids or [])
            ]

            # Evidence correlated to this track's time window
            evidence = (
                db.query(EvidenceModel)
                .filter(
                    EvidenceModel.video_id == video_id,
                    EvidenceModel.validation_status == "VALID",
                    EvidenceModel.timestamp_seconds >= track.first_seen - 1.0,
                    EvidenceModel.timestamp_seconds <= track.last_seen + 1.0,
                )
                .order_by(EvidenceModel.timestamp_seconds.asc())
                .all()
            )

            # Object lifecycle: summarize appearance from trajectory
            lifecycle = self._build_object_lifecycle(track)

            track_data = {
                "track_id": track.track_id,
                "video_id": track.video_id,
                "object_class": track.object_class,
                "first_seen": round(track.first_seen, 2),
                "last_seen": round(track.last_seen, 2),
                "duration_seconds": round(track.duration_seconds, 2),
                "detection_count": track.detection_count,
                "max_confidence": round(track.max_confidence, 4),
                "color": track.color,
                "active": bool(track.active),
            }

            return {
                "video_id": video_id,
                "track_id": track_id,
                "is_supported": True,
                "result_type": "track_investigation",
                "message": (
                    f"Track {track_id} ({track.object_class}) was observed from "
                    f"{track.first_seen:.1f}s to {track.last_seen:.1f}s "
                    f"across {track.detection_count} detections."
                    " (Anonymous track — no identity inference.)"
                ),
                "track": track_data,
                "lifecycle": lifecycle,
                "security_events": [
                    {
                        "id": e.id,
                        "event_type": e.event_type,
                        "timestamp": round(e.timestamp_seconds, 2),
                        "description": e.description,
                        "severity": e.severity,
                    }
                    for e in sec_events
                ],
                "related_incidents": [
                    {
                        "incident_id": inc.id,
                        "incident_category": inc.incident_category,
                        "incident_subcategory": inc.incident_subcategory,
                        "start_time": round(inc.start_time, 2),
                        "end_time": round(inc.end_time, 2),
                        "validation_decision": inc.validation_decision,
                        "assessment_score": round(inc.assessment_score, 4),
                        "storyline": inc.storyline or "",
                    }
                    for inc in related_incidents
                ],
                "evidence": [
                    {
                        "evidence_id": ev.id,
                        "timestamp": round(ev.timestamp_seconds, 2),
                        "evidence_type": ev.evidence_type,
                        "has_snapshot": bool(ev.snapshot_path),
                        "has_clip": bool(ev.clip_path),
                    }
                    for ev in evidence
                ],
            }
        finally:
            db.close()

    def _build_object_lifecycle(self, track: TrackModel) -> Dict[str, Any]:
        """
        Build an observational object lifecycle summary for a track.
        Uses observational language — does NOT infer intent.
        """
        phases = []

        phases.append({
            "phase": "DETECTED",
            "timestamp": round(track.first_seen, 2),
            "note": f"First detected at {track.first_seen:.1f}s.",
        })

        # Check trajectory for stationary periods if available
        if track.trajectory and isinstance(track.trajectory, list) and len(track.trajectory) > 2:
            try:
                positions = [(t[1], t[2]) for t in track.trajectory if len(t) >= 3]
                if positions:
                    # Check if object was mostly stationary (low movement variance)
                    xs = [p[0] for p in positions]
                    ys = [p[1] for p in positions]
                    x_range = max(xs) - min(xs) if xs else 0
                    y_range = max(ys) - min(ys) if ys else 0
                    if x_range < 30 and y_range < 30 and len(positions) > 3:
                        phases.append({
                            "phase": "STATIONARY_PERIOD",
                            "note": "Object remained approximately stationary over multiple frames.",
                        })
            except (TypeError, IndexError):
                pass

        if track.duration_seconds > 1.0:
            phases.append({
                "phase": "PRESENT",
                "timestamp_start": round(track.first_seen, 2),
                "timestamp_end": round(track.last_seen, 2),
                "note": f"Present for {track.duration_seconds:.1f}s across {track.detection_count} detection frames.",
            })

        phases.append({
            "phase": "LAST_SEEN",
            "timestamp": round(track.last_seen, 2),
            "note": f"Last detected at {track.last_seen:.1f}s. Active: {bool(track.active)}.",
        })

        return {
            "track_id": track.track_id,
            "object_class": track.object_class,
            "phases": phases,
            "observational_note": (
                "This lifecycle is an observational summary of detection activity only. "
                "No identity, intent, or behavioral conclusion is inferred."
            ),
        }

    def investigate_zone(
        self,
        video_id: str,
        zone_name: str,
    ) -> Dict[str, Any]:
        """
        Phase 17: Zone-focused investigation.
        Returns all security events and incidents associated with a specific zone name.
        Video-isolated; zone_name matched case-insensitively.
        """
        db = SessionLocal()
        try:
            # Security events in this zone
            sec_events = (
                db.query(SecurityEventModel)
                .filter(
                    SecurityEventModel.video_id == video_id,
                    SecurityEventModel.zone_name.ilike(f"%{zone_name}%"),
                )
                .order_by(SecurityEventModel.timestamp_seconds.asc())
                .all()
            )

            # Correlated incidents mentioning this zone
            all_incidents = (
                db.query(CorrelatedIncidentModel)
                .filter(CorrelatedIncidentModel.video_id == video_id)
                .all()
            )
            zone_incidents = [
                inc for inc in all_incidents
                if zone_name.lower() in (inc.storyline or "").lower()
                or any(
                    zone_name.lower() in str(z).lower()
                    for z in (inc.zone_ids or [])
                )
            ]

            track_ids_in_zone = list({e.track_id for e in sec_events if e.track_id})

            events_serialized = [
                {
                    "id": e.id,
                    "event_type": e.event_type,
                    "timestamp": round(e.timestamp_seconds, 2),
                    "duration_seconds": round(e.duration_seconds or 0.0, 2),
                    "track_id": e.track_id,
                    "object_class": e.object_class,
                    "description": e.description,
                    "severity": e.severity,
                    "validation_decision": (
                        "REVIEW_REQUIRED" if e.human_verification_required else "ACCEPTED"
                    ),
                }
                for e in sec_events
            ]

            incidents_serialized = [
                {
                    "incident_id": inc.id,
                    "incident_category": inc.incident_category,
                    "start_time": round(inc.start_time, 2),
                    "end_time": round(inc.end_time, 2),
                    "validation_decision": inc.validation_decision,
                    "storyline": inc.storyline or "",
                }
                for inc in zone_incidents
            ]

            if sec_events or zone_incidents:
                msg = (
                    f"Found {len(sec_events)} security event(s) and {len(zone_incidents)} "
                    f"correlated incident(s) in zone '{zone_name}'."
                )
            else:
                msg = f"No activity detected in zone '{zone_name}' for this video."

            return {
                "video_id": video_id,
                "zone_name": zone_name,
                "is_supported": True,
                "result_type": "zone_investigation",
                "message": msg,
                "security_events": events_serialized,
                "correlated_incidents": incidents_serialized,
                "track_ids_observed": track_ids_in_zone,
            }
        finally:
            db.close()

    def get_unified_timeline(
        self,
        video_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        include_rejected: bool = False,
    ) -> Dict[str, Any]:
        """
        Phase 17: Unified forensic timeline.

        Merges all Sentinel intelligence layers into a single chronological sequence:
        - Grouped events (detection clusters)
        - Security events
        - Correlated incidents
        - Evidence markers

        Each entry carries layer type, playback timestamp, and detail.
        Results are bounded to [start_time, end_time] if provided.
        """
        db = SessionLocal()
        try:
            timeline_entries = []

            # Layer 1: Grouped events (detection timeline)
            ge_q = db.query(GroupedEventModel).filter(GroupedEventModel.video_id == video_id)
            if start_time is not None:
                ge_q = ge_q.filter(GroupedEventModel.end_time >= start_time)
            if end_time is not None:
                ge_q = ge_q.filter(GroupedEventModel.start_time <= end_time)
            grouped_events = ge_q.order_by(GroupedEventModel.start_time.asc()).all()
            for ge in grouped_events:
                timeline_entries.append({
                    "layer": "detection_event",
                    "timestamp": round(ge.start_time, 2),
                    "end_timestamp": round(ge.end_time, 2),
                    "id": ge.id,
                    "label": ge.event_type,
                    "total_detections": ge.total_detections,
                    "priority": ge.priority,
                })

            # Layer 2: Security events
            se_q = db.query(SecurityEventModel).filter(SecurityEventModel.video_id == video_id)
            if start_time is not None:
                se_q = se_q.filter(SecurityEventModel.timestamp_seconds >= start_time)
            if end_time is not None:
                se_q = se_q.filter(SecurityEventModel.timestamp_seconds <= end_time)
            sec_events = se_q.order_by(SecurityEventModel.timestamp_seconds.asc()).all()
            for se in sec_events:
                timeline_entries.append({
                    "layer": "security_event",
                    "timestamp": round(se.timestamp_seconds, 2),
                    "end_timestamp": round(se.timestamp_seconds + (se.duration_seconds or 0.0), 2),
                    "id": se.id,
                    "label": se.event_type.replace("_", " "),
                    "severity": se.severity,
                    "track_id": se.track_id,
                    "description": se.description,
                    "validation_decision": "REVIEW_REQUIRED" if se.human_verification_required else "ACCEPTED",
                })

            # Layer 3: Correlated incidents
            ci_q = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.video_id == video_id)
            if start_time is not None:
                ci_q = ci_q.filter(CorrelatedIncidentModel.end_time >= start_time)
            if end_time is not None:
                ci_q = ci_q.filter(CorrelatedIncidentModel.start_time <= end_time)
            if not include_rejected:
                ci_q = ci_q.filter(CorrelatedIncidentModel.validation_decision != "REJECTED")
            incidents = ci_q.order_by(CorrelatedIncidentModel.start_time.asc()).all()
            for inc in incidents:
                timeline_entries.append({
                    "layer": "correlated_incident",
                    "timestamp": round(inc.start_time, 2),
                    "end_timestamp": round(inc.end_time, 2),
                    "id": inc.id,
                    "label": f"{inc.incident_category} — {inc.incident_subcategory or ''}",
                    "assessment_score": round(inc.assessment_score, 4),
                    "reliability_rating": inc.reliability_rating,
                    "validation_decision": inc.validation_decision,
                    "primary_track_ids": inc.primary_track_ids or [],
                })

            # Layer 4: Evidence markers
            ev_q = db.query(EvidenceModel).filter(
                EvidenceModel.video_id == video_id,
                EvidenceModel.validation_status == "VALID",
            )
            if start_time is not None:
                ev_q = ev_q.filter(EvidenceModel.timestamp_seconds >= start_time)
            if end_time is not None:
                ev_q = ev_q.filter(EvidenceModel.timestamp_seconds <= end_time)
            evidence = ev_q.order_by(EvidenceModel.timestamp_seconds.asc()).all()
            for ev in evidence:
                timeline_entries.append({
                    "layer": "evidence",
                    "timestamp": round(ev.timestamp_seconds, 2),
                    "end_timestamp": round(ev.end_time or ev.timestamp_seconds, 2),
                    "id": ev.id,
                    "label": f"Evidence ({ev.evidence_type})",
                    "has_snapshot": bool(ev.snapshot_path),
                    "has_clip": bool(ev.clip_path),
                    "event_id": ev.event_id,
                })

            # Sort by timestamp
            timeline_entries.sort(key=lambda x: x["timestamp"])

            return {
                "video_id": video_id,
                "is_supported": True,
                "result_type": "unified_timeline",
                "start_time_filter": start_time,
                "end_time_filter": end_time,
                "total_entries": len(timeline_entries),
                "timeline": timeline_entries,
                "layer_counts": {
                    "detection_events": len(grouped_events),
                    "security_events": len(sec_events),
                    "correlated_incidents": len(incidents),
                    "evidence": len(evidence),
                },
            }
        finally:
            db.close()


