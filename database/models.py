"""
SQLAlchemy Models for Sentinel Database Schema
"""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Index, Text, JSON, UniqueConstraint
from sqlalchemy.orm import relationship
from database.session import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class VideoModel(Base):
    __tablename__ = "videos"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    original_filename = Column(String(255), nullable=False)
    saved_filename = Column(String(255), nullable=True)
    storage_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, nullable=False, default=0)
    duration_seconds = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=True)
    status = Column(String(50), nullable=False, default="uploaded")
    camera_id = Column(String(36), ForeignKey("camera_sources.id", ondelete="SET NULL"), nullable=True, index=True)
    uploaded_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    processed_at = Column(DateTime, nullable=True)

    events = relationship("EventModel", back_populates="video", cascade="all, delete-orphan")
    grouped_events = relationship("GroupedEventModel", back_populates="video", cascade="all, delete-orphan")
    evidence = relationship("EvidenceModel", back_populates="video", cascade="all, delete-orphan")
    tracks = relationship("TrackModel", back_populates="video", cascade="all, delete-orphan")
    vehicle_attributes = relationship("VehicleAttributeModel", back_populates="video", cascade="all, delete-orphan")
    face_detections = relationship("FaceDetectionModel", back_populates="video", cascade="all, delete-orphan")
    security_events = relationship("SecurityEventModel", back_populates="video", cascade="all, delete-orphan")
    reports = relationship("ReportModel", back_populates="video", cascade="all, delete-orphan")
    specialized_observations = relationship("SpecializedObservationModel", back_populates="video", cascade="all, delete-orphan")
    correlated_incidents = relationship("CorrelatedIncidentModel", back_populates="video", cascade="all, delete-orphan")


class EventModel(Base):
    __tablename__ = "events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(100), nullable=False, default="object_detected")
    object_class = Column(String(100), nullable=False, index=True)
    class_id = Column(Integer, nullable=False)
    timestamp_seconds = Column(Float, nullable=False, index=True)
    confidence = Column(Float, nullable=False, index=True)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    frame_number = Column(Integer, nullable=False)
    validation_status = Column(String(20), nullable=False, default="VALID", index=True)
    validation_reason = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    video = relationship("VideoModel", back_populates="events")


class GroupedEventModel(Base):
    __tablename__ = "grouped_events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)  # e.g., VEHICLE_DETECTED, PERSON_DETECTED
    start_time = Column(Float, nullable=False, index=True)
    end_time = Column(Float, nullable=False, index=True)
    duration_seconds = Column(Float, nullable=False)
    objects_summary = Column(JSON, nullable=False)  # List of {"class": "car", "count": 3}
    total_detections = Column(Integer, nullable=False, default=1)
    max_confidence = Column(Float, nullable=False, index=True)
    priority = Column(String(20), nullable=False, default="NORMAL")  # LOW, NORMAL, HIGH
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    video = relationship("VideoModel", back_populates="grouped_events")


class EvidenceModel(Base):
    __tablename__ = "evidence"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    event_id = Column(String(36), nullable=True, index=True)  # Links to EventModel.id or GroupedEventModel.id
    evidence_type = Column(String(50), nullable=False, default="snapshot_and_clip")  # snapshot_and_clip, snapshot_only, clip_only
    validation_status = Column(String(20), nullable=False, default="VALID", index=True)
    timestamp_seconds = Column(Float, nullable=False, index=True)
    source_video_name = Column(String(255), nullable=False)
    snapshot_path = Column(String(512), nullable=True)
    annotated_snapshot_path = Column(String(512), nullable=True)
    clip_path = Column(String(512), nullable=True)
    object_class = Column(String(100), nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    bounding_box = Column(JSON, nullable=True)  # {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
    start_time = Column(Float, nullable=True)
    end_time = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    pre_seconds = Column(Float, nullable=True, default=3.0)
    post_seconds = Column(Float, nullable=True, default=3.0)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    video = relationship("VideoModel", back_populates="evidence")


# ---------------------------------------------------------------------------
# Phase 8: Advanced Security Intelligence Models
# ---------------------------------------------------------------------------

class TrackModel(Base):
    """Multi-frame tracked object records."""
    __tablename__ = "tracks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    track_id = Column(String(64), nullable=False, index=True)  # e.g., 'TRACK-001'
    object_class = Column(String(100), nullable=False, index=True)
    first_seen = Column(Float, nullable=False, index=True)
    last_seen = Column(Float, nullable=False, index=True)
    duration_seconds = Column(Float, nullable=False)
    detection_count = Column(Integer, nullable=False, default=1)
    max_confidence = Column(Float, nullable=False)
    current_bbox = Column(JSON, nullable=True)
    trajectory = Column(JSON, nullable=True)  # [(timestamp, cx, cy), ...]
    color = Column(String(50), nullable=True, index=True)
    color_confidence = Column(Float, nullable=True)
    active = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    video = relationship("VideoModel", back_populates="tracks")


class VehicleAttributeModel(Base):
    """Computer-vision vehicle color and attribute records."""
    __tablename__ = "vehicle_attributes"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    track_id = Column(String(64), nullable=True, index=True)
    event_id = Column(String(36), nullable=True)
    object_class = Column(String(100), nullable=False, default="car")
    color = Column(String(50), nullable=False, index=True)  # Controlled vocabulary
    confidence = Column(Float, nullable=False)
    timestamp_seconds = Column(Float, nullable=False, index=True)
    bounding_box = Column(JSON, nullable=True)
    color_space_metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    video = relationship("VideoModel", back_populates="vehicle_attributes")


class FaceDetectionModel(Base):
    """
    Anonymous face region detection records.
    SAFETY: Strictly visual bounding boxes only. Zero identity or recognition attributes.
    """
    __tablename__ = "face_detections"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    track_id = Column(String(64), nullable=True, index=True)
    event_id = Column(String(36), nullable=True)
    timestamp_seconds = Column(Float, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    video = relationship("VideoModel", back_populates="face_detections")


class SecurityZoneModel(Base):
    """User-defined security zones with polygon boundary definitions."""
    __tablename__ = "security_zones"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=True, index=True)
    name = Column(String(100), nullable=False)
    polygon_coordinates = Column(JSON, nullable=False)  # [[x, y], ...]
    target_classes = Column(JSON, nullable=False)      # ["person", ...]
    enabled = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class SecurityEventModel(Base):
    """
    Advanced security intelligence events:
    POTENTIAL_INTRUSION, PROLONGED_PRESENCE, POTENTIAL_ABANDONED_OBJECT,
    HIGH_ACTIVITY_PERIOD, OBSERVATIONAL_ANOMALY.
    """
    __tablename__ = "security_events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), nullable=False, default="NORMAL")  # LOW, NORMAL, HIGH
    timestamp_seconds = Column(Float, nullable=False, index=True)
    duration_seconds = Column(Float, nullable=False, default=0.0)
    track_id = Column(String(64), nullable=True, index=True)
    object_class = Column(String(100), nullable=True, index=True)
    zone_name = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=False)
    bounding_box = Column(JSON, nullable=True)
    description = Column(Text, nullable=False)
    observable_signals = Column(JSON, nullable=True)
    evidence_id = Column(String(36), nullable=True)
    detector_name = Column(String(100), nullable=True)
    detector_version = Column(String(20), nullable=True)
    category = Column(String(50), nullable=True)
    human_verification_required = Column(Integer, nullable=False, default=1)
    incident_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    video = relationship("VideoModel", back_populates="security_events")


class ReportModel(Base):
    """
    Phase 9: Professional Incident Dossier Reports.
    Stores generated PDF report metadata, safely linked to VideoModel.
    """
    __tablename__ = "reports"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    report_id = Column(String(64), nullable=False, unique=True, index=True)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False, default="Incident Dossier")
    report_type = Column(String(50), nullable=False, default="INCIDENT_DOSSIER")
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, nullable=False, default=0)
    page_count = Column(Integer, nullable=False, default=1)
    status = Column(String(50), nullable=False, default="completed")
    metadata_payload = Column(JSON, nullable=True)
    generated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    video = relationship("VideoModel", back_populates="reports")


# ---------------------------------------------------------------------------
# Phase 15: Specialized Visual Detection Models
# ---------------------------------------------------------------------------

class SpecializedObservationModel(Base):
    """
    Records raw and validated observations from specialized visual detectors:
    Fire, Smoke, Weapon/Object, Pose telemetry.
    """
    __tablename__ = "specialized_observations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    event_id = Column(String(36), nullable=True, index=True)
    detector_name = Column(String(100), nullable=False, index=True)
    detector_version = Column(String(20), nullable=False, default="1.0.0")
    class_name = Column(String(100), nullable=False, index=True)
    timestamp_seconds = Column(Float, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    evidence_strength = Column(Float, nullable=False, default=0.5)
    validation_status = Column(String(20), nullable=False, default="RAW", index=True)
    bounding_box = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    episode_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    video = relationship("VideoModel", back_populates="specialized_observations")



# ---------------------------------------------------------------------------
# Phase 16: Advanced Incident Correlation, Contextual Fusion, & Storylines
# ---------------------------------------------------------------------------

class CorrelatedIncidentModel(Base):
    """
    Phase 16: Correlated security incident combining multi-signal candidates,
    persistent tracks, physical relationships, and deterministic storylines.
    """
    __tablename__ = "correlated_incidents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    incident_category = Column(String(50), nullable=False, index=True)
    incident_subcategory = Column(String(100), nullable=True, index=True)
    start_time = Column(Float, nullable=False, index=True)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False, default=0.0)
    assessment_score = Column(Float, nullable=False, default=0.5)
    evidence_strength = Column(Float, nullable=False, default=0.5)
    reliability_rating = Column(String(20), nullable=False, default="LOW")
    validation_decision = Column(String(30), nullable=False, default="REVIEW_REQUIRED")
    storyline = Column(Text, nullable=False)

    primary_track_ids = Column(JSON, nullable=True)
    supporting_track_ids = Column(JSON, nullable=True)
    involved_object_classes = Column(JSON, nullable=True)
    source_candidate_ids = Column(JSON, nullable=True)
    source_detector_ids = Column(JSON, nullable=True)
    supporting_signal_ids = Column(JSON, nullable=True)
    evidence_ids = Column(JSON, nullable=True)
    zone_ids = Column(JSON, nullable=True)
    negative_evidence = Column(JSON, nullable=True)
    contextual_factors = Column(JSON, nullable=True)
    provenance = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    video = relationship("VideoModel", back_populates="correlated_incidents")


Index("idx_events_video_timestamp", EventModel.video_id, EventModel.timestamp_seconds)
Index("idx_events_video_class", EventModel.video_id, EventModel.object_class)
Index("idx_grouped_video_time", GroupedEventModel.video_id, GroupedEventModel.start_time)
Index("idx_evidence_video_timestamp", EvidenceModel.video_id, EvidenceModel.timestamp_seconds)
Index("idx_tracks_video_time", TrackModel.video_id, TrackModel.first_seen)
Index("idx_vehicle_attr_video_color", VehicleAttributeModel.video_id, VehicleAttributeModel.color)
Index("idx_sec_events_video_type", SecurityEventModel.video_id, SecurityEventModel.event_type)
Index("idx_reports_video_time", ReportModel.video_id, ReportModel.generated_at)
Index("idx_spec_obs_video_time", SpecializedObservationModel.video_id, SpecializedObservationModel.timestamp_seconds)
Index("idx_spec_obs_video_class", SpecializedObservationModel.video_id, SpecializedObservationModel.class_name)
Index("idx_corr_inc_video_time", CorrelatedIncidentModel.video_id, CorrelatedIncidentModel.start_time)
Index("idx_corr_inc_video_cat", CorrelatedIncidentModel.video_id, CorrelatedIncidentModel.incident_category)


# ---------------------------------------------------------------------------
# Phase 17: Advanced Forensic Investigation — Evidence Bundle Persistence
# ---------------------------------------------------------------------------

class InvestigationBundleModel(Base):
    """
    Phase 17: Persistent evidence bundles created by investigators.

    Bundles collect selected incidents, events, tracks, and evidence IDs
    into a reproducible, video-isolated record. Physical evidence files
    are referenced by ID only — never duplicated.
    """
    __tablename__ = "investigation_bundles"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    bundle_name = Column(String(255), nullable=False, default="Investigation Bundle")
    selected_incident_ids = Column(JSON, nullable=True)   # List[str] — CorrelatedIncidentModel IDs
    selected_event_ids = Column(JSON, nullable=True)      # List[str] — SecurityEventModel IDs
    selected_track_ids = Column(JSON, nullable=True)      # List[str] — track_id strings
    selected_evidence_ids = Column(JSON, nullable=True)   # List[str] — EvidenceModel IDs
    storyline_text = Column(Text, nullable=True)          # Investigator-authored summary
    notes = Column(Text, nullable=True)                   # Free-text investigator notes
    provenance = Column(JSON, nullable=True)              # Query provenance record
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    video = relationship("VideoModel")


Index("idx_inv_bundle_video_time", InvestigationBundleModel.video_id, InvestigationBundleModel.created_at)


# ---------------------------------------------------------------------------
# Phase 18: Multi-Camera Tracking & Cross-Camera Scene Intelligence
# ---------------------------------------------------------------------------

class SurveillanceSessionModel(Base):
    """
    Phase 18: A named surveillance session grouping multiple camera sources
    belonging to the same site/operation.

    Sessions provide the top-level boundary for cross-camera reasoning.
    No cross-session data leakage is permitted.
    """
    __tablename__ = "surveillance_sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    site_name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default="active")  # active, archived
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    camera_sources = relationship("CameraSourceModel", back_populates="session", cascade="all, delete-orphan")
    associations = relationship("CrossCameraAssociationModel", back_populates="session", cascade="all, delete-orphan")


class CameraSourceModel(Base):
    """
    Phase 18: One physical CCTV camera source.
    Represents physical sensor hardware with persistent identity (e.g. CAM-NORTH-01).
    Spatial hints are free-text only — no GPS coordinates, no biometric anchors.
    """
    __tablename__ = "camera_sources"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("surveillance_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="SET NULL"), nullable=True, index=True)
    camera_label = Column(String(100), nullable=False, index=True)          # e.g. "CAM-NORTH-01"
    position_hint = Column(String(255), nullable=True)          # e.g. "North Gate"
    field_of_view_hint = Column(String(255), nullable=True)     # e.g. "Facing South"
    adjacency_hints = Column(JSON, nullable=True)               # List[str] — labels of physically adjacent cameras
    status = Column(String(50), nullable=False, default="ACTIVE")
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    session = relationship("SurveillanceSessionModel", back_populates="camera_sources")
    video = relationship("VideoModel", foreign_keys=[video_id])


class CrossCameraAssociationModel(Base):
    """
    Phase 18: Anonymous cross-camera track association hypothesis.

    Records the hypothesis that a tracked object seen in source_camera also
    appears in target_camera, based solely on observable attributes
    (class, color, size-ratio, trajectory direction, temporal gap).

    SAFETY INVARIANTS:
    - evidence_basis may NEVER contain facial features, biometric data, or names.
    - analyst_review_required is ALWAYS True on creation.
    - confidence < 0.40 → no record is created (correct abstention).
    - This is NOT facial recognition. This is NOT person identification.
    - Continuity is an evidence-supported hypothesis, never an identity assertion.
    """
    __tablename__ = "cross_camera_associations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("surveillance_sessions.id", ondelete="CASCADE"), nullable=False, index=True)

    # Source camera (where the track was first observed)
    source_camera_id = Column(String(36), ForeignKey("camera_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    source_video_id = Column(String(36), nullable=False, index=True)   # denormalized for fast queries
    source_track_id = Column(String(64), nullable=False, index=True)
    source_last_seen = Column(Float, nullable=True)                     # timestamp in source video

    # Target camera (where the track is hypothesized to reappear)
    target_camera_id = Column(String(36), ForeignKey("camera_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    target_video_id = Column(String(36), nullable=False, index=True)   # denormalized for fast queries
    target_track_id = Column(String(64), nullable=False, index=True)
    target_first_seen = Column(Float, nullable=True)                    # timestamp in target video

    # Association quality metrics
    confidence = Column(Float, nullable=False)                          # 0.0–1.0, must be >= 0.40
    association_type = Column(String(30), nullable=False, default="POSSIBLE_SAME")
    # SAME_OBJECT (>=0.80), PROBABLE_SAME (0.60-0.79), POSSIBLE_SAME (0.40-0.59)

    attribute_match_score = Column(Float, nullable=True)                # similarity from AttributeMatcher
    trajectory_compatibility_score = Column(Float, nullable=True)       # direction compatibility
    temporal_gap_seconds = Column(Float, nullable=True)                 # gap between source_last_seen + target_first_seen
    temporal_plausibility_score = Column(Float, nullable=True)          # score from TemporalCompatibilityReasoner

    # Evidence provenance — observable attributes only, NO biometrics
    evidence_basis = Column(JSON, nullable=True)                        # List[{signal_type, description, score}]

    # Analyst workflow
    analyst_review_required = Column(Integer, nullable=False, default=1)  # Always 1 on creation
    analyst_verdict = Column(String(20), nullable=False, default="PENDING")  # PENDING, CONFIRMED, REJECTED
    analyst_notes = Column(Text, nullable=True)
    analyst_verdict_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    session = relationship("SurveillanceSessionModel", back_populates="associations")
    source_camera = relationship("CameraSourceModel", foreign_keys=[source_camera_id])
    target_camera = relationship("CameraSourceModel", foreign_keys=[target_camera_id])


Index("idx_session_cameras", CameraSourceModel.session_id)
Index("idx_camera_video", CameraSourceModel.session_id, CameraSourceModel.video_id)
Index("idx_assoc_session", CrossCameraAssociationModel.session_id)
Index("idx_assoc_session_cams", CrossCameraAssociationModel.session_id, CrossCameraAssociationModel.source_camera_id, CrossCameraAssociationModel.target_camera_id)
Index("idx_assoc_session_src_vid", CrossCameraAssociationModel.session_id, CrossCameraAssociationModel.source_video_id)
Index("idx_assoc_src_track", CrossCameraAssociationModel.source_video_id, CrossCameraAssociationModel.source_track_id)
Index("idx_assoc_tgt_track", CrossCameraAssociationModel.target_video_id, CrossCameraAssociationModel.target_track_id)


# ---------------------------------------------------------------------------
# Phase 19: Forensic Case Management & Investigation Workspace Models
# ---------------------------------------------------------------------------

class CaseModel(Base):
    """
    Phase 19: Central Forensic Case record grouping videos, cameras,
    incidents, evidence, bookmarks, notes, and annotations.
    """
    __tablename__ = "cases"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    case_number = Column(String(50), nullable=False, unique=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default="OPEN", index=True)  # OPEN, INVESTIGATING, REVIEW, CLOSED
    priority = Column(String(20), nullable=False, default="MEDIUM", index=True)  # LOW, MEDIUM, HIGH, CRITICAL
    assigned_investigator = Column(String(100), nullable=True)
    tags = Column(JSON, nullable=True)  # List[str]
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    videos = relationship("CaseVideoModel", back_populates="case", cascade="all, delete-orphan")
    cameras = relationship("CaseCameraModel", back_populates="case", cascade="all, delete-orphan")
    incidents = relationship("CaseIncidentModel", back_populates="case", cascade="all, delete-orphan")
    evidence_links = relationship("CaseEvidenceModel", back_populates="case", cascade="all, delete-orphan")
    bookmarks = relationship("CaseBookmarkModel", back_populates="case", cascade="all, delete-orphan")
    notes = relationship("CaseNoteModel", back_populates="case", cascade="all, delete-orphan")
    annotations = relationship("CaseAnnotationModel", back_populates="case", cascade="all, delete-orphan")
    activities = relationship("CaseActivityModel", back_populates="case", cascade="all, delete-orphan")


class CaseVideoModel(Base):
    """Links a processed VideoModel to a Case."""
    __tablename__ = "case_videos"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    notes = Column(Text, nullable=True)
    added_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    case = relationship("CaseModel", back_populates="videos")
    video = relationship("VideoModel")

    __table_args__ = (UniqueConstraint("case_id", "video_id", name="uq_case_video"),)


class CaseCameraModel(Base):
    """Links a CameraSourceModel to a Case with an optional temporal clock offset."""
    __tablename__ = "case_cameras"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    camera_id = Column(String(36), ForeignKey("camera_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    clock_offset_seconds = Column(Float, nullable=False, default=0.0)
    notes = Column(Text, nullable=True)
    added_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    case = relationship("CaseModel", back_populates="cameras")
    camera = relationship("CameraSourceModel")

    __table_args__ = (UniqueConstraint("case_id", "camera_id", name="uq_case_camera"),)


class CaseIncidentModel(Base):
    """Explicitly links a correlated incident or security event to a Case."""
    __tablename__ = "case_incidents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    incident_id = Column(String(64), nullable=False, index=True)
    incident_type = Column(String(50), nullable=False, default="CORRELATED")  # CORRELATED, SECURITY_EVENT
    notes = Column(Text, nullable=True)
    added_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    case = relationship("CaseModel", back_populates="incidents")

    __table_args__ = (UniqueConstraint("case_id", "incident_id", name="uq_case_incident"),)


class CaseEvidenceModel(Base):
    """Explicitly links a preserved EvidenceModel to a Case."""
    __tablename__ = "case_evidence"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_id = Column(String(36), ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False, index=True)
    notes = Column(Text, nullable=True)
    added_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    case = relationship("CaseModel", back_populates="evidence_links")
    evidence = relationship("EvidenceModel")

    __table_args__ = (UniqueConstraint("case_id", "evidence_id", name="uq_case_evidence"),)


class CaseBookmarkModel(Base):
    """Investigator bookmark pinning a critical timestamp and camera within a Case."""
    __tablename__ = "case_bookmarks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    camera_id = Column(String(36), nullable=True, index=True)
    timestamp_seconds = Column(Float, nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    linked_incident_id = Column(String(64), nullable=True, index=True)
    linked_track_id = Column(String(64), nullable=True, index=True)
    linked_evidence_id = Column(String(36), nullable=True, index=True)
    author = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    case = relationship("CaseModel", back_populates="bookmarks")
    video = relationship("VideoModel")


class CaseNoteModel(Base):
    """
    Investigator note associated with a Case, Incident, Track, or Timestamp.
    Strictly classified as ANALYST_NOTE to prevent confusion with machine observations.
    """
    __tablename__ = "case_notes"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    author = Column(String(100), nullable=False, default="Investigator")
    associated_type = Column(String(50), nullable=False, default="CASE")  # CASE, INCIDENT, EVIDENCE, TRACK, TIMESTAMP, CAMERA
    associated_id = Column(String(64), nullable=True, index=True)
    timestamp_seconds = Column(Float, nullable=True)
    note_classification = Column(String(50), nullable=False, default="ANALYST_NOTE")
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    case = relationship("CaseModel", back_populates="notes")


class CaseAnnotationModel(Base):
    """
    Visual overlay annotation on video player.
    SAFETY INVARIANT: Stored strictly as metadata / overlay — NEVER modifies raw CCTV media.
    """
    __tablename__ = "case_annotations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    camera_id = Column(String(36), nullable=True)
    timestamp_seconds = Column(Float, nullable=False, index=True)
    end_timestamp_seconds = Column(Float, nullable=True)
    annotation_type = Column(String(50), nullable=False, default="REGION")  # POINT, REGION, TIMESTAMP_MARKER, TEXT_NOTE, INCIDENT_MARKER
    data = Column(JSON, nullable=False)  # Bounding box coordinates, point, label, color, text
    author = Column(String(100), nullable=False, default="Investigator")
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    case = relationship("CaseModel", back_populates="annotations")
    video = relationship("VideoModel")


class CaseActivityModel(Base):
    """Immutable audit trail for investigation actions taken within a Case."""
    __tablename__ = "case_activities"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    action_type = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    actor = Column(String(100), nullable=False, default="Investigator")
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    case = relationship("CaseModel", back_populates="activities")


Index("idx_case_status_priority", CaseModel.status, CaseModel.priority)
Index("idx_case_bookmarks_time", CaseBookmarkModel.case_id, CaseBookmarkModel.timestamp_seconds)
Index("idx_case_annotations_time", CaseAnnotationModel.case_id, CaseAnnotationModel.timestamp_seconds)
Index("idx_case_activities_time", CaseActivityModel.case_id, CaseActivityModel.created_at)

