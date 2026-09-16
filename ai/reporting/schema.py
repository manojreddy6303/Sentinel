"""
Data schemas for Sentinel Incident Dossier Generation (Phase 9)
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ReportVideoMetadata(BaseModel):
    video_id: str
    original_filename: str
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    status: str = "processed"
    resolution: Optional[str] = None
    codec: Optional[str] = None
    sampling_rate_fps: Optional[float] = 1.0
    analyzed_frames_count: Optional[int] = None
    uploaded_at: Optional[str] = None
    processed_at: Optional[str] = None


class ReportDetectionStats(BaseModel):
    total_detections: int = 0  # Validated detection observations
    total_raw_detections: int = 0
    total_uncertain_detections: int = 0
    total_rejected_detections: int = 0
    class_counts: Dict[str, int] = Field(default_factory=dict)  # Validated observations by class
    class_raw_counts: Dict[str, int] = Field(default_factory=dict)
    class_uncertain_counts: Dict[str, int] = Field(default_factory=dict)
    class_rejected_counts: Dict[str, int] = Field(default_factory=dict)
    class_track_counts: Dict[str, int] = Field(default_factory=dict)
    class_avg_confidences: Dict[str, float] = Field(default_factory=dict)
    total_tracks: int = 0
    validated_tracks: int = 0
    total_security_events: int = 0
    total_evidence_items: int = 0
    total_face_detections: int = 0


class ReportTimelineEvent(BaseModel):
    timestamp_seconds: float
    event_type: str
    object_class: Optional[str] = None
    confidence: Optional[float] = None
    track_id: Optional[str] = None
    description: Optional[str] = None
    evidence_id: Optional[str] = None
    source: str = "detection"  # "grouped_event" | "security_event" | "detection"


class ReportSecurityEvent(BaseModel):
    id: str
    event_type: str
    severity: str = "NORMAL"
    timestamp_seconds: float
    duration_seconds: float = 0.0
    track_id: Optional[str] = None
    object_class: Optional[str] = None
    zone_name: Optional[str] = None
    confidence: float = 0.0
    description: str = ""
    observable_signals: Optional[List[str]] = None
    evidence_id: Optional[str] = None


class ReportEvidenceItem(BaseModel):
    id: str
    evidence_type: str
    timestamp_seconds: float
    object_class: Optional[str] = None
    confidence: Optional[float] = None
    track_id: Optional[str] = None
    local_image_path: Optional[str] = None
    has_image: bool = False
    is_annotated: bool = False
    clip_available: bool = False
    notes: Optional[str] = None


class ReportInvestigationFinding(BaseModel):
    query: str
    finding: str
    timestamp_reference: Optional[str] = None
    confidence: Optional[float] = None


class ReportSpecializedEvent(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: str
    event_type: str
    timestamp_seconds: float
    duration_seconds: float = 0.0
    detector_name: str = ""
    model_name: Optional[str] = None
    evidence_strength: float = 0.0
    validation_status: str = "REVIEW_REQUIRED"
    review_required: bool = True
    description: str = ""
    supporting_observations: List[str] = Field(default_factory=list)
    evidence_id: Optional[str] = None


class ReportCorrelatedIncident(BaseModel):
    incident_id: str
    incident_category: str
    incident_subcategory: Optional[str] = None
    start_time: float
    end_time: float
    duration: float
    primary_track_ids: List[str] = Field(default_factory=list)
    supporting_track_ids: List[str] = Field(default_factory=list)
    involved_object_classes: List[str] = Field(default_factory=list)
    assessment_score: float = 0.0
    evidence_strength: float = 0.0
    reliability_rating: str = "LOW"
    validation_decision: str = "REVIEW_REQUIRED"
    storyline: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    negative_evidence: List[str] = Field(default_factory=list)
    contextual_factors: Dict[str, Any] = Field(default_factory=dict)


class ReportDataPayload(BaseModel):
    report_id: str
    title: str = "SECURITY INCIDENT DOSSIER"
    classification: str = "CONFIDENTIAL // SECURITY ANALYSIS"
    generated_at_iso: str
    video: ReportVideoMetadata
    stats: ReportDetectionStats
    timeline: List[ReportTimelineEvent] = Field(default_factory=list)
    security_events: List[ReportSecurityEvent] = Field(default_factory=list)
    theft_events: List[ReportSecurityEvent] = Field(default_factory=list)
    specialized_events: List[ReportSpecializedEvent] = Field(default_factory=list)
    correlated_incidents: List[ReportCorrelatedIncident] = Field(default_factory=list)
    evidence: List[ReportEvidenceItem] = Field(default_factory=list)
    vehicle_attributes: List[Dict[str, Any]] = Field(default_factory=list)
    anonymous_faces_count: int = 0
    zones: List[Dict[str, Any]] = Field(default_factory=list)
    investigation_findings: List[ReportInvestigationFinding] = Field(default_factory=list)
    # Phase 17: Optional structured forensic investigation context
    forensic_investigation_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Phase 17: Optional structured investigation result to render "
            "as a Forensic Investigation Findings section in the PDF dossier. "
            "All content is grounded in database records only."
        ),
    )
    sha256_hash: Optional[str] = None
    human_verification_notice: str = (
        "MANDATORY HUMAN VERIFICATION NOTICE: All detections, track trajectories, "
        "and behavioral pattern alerts generated by Sentinel represent observational machine "
        "analysis. They must be reviewed and verified by qualified security and investigative "
        "personnel before initiating operational or legal actions. Sentinel does not establish "
        "criminal culpability or individual identity."
    )

