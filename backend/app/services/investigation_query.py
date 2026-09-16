"""
Phase 17: Canonical Structured Investigation Query Model

Provides a typed, validated query representation for all Phase 17 forensic
investigation operations. Every filter is optional except video_id (enforced
for absolute video isolation). Timestamps are clamped safely.

SECURITY: video_id is mandatory and enforced at the model level.
No query may cross video boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_VALIDATION_DECISIONS = frozenset(
    ["ACCEPTED", "REVIEW_REQUIRED", "REJECTED", "ABSTAINED", "SUPERSEDED"]
)

VALID_RELIABILITY_LEVELS = frozenset(["HIGH", "MEDIUM", "LOW"])

VALID_INCIDENT_CATEGORIES = frozenset(
    ["VEHICLE", "PROPERTY", "PERSON", "CROWD", "ZONE", "SPECIALIZED", "ENVIRONMENT"]
)

VALID_SORT_ORDERS = frozenset(
    ["timestamp_asc", "timestamp_desc", "score_desc", "score_asc", "reliability_desc"]
)

MAX_RESULT_LIMIT = 200
DEFAULT_RESULT_LIMIT = 50

# Window (seconds) for "around X seconds" temporal queries
DEFAULT_AROUND_WINDOW_SECONDS = 15.0


class InvestigationQueryValidationError(ValueError):
    """Raised when an InvestigationQuery parameter fails validation."""
    pass


# ---------------------------------------------------------------------------
# Core Query Model
# ---------------------------------------------------------------------------

@dataclass
class InvestigationQuery:
    """
    Canonical structured representation of a forensic investigation query.

    All filters are optional except video_id.
    video_id enforces absolute video isolation — no query may cross boundaries.

    Fields
    ------
    video_id : str
        Required. UUID of the video being investigated.
    time_start : float, optional
        Start of temporal window (seconds, inclusive, ≥ 0).
    time_end : float, optional
        End of temporal window (seconds, inclusive, ≥ time_start).
    incident_categories : list of str, optional
        Filter by incident category (VEHICLE, PROPERTY, PERSON, CROWD, ZONE,
        SPECIALIZED, ENVIRONMENT). Case-insensitive.
    validation_decisions : list of str, optional
        Filter by validation decision. May contain ACCEPTED, REVIEW_REQUIRED,
        REJECTED, ABSTAINED, SUPERSEDED.
    min_assessment_score : float [0.0, 1.0], optional
    max_assessment_score : float [0.0, 1.0], optional
    reliability_levels : list of str, optional
        HIGH, MEDIUM, LOW.
    track_ids : list of str, optional
        Filter to specific anonymous track IDs within this video only.
    object_classes : list of str, optional
        Filter by YOLO object class (person, car, suitcase, …).
    relationship_types : list of str, optional
        Filter by inter-object relationship type.
    zone_ids : list of str, optional
        Filter to incidents/events within specific zone IDs.
    evidence_required : bool, optional
        If True, only return items with linked evidence.
    correlated_only : bool, optional
        If True, only return correlated (multi-signal fused) incidents.
    review_required_only : bool, optional
        Shorthand: filter validation_decisions to REVIEW_REQUIRED only.
    rejected_only : bool, optional
        Shorthand: filter validation_decisions to REJECTED only.
    search_text : str, optional
        Free-text substring match against storyline/description fields.
        Never passed directly to SQL — matched via Python-level filtering.
    sort_order : str, optional
        One of timestamp_asc, timestamp_desc, score_desc, score_asc,
        reliability_desc. Default: timestamp_asc.
    result_limit : int, optional
        Maximum results returned. Clamped to MAX_RESULT_LIMIT (200).
    result_offset : int, optional
        Pagination offset.
    video_duration_seconds : float, optional
        If known, used to clamp time_end safely. Never changes scores.
    """

    video_id: str

    # Temporal window
    time_start: Optional[float] = None
    time_end: Optional[float] = None

    # Incident filters
    incident_categories: List[str] = field(default_factory=list)
    validation_decisions: List[str] = field(default_factory=list)
    min_assessment_score: Optional[float] = None
    max_assessment_score: Optional[float] = None
    reliability_levels: List[str] = field(default_factory=list)

    # Track / object filters
    track_ids: List[str] = field(default_factory=list)
    object_classes: List[str] = field(default_factory=list)
    relationship_types: List[str] = field(default_factory=list)

    # Zone filters
    zone_ids: List[str] = field(default_factory=list)

    # Evidence / correlation flags
    evidence_required: Optional[bool] = None
    correlated_only: Optional[bool] = None
    review_required_only: bool = False
    rejected_only: bool = False

    # Text search (Python-level, never SQL)
    search_text: Optional[str] = None

    # Pagination / ordering
    sort_order: str = "timestamp_asc"
    result_limit: int = DEFAULT_RESULT_LIMIT
    result_offset: int = 0

    # Context for clamping
    video_duration_seconds: Optional[float] = None

    # Internal: raw query for provenance
    raw_query_text: str = ""

    def validate(self) -> "InvestigationQuery":
        """
        Validate all fields in-place and return self.
        Raises InvestigationQueryValidationError on invalid input.
        This method is safe to call multiple times.
        """
        # 1. video_id: required, alphanumeric + hyphen/underscore only (no traversal)
        if not self.video_id or not re.match(r"^[a-zA-Z0-9_\-]+$", self.video_id):
            raise InvestigationQueryValidationError(
                "video_id is required and must contain only alphanumeric characters, hyphens, or underscores."
            )

        # 2. Temporal bounds
        if self.time_start is not None:
            self.time_start = float(self.time_start)
            if self.time_start < 0.0:
                self.time_start = 0.0  # Clamp negative timestamps safely

        if self.time_end is not None:
            self.time_end = float(self.time_end)
            if self.time_end < 0.0:
                self.time_end = 0.0

            # Clamp to video duration if known
            if self.video_duration_seconds is not None and self.time_end > self.video_duration_seconds:
                self.time_end = self.video_duration_seconds

        # 3. Normalize start <= end
        if self.time_start is not None and self.time_end is not None:
            if self.time_start > self.time_end:
                self.time_start, self.time_end = self.time_end, self.time_start

        # 4. Incident categories
        self.incident_categories = [
            c.upper() for c in self.incident_categories
            if isinstance(c, str) and c.upper() in VALID_INCIDENT_CATEGORIES
        ]

        # 5. Validation decisions — apply shorthands
        if self.review_required_only:
            self.validation_decisions = ["REVIEW_REQUIRED"]
        elif self.rejected_only:
            self.validation_decisions = ["REJECTED"]
        else:
            self.validation_decisions = [
                d.upper() for d in self.validation_decisions
                if isinstance(d, str) and d.upper() in VALID_VALIDATION_DECISIONS
            ]

        # 6. Assessment score bounds
        if self.min_assessment_score is not None:
            self.min_assessment_score = float(self.min_assessment_score)
            if not (0.0 <= self.min_assessment_score <= 1.0):
                raise InvestigationQueryValidationError(
                    f"min_assessment_score must be between 0.0 and 1.0 (got {self.min_assessment_score})"
                )

        if self.max_assessment_score is not None:
            self.max_assessment_score = float(self.max_assessment_score)
            if not (0.0 <= self.max_assessment_score <= 1.0):
                raise InvestigationQueryValidationError(
                    f"max_assessment_score must be between 0.0 and 1.0 (got {self.max_assessment_score})"
                )

        if (
            self.min_assessment_score is not None
            and self.max_assessment_score is not None
            and self.min_assessment_score > self.max_assessment_score
        ):
            self.min_assessment_score, self.max_assessment_score = (
                self.max_assessment_score, self.min_assessment_score,
            )

        # 7. Reliability levels
        self.reliability_levels = [
            r.upper() for r in self.reliability_levels
            if isinstance(r, str) and r.upper() in VALID_RELIABILITY_LEVELS
        ]

        # 8. Track IDs — normalize format TRACK-NNN, keep video-scoped
        normalized_tracks = []
        for t in self.track_ids:
            if not isinstance(t, str):
                continue
            t_norm = t.strip().upper().replace(" ", "-").replace("_", "-")
            # Allow TRACK-NNN or arbitrary track references
            if t_norm:
                normalized_tracks.append(t_norm)
        self.track_ids = normalized_tracks

        # 9. Object classes — lowercase, strip
        self.object_classes = [
            c.strip().lower() for c in self.object_classes
            if isinstance(c, str) and c.strip()
        ]

        # 10. Sort order
        if self.sort_order not in VALID_SORT_ORDERS:
            self.sort_order = "timestamp_asc"

        # 11. Result limit — clamp
        self.result_limit = max(1, min(int(self.result_limit), MAX_RESULT_LIMIT))

        # 12. Offset
        self.result_offset = max(0, int(self.result_offset))

        # 13. Search text — strip, truncate to 500 chars
        if self.search_text:
            self.search_text = self.search_text.strip()[:500] or None

        return self

    def to_provenance(self) -> dict:
        """Return a serializable provenance record for this query."""
        return {
            "video_id": self.video_id,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "incident_categories": self.incident_categories,
            "validation_decisions": self.validation_decisions,
            "min_assessment_score": self.min_assessment_score,
            "max_assessment_score": self.max_assessment_score,
            "reliability_levels": self.reliability_levels,
            "track_ids": self.track_ids,
            "object_classes": self.object_classes,
            "zone_ids": self.zone_ids,
            "evidence_required": self.evidence_required,
            "correlated_only": self.correlated_only,
            "sort_order": self.sort_order,
            "result_limit": self.result_limit,
            "result_offset": self.result_offset,
            "raw_query_text": self.raw_query_text,
        }


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def build_temporal_query(
    video_id: str,
    time_point: float,
    window_seconds: float = DEFAULT_AROUND_WINDOW_SECONDS,
    video_duration_seconds: Optional[float] = None,
    raw_query_text: str = "",
) -> InvestigationQuery:
    """
    Build an InvestigationQuery centered around a specific time point.
    The window is clamped to [0, video_duration].
    """
    half = window_seconds / 2.0
    start = max(0.0, time_point - half)
    end = time_point + half
    q = InvestigationQuery(
        video_id=video_id,
        time_start=start,
        time_end=end,
        video_duration_seconds=video_duration_seconds,
        raw_query_text=raw_query_text,
    )
    return q.validate()


def build_track_query(
    video_id: str,
    track_id: str,
    raw_query_text: str = "",
) -> InvestigationQuery:
    """Build an InvestigationQuery scoped to a single anonymous track."""
    q = InvestigationQuery(
        video_id=video_id,
        track_ids=[track_id],
        raw_query_text=raw_query_text,
    )
    return q.validate()


def build_review_required_query(
    video_id: str,
    raw_query_text: str = "",
) -> InvestigationQuery:
    """Build a query returning only REVIEW_REQUIRED incidents."""
    q = InvestigationQuery(
        video_id=video_id,
        review_required_only=True,
        raw_query_text=raw_query_text,
    )
    return q.validate()
