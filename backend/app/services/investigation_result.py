"""
Phase 17: Canonical Structured Investigation Result Model

Provides a typed result representation returned by all Phase 17 investigation
operations. Every result is video-scoped and traceable to source DB records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InvestigationResult:
    """
    Canonical structured result for a Phase 17 forensic investigation query.

    All records are video-scoped — source_video_id matches the query video_id.
    Every factual field traces back to a canonical DB record (incident ID,
    event ID, track ID, evidence ID).

    Fields
    ------
    query : dict
        Provenance copy of the InvestigationQuery that produced this result.
    interpretation : str
        Human-readable interpretation of what was searched.
    matched_incidents : list of dict
        CorrelatedIncidentModel records matching the query.
    matched_events : list of dict
        SecurityEventModel records matching the query.
    matched_tracks : list of dict
        TrackModel records matching the query.
    matched_evidence : list of dict
        EvidenceModel records matching the query.
    timeline : list of dict
        Unified chronological timeline merging events + incidents + evidence.
    relationships : list of dict
        Inter-object or inter-track relationships observed.
    negative_evidence : list of dict
        Recorded absence of expected signals (from correlated incidents).
    total_results : int
        Total number of records matched before pagination.
    truncated : bool
        True if results were truncated by result_limit.
    source_video_id : str
        Canonical video ID — always matches query.video_id.
    provenance : dict
        Query provenance record for audit trail.
    grounded_answer : str
        Human-readable answer generated from retrieved records only.
    diagnostics : dict
        Performance and diagnostic metrics (timing, fallback_used, etc.).
    """

    query: Dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    matched_incidents: List[Dict[str, Any]] = field(default_factory=list)
    matched_events: List[Dict[str, Any]] = field(default_factory=list)
    matched_tracks: List[Dict[str, Any]] = field(default_factory=list)
    matched_evidence: List[Dict[str, Any]] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    negative_evidence: List[Dict[str, Any]] = field(default_factory=list)
    total_results: int = 0
    truncated: bool = False
    source_video_id: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    grounded_answer: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary for API responses."""
        return {
            "query": self.query,
            "interpretation": self.interpretation,
            "matched_incidents": self.matched_incidents,
            "matched_events": self.matched_events,
            "matched_tracks": self.matched_tracks,
            "matched_evidence": self.matched_evidence,
            "timeline": self.timeline,
            "relationships": self.relationships,
            "negative_evidence": self.negative_evidence,
            "total_results": self.total_results,
            "truncated": self.truncated,
            "source_video_id": self.source_video_id,
            "provenance": self.provenance,
            "grounded_answer": self.grounded_answer,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def empty(
        cls,
        video_id: str,
        interpretation: str = "",
        query: Optional[Dict[str, Any]] = None,
        grounded_answer: str = "No validated evidence was found for that query.",
    ) -> "InvestigationResult":
        """Return an empty result (no matches) for graceful no-result handling."""
        return cls(
            source_video_id=video_id,
            interpretation=interpretation,
            query=query or {},
            grounded_answer=grounded_answer,
            total_results=0,
            truncated=False,
        )
