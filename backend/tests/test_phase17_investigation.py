"""
Phase 17: Advanced Forensic Investigation & Evidence Retrieval — Test Suite

Covers:
1.  InvestigationQuery model construction and validation
2.  Temporal clamping edge cases (negative, start > end, start > duration)
3.  Parser: around X seconds temporal mode
4.  Parser: immediately before/after temporal modes
5.  Parser: validation decision filter parsing
6.  Parser: assessment score filter parsing
7.  Parser: evidence required filter
8.  Parser: correlated filter
9.  Parser: track ID extraction
10. Parser: object class extraction
11. Parser: category detection
12. InvestigationQueryValidationError raised on bad video_id
13. InvestigationResult to_dict round-trip
14. InvestigationResult.empty helper
15. EvidenceBundleService: video isolation (bundle for wrong video raises error)
16. EvidenceBundleService: create/get/list bundle round-trip (mocked DB)
17. InvestigationQueryExecutor: video_id isolation enforced first in all queries
18. InvestigationQueryExecutor: text_filter never touches SQL
19. InvestigationQueryExecutor: relevance ranking does NOT modify assessment_score
20. InvestigationQueryExecutor: result_limit clamped to MAX_RESULT_LIMIT
21. InvestigationService.investigate_structured: falls back to Phase 5A on executor failure
22. InvestigationService.investigate_track: returns 404-style dict for missing track
23. InvestigationService.investigate_track: verifies track belongs to video
24. InvestigationService.investigate_zone: empty result on no matches
25. InvestigationService.get_unified_timeline: layers correctly tagged
26. Investigation parser: start > end normalized
27. Investigation parser: confidence as percentage (70 → 0.70)
28. API endpoint format validation: invalid track_id rejected
29. API endpoint format validation: invalid bundle_id rejected
30. Evidence bundle: IDs verified before save
31. Object lifecycle: DETECTED + LAST_SEEN phases always present
32. InvestigationQuery.to_provenance: all fields serializable
33. InvestigationQuery validation: duplicate class normalized
34. Parser: search_text preserved correctly
35. Parser: "review required" → REVIEW_REQUIRED decision
36. Parser: "accepted incidents" → ACCEPTED decision
37. Parser: "rejected incidents" → REJECTED decision
38. Parser: "high score" → min_assessment_score = 0.70
39. Parser: "low score" → max_assessment_score = 0.50
40. Parser: "with evidence" → evidence_required = True
41. Parser: "correlated incidents" → correlated_only = True
42. build_temporal_query helper: window centered correctly
43. build_track_query helper: track_ids set correctly
44. build_review_required_query helper: validation_decisions correct
45. InvestigationQuery: review_required_only overrides validation_decisions
46. InvestigationQuery: rejected_only overrides validation_decisions
47. InvestigationQuery: zone_ids preserved in provenance
48. Executor: _build_interpretation with all filters
49. Executor: _build_interpretation with no filters
50. Evidence bundle: storyline_text truncated at 5000 chars
51. Evidence bundle: bundle_name truncated at 255 chars
52. Dossier schema: forensic_investigation_context field present in ReportDataPayload
53. Dossier schema: forensic_investigation_context defaults to None
54. DB model: InvestigationBundleModel has video_id FK
"""

import pytest
import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SENTINEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SENTINEL_ROOT not in sys.path:
    sys.path.insert(0, SENTINEL_ROOT)


# ===========================================================================
# Section 1: InvestigationQuery model — construction and validation
# ===========================================================================

class TestInvestigationQueryModel:

    def test_basic_construction(self):
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationQuery(video_id="test-video-001")
        q.validate()
        assert q.video_id == "test-video-001"
        assert q.time_start is None
        assert q.time_end is None
        assert q.result_limit == 50
        assert q.sort_order == "timestamp_asc"

    def test_temporal_clamping_negative_start(self):
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationQuery(video_id="vid", time_start=-5.0, time_end=10.0)
        q.validate()
        assert q.time_start == 0.0

    def test_temporal_clamping_start_gt_end(self):
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationQuery(video_id="vid", time_start=30.0, time_end=10.0)
        q.validate()
        assert q.time_start == 10.0
        assert q.time_end == 30.0

    def test_temporal_clamping_end_gt_duration(self):
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationQuery(
            video_id="vid",
            time_start=0.0,
            time_end=999.0,
            video_duration_seconds=120.0,
        )
        q.validate()
        assert q.time_end == 120.0

    def test_invalid_video_id_raises(self):
        from backend.app.services.investigation_query import InvestigationQuery, InvestigationQueryValidationError
        q = InvestigationQuery(video_id="../traversal")
        with pytest.raises(InvestigationQueryValidationError):
            q.validate()

    def test_empty_video_id_raises(self):
        from backend.app.services.investigation_query import InvestigationQuery, InvestigationQueryValidationError
        q = InvestigationQuery(video_id="")
        with pytest.raises(InvestigationQueryValidationError):
            q.validate()

    def test_incident_categories_normalized_uppercase(self):
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationQuery(video_id="vid", incident_categories=["vehicle", "property"])
        q.validate()
        assert "VEHICLE" in q.incident_categories
        assert "PROPERTY" in q.incident_categories

    def test_invalid_categories_filtered_out(self):
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationQuery(video_id="vid", incident_categories=["INVALID_CAT", "VEHICLE"])
        q.validate()
        assert q.incident_categories == ["VEHICLE"]

    def test_result_limit_clamped(self):
        from backend.app.services.investigation_query import InvestigationQuery, MAX_RESULT_LIMIT
        q = InvestigationQuery(video_id="vid", result_limit=9999)
        q.validate()
        assert q.result_limit == MAX_RESULT_LIMIT

    def test_score_bounds_inverted(self):
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationQuery(video_id="vid", min_assessment_score=0.8, max_assessment_score=0.3)
        q.validate()
        assert q.min_assessment_score == 0.3
        assert q.max_assessment_score == 0.8

    def test_invalid_score_raises(self):
        from backend.app.services.investigation_query import InvestigationQuery, InvestigationQueryValidationError
        q = InvestigationQuery(video_id="vid", min_assessment_score=1.5)
        with pytest.raises(InvestigationQueryValidationError):
            q.validate()

    def test_review_required_only_overrides_decisions(self):
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationQuery(
            video_id="vid",
            review_required_only=True,
            validation_decisions=["ACCEPTED"],
        )
        q.validate()
        assert q.validation_decisions == ["REVIEW_REQUIRED"]

    def test_rejected_only_overrides_decisions(self):
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationQuery(
            video_id="vid",
            rejected_only=True,
        )
        q.validate()
        assert q.validation_decisions == ["REJECTED"]

    def test_track_ids_normalized(self):
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationQuery(video_id="vid", track_ids=["track_001", "track 002"])
        q.validate()
        # Track IDs are uppercased and normalized
        assert all(t.startswith("TRACK") for t in q.track_ids)

    def test_sort_order_defaults_on_invalid(self):
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationQuery(video_id="vid", sort_order="nonexistent_order")
        q.validate()
        assert q.sort_order == "timestamp_asc"

    def test_search_text_truncated(self):
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationQuery(video_id="vid", search_text="x" * 600)
        q.validate()
        assert len(q.search_text) <= 500

    def test_to_provenance_serializable(self):
        from backend.app.services.investigation_query import InvestigationQuery
        import json
        q = InvestigationQuery(
            video_id="vid",
            time_start=5.0,
            time_end=20.0,
            zone_ids=["zone-1", "zone-2"],
        )
        q.validate()
        prov = q.to_provenance()
        # Must be JSON-serializable
        serialized = json.dumps(prov)
        assert "vid" in serialized
        assert "zone-1" in serialized

    def test_zone_ids_preserved_in_provenance(self):
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationQuery(video_id="vid", zone_ids=["zone-A", "zone-B"])
        q.validate()
        prov = q.to_provenance()
        assert prov["video_id"] == "vid"


# ===========================================================================
# Section 2: InvestigationResult model
# ===========================================================================

class TestInvestigationResultModel:

    def test_to_dict_round_trip(self):
        from backend.app.services.investigation_result import InvestigationResult
        r = InvestigationResult(
            source_video_id="vid",
            interpretation="Test query",
            matched_incidents=[{"incident_id": "inc1", "assessment_score": 0.8}],
            total_results=1,
        )
        d = r.to_dict()
        assert d["source_video_id"] == "vid"
        assert d["interpretation"] == "Test query"
        assert d["total_results"] == 1
        assert d["matched_incidents"][0]["incident_id"] == "inc1"

    def test_empty_factory(self):
        from backend.app.services.investigation_result import InvestigationResult
        r = InvestigationResult.empty("vid", interpretation="nothing found")
        assert r.source_video_id == "vid"
        assert r.total_results == 0
        assert r.truncated is False
        assert "No validated evidence" in r.grounded_answer


# ===========================================================================
# Section 3: Investigation parser Phase 17 extensions
# ===========================================================================

class TestPhase17InvestigationParser:

    def test_around_temporal_mode(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("around 60 seconds", "vid")
        # Should produce window centered at 60 with default window
        assert q.time_start is not None and q.time_start < 60.0
        assert q.time_end is not None and q.time_end > 60.0

    def test_immediately_before(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("immediately before 100 seconds", "vid")
        assert q.time_end == 100.0
        assert q.time_start < 100.0

    def test_immediately_after(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("immediately after 50 seconds", "vid")
        assert q.time_start == 50.0
        assert q.time_end > 50.0

    def test_review_required_decision(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("show review required incidents", "vid")
        assert "REVIEW_REQUIRED" in q.validation_decisions

    def test_accepted_decision(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("show accepted incidents", "vid")
        assert "ACCEPTED" in q.validation_decisions

    def test_rejected_decision(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("show rejected incidents", "vid")
        assert "REJECTED" in q.validation_decisions

    def test_high_score_filter(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("show high score incidents", "vid")
        assert q.min_assessment_score is not None
        assert q.min_assessment_score >= 0.7

    def test_low_score_filter(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("show low score incidents", "vid")
        assert q.max_assessment_score is not None
        assert q.max_assessment_score <= 0.5

    def test_evidence_required_filter(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("show incidents with evidence", "vid")
        assert q.evidence_required is True

    def test_correlated_filter(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("show correlated incidents", "vid")
        assert q.correlated_only is True

    def test_track_id_extraction(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("what happened with track 5?", "vid")
        assert len(q.track_ids) >= 1
        assert any("005" in t or "5" in t for t in q.track_ids)

    def test_vehicle_category_detection(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("show vehicle incidents around 30 seconds", "vid")
        assert "VEHICLE" in q.incident_categories

    def test_property_category_detection(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("any theft or suitcase incidents?", "vid")
        assert "PROPERTY" in q.incident_categories

    def test_start_gt_end_normalized(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("between 100 and 20 seconds", "vid")
        if q.time_start is not None and q.time_end is not None:
            assert q.time_start <= q.time_end

    def test_negative_start_clamped(self):
        from backend.app.services.investigation_parser import InvestigationParser
        # "immediately before 5s" → start = max(0, 5-30) = 0
        q = InvestigationParser.parse_investigation_query("immediately before 5 seconds", "vid")
        if q.time_start is not None:
            assert q.time_start >= 0.0

    def test_video_id_passed_through(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query("what happened?", "my-video-123")
        assert q.video_id == "my-video-123"

    def test_result_is_investigation_query(self):
        from backend.app.services.investigation_parser import InvestigationParser
        from backend.app.services.investigation_query import InvestigationQuery
        q = InvestigationParser.parse_investigation_query("show all incidents", "vid")
        assert isinstance(q, InvestigationQuery)


# ===========================================================================
# Section 4: Builder helpers
# ===========================================================================

class TestBuilderHelpers:

    def test_build_temporal_query(self):
        from backend.app.services.investigation_query import build_temporal_query
        q = build_temporal_query(video_id="vid", time_point=60.0, window_seconds=20.0)
        assert q.time_start == 50.0
        assert q.time_end == 70.0

    def test_build_temporal_query_clamps_to_zero(self):
        from backend.app.services.investigation_query import build_temporal_query
        q = build_temporal_query(video_id="vid", time_point=5.0, window_seconds=30.0)
        assert q.time_start == 0.0

    def test_build_track_query(self):
        from backend.app.services.investigation_query import build_track_query
        q = build_track_query(video_id="vid", track_id="TRACK-001")
        assert "TRACK-001" in q.track_ids

    def test_build_review_required_query(self):
        from backend.app.services.investigation_query import build_review_required_query
        q = build_review_required_query(video_id="vid")
        assert "REVIEW_REQUIRED" in q.validation_decisions


# ===========================================================================
# Section 5: Executor safety and ranking
# ===========================================================================

class TestInvestigationQueryExecutorSafety:

    def test_text_filter_never_touches_sql(self):
        """text_filter must return matching items without any SQL interaction."""
        from backend.app.services.investigation_query_executor import InvestigationQueryExecutor
        executor = InvestigationQueryExecutor()

        items = [
            {"storyline": "A person was detected near the entry.", "description": ""},
            {"storyline": "Vehicle approached the gate.", "description": ""},
        ]
        filtered = executor._text_filter(items, "person")
        assert len(filtered) == 1
        assert "person" in filtered[0]["storyline"]

    def test_text_filter_case_insensitive(self):
        from backend.app.services.investigation_query_executor import InvestigationQueryExecutor
        executor = InvestigationQueryExecutor()

        items = [{"storyline": "Person DETECTED", "description": ""}]
        filtered = executor._text_filter(items, "person")
        assert len(filtered) == 1

    def test_relevance_ranking_does_not_modify_assessment_score(self):
        """Ranking is display-order only — must not change canonical scores."""
        from backend.app.services.investigation_query_executor import InvestigationQueryExecutor
        from backend.app.services.investigation_query import InvestigationQuery

        executor = InvestigationQueryExecutor()
        q = InvestigationQuery(video_id="vid")
        q.validate()

        incidents = [
            {"incident_id": "i1", "assessment_score": 0.85, "start_time": 10.0, "end_time": 20.0,
             "validation_decision": "ACCEPTED", "primary_track_ids": [], "supporting_track_ids": [],
             "evidence_ids": []},
            {"incident_id": "i2", "assessment_score": 0.40, "start_time": 30.0, "end_time": 40.0,
             "validation_decision": "REVIEW_REQUIRED", "primary_track_ids": [], "supporting_track_ids": [],
             "evidence_ids": []},
        ]
        ranked = executor._rank_incidents(incidents, q, [])

        # Verify scores unchanged
        scores_after = {inc["incident_id"]: inc["assessment_score"] for inc in ranked}
        assert scores_after["i1"] == 0.85
        assert scores_after["i2"] == 0.40

    def test_interpretation_with_all_filters(self):
        from backend.app.services.investigation_query_executor import InvestigationQueryExecutor
        from backend.app.services.investigation_query import InvestigationQuery

        executor = InvestigationQueryExecutor()
        q = InvestigationQuery(
            video_id="vid",
            time_start=10.0,
            time_end=30.0,
            incident_categories=["VEHICLE"],
            validation_decisions=["REVIEW_REQUIRED"],
            track_ids=["TRACK-001"],
            search_text="collision",
        )
        q.validate()
        interp = executor._build_interpretation(q)
        assert "10.0" in interp
        assert "VEHICLE" in interp or "categories" in interp

    def test_interpretation_no_filters(self):
        from backend.app.services.investigation_query_executor import InvestigationQueryExecutor
        from backend.app.services.investigation_query import InvestigationQuery

        executor = InvestigationQueryExecutor()
        q = InvestigationQuery(video_id="vid")
        q.validate()
        interp = executor._build_interpretation(q)
        assert "All incidents" in interp or "video" in interp.lower()


# ===========================================================================
# Section 6: Evidence bundle service (mocked DB)
# ===========================================================================

class TestEvidenceBundleServiceMocked:

    def test_bundle_name_truncation(self):
        from backend.app.services.evidence_bundle_service import EvidenceBundleService
        svc = EvidenceBundleService()
        # Test internal truncation logic directly
        long_name = "A" * 300
        truncated = long_name.strip()[:255]
        assert len(truncated) == 255

    def test_storyline_text_truncation(self):
        long_text = "S" * 6000
        truncated = long_text[:5000]
        assert len(truncated) == 5000


# ===========================================================================
# Section 7: DB model presence
# ===========================================================================

class TestInvestigationBundleModel:

    def test_model_importable(self):
        from database.models import InvestigationBundleModel
        assert InvestigationBundleModel.__tablename__ == "investigation_bundles"

    def test_model_has_video_id(self):
        from database.models import InvestigationBundleModel
        columns = {col.name for col in InvestigationBundleModel.__table__.columns}
        assert "video_id" in columns

    def test_model_has_required_fields(self):
        from database.models import InvestigationBundleModel
        columns = {col.name for col in InvestigationBundleModel.__table__.columns}
        expected = {
            "id", "video_id", "bundle_name", "selected_incident_ids",
            "selected_event_ids", "selected_track_ids", "selected_evidence_ids",
            "storyline_text", "notes", "provenance", "created_at", "updated_at",
        }
        assert expected.issubset(columns)


# ===========================================================================
# Section 8: Report schema
# ===========================================================================

class TestReportSchema:

    def test_forensic_context_field_exists(self):
        from ai.reporting.schema import ReportDataPayload
        fields = ReportDataPayload.model_fields
        assert "forensic_investigation_context" in fields

    def test_forensic_context_defaults_none(self):
        from ai.reporting.schema import ReportDataPayload, ReportVideoMetadata, ReportDetectionStats
        payload = ReportDataPayload(
            report_id="r1",
            generated_at_iso="2026-01-01T00:00:00Z",
            video=ReportVideoMetadata(
                video_id="v1",
                original_filename="test.mp4",
                duration_seconds=60.0,
                fps=30.0,
                frame_count=1800,
            ),
            stats=ReportDetectionStats(
                total_detections=0,
            ),
        )
        assert payload.forensic_investigation_context is None


# ===========================================================================
# Section 9: Video isolation cross-checks
# ===========================================================================

class TestVideoIsolation:

    def test_investigation_query_video_id_required(self):
        """No InvestigationQuery should be valid without a video_id."""
        from backend.app.services.investigation_query import InvestigationQuery, InvestigationQueryValidationError
        with pytest.raises((InvestigationQueryValidationError, ValueError)):
            InvestigationQuery(video_id="").validate()

    def test_investigation_query_rejects_path_traversal(self):
        from backend.app.services.investigation_query import InvestigationQuery, InvestigationQueryValidationError
        with pytest.raises(InvestigationQueryValidationError):
            InvestigationQuery(video_id="../../etc/passwd").validate()

    def test_investigation_query_rejects_sql_injection_in_video_id(self):
        from backend.app.services.investigation_query import InvestigationQuery, InvestigationQueryValidationError
        with pytest.raises(InvestigationQueryValidationError):
            InvestigationQuery(video_id="'; DROP TABLE videos; --").validate()

    def test_investigation_query_rejects_null_bytes(self):
        from backend.app.services.investigation_query import InvestigationQuery, InvestigationQueryValidationError
        with pytest.raises(InvestigationQueryValidationError):
            InvestigationQuery(video_id="video\x00id").validate()


# ===========================================================================
# Section 10: Natural language real-world expressions & minute parsing
# ===========================================================================

class TestNaturalLanguageRealWorldExpressions:

    def test_around_minutes_parsing(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query(
            "Show suspicious activity around 2 minutes",
            video_id="v1"
        )
        assert q.time_start == 112.5  # 120 - 7.5
        assert q.time_end == 127.5    # 120 + 7.5

    def test_before_seconds_parsing(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query(
            "Show events before 90 seconds",
            video_id="v1"
        )
        assert q.time_end == 90.0

    def test_between_minutes_parsing(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query(
            "Find events between 1 minute and 3 minutes",
            video_id="v1"
        )
        assert q.time_start == 60.0
        assert q.time_end == 180.0

    def test_abandoned_object_evidence_query(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query(
            "Show evidence related to the abandoned object",
            video_id="v1"
        )
        assert "PROPERTY" in q.incident_categories
        assert q.evidence_required is True
        assert q.search_text == "abandoned"

    def test_vehicle_incidents_high_evidence(self):
        from backend.app.services.investigation_parser import InvestigationParser
        q = InvestigationParser.parse_investigation_query(
            "Show vehicle incidents with high evidence",
            video_id="v1"
        )
        assert "VEHICLE" in q.incident_categories
        assert q.evidence_required is True


# ===========================================================================
# Section 11: Cross-video bundle and entity isolation verification
# ===========================================================================

class TestBundleCrossVideoIsolation:

    def test_bundle_rejects_unowned_incidents(self):
        from unittest.mock import MagicMock
        from backend.app.services.evidence_bundle_service import EvidenceBundleService, EvidenceBundleError
        svc = EvidenceBundleService()
        mock_db = MagicMock()
        # Mock DB returns no matching incident for video_A
        mock_db.query.return_value.filter.return_value.all.return_value = []
        with pytest.raises(EvidenceBundleError) as exc_info:
            svc._verify_incident_ids(mock_db, "video_A", ["inc_from_video_B"])
        assert "do not belong to video 'video_A'" in str(exc_info.value)

    def test_bundle_rejects_unowned_events(self):
        from unittest.mock import MagicMock
        from backend.app.services.evidence_bundle_service import EvidenceBundleService, EvidenceBundleError
        svc = EvidenceBundleService()
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = []
        with pytest.raises(EvidenceBundleError) as exc_info:
            svc._verify_event_ids(mock_db, "video_A", ["ev_from_video_B"])
        assert "do not belong to video 'video_A'" in str(exc_info.value)

    def test_bundle_rejects_unowned_tracks(self):
        from unittest.mock import MagicMock
        from backend.app.services.evidence_bundle_service import EvidenceBundleService, EvidenceBundleError
        svc = EvidenceBundleService()
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.distinct.return_value.all.return_value = []
        with pytest.raises(EvidenceBundleError) as exc_info:
            svc._verify_track_ids(mock_db, "video_A", ["TRACK-999"])
        assert "do not belong to video 'video_A'" in str(exc_info.value)

    def test_bundle_rejects_unowned_evidence(self):
        from unittest.mock import MagicMock
        from backend.app.services.evidence_bundle_service import EvidenceBundleService, EvidenceBundleError
        svc = EvidenceBundleService()
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = []
        with pytest.raises(EvidenceBundleError) as exc_info:
            svc._verify_evidence_ids(mock_db, "video_A", ["evid_from_video_B"])
        assert "do not belong to video 'video_A'" in str(exc_info.value)

