"""
Sentinel Phase 7 Automated Test Suite
LLM-Assisted, Evidence-Grounded Video Investigation Platform

Covers:
1. LLM provider configuration & factory
2. Missing API key fallback
3. Structured LLM response validation
4. Invalid structured response rejection
5. Object-class validation and synonym mapping
6. Timestamp boundary validation
7. Confidence validation and percentage normalization
8. Grounded response generation
9. Empty database result handling
10. Source/event traceability
11. Evidence reference integration
12. Unsupported identity question (guardrail)
13. Criminal / threat attribution (guardrail)
14. Vehicle colour hallucination prevention
15. AI provider failure fallback
16. Intelligent Video Summary capability
17. Observational Activity Analysis capability
18. API endpoint: POST /api/videos/{video_id}/ai-investigate
19. Existing Phase 5A investigation endpoint regression check
20. Existing Phase 6 evidence endpoint regression check
"""

import os
import sys
import uuid
import json
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from database.session import init_db, SessionLocal
from database.models import VideoModel, EventModel, GroupedEventModel, EvidenceModel
from ai.investigation.provider import (
    LLMProvider,
    GeminiProvider,
    MockLLMProvider,
    get_llm_provider,
    LLMProviderError,
)
from ai.investigation.validator import StructuredIntentValidator, ValidationError
from ai.investigation.orchestrator import InvestigationOrchestrator

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test Fixture: Seeded Video with Events, Detections, and Evidence
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seeded_video_id():
    """Seeds a test video with known detections, grouped events, and evidence."""
    init_db()
    video_id = f"test_p7_{uuid.uuid4().hex[:10]}"

    # Persist metadata sidecar
    meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
    meta = {
        "video_id": video_id,
        "filename": "surveillance_phase7.mp4",
        "saved_filename": f"{video_id}_surveillance_phase7.mp4",
        "storage_path": str(settings.STORAGE_UPLOADS_DIR / f"{video_id}_surveillance_phase7.mp4"),
        "file_size_bytes": 1048576,
        "duration_seconds": 20.0,
        "fps": 30.0,
        "frames_processed": 20,
        "detections_count": 8,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    db = SessionLocal()
    try:
        vid = VideoModel(
            id=video_id,
            original_filename="surveillance_phase7.mp4",
            saved_filename=f"{video_id}_surveillance_phase7.mp4",
            storage_path=str(settings.STORAGE_UPLOADS_DIR / f"{video_id}_surveillance_phase7.mp4"),
            file_size_bytes=1048576,
            duration_seconds=20.0,
            fps=30.0,
            frame_count=20,
            status="processed",
        )
        db.add(vid)

        # 4 person detections (around 2-3s)
        for i in range(4):
            db.add(
                EventModel(
                    video_id=video_id,
                    object_class="person",
                    class_id=0,
                    timestamp_seconds=2.0 + (i * 0.3),
                    confidence=0.85,
                    bbox_x1=100.0,
                    bbox_y1=150.0,
                    bbox_x2=200.0,
                    bbox_y2=350.0,
                    frame_number=i * 10,
                )
            )

        # 4 car detections (around 8-9s)
        car_event_id = None
        for i in range(4):
            det = EventModel(
                video_id=video_id,
                object_class="car",
                class_id=2,
                timestamp_seconds=8.0 + (i * 0.25),
                confidence=0.81,
                bbox_x1=300.0,
                bbox_y1=200.0,
                bbox_x2=500.0,
                bbox_y2=400.0,
                frame_number=80 + (i * 10),
            )
            db.add(det)
            if i == 0:
                db.flush()
                car_event_id = det.id

        # 2 Grouped Events
        db.add(
            GroupedEventModel(
                video_id=video_id,
                event_type="PERSON_DETECTED",
                start_time=2.0,
                end_time=3.0,
                duration_seconds=1.0,
                objects_summary=[{"class": "person", "count": 4}],
                total_detections=4,
                max_confidence=0.85,
                priority="NORMAL",
            )
        )
        db.add(
            GroupedEventModel(
                video_id=video_id,
                event_type="VEHICLE_DETECTED",
                start_time=8.0,
                end_time=9.0,
                duration_seconds=1.0,
                objects_summary=[{"class": "car", "count": 4}],
                total_detections=4,
                max_confidence=0.81,
                priority="HIGH",
            )
        )

        # 1 Evidence record around 8.0s
        db.add(
            EvidenceModel(
                id=f"ev_p7_{uuid.uuid4().hex[:8]}",
                video_id=video_id,
                event_id=car_event_id,
                evidence_type="snapshot_and_clip",
                timestamp_seconds=8.0,
                source_video_name="surveillance_phase7.mp4",
                snapshot_path=str(settings.STORAGE_EVIDENCE_DIR / "dummy_snapshot.jpg"),
                annotated_snapshot_path=str(settings.STORAGE_EVIDENCE_DIR / "dummy_annotated.jpg"),
                clip_path=str(settings.STORAGE_EVIDENCE_DIR / "dummy_clip.mp4"),
                object_class="car",
                confidence=0.81,
                start_time=5.0,
                end_time=11.0,
                duration_seconds=6.0,
            )
        )
        db.commit()
    finally:
        db.close()

    yield video_id

    # Cleanup sidecar file
    if meta_path.exists():
        meta_path.unlink()


# ---------------------------------------------------------------------------
# Tests 1 - 7: Provider Architecture & Structured Validation
# ---------------------------------------------------------------------------

def test_1_llm_provider_configuration():
    """1. Test LLM provider configuration and factory instantiation."""
    mock_prov = get_llm_provider("mock")
    assert isinstance(mock_prov, MockLLMProvider)
    assert mock_prov.is_available() is True

    gemini_prov = get_llm_provider("gemini", api_key="test_key", model="gemini-1.5-pro")
    assert isinstance(gemini_prov, GeminiProvider)
    assert gemini_prov.api_key == "test_key"
    assert gemini_prov.model == "gemini-1.5-pro"
    assert gemini_prov.is_available() is True


def test_2_missing_api_key_fallback():
    """2. Test graceful fallback behavior when API key is not configured."""
    empty_prov = GeminiProvider(api_key="")
    assert empty_prov.is_available() is False

    # Orchestrator uses fallback mode without crashing
    orch = InvestigationOrchestrator(provider=empty_prov)
    res = orch.process_investigation("any_id", "Show cars between 8 and 12 seconds")
    assert res["mode"] == "deterministic_fallback"
    assert res["is_supported"] is True


def test_3_structured_llm_response_validation():
    """3. Test structured intent validation for valid inputs."""
    raw = {
        "intent": "investigate",
        "object_classes": ["car"],
        "start_time": 8.0,
        "end_time": 12.0,
        "min_confidence": 0.75,
        "result_type": "detections",
    }
    sanitized = StructuredIntentValidator.validate_and_sanitize(raw)
    assert sanitized["is_supported"] is True
    assert sanitized["object_class"] == "car"
    assert sanitized["start_time"] == 8.0
    assert sanitized["end_time"] == 12.0
    assert sanitized["min_confidence"] == 0.75
    assert sanitized["result_type"] == "detections"


def test_4_invalid_structured_response_rejection():
    """4. Reject invalid non-dict or malformed parameters."""
    with pytest.raises(ValidationError):
        StructuredIntentValidator.validate_and_sanitize("not a dict")  # type: ignore

    with pytest.raises(ValidationError):
        StructuredIntentValidator.validate_and_sanitize({"object_classes": 12345})  # type: ignore


def test_5_object_class_validation_and_synonyms():
    """5. Verify object-class validation and synonym mapping."""
    # "pedestrians" -> "person"
    res1 = StructuredIntentValidator.validate_and_sanitize({"object_classes": ["pedestrians"]})
    assert res1["object_class"] == "person"

    # "automobile" -> "car"
    res2 = StructuredIntentValidator.validate_and_sanitize({"object_classes": ["automobile"]})
    assert res2["object_class"] == "car"

    # "vehicles" -> "vehicle_group"
    res3 = StructuredIntentValidator.validate_and_sanitize({"object_classes": ["vehicles"]})
    assert res3["object_class"] == "vehicle_group"

    # Unsupported object class -> None (filtered)
    res4 = StructuredIntentValidator.validate_and_sanitize({"object_classes": ["submarine"]})
    assert res4["object_class"] is None


def test_6_timestamp_validation():
    """6. Validate timestamp boundaries and negative rejection."""
    with pytest.raises(ValidationError):
        StructuredIntentValidator.validate_and_sanitize({"start_time": -5.0})

    # Auto-swap inverted timestamps (start 12 > end 8)
    res = StructuredIntentValidator.validate_and_sanitize({"start_time": 12.0, "end_time": 8.0})
    assert res["start_time"] == 8.0
    assert res["end_time"] == 12.0


def test_7_confidence_validation():
    """7. Validate confidence range [0.0, 1.0] and percentage normalization."""
    # Percentage 85 -> 0.85
    res = StructuredIntentValidator.validate_and_sanitize({"min_confidence": 85})
    assert res["min_confidence"] == 0.85

    # Out of range (150) -> raises ValidationError
    with pytest.raises(ValidationError):
        StructuredIntentValidator.validate_and_sanitize({"min_confidence": 150})


# ---------------------------------------------------------------------------
# Tests 8 - 15: Grounded Retrieval & Guardrails
# ---------------------------------------------------------------------------

def test_8_grounded_response_generation(seeded_video_id):
    """8. Grounded synthesis strictly cites database records and timestamps."""
    provider = MockLLMProvider()
    orch = InvestigationOrchestrator(provider=provider)

    res = orch.process_investigation(seeded_video_id, "What happened around 8 seconds?")
    assert res["mode"] == "ai_assisted"
    assert res["is_supported"] is True
    assert "car" in res["answer"].lower()
    assert "8.0" in res["answer"]
    assert res["count"] > 0


def test_9_empty_database_result_handling(seeded_video_id):
    """9. Empty database results return clear zero-hallucination message."""
    provider = MockLLMProvider()
    orch = InvestigationOrchestrator(provider=provider)

    # Inquire about 15-18 seconds (where no detections were seeded)
    res = orch.process_investigation(seeded_video_id, "Show cars between 15 and 18 seconds")
    assert res["is_supported"] is True
    assert res["count"] == 0
    assert "no matching sentinel data was found" in res["answer"].lower()


def test_10_source_and_event_traceability(seeded_video_id):
    """10. Verify structured sources expose detection and event IDs with bounding boxes."""
    provider = MockLLMProvider()
    orch = InvestigationOrchestrator(provider=provider)

    res = orch.process_investigation(seeded_video_id, "Show cars between 8 and 10 seconds")
    sources = res["sources"]
    assert "detections" in sources
    assert len(sources["detections"]) > 0
    first_det = sources["detections"][0]
    assert "event_id" in first_det
    assert "timestamp" in first_det
    assert "bounding_box" in first_det
    assert first_det["object_class"] == "car"


def test_11_evidence_reference_integration(seeded_video_id):
    """11. Evidence records correlating to queried time window are included."""
    provider = MockLLMProvider()
    orch = InvestigationOrchestrator(provider=provider)

    res = orch.process_investigation(seeded_video_id, "What happened around 8 seconds?")
    evidence_list = res["sources"]["evidence"]
    assert len(evidence_list) >= 1
    assert evidence_list[0]["timestamp"] == 8.0
    assert evidence_list[0]["has_snapshot"] is True
    assert evidence_list[0]["has_clip"] is True


def test_12_unsupported_identity_guardrail(seeded_video_id):
    """12. Rejects facial recognition / identity inquiries."""
    orch = InvestigationOrchestrator(provider=MockLLMProvider())

    for q in ["Who is this person?", "What is his name?", "Identify the man"]:
        res = orch.process_investigation(seeded_video_id, q)
        assert res["is_supported"] is False
        assert "facial recognition" in res["answer"].lower() or "identity" in res["answer"].lower()


def test_13_criminal_attribution_guardrail(seeded_video_id):
    """13. Prevents criminal attribution or suspect labeling."""
    orch = InvestigationOrchestrator(provider=MockLLMProvider())

    res = orch.process_investigation(seeded_video_id, "Who is the criminal?")
    assert res["is_supported"] is False
    assert "criminal culpability" in res["answer"].lower() or "observational" in res["answer"].lower()


def test_14_vehicle_colour_hallucination_prevention():
    """14. Vehicle color adjectives are stripped to prevent hallucination."""
    res = StructuredIntentValidator.validate_and_sanitize({
        "object_classes": ["red car", "blue truck"]
    })
    # Must be "car", NOT "red car"
    assert res["object_class"] == "car"


def test_15_ai_provider_failure_fallback(seeded_video_id):
    """15. Graceful fallback when provider raises LLMProviderError."""
    failing_provider = MockLLMProvider(should_fail=True, failure_message="Connection timed out")
    orch = InvestigationOrchestrator(provider=failing_provider)

    res = orch.process_investigation(seeded_video_id, "Show cars between 8 and 12 seconds")
    # System must not crash, must return deterministic response
    assert res["mode"] == "deterministic_fallback"
    assert res["is_supported"] is True
    assert res["count"] > 0


# ---------------------------------------------------------------------------
# Tests 16 - 20: Video Summary, Activity Analysis & API Endpoints
# ---------------------------------------------------------------------------

def test_16_video_summary_capability(seeded_video_id):
    """16. Intelligent Video Summary aggregates actual database metrics."""
    orch = InvestigationOrchestrator(provider=MockLLMProvider())

    res = orch.process_investigation(seeded_video_id, "Summarize this video")
    assert res["is_supported"] is True
    assert "VIDEO ACTIVITY SUMMARY" in res["answer"]
    assert "person" in res["answer"].lower()
    assert "car" in res["answer"].lower()
    assert res.get("summary") is not None
    assert res["summary"]["total_detections"] == 8


def test_17_activity_analysis_capability(seeded_video_id):
    """17. Observational activity analysis identifies clustering without criminal labeling."""
    orch = InvestigationOrchestrator(provider=MockLLMProvider())

    res = orch.process_investigation(seeded_video_id, "Was there any noteworthy activity?")
    assert res["is_supported"] is True
    assert "noteworthy activity" in res["answer"].lower()
    assert "does not establish criminal" in res["answer"].lower()


def test_18_api_ai_investigate_endpoint(seeded_video_id):
    """18. API POST /api/videos/{video_id}/ai-investigate returns HTTP 200 with structured data."""
    # Test valid query
    resp = client.post(
        f"/api/videos/{seeded_video_id}/ai-investigate",
        json={"query": "What happened around 8 seconds?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["video_id"] == seeded_video_id
    assert "answer" in data
    assert "sources" in data
    assert "limitations" in data

    # Test invalid video_id (path traversal)
    bad_resp = client.post(
        "/api/videos/../../etc/passwd/ai-investigate",
        json={"query": "test query"},
    )
    assert bad_resp.status_code in [400, 404]


def test_19_existing_phase5a_compatibility(seeded_video_id):
    """19. Existing Phase 5A POST /api/videos/{video_id}/investigate remains functional."""
    resp = client.post(
        f"/api/videos/{seeded_video_id}/investigate",
        json={"query": "Show cars between 8 and 12 seconds"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["result_type"] == "detections"
    assert data["count"] == 4


def test_20_existing_phase6_evidence_compatibility(seeded_video_id):
    """20. Existing Phase 6 evidence listing endpoint remains functional."""
    resp = client.get(f"/api/videos/{seeded_video_id}/evidence")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_evidence"] >= 1
    assert data["evidence"][0]["video_id"] == seeded_video_id
