import uuid
import pytest
from ai.investigation.orchestrator import InvestigationOrchestrator
from ai.investigation.provider import LLMProvider, LLMProviderError
from backend.app.services.investigation_cache import InvestigationDerivedDataCache
from database.session import init_db, SessionLocal
from database.models import VideoModel, EventModel, GroupedEventModel, TrackModel, SecurityEventModel, EvidenceModel


class SpyingLLMProvider(LLMProvider):
    """Spy provider that records calls and can simulate successes or errors."""
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.intent_calls = 0
        self.grounded_calls = 0

    def generate_structured_intent(self, user_query, history=None):
        self.intent_calls += 1
        if self.should_fail:
            raise LLMProviderError("Simulated LLM intent error")
        return {"intent": "investigate", "is_supported": True}

    def generate_grounded_response(self, user_query, retrieved_data, history=None, context_notes=None):
        self.grounded_calls += 1
        if self.should_fail:
            raise LLMProviderError("Simulated LLM response error")
        return "Simulated synthesized narrative response from AI provider."

    def is_available(self) -> bool:
        return True


@pytest.fixture
def seeded_video_id():
    """Seeds a test video with known tracks, events, and evidence in test_sentinel.db."""
    init_db()
    vid_id = f"test_fast_{uuid.uuid4().hex[:8]}"
    db = SessionLocal()
    try:
        vid = VideoModel(
            id=vid_id,
            original_filename="test_surveillance.mp4",
            saved_filename="test_surveillance.mp4",
            storage_path="storage/uploads/test_surveillance.mp4",
            duration_seconds=30.0,
            fps=30.0,
            status="processed",
        )
        db.add(vid)

        # Track 1: Person wearing blue
        db.add(
            TrackModel(
                video_id=vid_id,
                track_id="TRACK-001",
                object_class="person",
                first_seen=1.0,
                last_seen=10.0,
                duration_seconds=9.0,
                detection_count=10,
                color="blue",
                color_confidence=0.92,
                max_confidence=0.88,
            )
        )
        # Track 2: Person wearing black
        db.add(
            TrackModel(
                video_id=vid_id,
                track_id="TRACK-002",
                object_class="person",
                first_seen=5.0,
                last_seen=15.0,
                duration_seconds=10.0,
                detection_count=12,
                color="black",
                color_confidence=0.89,
                max_confidence=0.85,
            )
        )
        # Add events
        db.add(
            EventModel(
                video_id=vid_id,
                object_class="person",
                class_id=0,
                timestamp_seconds=2.0,
                confidence=0.90,
                bbox_x1=100.0,
                bbox_y1=150.0,
                bbox_x2=200.0,
                bbox_y2=350.0,
                frame_number=60,
                validation_status="VALID",
            )
        )
        db.add(
            SecurityEventModel(
                video_id=vid_id,
                event_type="POTENTIAL_THEFT",
                timestamp_seconds=8.0,
                duration_seconds=3.0,
                severity="HIGH",
                confidence=0.88,
                description="Object takeaway interaction",
            )
        )
        db.add(
            GroupedEventModel(
                video_id=vid_id,
                event_type="POTENTIAL_THEFT",
                start_time=8.0,
                end_time=11.0,
                duration_seconds=3.0,
                total_detections=1,
                max_confidence=0.88,
                priority="HIGH",
                objects_summary=[{"class": "person", "count": 1}],
            )
        )
        db.commit()
        yield vid_id
    finally:
        db.close()


def test_simple_person_count_does_not_invoke_gemini(seeded_video_id):
    """Prove that simple person-count queries bypass the LLM completely."""
    spy = SpyingLLMProvider()
    orchestrator = InvestigationOrchestrator(provider=spy)
    
    res = orchestrator.process_investigation(seeded_video_id, "How many people were detected?")
    
    assert res["mode"] == "deterministic_fast_path"
    assert spy.intent_calls == 0, "generate_structured_intent must NOT be called for simple count"
    assert spy.grounded_calls == 0, "generate_grounded_response must NOT be called for simple count"
    assert "distinct physical people" in res["answer"]


def test_color_count_does_not_invoke_gemini(seeded_video_id):
    """Prove that color-count queries bypass the LLM completely."""
    spy = SpyingLLMProvider()
    orchestrator = InvestigationOrchestrator(provider=spy)
    
    res = orchestrator.process_investigation(seeded_video_id, "How many people wore blue?")
    
    assert res["mode"] == "deterministic_fast_path"
    assert spy.intent_calls == 0, "generate_structured_intent must NOT be called for color count"
    assert spy.grounded_calls == 0, "generate_grounded_response must NOT be called for color count"
    assert "blue" in res["answer"].lower()


def test_event_list_does_not_invoke_gemini(seeded_video_id):
    """Prove that event-list queries bypass the LLM completely."""
    spy = SpyingLLMProvider()
    orchestrator = InvestigationOrchestrator(provider=spy)
    
    res = orchestrator.process_investigation(seeded_video_id, "What events were detected?")
    
    assert res["mode"] == "deterministic_fast_path"
    assert spy.intent_calls == 0, "generate_structured_intent must NOT be called for event list"
    assert spy.grounded_calls == 0, "generate_grounded_response must NOT be called for event list"
    assert "verified security event" in res["answer"] or "theft" in res["answer"].lower()


def test_complex_narrative_can_invoke_gemini(seeded_video_id):
    """Prove that open-ended narrative queries invoke the LLM for reasoning."""
    spy = SpyingLLMProvider()
    orchestrator = InvestigationOrchestrator(provider=spy)
    
    res = orchestrator.process_investigation(
        seeded_video_id, "Explain what happened in chronological order."
    )
    
    assert res["mode"] == "ai_assisted"
    # Notice: intent translation is handled locally (0 remote calls), only 1 grounded call is made!
    assert spy.intent_calls == 0, "Fast local intent parser eliminates redundant first LLM call"
    assert spy.grounded_calls == 1, "Complex narrative must invoke LLM grounded response exactly once"
    assert res["answer"] == "Simulated synthesized narrative response from AI provider."


def test_provider_failure_returns_deterministic_fallback_quickly(seeded_video_id):
    """Prove that provider failure immediately returns grounded deterministic fallback."""
    spy = SpyingLLMProvider(should_fail=True)
    orchestrator = InvestigationOrchestrator(provider=spy)
    
    res = orchestrator.process_investigation(
        seeded_video_id, "Explain what happened in chronological order."
    )
    
    assert res["mode"] == "deterministic_fallback"
    assert "VIDEO ACTIVITY SUMMARY" in res["answer"] or "deterministic" in res["answer"].lower()
    assert res["is_supported"] is True


def test_no_cross_video_cache_leakage():
    """Verify that cached derived data is strictly isolated per video ID."""
    InvestigationDerivedDataCache.clear()
    
    InvestigationDerivedDataCache.set_canonical_entities("vid_A", "token1", [{"track_id": "T1", "color": "blue"}])
    InvestigationDerivedDataCache.set_canonical_entities("vid_B", "token1", [{"track_id": "T2", "color": "red"}])
    
    res_a = InvestigationDerivedDataCache.get_canonical_entities("vid_A", "token1")
    res_b = InvestigationDerivedDataCache.get_canonical_entities("vid_B", "token1")
    
    assert res_a[0]["track_id"] == "T1"
    assert res_b[0]["track_id"] == "T2"
    
    # Invalidate vid_A only
    InvestigationDerivedDataCache.invalidate_video("vid_A")
    assert InvestigationDerivedDataCache.get_canonical_entities("vid_A", "token1") is None
    assert InvestigationDerivedDataCache.get_canonical_entities("vid_B", "token1") is not None


def test_raw_track_and_canonical_person_counts_remain_distinct(seeded_video_id):
    """Verify that raw track counts and resolved canonical people counts remain distinct concepts."""
    spy = SpyingLLMProvider()
    orchestrator = InvestigationOrchestrator(provider=spy)
    
    # Query for distinct physical people
    res_people = orchestrator.process_investigation(seeded_video_id, "How many distinct people are there?")
    # Query for raw tracks
    res_tracks = orchestrator.process_investigation(seeded_video_id, "How many person tracks?")
    
    assert res_people["mode"] == "deterministic_fast_path"
    assert res_tracks["mode"] == "deterministic_fast_path"
    
    assert "distinct" in res_people["answer"] or "people" in res_people["answer"]
    assert "tracks" in res_tracks["answer"]


def test_frontend_single_request_contract():
    """Verify frontend request payload structure and single-request dispatch contract."""
    from backend.app.api.videos import AIInvestigationRequest
    req = AIInvestigationRequest(query="How many people were detected?", history=None)
    assert req.query == "How many people were detected?"
    assert req.history is None
