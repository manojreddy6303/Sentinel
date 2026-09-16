"""
backend/tests/test_phase19_api.py

Integration tests for Phase 19 REST API endpoints in app.api.cases:
- Case CRUD over HTTP
- Entity linking (videos, cameras, incidents) over HTTP
- Bookmarks, Notes, and Annotations over HTTP
- Unified Timeline, Incident Replay, Focus Mode, Explainability, Topology, Storyline, Activities, Export over HTTP
"""
import uuid
import pytest
from fastapi.testclient import TestClient
from database.session import SessionLocal, init_db
from database.models import VideoModel, CorrelatedIncidentModel
from backend.app.main import app

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


@pytest.fixture
def test_video():
    db = SessionLocal()
    try:
        vid = str(uuid.uuid4())
        v = VideoModel(id=vid, original_filename="api_test_video.mp4", duration_seconds=120.0, storage_path=f"storage/{vid}.mp4", status="processed")
        db.add(v)
        corr_id = f"CORR-{uuid.uuid4().hex[:8]}"
        corr = CorrelatedIncidentModel(
            id=corr_id,
            video_id=vid,
            incident_category="theft",
            start_time=20.0,
            end_time=35.0,
            duration=15.0,
            assessment_score=0.65,
            evidence_strength=0.92,
            validation_decision="REVIEW_REQUIRED",
            storyline="Object takeaway observed at 20.0s.",
        )
        db.add(corr)
        db.commit()
        return vid, corr_id
    finally:
        db.close()


def test_api_case_lifecycle(test_video):
    vid, corr_id = test_video

    # 1. Create Case
    payload = {
        "title": "API Test Investigation",
        "description": "Full lifecycle API test",
        "priority": "HIGH",
        "assigned_investigator": "Agent Fox",
        "tags": ["Theft", "Critical"],
        "initial_video_ids": [vid],
    }
    res = client.post("/api/cases", json=payload)
    assert res.status_code == 201, res.text
    case_data = res.json()
    case_id = case_data["id"]
    assert case_data["case_number"].startswith("CASE-")
    assert case_data["status"] == "OPEN"
    assert case_data["priority"] == "HIGH"

    # 2. Get Case
    res_get = client.get(f"/api/cases/{case_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == case_id

    # 3. Update Case
    res_patch = client.patch(f"/api/cases/{case_id}", json={"status": "INVESTIGATING", "priority": "CRITICAL"})
    assert res_patch.status_code == 200
    assert res_patch.json()["status"] == "INVESTIGATING"
    assert res_patch.json()["priority"] == "CRITICAL"

    # 4. Bookmarks
    res_bm = client.post(
        f"/api/cases/{case_id}/bookmarks",
        json={"video_id": vid, "timestamp_seconds": 22.5, "title": "Suspect entry pin", "author": "Agent Fox"},
    )
    assert res_bm.status_code == 201
    bm_id = res_bm.json()["id"]

    res_bms = client.get(f"/api/cases/{case_id}/bookmarks")
    assert res_bms.status_code == 200
    assert len(res_bms.json()["bookmarks"]) >= 1

    # 5. Notes
    res_note = client.post(
        f"/api/cases/{case_id}/notes",
        json={"content": "Reviewed initial footage; suspect confirmed present.", "author": "Agent Fox", "associated_type": "CASE"},
    )
    assert res_note.status_code == 201
    note_id = res_note.json()["id"]
    assert res_note.json()["note_classification"] == "ANALYST_NOTE"

    # 6. Annotations
    res_ann = client.post(
        f"/api/cases/{case_id}/annotations",
        json={"video_id": vid, "timestamp_seconds": 20.0, "annotation_type": "REGION", "data": {"bbox": [10, 10, 80, 80], "label": "Vehicle"}},
    )
    assert res_ann.status_code == 201
    ann_id = res_ann.json()["id"]

    # 7. Timeline
    res_tl = client.get(f"/api/cases/{case_id}/timeline")
    assert res_tl.status_code == 200
    assert res_tl.json()["total_entries"] >= 1

    # 8. Replay
    res_rep = client.get(f"/api/cases/{case_id}/incidents/{corr_id}/replay?pre_roll_seconds=5&post_roll_seconds=5")
    assert res_rep.status_code == 200
    assert res_rep.json()["replay_context"]["replay_start"] == 15.0
    assert res_rep.json()["replay_context"]["replay_end"] == 40.0

    # 9. Focus
    res_focus = client.get(f"/api/cases/{case_id}/incidents/{corr_id}/focus")
    assert res_focus.status_code == 200
    assert "explanation" in res_focus.json()

    # 10. Explainability
    res_exp = client.get(f"/api/cases/{case_id}/incidents/{corr_id}/explain")
    assert res_exp.status_code == 200
    assert res_exp.json()["final_assessment_score"] <= 0.65
    assert res_exp.json()["human_verification_required"] is True

    # 11. Topology
    res_top = client.get(f"/api/cases/{case_id}/topology")
    assert res_top.status_code == 200

    # 12. Storyline
    res_story = client.get(f"/api/cases/{case_id}/storyline")
    assert res_story.status_code == 200

    # 13. Activities
    res_acts = client.get(f"/api/cases/{case_id}/activities")
    assert res_acts.status_code == 200
    assert len(res_acts.json()["activities"]) >= 1

    # 14. Export
    res_exp_all = client.get(f"/api/cases/{case_id}/export")
    assert res_exp_all.status_code == 200
    pkg = res_exp_all.json()
    assert pkg["case_metadata"]["id"] == case_id
    assert "evidence_index" in pkg
    assert "activity_audit_trail" in pkg

    # Clean up bookmark, note, annotation
    client.delete(f"/api/cases/{case_id}/bookmarks/{bm_id}")
    client.delete(f"/api/cases/{case_id}/notes/{note_id}")
    client.delete(f"/api/cases/{case_id}/annotations/{ann_id}")

    # Delete case
    res_del = client.delete(f"/api/cases/{case_id}")
    assert res_del.status_code == 204
