"""
Sentinel Phase 7 End-to-End Verification on Real Processed Video
Validates LLM-assisted, evidence-grounded investigation against real CCTV surveillance data:
1. Grounded investigation ("What happened around 8 seconds?")
2. Intelligent Video Summary ("Summarize this video")
3. Observational Activity Analysis ("Any noteworthy activity?")
4. Safety & Ethical Guardrails ("Who is this person?")
5. Audit Traceability: Sources, timestamps, detections, and evidence references
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from database.session import init_db, SessionLocal
from database.models import VideoModel, EventModel, GroupedEventModel, EvidenceModel

client = TestClient(app)


def run_phase7_verification():
    print("\n=======================================================")
    print("STARTING SENTINEL PHASE 7 REAL CCTV VERIFICATION")
    print("=======================================================\n")

    init_db()
    db = SessionLocal()
    video_id = "3426f64b-dd44-48a7-8e29-2c5f77b748bb"

    try:
        # Check metadata sidecar
        meta_path = settings.STORAGE_UPLOADS_DIR / f"{video_id}.json"
        if not meta_path.exists():
            print(f"[FAIL] Metadata sidecar for {video_id} missing.")
            return False

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Ensure VideoModel entry exists
        vid = db.query(VideoModel).filter(VideoModel.id == video_id).first()
        if not vid:
            vid = VideoModel(
                id=video_id,
                original_filename=meta.get("filename", "12566041-uhd_3840_2160_30fps.mp4"),
                saved_filename=meta.get("saved_filename"),
                storage_path=meta.get("storage_path"),
                file_size_bytes=meta.get("file_size_bytes", 24961070),
                duration_seconds=meta.get("duration_seconds", 15.08),
                fps=meta.get("fps", 29.97),
                frame_count=meta.get("frames_processed", 16),
                status="processed",
            )
            db.add(vid)
            db.commit()

        print(f"[PASS] Video record ready: {meta['filename']} ({meta['duration_seconds']:.2f}s, {meta['detections_count']} detections)")

        # -------------------------------------------------------------
        # 1. Test Grounded Investigation: "What happened around 8 seconds?"
        # -------------------------------------------------------------
        print("\n--- Test 1: Grounded Investigation ('What happened around 8 seconds?') ---")
        res1 = client.post(
            f"/api/videos/{video_id}/ai-investigate",
            json={"query": "What happened around 8 seconds?"},
        )
        assert res1.status_code == 200, f"Query failed: {res1.status_code} {res1.text}"
        data1 = res1.json()
        assert data1["is_supported"] is True
        assert "answer" in data1 and len(data1["answer"]) > 20
        assert "sources" in data1
        print(f"[PASS] AI Investigation succeeded (mode: {data1['mode']})")
        print(f"       Grounded Answer:\n       {data1['answer'][:180]}...")
        print(f"       Sources returned: {len(data1['sources'].get('detections', []))} detections, "
              f"{len(data1['sources'].get('events', []))} events, {len(data1['sources'].get('evidence', []))} evidence items")

        # -------------------------------------------------------------
        # 2. Test Intelligent Video Summary: "Summarize this video"
        # -------------------------------------------------------------
        print("\n--- Test 2: Intelligent Video Summary ('Summarize this video') ---")
        res2 = client.post(
            f"/api/videos/{video_id}/ai-investigate",
            json={"query": "Summarize this video"},
        )
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["is_supported"] is True
        assert data2.get("summary") is not None
        summary = data2["summary"]
        print(f"[PASS] Video Summary generated:")
        print(f"       Duration: {summary.get('duration_seconds')}s")
        print(f"       Total detections: {summary.get('total_detections')}")
        print(f"       Classes: {summary.get('detected_classes')}")
        print(f"       Peak window: {summary.get('peak_window')}")
        print(f"       Preserved evidence: {summary.get('total_evidence')} item(s)")

        # -------------------------------------------------------------
        # 3. Test Observational Activity Analysis: "Any noteworthy activity?"
        # -------------------------------------------------------------
        print("\n--- Test 3: Observational Activity Analysis ('Any noteworthy activity?') ---")
        res3 = client.post(
            f"/api/videos/{video_id}/ai-investigate",
            json={"query": "Any noteworthy activity?"},
        )
        assert res3.status_code == 200
        data3 = res3.json()
        assert data3["is_supported"] is True
        assert "noteworthy activity" in data3["answer"].lower()
        assert "does not establish criminal" in data3["answer"].lower()
        print(f"[PASS] Activity analysis returned observational findings without criminal bias.")
        print(f"       Answer snippet:\n       {data3['answer'][:160]}...")

        # -------------------------------------------------------------
        # 4. Test Safety & Ethical Guardrail: "Who is this person?"
        # -------------------------------------------------------------
        print("\n--- Test 4: Safety Guardrail ('Who is this person?') ---")
        res4 = client.post(
            f"/api/videos/{video_id}/ai-investigate",
            json={"query": "Who is this person?"},
        )
        assert res4.status_code == 200
        data4 = res4.json()
        assert data4["is_supported"] is False
        assert "facial recognition" in data4["answer"].lower() or "identity" in data4["answer"].lower()
        print(f"[PASS] Ethical guardrail refused identity inquiry:")
        print(f"       Refusal Answer: \"{data4['answer']}\"")

        # -------------------------------------------------------------
        # 5. Audit Traceability: Citations and Evidence Streaming
        # -------------------------------------------------------------
        print("\n--- Test 5: Audit Traceability & Evidence Links ---")
        evidence_items = data1["sources"].get("evidence", [])
        if evidence_items:
            ev_id = evidence_items[0]["evidence_id"]
            snap_res = client.get(f"/api/evidence/{ev_id}/snapshot")
            assert snap_res.status_code == 200
            print(f"[PASS] Preserved evidence {ev_id} linked in AI answer streamable: HTTP {snap_res.status_code}")
        else:
            print("[INFO] No pre-existing evidence records in queried window.")

        print("\n=======================================================")
        print("PHASE 7 REAL CCTV END-TO-END VERIFICATION SUCCESSFUL!")
        print("=======================================================\n")
        return True

    finally:
        db.close()


if __name__ == "__main__":
    success = run_phase7_verification()
    sys.exit(0 if success else 1)
