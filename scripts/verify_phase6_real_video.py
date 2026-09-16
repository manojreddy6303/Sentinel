"""
Sentinel Phase 6 End-to-End Verification on Real Processed Video
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

def run_verification():
    init_db()
    db = SessionLocal()
    video_id = "3426f64b-dd44-48a7-8e29-2c5f77b748bb"

    try:
        # Check sidecar
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
            print(f"[INFO] Synced VideoModel row for {video_id} into DB.")

        print(f"[PASS] Video record ready: {meta['filename']} ({meta['duration_seconds']:.2f}s, {meta['detections_count']} detections)")

        # 1. Find detection around 8 seconds
        det_8s = (
            db.query(EventModel)
            .filter(EventModel.video_id == video_id, EventModel.timestamp_seconds >= 8.0, EventModel.timestamp_seconds <= 9.0)
            .first()
        )
        if not det_8s:
            # Fallback to any detection
            det_8s = db.query(EventModel).filter(EventModel.video_id == video_id).first()

        print(f"[INFO] Selected detection: {det_8s.object_class} at {det_8s.timestamp_seconds:.2f}s (conf: {det_8s.confidence:.2f})")

        # 2. Capture Detection Evidence via API
        print("[INFO] Extracting detection evidence (snapshot + annotated + clip)...")
        res = client.post(
            f"/api/videos/{video_id}/evidence",
            json={
                "timestamp": det_8s.timestamp_seconds,
                "event_id": det_8s.id,
                "evidence_type": "snapshot_and_clip",
                "pre_seconds": 3.0,
                "post_seconds": 3.0,
            },
        )
        if res.status_code not in [200, 201]:
            print(f"[FAIL] Detection evidence extraction failed: {res.status_code} {res.text}")
            return False

        ev_det = res.json()
        ev_det_id = ev_det["evidence_id"]
        print(f"[PASS] Detection evidence created: {ev_det_id}")
        print(f"       has_snapshot: {ev_det['has_snapshot']}, has_annotated: {ev_det['has_annotated']}, has_clip: {ev_det['has_clip']}")
        print(f"       window: {ev_det.get('start_time')}s -> {ev_det.get('end_time')}s (duration: {ev_det.get('duration_seconds')}s)")

        # 3. Verify files on disk
        rec_det = db.query(EvidenceModel).filter(EvidenceModel.id == ev_det_id).first()
        assert rec_det.snapshot_path and Path(rec_det.snapshot_path).exists(), "Snapshot missing on disk"
        assert rec_det.annotated_snapshot_path and Path(rec_det.annotated_snapshot_path).exists(), "Annotated snapshot missing"
        assert rec_det.clip_path and Path(rec_det.clip_path).exists(), "Clip missing on disk"
        print(f"[PASS] Files verified on disk:")
        print(f"       Snapshot: {Path(rec_det.snapshot_path).name} ({os.path.getsize(rec_det.snapshot_path)} bytes)")
        print(f"       Annotated: {Path(rec_det.annotated_snapshot_path).name} ({os.path.getsize(rec_det.annotated_snapshot_path)} bytes)")
        print(f"       Clip: {Path(rec_det.clip_path).name} ({os.path.getsize(rec_det.clip_path)} bytes)")

        # 4. Find Grouped Event
        grp = db.query(GroupedEventModel).filter(GroupedEventModel.video_id == video_id).first()
        if grp:
            print(f"[INFO] Selected grouped event: {grp.event_type} ({grp.start_time:.2f}s - {grp.end_time:.2f}s, {grp.total_detections} dets)")
            res_grp = client.post(
                f"/api/videos/{video_id}/evidence",
                json={
                    "timestamp": grp.start_time,
                    "event_id": grp.id,
                    "evidence_type": "snapshot_and_clip",
                    "pre_seconds": 2.0,
                    "post_seconds": 2.0,
                },
            )
            if res_grp.status_code not in [200, 201]:
                print(f"[FAIL] Grouped event evidence extraction failed: {res_grp.status_code} {res_grp.text}")
                return False
            ev_grp = res_grp.json()
            print(f"[PASS] Grouped event evidence created: {ev_grp['evidence_id']}")

        # 5. Verify Vault List API
        vault_res = client.get(f"/api/videos/{video_id}/evidence")
        assert vault_res.status_code == 200
        vault_data = vault_res.json()
        print(f"[PASS] Evidence Vault queried: {vault_data.get('total_evidence')} items preserved for video {video_id}")

        # 6. Verify Binary File Streaming Endpoints
        snap_stream = client.get(f"/api/evidence/{ev_det_id}/snapshot")
        assert snap_stream.status_code == 200
        assert snap_stream.headers["content-type"].startswith("image/")
        print(f"[PASS] Snapshot streaming endpoint verified: {snap_stream.headers['content-type']}")

        ann_stream = client.get(f"/api/evidence/{ev_det_id}/annotated")
        assert ann_stream.status_code == 200
        assert ann_stream.headers["content-type"].startswith("image/")
        print(f"[PASS] Annotated snapshot streaming endpoint verified: {ann_stream.headers['content-type']}")

        clip_stream = client.get(f"/api/evidence/{ev_det_id}/clip")
        assert clip_stream.status_code == 200
        assert clip_stream.headers["content-type"].startswith("video/")
        print(f"[PASS] Clip streaming endpoint verified: {clip_stream.headers['content-type']}")

        print("\n=======================================================")
        print("PHASE 6 END-TO-END VERIFICATION SUCCESSFUL ON REAL CCTV VIDEO")
        print("=======================================================")
        return True

    finally:
        db.close()

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
