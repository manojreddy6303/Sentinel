"""Deep audit of video provenance, test fixture origins, and validation status distribution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.session import SessionLocal
from database.models import (VideoModel, CameraSourceModel, EvidenceModel, 
    CorrelatedIncidentModel, CaseModel, SpecializedObservationModel, 
    EventModel, SurveillanceSessionModel, ReportModel, CaseVideoModel)
from sqlalchemy import func

db = SessionLocal()

print("=== VIDEO PROVENANCE ANALYSIS ===")
print("Looking for test-like filenames...")
test_patterns = ["test", "mock", "stale", "synthetic", "phase", "fixture", "sample", "fake", "dummy"]
all_vids = db.query(VideoModel).all()
test_vids = []
real_vids = []
uuid_vids = []

for v in all_vids:
    name = v.original_filename.lower()
    is_test = any(p in name for p in test_patterns)
    is_uuid = len(v.original_filename.replace("-","").replace(".mp4","")) == 32 and v.original_filename.endswith(".mp4")
    if is_test:
        test_vids.append(v)
    elif is_uuid:
        uuid_vids.append(v)
    else:
        real_vids.append(v)

print(f"Total videos: {len(all_vids)}")
print(f"Test-named videos: {len(test_vids)}")
print(f"UUID-named videos (likely test fixtures): {len(uuid_vids)}")
print(f"Other videos (possible real): {len(real_vids)}")
print()

print("=== REAL-LOOKING VIDEOS ===")
for v in real_vids[:30]:
    ev_count = db.query(EventModel).filter(EventModel.video_id == v.id).count()
    print(f"  {v.original_filename} | {v.status} | events={ev_count} | {v.uploaded_at}")

print()
print("=== TEST-NAMED VIDEOS (first 20) ===")
for v in test_vids[:20]:
    ev_count = db.query(EventModel).filter(EventModel.video_id == v.id).count()
    print(f"  {v.original_filename} | {v.status} | events={ev_count} | {v.uploaded_at}")

print()
print("=== UUID-NAMED VIDEOS (first 5) ===")
for v in uuid_vids[:5]:
    ev_count = db.query(EventModel).filter(EventModel.video_id == v.id).count()
    print(f"  {v.original_filename} | events={ev_count} | {v.uploaded_at}")

print()
print("=== UNCERTAIN EVENTS ===")
uncert = db.query(EventModel).filter(EventModel.validation_status=="UNCERTAIN").limit(5).all()
for e in uncert:
    vid = db.query(VideoModel).filter(VideoModel.id == e.video_id).first()
    print(f"  Event {e.id[:8]} | class={e.object_class} | validation={e.validation_status} | video={vid.original_filename if vid else 'UNKNOWN'}")

print()
print("=== CASE ANALYSIS ===")
cases = db.query(CaseModel).all()
print(f"Total cases: {len(cases)}")
for c in cases:
    vid_links = db.query(CaseVideoModel).filter(CaseVideoModel.case_id == c.id).count()
    print(f"  {c.case_number} | {c.title} | {c.status} | linked_videos={vid_links}")

print()
print("=== SESSION ANALYSIS ===")
for s in db.query(SurveillanceSessionModel).all():
    cam_count = db.query(CameraSourceModel).filter(CameraSourceModel.session_id == s.id).count()
    print(f"  {s.name} | {s.site_name} | cameras={cam_count} | {s.created_at}")

print()
print("=== REPORT ANALYSIS ===")
# Get sample reports to understand their source
from database.models import ReportModel
reports = db.query(ReportModel).limit(10).all()
for r in reports:
    vid = db.query(VideoModel).filter(VideoModel.id == r.video_id).first() if hasattr(r, 'video_id') else None
    attrs = {col: getattr(r, col, None) for col in ['id', 'video_id', 'report_type', 'status', 'created_at']}
    print(f"  Report: {attrs}")

db.close()
