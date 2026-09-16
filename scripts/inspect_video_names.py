"""Inspect what videos are kept and their upload patterns to refine purge heuristics."""
import sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.session import SessionLocal
from database.models import VideoModel, EventModel, CaseVideoModel, CaseModel

db = SessionLocal()

# Get unique filenames and their counts/upload times
all_vids = db.query(VideoModel).order_by(VideoModel.uploaded_at.asc()).all()

print("=== ALL UNIQUE FILENAMES ===")
name_counter = Counter(v.original_filename for v in all_vids)
for name, count in sorted(name_counter.items(), key=lambda x: -x[1]):
    print(f"  count={count:4d}  {name}")

print()
print("=== VIDEOS REFERENCED BY REAL CASES (linked via CaseVideoModel) ===")
# Find which video IDs are referenced by the meaningful cases
meaningful_cases = [c for c in db.query(CaseModel).all() 
                    if not any(t in (c.title or "") for t in [
                        "Warehouse Perimeter Breach Investigation",
                        "Vehicle Theft Case",
                        "Crowd Loitering Review",
                        "Multi-Video Investigation",
                    ])]
print(f"Meaningful (non-template) cases: {len(meaningful_cases)}")
linked_vids = set()
for c in meaningful_cases:
    links = db.query(CaseVideoModel).filter(CaseVideoModel.case_id == c.id).all()
    for link in links:
        linked_vids.add(link.video_id)
        vid = db.query(VideoModel).filter(VideoModel.id == link.video_id).first()
        print(f"  Case '{c.title}' -> {vid.original_filename if vid else link.video_id}")

print()
print("=== VIDEOS WITH ACTUAL EVENTS (non-zero detections) ===")
for v in all_vids:
    ev = db.query(EventModel).filter(EventModel.video_id == v.id).count()
    if ev > 50:
        print(f"  {v.original_filename[:60]:<62} events={ev} uploaded={v.uploaded_at}")

db.close()
