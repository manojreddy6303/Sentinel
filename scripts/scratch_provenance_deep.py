import sys
sys.path.insert(0, '.')
from database.session import SessionLocal
from database.models import (
    VideoModel, CaseModel, CaseVideoModel, SpecializedObservationModel,
    EvidenceModel, CorrelatedIncidentModel, EventModel, TrackModel
)

db = SessionLocal()

print('--- ALL CASES ---')
for c in db.query(CaseModel).all():
    vids = db.query(CaseVideoModel).filter(CaseVideoModel.case_id == c.id).all()
    vid_ids = [v.video_id for v in vids]
    print(f'Case: id={c.id} | num={c.case_number} | title="{c.title}" | created={c.created_at} | vids={vid_ids}')

print('\n--- ALL BURGLARY VIDEOS ---')
for v in db.query(VideoModel).filter(VideoModel.original_filename.like('%Burglary%')).all():
    ev_count = db.query(EvidenceModel).filter(EvidenceModel.video_id == v.id).count()
    inc_count = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.video_id == v.id).count()
    raw_count = db.query(EventModel).filter(EventModel.video_id == v.id).count()
    track_count = db.query(TrackModel).filter(TrackModel.video_id == v.id).count()
    print(f'Video: id={v.id} | file={v.original_filename} | path={v.storage_path} | uploaded={v.uploaded_at} | raw={raw_count} | tracks={track_count} | incidents={inc_count} | evidence={ev_count}')

print('\n--- ALL SPECIALIZED OBSERVATIONS ---')
for obs in db.query(SpecializedObservationModel).all():
    vid = db.query(VideoModel).filter(VideoModel.id == obs.video_id).first()
    fname = vid.original_filename if vid else "none"
    print(f'Obs: id={obs.id} | video_id={obs.video_id} ({fname}) | detector={obs.detector_name} | class={obs.class_name} | status={obs.validation_status} | conf={obs.confidence} | ts={obs.timestamp_seconds}')

print('\n--- ALL EVIDENCE ITEMS ---')
for ev in db.query(EvidenceModel).all():
    vid = db.query(VideoModel).filter(VideoModel.id == ev.video_id).first()
    fname = vid.original_filename if vid else "none"
    print(f'Evidence: id={ev.id} | video_id={ev.video_id} ({fname}) | type={ev.evidence_type} | ts={ev.timestamp_seconds} | status={ev.validation_status} | path={ev.file_path} | incident_id={ev.incident_id}')

db.close()
