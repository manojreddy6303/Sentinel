from database.session import SessionLocal, init_db
from database.models import (
    VideoModel,
    EventModel,
    TrackModel,
    VehicleAttributeModel,
    FaceDetectionModel,
    SecurityEventModel,
    CorrelatedIncidentModel,
    SpecializedObservationModel,
    EvidenceModel,
)

init_db()
db = SessionLocal()

videos = [
    ('0d4d92f9-19f8-42e3-925f-1931cb557705', 'BURGLARY'),
    ('8edd2faf-8a6e-4c7d-9b58-292e45b06b92', 'HIGHWAY'),
    ('3426f64b-dd44-48a7-8e29-2c5f77b748bb', '4K CCTV'),
]

for vid_key, name in videos:
    v = db.query(VideoModel).filter(VideoModel.id == vid_key).first()
    vid = v.id if v else "NOT FOUND"
    print(f"=== {name} ({vid}) ===")
    if not v:
        continue
    raw = db.query(EventModel).filter(EventModel.video_id == vid).all()
    val = [e for e in raw if getattr(e, 'validation_status', '') == 'VALID']
    rej = [e for e in raw if getattr(e, 'validation_status', '') == 'REJECTED']
    tracks = db.query(TrackModel).filter(TrackModel.video_id == vid).all()
    val_tracks = [t for t in tracks if getattr(t, 'validation_status', 'VALID') == 'VALID']
    sec_events = db.query(SecurityEventModel).filter(SecurityEventModel.video_id == vid).all()
    incidents = db.query(CorrelatedIncidentModel).filter(CorrelatedIncidentModel.video_id == vid).all()
    veh_attr = db.query(VehicleAttributeModel).filter(VehicleAttributeModel.video_id == vid).all()
    faces = db.query(FaceDetectionModel).filter(FaceDetectionModel.video_id == vid).all()
    spec = db.query(SpecializedObservationModel).filter(SpecializedObservationModel.video_id == vid).all()
    evidence = db.query(EvidenceModel).filter(EvidenceModel.video_id == vid).all()

    smoke = [s for s in spec if s.class_name == 'smoke']
    fire = [s for s in spec if s.class_name == 'fire']
    weapon = [s for s in spec if s.class_name == 'weapon']

    theft_events = [e for e in sec_events if 'THEFT' in (e.event_type or '')]
    theft_incidents = [i for i in incidents if 'theft' in (i.incident_subcategory or '').lower() or 'takeaway' in (i.storyline or '').lower()]

    collision_events = [e for e in sec_events if 'COLLISION' in (e.event_type or '')]
    collision_incidents = [i for i in incidents if 'collision' in (i.incident_subcategory or '').lower() or 'collision' in (i.storyline or '').lower()]

    rev_req = [inc for inc in incidents if inc.validation_decision == 'REVIEW_REQUIRED']
    max_rev_req = max([inc.assessment_score for inc in rev_req]) if rev_req else 0.0

    print(f"  Raw Detections: {len(raw)}")
    print(f"  Validated Detections: {len(val)}")
    print(f"  Rejected Detections: {len(rej)}")
    print(f"  Total Tracks: {len(tracks)}")
    print(f"  Validated Tracks: {len(val_tracks)}")
    print(f"  Security Events: {len(sec_events)} (theft: {len(theft_events)}, collision: {len(collision_events)})")
    print(f"  Correlated Incidents: {len(incidents)} (theft: {len(theft_incidents)}, collision: {len(collision_incidents)})")
    print(f"  Vehicle Attributes: {len(veh_attr)}")
    print(f"  Face Observations: {len(faces)}")
    print(f"  Smoke Obs: {len(smoke)}, Fire Obs: {len(fire)}, Weapon Obs: {len(weapon)}")
    print(f"  Validated Evidence: {len(evidence)}")
    print(f"  REVIEW_REQUIRED Incidents: {len(rev_req)}, Max Score: {max_rev_req:.4f}")

db.close()
