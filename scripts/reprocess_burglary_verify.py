"""
Reprocess uccrime_Burglary010_x264.mp4 with Phase 8 intelligence pipeline.

Verifies:
1. POTENTIAL_THEFT spatial grounding at ~147s (person + object bbox stored)
2. No false BUS tracks (validated bus tracks require >= 2 detections OR conf >= 0.65)
3. Evidence annotated snapshot shows correct multi-box layout

Usage:
    cd C:\\Sentinel
    python scripts/reprocess_burglary_verify.py
"""

import sys
import os
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("reprocess_burglary")

VIDEO_ID = "e71e62d8-5428-4720-93f5-19342d584a9d"
VIDEO_PATH = str(PROJECT_ROOT / "storage" / "uploads" / f"{VIDEO_ID}_uccrime_Burglary010_x264.mp4")


def main():
    from database.session import SessionLocal
    from database.models import VideoModel, EventModel, SecurityEventModel, EvidenceModel
    from ai.intelligence_pipeline import SecurityIntelligencePipeline
    from ai.intelligence_repository import get_intelligence_repository
    from backend.app.services.evidence_service import EvidenceService

    # Step 1: Validate video exists in DB
    db = SessionLocal()
    try:
        video = db.query(VideoModel).filter(VideoModel.id == VIDEO_ID).first()
        if not video:
            logger.error(f"Video {VIDEO_ID} not found in database.")
            sys.exit(1)
        logger.info(f"Video: {video.original_filename} | {video.duration_seconds:.1f}s | {video.fps:.1f}fps")
    finally:
        db.close()

    if not os.path.exists(VIDEO_PATH):
        logger.error(f"Video file not found: {VIDEO_PATH}")
        sys.exit(1)

    # Step 2: Load raw detection events from DB
    db = SessionLocal()
    try:
        raw_events_rows = db.query(EventModel).filter(EventModel.video_id == VIDEO_ID).all()
        raw_events = []
        for ev in raw_events_rows:
            raw_events.append({
                "id": ev.id,
                "timestamp": ev.timestamp_seconds,
                "object_class": ev.object_class,
                "confidence": ev.confidence,
                "bounding_box": {
                    "x1": ev.bbox_x1,
                    "y1": ev.bbox_y1,
                    "x2": ev.bbox_x2,
                    "y2": ev.bbox_y2,
                },
            })
        logger.info(f"Loaded {len(raw_events)} raw detection events.")
    finally:
        db.close()

    fps = video.fps or 30.0
    duration = video.duration_seconds or 0.0

    # Step 3: Run Phase 8 intelligence pipeline
    logger.info("Running Phase 8 security intelligence pipeline...")
    pipeline = SecurityIntelligencePipeline()
    results = pipeline.process_video_intelligence(
        video_id=VIDEO_ID,
        video_path=VIDEO_PATH,
        raw_events=raw_events,
        fps=fps,
        duration_seconds=duration,
        sample_rate_fps=1.0,
    )

    tracks = results["tracks"]
    all_tracks = results["all_tracks"]
    security_events = results["security_events"]

    logger.info(f"Pipeline complete: {len(tracks)} validated tracks / {len(all_tracks)} total tracks")
    logger.info(f"Security events: {len(security_events)}")

    # Step 4: Check for false BUS detections
    logger.info("\n--- BUS TRACK VALIDATION ---")
    bus_tracks = [t for t in all_tracks if t.object_class == "bus"]
    validated_bus = [t for t in bus_tracks if t.is_validated]
    rejected_bus = [t for t in bus_tracks if not t.is_validated]

    if bus_tracks:
        logger.info(f"Total bus tracks detected: {len(bus_tracks)}")
        for bt in bus_tracks:
            status = "VALIDATED" if bt.is_validated else "REJECTED (false positive)"
            logger.info(
                f"  [{status}] {bt.track_id} | count={bt.detection_count} | "
                f"conf={bt.confidence:.2f} | dur={bt.duration_seconds:.1f}s"
            )
        if rejected_bus:
            logger.info(f"  -> {len(rejected_bus)} false BUS detections suppressed by is_validated filter.")
    else:
        logger.info("  No bus detections in raw tracks for this video.")

    # Step 5: Find POTENTIAL_THEFT events and verify spatial grounding
    logger.info("\n--- POTENTIAL_THEFT GROUNDING CHECK ---")
    theft_events = [e for e in security_events if e.event_type == "POTENTIAL_THEFT"]
    logger.info(f"POTENTIAL_THEFT events generated: {len(theft_events)}")

    for ev in theft_events:
        logger.info(f"\n  Event ID: {ev.event_id}")
        logger.info(f"  Timestamp: {ev.timestamp:.2f}s")
        logger.info(f"  Confidence: {int(ev.confidence * 100)}%")
        logger.info(f"  Person track: {ev.person_track_id}")
        logger.info(f"  Object track: {ev.object_track_id} ({ev.object_class})")

        if ev.bounding_box:
            logger.info(f"  Person BBox: [{ev.bounding_box.x1:.1f},{ev.bounding_box.y1:.1f} -> {ev.bounding_box.x2:.1f},{ev.bounding_box.y2:.1f}]")
        else:
            logger.warning("  Person BBox: MISSING — spatial grounding failed for person!")

        if ev.object_bounding_box:
            logger.info(f"  Object BBox: [{ev.object_bounding_box.x1:.1f},{ev.object_bounding_box.y1:.1f} -> {ev.object_bounding_box.x2:.1f},{ev.object_bounding_box.y2:.1f}]")
            if ev.is_prior_object_observation:
                logger.info(f"  Object BBox note: Prior observation at {ev.object_observation_timestamp:.1f}s")
        else:
            logger.warning("  Object BBox: MISSING — object not spatially grounded (may have disappeared before interaction_start)")

        for sig in ev.observable_signals:
            logger.info(f"    Signal: {sig}")

    # Step 6: Persist intelligence results
    logger.info("\n--- PERSISTING INTELLIGENCE RESULTS ---")
    repo = get_intelligence_repository()
    repo.save_intelligence_results(
        video_id=VIDEO_ID,
        tracks=tracks,
        vehicle_attributes=results["vehicle_attributes"],
        face_detections=results["face_detections"],
        security_events=security_events,
    )
    logger.info("Intelligence results saved to database.")

    # Step 7: Verify saved POTENTIAL_THEFT events from DB
    logger.info("\n--- VERIFYING PERSISTED THEFT EVENTS ---")
    db = SessionLocal()
    try:
        saved_thefts = db.query(SecurityEventModel).filter(
            SecurityEventModel.video_id == VIDEO_ID,
            SecurityEventModel.event_type == "POTENTIAL_THEFT",
        ).all()
        logger.info(f"Saved POTENTIAL_THEFT events in DB: {len(saved_thefts)}")
        for st in saved_thefts:
            bbox = st.bounding_box or {}
            has_person = "x1" in bbox
            has_object = "object_bbox" in bbox
            has_grounding = has_person and has_object
            status = "GROUNDED" if has_grounding else ("PERSON_ONLY" if has_person else "UNGROUNDED")
            logger.info(
                f"  [{status}] {st.id} @ {st.timestamp_seconds:.1f}s | conf={st.confidence:.2f} | "
                f"person_track={bbox.get('person_track_id','?')} | object_class={st.object_class}"
            )
            if "object_bbox" in bbox:
                ob = bbox["object_bbox"]
                logger.info(f"    Object BBox persisted: [{ob.get('x1',0):.1f},{ob.get('y1',0):.1f} -> {ob.get('x2',0):.1f},{ob.get('y2',0):.1f}]")
    finally:
        db.close()

    # Step 8: Generate evidence for the POTENTIAL_THEFT event
    logger.info("\n--- GENERATING EVIDENCE ---")
    db = SessionLocal()
    try:
        # Delete previous evidence for this video to force regeneration
        old_ev = db.query(EvidenceModel).filter(EvidenceModel.video_id == VIDEO_ID).all()
        if old_ev:
            logger.info(f"Removing {len(old_ev)} old evidence records for fresh generation.")
            db.query(EvidenceModel).filter(EvidenceModel.video_id == VIDEO_ID).delete()
            db.commit()
    finally:
        db.close()

    # Re-fetch the saved theft event IDs
    db = SessionLocal()
    try:
        saved_thefts = db.query(SecurityEventModel).filter(
            SecurityEventModel.video_id == VIDEO_ID,
            SecurityEventModel.event_type.in_(["POTENTIAL_THEFT", "OBSERVATIONAL_ANOMALY"]),
        ).order_by(SecurityEventModel.timestamp_seconds).all()
        theft_event_ids = [(s.id, s.event_type, s.timestamp_seconds) for s in saved_thefts]
    finally:
        db.close()

    ev_svc = EvidenceService()
    for ev_id, ev_type, ev_ts in theft_event_ids[:3]:  # Generate evidence for top 3 theft events
        try:
            ev_result = ev_svc.create_evidence(
                video_id=VIDEO_ID,
                timestamp=ev_ts,
                event_id=ev_id,
                evidence_type="snapshot_and_clip",
                pre_seconds=3.0,
                post_seconds=3.0,
            )
            logger.info(f"  Evidence {ev_result['evidence_id']} for {ev_type} @ {ev_ts:.1f}s")
            logger.info(f"    Snapshot: {ev_result['has_snapshot']} | Annotated: {ev_result['has_annotated']} | Clip: {ev_result['has_clip']}")
            if ev_result["has_annotated"]:
                logger.info(f"    Annotated snapshot URL: {ev_result['annotated_snapshot_url']}")
        except Exception as exc:
            logger.warning(f"  Evidence generation failed for {ev_id}: {exc}")

    logger.info("\n=== REPROCESSING COMPLETE ===")
    logger.info(f"Verified for: uccrime_Burglary010_x264.mp4 (ID: {VIDEO_ID})")
    logger.info(f"  Validated tracks: {len(tracks)}")
    logger.info(f"  POTENTIAL_THEFT events: {len(theft_events)}")
    logger.info(f"  False BUS detections suppressed: {len(rejected_bus)}")


if __name__ == "__main__":
    main()
