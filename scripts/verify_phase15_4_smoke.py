"""
Phase 15.4 Smoke Integrity Verification Script
"""
import sys, os, logging
from pathlib import Path
PROJECT_ROOT = Path(r'C:\Sentinel')
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('phase15_4_verify')
VIDEO_ID = '0d4d92f9-19f8-42e3-925f-1931cb557705'
VIDEO_PATH = str(PROJECT_ROOT / 'storage' / 'uploads' / f'{VIDEO_ID}_uccrime_Burglary010_x264.mp4')

def main():
    from database.session import SessionLocal
    from database.models import VideoModel, EventModel
    from ai.intelligence_pipeline import SecurityIntelligencePipeline
    if not os.path.exists(VIDEO_PATH):
        logger.error(f'Video file not found: {VIDEO_PATH}'); sys.exit(1)
    db = SessionLocal()
    try:
        video = db.query(VideoModel).filter(VideoModel.id == VIDEO_ID).first()
        if not video:
            logger.error(f'Video {VIDEO_ID} not in DB'); sys.exit(1)
        logger.info(f'Video: {video.original_filename} | {video.duration_seconds:.1f}s | {video.fps:.1f}fps')
        fps = float(video.fps or 30.0); duration = float(video.duration_seconds or 0.0)
        rows = db.query(EventModel).filter(EventModel.video_id == VIDEO_ID).all()
        raw_events = [{'id':r.id,'timestamp':r.timestamp_seconds,'object_class':r.object_class,'confidence':r.confidence,'bounding_box':{'x1':r.bbox_x1,'y1':r.bbox_y1,'x2':r.bbox_x2,'y2':r.bbox_y2}} for r in rows]
        logger.info(f'Loaded {len(raw_events)} raw events')
    finally: db.close()
    logger.info('Running Phase 15 pipeline...')
    pipeline = SecurityIntelligencePipeline()
    results = pipeline.process_video_intelligence(video_id=VIDEO_ID, video_path=VIDEO_PATH, raw_events=raw_events, fps=fps, duration_seconds=duration, sample_rate_fps=1.0)
    spec_obs = results.get('specialized_observations', [])
    spec_eps = results.get('specialized_episodes', []) or []
    incidents = results.get('incidents', [])
    sec_events = results.get('security_events', [])
    raw_smoke = [o for o in spec_obs if getattr(o,'class_name','')=='smoke']
    val_smoke = [o for o in raw_smoke if str(getattr(o,'validation_status','')).upper()=='VALID']
    smoke_eps = [e for e in spec_eps if getattr(e,'class_name','')=='smoke']
    smoke_inc = [i for i in incidents if getattr(i,'event_type','')=='POTENTIAL_SMOKE']
    theft_evts = [e for e in sec_events if getattr(e,'event_type','')=='POTENTIAL_THEFT']
    logger.info('')
    logger.info('='*60)
    logger.info('PHASE 15.4 SMOKE INTEGRITY REPORT')
    logger.info('='*60)
    logger.info(f'  Total spec obs      : {len(spec_obs)}')
    logger.info(f'  Raw smoke obs       : {len(raw_smoke)}')
    logger.info(f'  Validated smoke obs : {len(val_smoke)}')
    logger.info(f'  Smoke episodes      : {len(smoke_eps)}')
    logger.info(f'  POTENTIAL_SMOKE inc : {len(smoke_inc)}')
    logger.info(f'  Total incidents     : {len(incidents)}')
    logger.info(f'  POTENTIAL_THEFT evts: {len(theft_evts)}')
    for ev in theft_evts:
        ts=getattr(ev,'timestamp',0.0)
        pb=bool(getattr(ev,'bounding_box',None)); ob=bool(getattr(ev,'object_bounding_box',None))
        logger.info(f'  THEFT @ {ts:.1f}s person_bbox={pb} object_bbox={ob}')
    failures=[]
    def chk(lbl,ok,actual,exp):
        if ok: logger.info(f'  [PASS] {lbl}: {actual}')
        else: logger.error(f'  [FAIL] {lbl}: got {actual} expected {exp}'); failures.append(lbl)
    chk('Raw smoke obs==0', len(raw_smoke)==0, len(raw_smoke), 0)
    chk('Val smoke obs==0', len(val_smoke)==0, len(val_smoke), 0)
    chk('Smoke episodes==0', len(smoke_eps)==0, len(smoke_eps), 0)
    chk('POTENTIAL_SMOKE inc==0', len(smoke_inc)==0, len(smoke_inc), 0)
    chk('THEFT event exists', len(theft_evts)>=1, len(theft_evts), '>=1')
    near147=[e for e in theft_evts if abs(getattr(e,'timestamp',9999.0)-147.0)<=20.0]
    chk('THEFT near 147s', len(near147)>=1, len(near147), '>=1')
    logger.info('')
    if not failures:
        logger.info('RESULT: ALL ASSERTIONS PASSED -- PHASE 15.4 PASS'); sys.exit(0)
    else:
        logger.error(f'RESULT: {len(failures)} FAILED -- PHASE 15.4 FAIL')
        for f in failures: logger.error(f'  FAILED: {f}')
        sys.exit(1)

if __name__ == '__main__': main()
