"""
Reproducible Benchmark for Sentinel Incident Correlation Engine & Intelligence Pipeline.

Measures:
- Correlation execution time across varying scales (100, 500, 1000 observations/candidates)
- Database retrieval and query latency
- Repetitions (20 trials)
- Median and p95 latency
- Hardware / OS environment specifications
"""

import os
import sys
import time
import platform
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.correlation.engine import AdvancedIncidentCorrelationEngine
from ai.schemas import BoundingBox, TrackedObject
from ai.incidents.schemas import IncidentCandidate, SupportingSignal, EvidenceCandidate
from database.session import SessionLocal
from database.models import CorrelatedIncidentModel, SecurityEventModel


def generate_synthetic_scene(num_candidates: int, num_tracks: int):
    candidates = []
    tracks = []
    
    for i in range(num_candidates):
        t_start = (i % 20) * 5.0
        c = IncidentCandidate(
            incident_id=f"bench_cand_{i:04d}",
            video_id="bench_video",
            category="vehicle" if i % 2 == 0 else "property",
            event_type="VEHICLE_COLLISION" if i % 2 == 0 else "THEFT_TAKEAWAY",
            start_time=t_start,
            end_time=t_start + 3.0,
            duration=3.0,
            confidence=0.85,
            severity="HIGH",
            validation_decision="ACCEPTED",
            track_ids=[f"TRACK_{i % 10:02d}", f"TRACK_{(i+1) % 10:02d}"],
            object_classes=["car"] if i % 2 == 0 else ["person", "suitcase"],
            supporting_signals=[SupportingSignal(signal_type="contact", description="bench signal", confidence=0.85)],
            contradictory_signals=[],
            evidence_candidates=[EvidenceCandidate(timestamp=t_start, reason="bench", target_track_id=f"TRACK_{i % 10:02d}")],
            human_verification_required=False,
            detector_name="bench_detector",
            detector_version="1.0.0",
        )
        candidates.append(c)

    for i in range(num_tracks):
        bb = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=200.0)
        trk = TrackedObject(
            track_id=f"TRACK_{i:02d}",
            object_class="car" if i % 2 == 0 else "person",
            first_seen=0.0,
            last_seen=100.0,
            history_bboxes=[bb],
            current_bbox=bb,
            confidence=0.9,
        )
        tracks.append(trk)

    return candidates, tracks


def run_benchmark():
    engine = AdvancedIncidentCorrelationEngine()
    
    print("=" * 70)
    print("SENTINEL INCIDENT CORRELATION ENGINE BENCHMARK")
    print("=" * 70)
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Processor: {platform.processor() or platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print("=" * 70)

    scales = [
        (100, 100),
        (500, 500),
        (1000, 1000)
    ]
    
    reps = 20

    for num_cand, num_obs in scales:
        candidates, observations = generate_synthetic_scene(num_cand, num_obs)
        latencies_ms = []
        
        # Warmup
        engine.correlate_incidents("bench_video", candidates, observations)
        
        for _ in range(reps):
            t0 = time.perf_counter()
            res = engine.correlate_incidents("bench_video", candidates, observations)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            
        incidents = res["correlated_incidents"]
            
        median = np.median(latencies_ms)
        p95 = np.percentile(latencies_ms, 95)
        p99 = np.percentile(latencies_ms, 99)
        min_lat = np.min(latencies_ms)
        max_lat = np.max(latencies_ms)
        
        print(f"\n[Scale: {num_cand} Candidates | {num_obs} Observations]")
        print(f"  Repetitions: {reps}")
        print(f"  Resulting Incidents: {len(incidents)}")
        print(f"  Correlation Time (ms):")
        print(f"    Median: {median:.2f} ms")
        print(f"    p95:    {p95:.2f} ms")
        print(f"    p99:    {p99:.2f} ms")
        print(f"    Min:    {min_lat:.2f} ms | Max: {max_lat:.2f} ms")

    # DB Query Benchmark
    print("\n[Database Query Retrieval Benchmark]")
    db = SessionLocal()
    try:
        db_latencies_ms = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _ = db.query(CorrelatedIncidentModel).limit(100).all()
            t1 = time.perf_counter()
            db_latencies_ms.append((t1 - t0) * 1000.0)
        
        db_median = np.median(db_latencies_ms)
        db_p95 = np.percentile(db_latencies_ms, 95)
        print(f"  DB Correlated Incident Query (Limit 100):")
        print(f"    Median: {db_median:.2f} ms")
        print(f"    p95:    {db_p95:.2f} ms")
    finally:
        db.close()

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_benchmark()
