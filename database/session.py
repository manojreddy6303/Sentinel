"""
Database Engine & Session Management for Sentinel
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()


def is_test_environment() -> bool:
    """Detect if code is executing under pytest or a test environment."""
    return (
        "pytest" in sys.modules
        or bool(os.getenv("PYTEST_CURRENT_TEST"))
        or os.getenv("SENTINEL_TESTING") == "1"
        or os.getenv("SENTINEL_ENV") == "test"
        or os.getenv("TESTING") == "1"
        or (bool(os.getenv("CI")) and os.getenv("SENTINEL_PROD") != "1")
    )


def assert_production_db_safety(db_url: str) -> None:
    """
    CRITICAL SAFETY GUARD:
    If executing in a test context, connecting to the primary production database
    (storage/sentinel.db) is strictly prohibited. Tests must execute against
    storage/test_sentinel.db or an isolated in-memory/temporary SQLite database.
    """
    if is_test_environment():
        normalized = db_url.replace("\\", "/").lower()
        if "storage/sentinel.db" in normalized and "test_sentinel.db" not in normalized:
            raise RuntimeError(
                "PRODUCTION DATABASE PROTECTION ACTIVATED: "
                "A test or synthetic verification execution attempted to connect to the "
                "production database 'storage/sentinel.db'. "
                "Tests must set DATABASE_URL=sqlite:///storage/test_sentinel.db"
            )


def get_database_url() -> str:
    """Return configured DATABASE_URL or SQLite fallback."""
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        assert_production_db_safety(env_url)
        return env_url
    
    project_root = Path(__file__).resolve().parent.parent
    if is_test_environment():
        test_db_path = project_root / "storage" / "test_sentinel.db"
        test_db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{test_db_path}"

    # Fallback to local SQLite DB inside storage directory
    db_path = project_root / "storage" / "sentinel.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{db_path}"
    assert_production_db_safety(url)
    return url


DB_URL = get_database_url()

# Enable connect_args for SQLite threading
connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}

engine = create_engine(DB_URL, connect_args=connect_args, pool_pre_ping=True)

if DB_URL.startswith("sqlite"):
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def rebind_engine(new_url: Optional[str] = None):
    """Dynamically rebind the engine and SessionLocal to a new database URL."""
    global DB_URL, engine, SessionLocal
    DB_URL = new_url or get_database_url()
    assert_production_db_safety(DB_URL)
    c_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
    engine = create_engine(DB_URL, connect_args=c_args, pool_pre_ping=True)
    if DB_URL.startswith("sqlite"):
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    SessionLocal.configure(bind=engine)

def init_db():
    """Initialize database tables."""
    try:
        from database.models import (
            VideoModel, EventModel, GroupedEventModel, EvidenceModel,
            TrackModel, VehicleAttributeModel, FaceDetectionModel,
            SecurityZoneModel, SecurityEventModel, ReportModel,
            SpecializedObservationModel, CorrelatedIncidentModel,
            InvestigationBundleModel,
            SurveillanceSessionModel, CameraSourceModel, CrossCameraAssociationModel,
            CaseModel, CaseVideoModel, CaseCameraModel, CaseIncidentModel, CaseEvidenceModel,
            CaseBookmarkModel, CaseNoteModel, CaseAnnotationModel, CaseActivityModel,
        )  # noqa
        Base.metadata.create_all(bind=engine)
        if DB_URL.startswith("sqlite"):
            from sqlalchemy import text
            with engine.connect() as conn:
                # Check events table columns
                res_events = conn.execute(text("PRAGMA table_info(events)")).fetchall()
                col_events = [r[1] for r in res_events]
                if "validation_status" not in col_events:
                    conn.execute(text("ALTER TABLE events ADD COLUMN validation_status VARCHAR(20) DEFAULT 'VALID' NOT NULL"))
                if "validation_reason" not in col_events:
                    conn.execute(text("ALTER TABLE events ADD COLUMN validation_reason VARCHAR(255)"))

                # Check camera_sources table columns and nullability
                res_cams = conn.execute(text("PRAGMA table_info(camera_sources)")).fetchall()
                col_cams = [r[1] for r in res_cams]
                if "status" not in col_cams:
                    conn.execute(text("ALTER TABLE camera_sources ADD COLUMN status VARCHAR(50) DEFAULT 'ACTIVE' NOT NULL"))
                    res_cams = conn.execute(text("PRAGMA table_info(camera_sources)")).fetchall()

                # Ensure session_id and video_id are nullable in SQLite
                notnull_session = any(r[1] == "session_id" and r[3] == 1 for r in res_cams)
                notnull_video = any(r[1] == "video_id" and r[3] == 1 for r in res_cams)
                if notnull_session or notnull_video:
                    conn.execute(text("PRAGMA foreign_keys=OFF"))
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS camera_sources_dg_tmp (
                            id VARCHAR(36) NOT NULL PRIMARY KEY,
                            session_id VARCHAR(36),
                            video_id VARCHAR(36),
                            camera_label VARCHAR(100) NOT NULL,
                            position_hint VARCHAR(255),
                            field_of_view_hint VARCHAR(255),
                            adjacency_hints JSON,
                            status VARCHAR(50) DEFAULT 'ACTIVE' NOT NULL,
                            created_at DATETIME NOT NULL,
                            FOREIGN KEY(session_id) REFERENCES surveillance_sessions(id) ON DELETE SET NULL,
                            FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE SET NULL
                        )
                    """))
                    conn.execute(text("""
                        INSERT INTO camera_sources_dg_tmp (id, session_id, video_id, camera_label, position_hint, field_of_view_hint, adjacency_hints, status, created_at)
                        SELECT id, session_id, video_id, camera_label, position_hint, field_of_view_hint, adjacency_hints, COALESCE(status, 'ACTIVE'), created_at
                        FROM camera_sources
                    """))
                    conn.execute(text("DROP TABLE camera_sources"))
                    conn.execute(text("ALTER TABLE camera_sources_dg_tmp RENAME TO camera_sources"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_camera_sources_camera_label ON camera_sources (camera_label)"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_camera_sources_session_id ON camera_sources (session_id)"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_camera_sources_video_id ON camera_sources (video_id)"))
                    conn.execute(text("PRAGMA foreign_keys=ON"))
                    conn.commit()

                # Check videos table columns
                res_vids = conn.execute(text("PRAGMA table_info(videos)")).fetchall()
                col_vids = [r[1] for r in res_vids]
                if "camera_id" not in col_vids:
                    conn.execute(text("ALTER TABLE videos ADD COLUMN camera_id VARCHAR(36)"))

                # Check evidence table columns
                res_ev = conn.execute(text("PRAGMA table_info(evidence)")).fetchall()
                col_ev = [r[1] for r in res_ev]
                if "validation_status" not in col_ev:
                    conn.execute(text("ALTER TABLE evidence ADD COLUMN validation_status VARCHAR(20) DEFAULT 'VALID' NOT NULL"))

                # Check security_events table columns (Phase 10 Universal Incident Intelligence)
                res_sec = conn.execute(text("PRAGMA table_info(security_events)")).fetchall()
                col_sec = [r[1] for r in res_sec]
                if "detector_name" not in col_sec:
                    conn.execute(text("ALTER TABLE security_events ADD COLUMN detector_name VARCHAR(100)"))
                if "detector_version" not in col_sec:
                    conn.execute(text("ALTER TABLE security_events ADD COLUMN detector_version VARCHAR(20)"))
                if "category" not in col_sec:
                    conn.execute(text("ALTER TABLE security_events ADD COLUMN category VARCHAR(50)"))
                if "human_verification_required" not in col_sec:
                    conn.execute(text("ALTER TABLE security_events ADD COLUMN human_verification_required INTEGER DEFAULT 1 NOT NULL"))
                if "incident_metadata" not in col_sec:
                    conn.execute(text("ALTER TABLE security_events ADD COLUMN incident_metadata JSON"))

                # Check specialized_observations table columns (Phase 15.2 Visual Episode Aggregation)
                res_spec = conn.execute(text("PRAGMA table_info(specialized_observations)")).fetchall()
                col_spec = [r[1] for r in res_spec]
                if "episode_id" not in col_spec:
                    conn.execute(text("ALTER TABLE specialized_observations ADD COLUMN episode_id VARCHAR(64)"))

                # Check correlated_incidents table columns (Phase 16 Advanced Incident Correlation)
                res_corr = conn.execute(text("PRAGMA table_info(correlated_incidents)")).fetchall()
                col_corr = [r[1] for r in res_corr]
                if res_corr:
                    if "duration" not in col_corr:
                        conn.execute(text("ALTER TABLE correlated_incidents ADD COLUMN duration FLOAT DEFAULT 0.0 NOT NULL"))
                    if "incident_subcategory" not in col_corr:
                        conn.execute(text("ALTER TABLE correlated_incidents ADD COLUMN incident_subcategory VARCHAR(100)"))
                    if "negative_evidence" not in col_corr:
                        conn.execute(text("ALTER TABLE correlated_incidents ADD COLUMN negative_evidence JSON"))
                    if "contextual_factors" not in col_corr:
                        conn.execute(text("ALTER TABLE correlated_incidents ADD COLUMN contextual_factors JSON"))
                    if "provenance" not in col_corr:
                        conn.execute(text("ALTER TABLE correlated_incidents ADD COLUMN provenance JSON"))
                conn.commit()
        logger.info(f"Database initialized using {DB_URL.split('://')[0]}")
    except Exception as exc:
        logger.error(f"Failed to initialize database: {exc}")
        raise
