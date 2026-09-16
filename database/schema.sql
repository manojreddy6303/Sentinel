-- ==============================================================================
-- SENTINEL RELATIONAL SCHEMA DRAFT (PostgreSQL)
-- Core MVP: UPLOAD -> DETECT -> TIMELINE -> ASK -> EVIDENCE -> REPORT
-- Note: Strictly no facial recognition or criminal identification fields.
-- ==============================================================================

-- Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Videos table: Metadata for ingested surveillance videos
CREATE TABLE IF NOT EXISTS videos (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    original_name VARCHAR(255) NOT NULL,
    storage_path VARCHAR(512) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    duration_seconds NUMERIC(10, 2),
    fps NUMERIC(6, 2),
    resolution_width INT,
    resolution_height INT,
    status VARCHAR(50) DEFAULT 'uploaded', -- uploaded, processing, completed, failed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Video Events: Temporal events extracted via Computer Vision (YOLO/OpenCV)
CREATE TABLE IF NOT EXISTS video_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    timestamp_start NUMERIC(10, 2) NOT NULL,
    timestamp_end NUMERIC(10, 2) NOT NULL,
    event_type VARCHAR(100) NOT NULL, -- e.g., 'object_detected', 'motion_detected', 'vehicle_present'
    label VARCHAR(100) NOT NULL,      -- e.g., 'person', 'car', 'backpack', 'bicycle'
    confidence NUMERIC(5, 4) NOT NULL, -- Detection confidence score (0.0000 - 1.0000)
    metadata JSONB DEFAULT '{}'::jsonb, -- Bounding box coordinates, trajectory, class metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Evidence Clips: Specific video segments extracted and bookmarked by investigator
CREATE TABLE IF NOT EXISTS evidence_clips (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    timestamp_start NUMERIC(10, 2) NOT NULL,
    timestamp_end NUMERIC(10, 2) NOT NULL,
    clip_storage_path VARCHAR(512) NOT NULL,
    tags VARCHAR(50)[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Investigation Queries: Natural language queries and LLM-assisted reasoning history
CREATE TABLE IF NOT EXISTS investigation_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    response_text TEXT NOT NULL,
    referenced_event_ids UUID[],
    referenced_clip_ids UUID[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Incident Reports: Generated summary reports of findings and evidence
CREATE TABLE IF NOT EXISTS incident_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    summary TEXT NOT NULL,
    incident_date TIMESTAMP WITH TIME ZONE,
    investigator_notes TEXT,
    report_file_path VARCHAR(512),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indices for rapid timeline searching
CREATE INDEX IF NOT EXISTS idx_video_events_video_id ON video_events(video_id);
CREATE INDEX IF NOT EXISTS idx_video_events_timestamps ON video_events(timestamp_start, timestamp_end);
CREATE INDEX IF NOT EXISTS idx_video_events_label ON video_events(label);
CREATE INDEX IF NOT EXISTS idx_evidence_clips_video_id ON evidence_clips(video_id);
