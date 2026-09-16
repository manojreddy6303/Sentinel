# Sentinel — Architecture Specification

## Overview

**SENTINEL** is a software-only AI-powered security video investigation platform. It provides security investigators and analysts with an interactive timeline and conversational interface over hours of recorded surveillance footage.

## Core MVP Pipeline

```
[ Upload Video ]
       │
       ▼
[ Sample & Detect (YOLO + OpenCV) ]
       │
       ▼
[ Aggregate Temporal Events ]
       │
       ▼
[ Searchable Timeline UI ]
       │
       ▼
[ Ask Questions (LLM / Gemini) ]
       │
       ▼
[ Extract Evidence Clips ]
       │
       ▼
[ Generate Incident Report ]
```

## Architectural Components

1. **Frontend (`frontend/`):**
   - Built with Next.js (App Router), React, TypeScript, and Tailwind CSS.
   - Provides video upload zone, interactive scrubbable event timeline, conversational search panel, evidence clip viewer, and report previewer.

2. **Backend (`backend/`):**
   - Built with Python and FastAPI.
   - Exposes RESTful endpoints for video management, task status, timeline event querying, and evidence clip retrieval.
   - Serves `GET /api/health` for monitoring and deployment verification.

3. **Computer Vision & AI (`ai/`):**
   - **Video Processing (`ai/video`):** Decodes video streams and samples frames at defined intervals.
   - **Detection (`ai/detection`):** Employs YOLO for object, person, and vehicle detection.
   - **Events (`ai/events`):** Groups frame detections into continuous chronological events.
   - **Evidence Extraction (`ai/extraction`):** Trims sub-clips bounded by event timestamps.
   - **Investigation (`ai/investigation`):** Powers natural language reasoning over timeline events using LLM.

4. **Database (`database/`):**
   - PostgreSQL stores video catalog records, timeline events, evidence clip bookmarks, and report summaries.

5. **Storage Layer (`storage/`):**
   - Initial local filesystem storage structured into `uploads/`, `evidence/`, and `reports/`.
   - Raw sampled frames are processed on-the-fly and not saved permanently.
   - Designed to be replaced with AWS S3, Google Cloud Storage, or MinIO.

## Compliance and Boundary Principles

- **No Facial Recognition:** Biometric facial classification or comparison algorithms are strictly out of scope.
- **No Criminal Determination:** The system surfaces timestamped occurrences; human investigators make all legal and operational assessments.
- **Probabilistic Scoring:** Detections are accompanied by confidence metrics and are subject to visual verification.
- **Decoupled Architecture:** Clean boundary between user interface and backend services.
