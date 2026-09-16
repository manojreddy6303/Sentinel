# SENTINEL — AI-Powered Security Video Investigation Platform

> **Software-only AI-powered surveillance video investigation platform.**  
> Accelerate security investigations by converting hours of raw surveillance footage into structured, searchable events, interactive timelines, and verifiable evidence dossiers.

---

## 1. What Sentinel Is

Sentinel is an investigative workbench designed for security analysts and investigators. Instead of manually scrubbing through hours or days of video footage, Sentinel automatically extracts timestamped visual events using computer vision, maps them onto an interactive timeline, allows natural language inquiry (e.g., *"Show every time a red vehicle entered the perimeter between 2:00 AM and 4:00 AM"*), extracts verifiable sub-clips as evidence, and generates standardized incident reports.

### Safety & Ethical Principles
- **No Facial Recognition:** Sentinel strictly excludes facial recognition, biometric matching, and demographic profiling.
- **No Criminal Identification:** The system surfaces timestamped occurrences and visual attributes; human investigators make all legal and operational assessments.
- **Probabilistic Assistance:** Computer vision detections are probabilistic aids with confidence scores and require human verification.
- **No Permanent Raw Frame Clutter:** Only event metadata and selected evidence clips are permanently stored; intermediate sampled frames are discarded after processing.

---

## 2. Planned MVP Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  01. UPLOAD  │ ──► │  02. DETECT  │ ──► │ 03. TIMELINE │
│  Ingest MP4  │     │ YOLO / CV2   │     │ Event stream │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  06. REPORT  │ ◄── │ 05. EVIDENCE │ ◄── │   04. ASK    │
│ Dossier PDF  │     │ Trimmed clip │     │  Gemini LLM  │
└──────────────┘     └──────────────┘     └──────────────┘
```

1. **UPLOAD:** Investigator uploads surveillance video footage to `storage/uploads/`.
2. **DETECT:** Modular vision subsystem samples frames and runs YOLO object/motion detection.
3. **TIMELINE:** Frame detections are aggregated into continuous chronological events and stored in PostgreSQL.
4. **ASK:** Investigator queries the timeline in natural language; LLM synthesizes answers with exact timestamp links.
5. **EVIDENCE:** Targeted video clips corresponding to key events are cut and saved to `storage/evidence/`.
6. **REPORT:** Summary findings, bookmarks, and evidence references are exported to standardized reports in `storage/reports/`.

---

## 3. Technology Stack

| Layer | Technologies | Role / Notes |
|---|---|---|
| **Frontend** | Next.js 15+ (App Router), React 19, TypeScript, Tailwind CSS | Security investigation dashboard UI |
| **Backend** | Python 3, FastAPI, Uvicorn, Pydantic | REST API for ingestion, metadata, and queries |
| **Computer Vision** | YOLO (Ultralytics), OpenCV | Object and motion detection (Modular placeholders in `ai/`) |
| **Generative AI** | Google Gemini / LLM | Natural language timeline querying and reasoning |
| **Database** | PostgreSQL | Video metadata, event timelines, clip bookmarks, queries |
| **Storage** | Local disk (`storage/`) | Development MVP storage; interface designed to swap to S3/Cloud Storage |

---

## 4. Current Setup Status (Foundation Phase)

The current phase establishes the **clean foundation and architecture** for Sentinel:
- [x] **Frontend:** Next.js + TypeScript + Tailwind CSS project initialized and successfully verified with clean production builds.
- [x] **Backend:** FastAPI structure (`backend/app/main.py`) operational with `GET /api/health` endpoint verified.
- [x] **AI Subsystem:** Modular architecture with interface placeholders created (`ai/video`, `ai/detection`, `ai/events`, `ai/extraction`, `ai/investigation`).
- [x] **Database:** Relational schema drafted (`database/schema.sql`) and documented (`database/README.md`).
- [x] **Storage:** Local storage directories initialized (`storage/uploads`, `storage/evidence`, `storage/reports`) with retention policy documentation.
- [x] **Configuration:** Root `.env.example` and `.gitignore` configured.
- [!] **Advanced Features:** Video decoding, YOLO execution, LLM prompts, multi-camera tracking, and automated reporting are intentionally staged for upcoming development phases.

---

## 5. Repository Structure

```
Sentinel/
├── frontend/                  # Next.js web application
│   ├── src/app/
│   │   ├── layout.tsx         # Sentinel root layout & theme
│   │   ├── page.tsx           # Foundation status dashboard
│   │   └── globals.css        # Global CSS & Tailwind styling
│   ├── package.json
│   └── tsconfig.json
│
├── backend/                   # FastAPI Python backend
│   ├── app/
│   │   ├── main.py            # FastAPI entry point & CORS
│   │   ├── api/
│   │   │   └── health.py      # GET /api/health endpoint
│   │   └── core/
│   │       └── config.py      # App settings & environment loader
│   ├── requirements.txt
│   └── README.md
│
├── ai/                        # Modular AI & Computer Vision subsystems
│   ├── video/                 # Frame sampling & OpenCV stream handling
│   ├── detection/             # YOLO object detection interfaces
│   ├── events/                # Temporal event aggregation
│   ├── extraction/            # Video evidence clip cutting
│   ├── investigation/         # LLM natural language timeline reasoning
│   └── README.md
│
├── database/                  # PostgreSQL database resources
│   ├── schema.sql             # Relational schema draft
│   ├── migrations/            # Versioned migration directory (.gitkeep)
│   └── README.md              # Database setup guide
│
├── storage/                   # Local MVP media storage
│   ├── uploads/               # Raw ingested videos (.gitkeep)
│   ├── evidence/              # Extracted evidence clips (.gitkeep)
│   ├── reports/               # Incident reports (.gitkeep)
│   └── README.md              # Data retention rules
│
├── docs/
│   └── architecture.md        # Detailed system design specification
│
├── .env.example               # Environment variables template
├── .gitignore                 # Root gitignore
└── README.md                  # Project overview and quickstart
```

---

## 6. How to Run the Frontend

From the repository root:

```bash
# Navigate to frontend
cd frontend

# Run development server (runs on http://localhost:3000)
npm run dev

# Or build and run production server
npm run build
npm run start
```

---

## 7. How to Run the Backend

From the repository root (`Sentinel/`):

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Run the FastAPI server (runs on http://127.0.0.1:8000)
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Verify Backend Health:
```bash
# Windows PowerShell:
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -Method Get

# Or cURL:
curl http://127.0.0.1:8000/api/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "Sentinel Backend",
  "version": "0.1.0",
  "environment": "development",
  "message": "Sentinel backend is running successfully."
}
```

Interactive API documentation will be accessible at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 4. Current Status: Phase 5A Complete

- [x] **Phase 1 — Foundation:** Next.js + FastAPI + SQLite/PostgreSQL architecture.
- [x] **Phase 2 — Video Upload:** Resumable chunked streaming ingestion with metadata sidecars.
- [x] **Phase 3 — OpenCV + YOLO Detection:** Frame sampling, YOLOv8n inference, confidence scores, and bounding boxes.
- [x] **Phase 4 & 4.1 — Event Intelligence & Timeline:** Temporal clustering, grouped surveillance events, individual detection visibility, and synchronized video player seeking.
- [x] **Phase 5A — Natural-Language Investigation Layer ("Ask Sentinel"):**
  - Deterministic natural-language parser translating inquiries into structured database queries.
  - Zero hallucination: answers derived exclusively from ground-truth database records (`EventModel`, `GroupedEventModel`).
  - Interactive "Ask Sentinel" console in surveillance dashboard.
  - One-click timestamp jumping to synchronized HTML5 surveillance player.
  - Strict ethical guardrails: rejects facial recognition, personal identity guessing, criminal attribution, and unsupported color inferences.

---

## 5. Phase 5A: Natural-Language Investigation Architecture

### Data & Query Flow
```
USER QUESTION ("Show cars between 8 and 12 seconds")
          │
          ▼
INVESTIGATION QUERY PARSER (Deterministic pattern & synonym parsing)
          │
          ▼
STRUCTURED QUERY FILTERS ({ object_class: "car", start_time: 8.0, end_time: 12.0 })
          │
          ▼
RELATIONAL DATABASE (SQLAlchemy query on EventModel / GroupedEventModel)
          │
          ▼
GROUND-TRUTH DETECTIONS / EVENTS (Chronologically sorted records)
          │
          ▼
INVESTIGATION RESPONSE (Count, filters, bounding boxes, timestamps)
          │
          ▼
ASK SENTINEL SURVEILLANCE UI (Clickable timestamp cards seeking video player)
```

### Supported Query Types & Examples

| Query Intent | Example Natural Query | Interpretation |
|---|---|---|
| **All Detections of Class** | `"Show all people detected"` | `object_class: "person"`, returns all matching detections |
| **Time-Filtered Detections** | `"Show cars between 8 and 12 seconds"` | `object_class: "car"`, `start_time: 8.0`, `end_time: 12.0` |
| **Vehicle Group Inquiries** | `"What vehicles were detected?"` | Matches `car`, `truck`, `bus`, `motorcycle`, `bicycle` |
| **Confidence Thresholds** | `"Show detections above 70% confidence"` | `min_confidence: 0.70` |
| **Exact Record Counts** | `"How many people were detected?"` | Returns count of database detection instances across video frames |
| **Grouped Timeline Events** | `"Show events between 10 and 15 seconds"` | `result_type: "events"`, returns clustered timeline events |

### Investigation API Endpoint

`POST /api/videos/{video_id}/investigate`

**Request:**
```json
{
  "query": "Show cars between 8 and 12 seconds"
}
```

**Response:**
```json
{
  "query": "Show cars between 8 and 12 seconds",
  "video_id": "3426f64b-dd44-48a7-8e29-2c5f77b748bb",
  "is_supported": true,
  "result_type": "detections",
  "interpreted_filters": {
    "object_class": "car",
    "start_time": 8.0,
    "end_time": 12.0,
    "min_confidence": null,
    "max_confidence": null
  },
  "count": 43,
  "message": "Found 43 matching detections.",
  "results": [
    {
      "event_id": "a7aa9534-5da3-4d44-8702-891e1d3e6069",
      "video_id": "3426f64b-dd44-48a7-8e29-2c5f77b748bb",
      "timestamp": 8.01,
      "object_class": "car",
      "confidence": 0.8098,
      "frame_number": 240,
      "bounding_box": { "x1": 620.1, "y1": 1233.2, "x2": 836.8, "y2": 1429.3 }
    }
  ]
}
```

### Ethical Guardrails & Boundaries
- **No Facial Recognition or Identity Attribution:** Questions such as *"Who is this person?"*, *"What is the person's name?"*, or *"Identify this individual"* are explicitly rejected with an informative refusal message.
- **No Automated Criminal Accusation:** Inquiries asking *"Is this person a criminal?"* or *"Did this person steal?"* are strictly blocked.
- **No Vehicle Color Guessing:** Queries like *"Show red car"* are rejected because standard YOLO COCO classes do not classify vehicle body paint without a dedicated color classifier.
- **Detection Count vs. Unique Identity:** Record count queries explicitly note that numbers reflect per-frame detections and do not indicate unique individuals.

### Future LLM Architecture Plan (Phase 7)
In upcoming phases, a Large Language Model (e.g., Google Gemini) can safely interface with Sentinel:
1. User provides natural language inquiry (including complex reasoning).
2. LLM translates inquiry into structured parameters adhering to `InvestigationQuery`.
3. `InvestigationService` queries the relational database for ground-truth facts.
4. LLM formulates a natural language summary referencing exclusively the verified database output.
*The LLM never directly queries or hallucinates detection records.*

---

## 6. Phase 6 — Evidence Extraction & Evidence Vault

Sentinel Phase 6 implements the complete chain of custody for evidence preservation:

```
DETECTION / GROUPED EVENT / ASK SENTINEL
                   ↓
         [Capture Evidence]
                   ↓
EVIDENCE EXTRACTION SERVICE (OpenCV mp4v)
                   ↓
  ├── Original Snapshot (.jpg)
  ├── Annotated Bounding Box Snapshot (.jpg)
  └── Sub-Clip with Pre/Post-Roll (.mp4)
                   ↓
  EVIDENCE METADATA RECORD (SQLAlchemy EvidenceModel)
                   ↓
            EVIDENCE VAULT
(Forensic cards, in-modal preview, player seek)
```

### Key Features
1. **Evidence Types:** Supports `snapshot_only`, `clip_only`, and `snapshot_and_clip` (default).
2. **Dual Snapshots:** Extracts both clean, unmanipulated original CCTV frame evidence and a secondary high-visibility annotated bounding box image (`#00e673` emerald border with class label, confidence %, and timestamp).
3. **Boundary-Clamped Clips:** Configurable pre-roll and post-roll buffers (default 3.0s prior, 3.0s following, total ~6s). Boundaries are dynamically clamped between `0.0s` and the video duration so clips never request negative timestamps or exceed the video length.
4. **Idempotency & Duplicate Mitigation:** Re-clicking `[Capture Evidence]` on the same detection or timestamp immediately resolves and returns the existing preserved evidence without wasting disk space or database rows.
5. **Secure Evidence Streaming:** Endpoints stream binary files via database record lookup (`/api/evidence/{id}/snapshot`, `/api/evidence/{id}/annotated`, `/api/evidence/{id}/clip`). Path traversal (`../`), direct filesystem exposure, and internal Windows paths are completely blocked.
6. **Zero Original Footage Mutation:** Extraction operates read-only on the uploaded CCTV footage. All derivative artifacts are securely namespaced in `storage/evidence/<evidence_id>_*`.

### Evidence APIs
- `POST /api/videos/{video_id}/evidence` — Extract snapshot and/or sub-clip evidence.
- `GET /api/videos/{video_id}/evidence` — List all preserved evidence items for a video.
- `GET /api/evidence/{evidence_id}` — Retrieve detailed forensic metadata for an evidence item.
- `GET /api/evidence/{evidence_id}/snapshot` — Stream JPEG original frame.
- `GET /api/evidence/{evidence_id}/annotated` — Stream JPEG bounding box annotated frame.
- `GET /api/evidence/{evidence_id}/clip` — Stream MP4 evidence video sub-clip.

### Ethical & Forensic Boundaries
- **Observational Provenance:** Evidence titles and notes describe facts (*"Person detected at 8.01s with 92% confidence"*), never legal or criminal conclusions (*"Thief identified"* or *"Guilty suspect"*).
- **No Identity Guesses:** Facial recognition and personal identity inference are permanently excluded.

---

## 7. Phase 7 — LLM-Assisted, Evidence-Grounded Video Investigation

Sentinel Phase 7 upgrades the platform from a keyword-based query engine into an **LLM-assisted, evidence-grounded investigative workbench**.

```
USER QUESTION ("What happened around 8 seconds?")
       │
       ▼
LLM UNDERSTANDING (Intent & Parameter Extraction)
       │
       ▼
STRUCTURED INVESTIGATION REQUEST (Type-safe JSON schema)
       │
       ▼
VALIDATION & ETHICAL GUARDRAILS ENFORCEMENT
       │
       ▼
EXISTING PHASE 5A INVESTIGATION SERVICE & DATABASE
       │
       ▼
GROUND-TRUTH RETRIEVAL (Events, Detections, Evidence records)
       │
       ▼
GROUNDED RESPONSE SYNTHESIS (Strictly constrained to retrieved context)
       │
       ▼
AUDITABLE ANSWER WITH TIMESTAMPS, CITATIONS & EVIDENCE LINKS
```

### Core Grounding Rule
**The LLM is NOT the source of truth.** The Sentinel relational database (`EventModel`, `GroupedEventModel`, `VideoModel`) and generated evidence records (`EvidenceModel`) are the sole sources of truth. The LLM acts as an **understanding translator** and **grounded synthesizer**. It never directly queries the database or invents detections, timestamps, confidence scores, evidence, or criminal attributions.

### Key Capabilities
1. **Natural Language Inquiry:** Comprehends complex, natural phrasing (*"Can you tell me what happened around 8 seconds?"*, *"Was there much activity between 8 and 12 seconds?"*, *"What objects were present during that period?"*).
2. **Intelligent Video Summary:** Automatically synthesizes high-level overviews (*"Summarize this video"*) including total duration, detection counts, breakdown by object classes, timeline event frequency, peak activity time windows, and preserved evidence tally.
3. **Observational Activity Analysis:** Detects elevated movement, temporal clustering, and multi-object co-occurrence (*"Any noteworthy activity?"*) while strictly using non-judgmental, observational terminology (*"Potentially noteworthy activity was observed around 8.01–9.01s because detection activity was higher than other periods in the video. This is an automated observational finding and does not establish criminal or malicious behavior."*).
4. **Audit Traceability & Citations:** Responses cite exact timestamps (`[08.01s]`), detection records (`[DET-XXXX]`), timeline events (`[EVENT-XXXX]`), and evidence items (`[EV-XXXX]`). Clicking citations in the dashboard immediately seeks the synchronized CCTV video player.
5. **Evidence-Aware Responses:** Directly links existing preserved evidence from the Evidence Vault, or provides a one-click **[Capture Evidence]** button when evidence has not yet been preserved for the surfaced window.
6. **Graceful Fallback:** If `LLM_API_KEY` is not provided or the LLM service experiences network issues, Sentinel **never crashes**. It automatically falls back to deterministic Phase 5A execution with the message: *"AI reasoning unavailable — deterministic investigation remains available."*

### Safety & Ethical Principles
- **No Facial Recognition or Personal Identity:** Inquiries such as *"Who is this?"* or *"What is his name?"* trigger an immediate safety refusal: *"Sentinel does not perform facial recognition or identity identification."*
- **No Criminal or Threat Attribution:** Does not label individuals as criminals, suspects, or thieves. Observations are restricted to measurable vision detections.
- **Vehicle Colour Hallucination Prevention:** Disregards unsupported vehicle color adjectives (e.g. "red car" is queried purely as "car") until dedicated attribute-extraction computer vision models are integrated.

### Configuration (`.env`):
```bash
LLM_PROVIDER=gemini                     # Provider: gemini or mock
LLM_API_KEY=your_gemini_api_key_here     # Google Gemini API key
GEMINI_API_KEY=your_gemini_api_key_here  # Alternate key alias
LLM_MODEL=gemini-1.5-flash              # Gemini model name
LLM_TEMPERATURE=0.1                     # Low temperature for deterministic synthesis
```

### New Phase 7 API:
- `POST /api/videos/{video_id}/ai-investigate` — Execute LLM-assisted, evidence-grounded investigation query with audit citations and evidence linkage.

---

## 8. How to Run the Application

From repository root (`c:\Sentinel`):

### Start Backend:
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Start Frontend:
```bash
cd frontend
npm run dev
```

### Run All Backend Tests (77/77 tests passing):
```bash
python -m pytest backend/tests/
```

### Run Real CCTV Verification Scripts:
```bash
# Phase 6 Evidence Verification:
python scripts/verify_phase6_real_video.py

# Phase 7 AI Investigation Verification:
python scripts/verify_phase7_real_video.py
```

