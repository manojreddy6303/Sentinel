# Sentinel Backend Service

FastAPI-powered backend service for the Sentinel Security Video Investigation Platform.

## Features (Current Foundation)

- REST API framework using FastAPI.
- Modular architecture (`core/`, `api/`, `main.py`).
- Built-in CORS support for Next.js frontend communication.
- Automated OpenAPI documentation at `/docs` and `/redoc`.
- Health check endpoint at `GET /api/health`.

## Installation & Setup

1. (Optional) Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the development server:
   From the repository root (`Sentinel/`):
   ```bash
   python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

4. Verify health endpoint:
   ```bash
   curl http://127.0.0.1:8000/api/health
   ```
