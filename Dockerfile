FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV, PyTorch, and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Set default production environment variables
ENV PYTHONPATH=.
ENV ENVIRONMENT=production
ENV SENTINEL_PROD=1
ENV DATABASE_URL=sqlite:////storage/sentinel.db
ENV STORAGE_BASE_DIR=/storage

EXPOSE 8000

CMD ["sh", "-c", "python scripts/init_railway_storage.py && uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
