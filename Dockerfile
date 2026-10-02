FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV, PyTorch, and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Build-time verification: validate PyTorch/Ultralytics versions, trusted YOLO model loading, and CPU inference
RUN python scripts/verify_yolo_build.py

# Set default production environment variables
ENV PYTHONPATH=.
ENV ENVIRONMENT=production
ENV SENTINEL_PROD=1
ENV DATABASE_URL=sqlite:////storage/sentinel.db
ENV STORAGE_BASE_DIR=/storage

# Phase 20.3: Memory Hardening & Thread Constraints for Railway 1GB containers
ENV OMP_NUM_THREADS=2
ENV TORCH_NUM_THREADS=2
ENV YOLO_BATCH_SIZE=4
ENV FRAME_CACHE_MAX_MEMORY_MB=64.0
ENV MAX_CONCURRENT_HEAVY_JOBS=1

EXPOSE 8000

CMD ["sh", "-c", "python scripts/init_railway_storage.py && uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
