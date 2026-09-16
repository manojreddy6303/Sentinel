# Sentinel Local Storage Architecture

This directory serves as the local filesystem storage driver for the Sentinel MVP.

## Storage Hierarchy

```
storage/
├── uploads/     # Raw surveillance video files uploaded by investigators
├── evidence/    # Trimmed video evidence clips linked to detected events
└── reports/     # Generated investigation reports (PDF/Markdown/JSON)
```

## Architectural Guidelines & Data Retention Policy

1. **Non-Permanent Intermediate Frames:**
   - The platform will process video streams by sampling keyframes.
   - **Do not store every processed frame permanently.** Raw frames are analyzed in-memory or transiently buffered during CV passes.
   - Only significant event metadata and trimmed evidence clips (`storage/evidence/`) are permanently persisted.

2. **Storage Interface Abstraction:**
   - File access and storage operations are decoupled through a storage interface abstraction.
   - Local disk storage is used for initial development and MVP.
   - The interface is designed to seamlessly swap with cloud object storage (e.g., AWS S3, Google Cloud Storage, or MinIO) without changing application business logic.
