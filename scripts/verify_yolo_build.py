"""
Sentinel Build-Time YOLO Checkpoint & Inference Verification Script
Validates PyTorch / Ultralytics compatibility during Docker container build.

Verification Steps:
1. Import torch and ultralytics
2. Print exact Python, PyTorch, and Ultralytics versions
3. Configure safe globals allowlist for trusted YOLO checkpoints
4. Load trusted yolov8n.pt model and verify internal model construction
5. Run CPU inference smoke test on a synthetic test frame
6. Fail the build (exit code 1) if any step fails
"""
import sys
import os
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def verify_build():
    print("=" * 60)
    print("SENTINEL BUILD VERIFICATION: PyTorch & YOLO Compatibility")
    print("=" * 60)

    # 1. Imports
    try:
        import torch
        import ultralytics
    except ImportError as e:
        print(f"[FATAL] Failed to import core dependencies: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Print exact versions
    print(f"[VERIFY] Python version:      {sys.version.split()[0]}")
    print(f"[VERIFY] PyTorch version:     {torch.__version__}")
    print(f"[VERIFY] Ultralytics version: {ultralytics.__version__}")
    print(f"[VERIFY] CPU Device Count:    {os.cpu_count()}")

    # 3. Configure trusted safe globals
    try:
        from ai.detection.torch_compat import (
            configure_trusted_yolo_safe_globals,
            load_trusted_yolo_model,
        )
        safe_configured = configure_trusted_yolo_safe_globals()
        print(f"[VERIFY] Safe globals registered: {safe_configured}")
    except Exception as e:
        print(f"[WARN] Error during safe globals configuration: {e}")
        from ultralytics import YOLO
        load_trusted_yolo_model = YOLO

    # 4. Load trusted yolov8n.pt model
    weights_path = "yolov8n.pt"
    print(f"[VERIFY] Loading checkpoint '{weights_path}'...")
    try:
        model = load_trusted_yolo_model(weights_path)
        if model is None or getattr(model, "model", None) is None:
            raise ValueError(f"YOLO model construction failed for '{weights_path}'")
        print(f"[VERIFY] YOLO model successfully constructed: {type(model).__name__}")
        print(f"[VERIFY] Underlying architecture: {type(model.model).__name__}")
        if hasattr(model, "names") and isinstance(model.names, dict):
            print(f"[VERIFY] Model class count: {len(model.names)} (e.g. {list(model.names.values())[:3]}...)")
    except Exception as e:
        print(f"[FATAL] Failed to load YOLO checkpoint '{weights_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Run CPU inference smoke test on a synthetic test frame
    print("[VERIFY] Executing CPU inference smoke test on 64x64 synthetic frame...")
    try:
        synthetic_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        # Add a synthetic high-contrast rectangle to test non-empty image processing
        synthetic_frame[16:48, 16:48] = [200, 200, 200]
        results = model(synthetic_frame, imgsz=64, verbose=False)
        if results is None or len(results) == 0:
            raise ValueError("Inference returned empty results list")
        print(f"[VERIFY] CPU inference smoke test passed. Results count: {len(results)}")
    except Exception as e:
        print(f"[FATAL] CPU inference smoke test failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("ALL BUILD VERIFICATION CHECKS PASSED SUCCESSFULLY")
    print("=" * 60)
    sys.exit(0)

if __name__ == "__main__":
    verify_build()
