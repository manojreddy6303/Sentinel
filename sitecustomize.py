"""
Python sitecustomize hook for Sentinel.
Automatically registers PyTorch safe globals for trusted Ultralytics checkpoints
whenever Python initializes in the project environment.
"""
try:
    from ai.detection.torch_compat import configure_trusted_yolo_safe_globals
    configure_trusted_yolo_safe_globals()
except Exception:
    pass
