"""
Tests for PyTorch 2.6+ / Ultralytics YOLO Checkpoint Compatibility.
Verifies safe-global allowlisting, trusted checkpoint validation,
clean model construction, and CPU inference.
"""
import os
import pytest
import numpy as np
import torch
from ai.detection.torch_compat import (
    configure_trusted_yolo_safe_globals,
    is_trusted_yolo_checkpoint,
    get_trusted_yolo_safe_classes,
    scoped_trusted_checkpoint_load,
    load_trusted_yolo_model,
)
from ai.detection.detector import YOLODetector


def test_trusted_safe_classes_collection():
    """Verify that trusted classes for YOLO unpickling are successfully discovered."""
    classes = get_trusted_yolo_safe_classes()
    assert len(classes) > 0, "No safe classes collected"
    class_names = [f"{c.__module__}.{c.__name__}" for c in classes]
    assert any("DetectionModel" in name for name in class_names), "DetectionModel must be in safe classes"
    assert any("Conv" in name for name in class_names), "Conv must be in safe classes"
    assert any("C2f" in name for name in class_names), "C2f must be in safe classes"


def test_safe_globals_registration():
    """Verify configure_trusted_yolo_safe_globals runs idempotently and without error."""
    result = configure_trusted_yolo_safe_globals()
    # On PyTorch 2.4+, result is True; on earlier PyTorch (<2.4), result is False
    if hasattr(torch.serialization, "add_safe_globals"):
        assert result is True
    else:
        assert result is False


def test_is_trusted_yolo_checkpoint():
    """Verify trusted vs untrusted checkpoint name discrimination."""
    assert is_trusted_yolo_checkpoint("yolov8n.pt") is True
    assert is_trusted_yolo_checkpoint("yolov8s.pt") is True
    assert is_trusted_yolo_checkpoint("yolov8n-pose.pt") is True
    assert is_trusted_yolo_checkpoint("yolov8m.pt") is True

    # Untrusted / arbitrary files
    assert is_trusted_yolo_checkpoint("arbitrary_weights.bin") is False
    assert is_trusted_yolo_checkpoint("malicious_model.pt") is False
    assert is_trusted_yolo_checkpoint("") is False
    assert is_trusted_yolo_checkpoint(None) is False


def test_scoped_trusted_load_restores_torch_load():
    """Verify that scoped_trusted_checkpoint_load restores torch.load upon exit."""
    orig_torch_load = torch.load
    with scoped_trusted_checkpoint_load("yolov8n.pt"):
        # Inside context
        pass
    # After exiting, torch.load must be exactly restored
    assert torch.load is orig_torch_load


def test_load_trusted_yolo_model():
    """Verify that yolov8n.pt loads cleanly into a valid YOLO instance."""
    model = load_trusted_yolo_model("yolov8n.pt")
    assert model is not None
    assert getattr(model, "model", None) is not None
    assert hasattr(model, "names")
    assert len(model.names) == 80


def test_yolo_cpu_inference_smoke():
    """Verify that loaded YOLO model performs CPU forward inference on a test frame."""
    model = load_trusted_yolo_model("yolov8n.pt")
    test_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    test_frame[16:48, 16:48] = [180, 180, 180]
    results = model(test_frame, imgsz=64, verbose=False)
    assert results is not None
    assert len(results) > 0


def test_yolo_detector_load_model():
    """Verify that Sentinel's YOLODetector initializes and loads model properly."""
    detector = YOLODetector(model_name="yolov8n.pt", confidence_threshold=0.25)
    model = detector._load_model()
    assert model is not None
    assert detector._model is model
    # Calling a second time returns cached model
    assert detector._load_model() is model
