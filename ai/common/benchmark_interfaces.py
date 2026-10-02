"""
Advanced Model Benchmark Interfaces (Phase 20.2 / Step 11)

Provides clean, unified abstract interfaces and adapters for future benchmarking:
- BaseObjectDetector (YOLO, RT-DETR, Grounding DINO, Selective SAHI, SAM 2)
- BaseMultiObjectTracker (ByteTrack, BoT-SORT)
- BaseVisualAttributeAnalyzer (Heuristic CV, CLIP / SigLIP, Local VLM)

CRITICAL ARCHITECTURAL SAFETY CONSTRAINT:
Advanced candidate models are interface-ready for empirical evaluation.
They are NOT set as default during this phase, preventing runtime regressions
or heavyweight unverified dependencies.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from ai.schemas import BoundingBox, TrackedObject


class BaseObjectDetector(ABC):
    """Abstract interface for object detection models."""

    @abstractmethod
    def detect(
        self,
        frame_bgr: np.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Execute object detection on a BGR frame.
        Returns list of detections with format:
        [{"object_class": str, "confidence": float, "bounding_box": dict}]
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata, architecture type, weights, and parameter count."""
        pass


class BaseMultiObjectTracker(ABC):
    """Abstract interface for multi-object tracking algorithms."""

    @abstractmethod
    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        timestamp: float,
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
    ) -> List[TrackedObject]:
        """Update active tracks with current frame detections."""
        pass

    @abstractmethod
    def finalize(self) -> List[TrackedObject]:
        """Finalize and return all tracks."""
        pass


class BaseVisualAttributeAnalyzer(ABC):
    """Abstract interface for visual attribute extraction models."""

    @abstractmethod
    def analyze(
        self,
        frame_bgr: np.ndarray,
        bbox: BoundingBox,
        timestamp: float,
        object_class: str,
        track_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze visual attributes for an object crop."""
        pass


# =========================================================================
# ADAPTER IMPLEMENTATIONS WRAPPING CURRENT PIPELINE COMPONENTS
# =========================================================================

class YOLOAdapter(BaseObjectDetector):
    """Adapter wrapping existing YOLOObjectDetector."""

    def __init__(self, detector=None):
        if detector is None:
            from ai.detection.detector import YOLOObjectDetector
            detector = YOLOObjectDetector()
        self.detector = detector

    def detect(
        self,
        frame_bgr: np.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        return self.detector.detect_objects(
            frame_bgr,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "YOLOv8n / YOLOv11",
            "family": "YOLO",
            "is_default": True,
            "status": "active_production",
        }


class SelectiveSAHIAdapter(BaseObjectDetector):
    """Adapter wrapping selective SAHI-gated detection."""

    def __init__(self, detector=None):
        if detector is None:
            from ai.detection.detector import YOLOObjectDetector
            detector = YOLOObjectDetector()
        self.detector = detector

    def detect(
        self,
        frame_bgr: np.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        # Uses the detector's built-in SAHI evaluation path
        return self.detector.detect_objects(
            frame_bgr,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Selective SAHI + YOLO",
            "family": "SAHI_Gated",
            "is_default": False,
            "status": "benchmark_ready",
        }


class ByteTrackAdapter(BaseMultiObjectTracker):
    """Adapter wrapping Sentinel's ByteTrack-style ObjectTracker."""

    def __init__(self, tracker=None):
        if tracker is None:
            from ai.tracking.tracker import ObjectTracker
            tracker = ObjectTracker()
        self.tracker = tracker

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        timestamp: float,
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
    ) -> List[TrackedObject]:
        return self.tracker.update(
            detections,
            frame_idx=frame_idx,
            timestamp=timestamp,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    def finalize(self) -> List[TrackedObject]:
        return self.tracker.finalize()


class PersonAttributeAdapter(BaseVisualAttributeAnalyzer):
    """Adapter wrapping PersonAttributeAnalyzer."""

    def __init__(self, analyzer=None):
        if analyzer is None:
            from ai.attributes.person_analyzer import PersonAttributeAnalyzer
            analyzer = PersonAttributeAnalyzer()
        self.analyzer = analyzer

    def analyze(
        self,
        frame_bgr: np.ndarray,
        bbox: BoundingBox,
        timestamp: float,
        object_class: str,
        track_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self.analyzer.analyze(
            frame_bgr=frame_bgr,
            person_bbox=bbox,
            timestamp=timestamp,
            track_id=track_id,
        )
        return result.to_dict()


class VehicleColorAdapter(BaseVisualAttributeAnalyzer):
    """Adapter wrapping VehicleColorAnalyzer."""

    def __init__(self, analyzer=None):
        if analyzer is None:
            from ai.attributes.color_analyzer import VehicleColorAnalyzer
            analyzer = VehicleColorAnalyzer()
        self.analyzer = analyzer

    def analyze(
        self,
        frame_bgr: np.ndarray,
        bbox: BoundingBox,
        timestamp: float,
        object_class: str,
        track_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self.analyzer.analyze(
            frame_bgr=frame_bgr,
            bbox=bbox,
            timestamp=timestamp,
            object_class=object_class,
            track_id=track_id,
        )
        return result.to_dict()


# =========================================================================
# FUTURE BENCHMARK CANDIDATE STUBS (PLUGGABLE & BENCHMARK-READY)
# =========================================================================

class RTDETRBenchmarkCandidate(BaseObjectDetector):
    """Pluggable candidate for RT-DETR evaluation."""

    def detect(self, frame_bgr: np.ndarray, confidence_threshold: float = 0.25, iou_threshold: float = 0.45):
        raise NotImplementedError("RT-DETR benchmark candidate: requires weight download and runtime benchmark.")

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": "RT-DETR-L", "family": "Transformer_Detector", "is_default": False, "status": "candidate"}


class GroundingDINOBenchmarkCandidate(BaseObjectDetector):
    """Pluggable candidate for Grounding DINO open-vocabulary evaluation."""

    def detect(self, frame_bgr: np.ndarray, confidence_threshold: float = 0.25, iou_threshold: float = 0.45):
        raise NotImplementedError("Grounding DINO candidate: requires open-vocabulary text query prompt configuration.")

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": "Grounding DINO", "family": "Open_Vocabulary_Detector", "is_default": False, "status": "candidate"}


class SAM2BenchmarkCandidate(BaseObjectDetector):
    """Pluggable candidate for SAM 2 zero-shot segmentation evaluation."""

    def detect(self, frame_bgr: np.ndarray, confidence_threshold: float = 0.25, iou_threshold: float = 0.45):
        raise NotImplementedError("SAM 2 candidate: requires prompt box/point configuration.")

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": "SAM 2", "family": "Segment_Anything", "is_default": False, "status": "candidate"}


class BoTSORTBenchmarkCandidate(BaseMultiObjectTracker):
    """Pluggable candidate for BoT-SORT camera-motion compensated tracking."""

    def update(self, detections, frame_idx, timestamp, frame_width=None, frame_height=None):
        raise NotImplementedError("BoT-SORT candidate: requires camera motion compensation module.")

    def finalize(self):
        return []


class CLIPVisualAttributeBenchmarkCandidate(BaseVisualAttributeAnalyzer):
    """Pluggable candidate for CLIP / SigLIP zero-shot attribute classification."""

    def analyze(self, frame_bgr, bbox, timestamp, object_class, track_id=None):
        raise NotImplementedError("CLIP candidate: requires open-source clip-vit-base weights.")


class LocalVLMBenchmarkCandidate(BaseVisualAttributeAnalyzer):
    """Pluggable candidate for local VLM (Moondream/PaliGemma/Qwen2-VL) visual reasoning."""

    def analyze(self, frame_bgr, bbox, timestamp, object_class, track_id=None):
        raise NotImplementedError("Local VLM candidate: requires local Ollama / vLLM multimodal server.")
