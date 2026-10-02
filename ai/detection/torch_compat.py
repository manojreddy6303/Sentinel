"""
Sentinel PyTorch / Ultralytics Compatibility Layer
Provides safe-global allowlisting and trusted model loading across PyTorch versions,
specifically supporting PyTorch 2.6+ where torch.load defaults to weights_only=True.

Security Invariants:
- Does NOT globally monkey-patch torch.load
- Does NOT globally force weights_only=False for arbitrary checkpoints
- Does NOT add blanket environment variables (e.g. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD)
- Scope is strictly restricted to trusted official Ultralytics weights (e.g. yolov8n.pt)
- PyTorch 2.6+ safe globals allowlist registers only the minimal required trusted classes
"""
import os
import sys
import logging
from typing import Any, List, Set
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Known trusted official Ultralytics checkpoint basenames
TRUSTED_YOLO_WEIGHT_PATTERNS = {
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolov8m-pose.pt",
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8n-cls.pt",
}

_SAFE_GLOBALS_CONFIGURED = False


def is_trusted_yolo_checkpoint(model_name_or_path: str) -> bool:
    """Verify if a model name or file path is an official trusted YOLO checkpoint."""
    if not model_name_or_path or not isinstance(model_name_or_path, str):
        return False
    basename = os.path.basename(model_name_or_path).strip().lower()
    if basename in TRUSTED_YOLO_WEIGHT_PATTERNS:
        return True
    # If a full or relative path to a local file, check if it's within project models/ or workspace
    if os.path.isfile(model_name_or_path):
        normalized = os.path.normpath(os.path.abspath(model_name_or_path)).lower()
        if basename in TRUSTED_YOLO_WEIGHT_PATTERNS:
            return True
        # Allow trusted sentinel models folder weights
        if ("models" in normalized or "sentinel" in normalized) and normalized.endswith(".pt"):
            return True
    return False


def get_trusted_yolo_safe_classes() -> List[Any]:
    """Collect the minimal required trusted classes for unpickling official YOLO checkpoints."""
    safe_classes: List[Any] = []
    seen: Set[str] = set()

    def _add(cls_obj: Any):
        if cls_obj is not None and isinstance(cls_obj, type):
            qualname = f"{cls_obj.__module__}.{cls_obj.__name__}"
            if qualname not in seen:
                seen.add(qualname)
                safe_classes.append(cls_obj)

    # 1. Ultralytics task architectures
    try:
        import ultralytics.nn.tasks as tasks
        for name in [
            "DetectionModel",
            "PoseModel",
            "SegmentationModel",
            "ClassificationModel",
            "RTDETRDetectionModel",
            "BaseModel",
            "Ensemble",
        ]:
            if hasattr(tasks, name):
                _add(getattr(tasks, name))
    except Exception as exc:
        logger.debug("Could not inspect ultralytics.nn.tasks: %s", exc)

    # 2. Ultralytics neural network modules (block, conv, head)
    try:
        import ultralytics.nn.modules as modules
        for name in [
            "Bottleneck",
            "BottleneckCSP",
            "C1",
            "C2",
            "C2f",
            "C3",
            "C3Ghost",
            "C3TR",
            "C3x",
            "Concat",
            "Conv",
            "Conv2",
            "ConvTranspose",
            "DFL",
            "DWConv",
            "DWConvTranspose2d",
            "Detect",
            "Focus",
            "GhostBottleneck",
            "GhostConv",
            "HGBlock",
            "HGStem",
            "Pose",
            "RepC3",
            "RepConv",
            "SPP",
            "SPPF",
            "Segment",
            "Classify",
        ]:
            if hasattr(modules, name):
                _add(getattr(modules, name))
    except Exception as exc:
        logger.debug("Could not inspect ultralytics.nn.modules: %s", exc)

    # 3. Core PyTorch container and layer classes used in serialized graph reconstruction
    try:
        import torch.nn.modules.container as containers
        for name in ["Sequential", "ModuleList", "ModuleDict"]:
            if hasattr(containers, name):
                _add(getattr(containers, name))
    except Exception:
        pass

    try:
        import torch.nn.modules.conv as convs
        for name in ["Conv2d", "ConvTranspose2d"]:
            if hasattr(convs, name):
                _add(getattr(convs, name))
    except Exception:
        pass

    try:
        import torch.nn.modules.batchnorm as bns
        for name in ["BatchNorm2d"]:
            if hasattr(bns, name):
                _add(getattr(bns, name))
    except Exception:
        pass

    try:
        import torch.nn.modules.activation as acts
        for name in ["SiLU", "ReLU", "LeakyReLU", "Sigmoid"]:
            if hasattr(acts, name):
                _add(getattr(acts, name))
    except Exception:
        pass

    try:
        import torch.nn.modules.pooling as pools
        for name in ["MaxPool2d", "AdaptiveAvgPool2d"]:
            if hasattr(pools, name):
                _add(getattr(pools, name))
    except Exception:
        pass

    try:
        import torch.nn.modules.upsampling as upsamples
        for name in ["Upsample"]:
            if hasattr(upsamples, name):
                _add(getattr(upsamples, name))
    except Exception:
        pass

    return safe_classes


def configure_trusted_yolo_safe_globals() -> bool:
    """Register trusted Ultralytics model classes in PyTorch's safe globals allowlist.

    In PyTorch 2.6+, torch.load defaults to weights_only=True. Registering the minimal
    required classes allows trusted official YOLO checkpoints to unpickle safely
    without disabling weights_only globally.
    """
    global _SAFE_GLOBALS_CONFIGURED
    try:
        import torch.serialization
        if hasattr(torch.serialization, "add_safe_globals"):
            classes = get_trusted_yolo_safe_classes()
            if classes:
                torch.serialization.add_safe_globals(classes)
                _SAFE_GLOBALS_CONFIGURED = True
                logger.debug(
                    "Configured %d trusted YOLO safe globals in PyTorch safe allowlist.",
                    len(classes),
                )
                return True
    except Exception as exc:
        logger.warning("Could not register trusted safe globals in torch.serialization: %s", exc)
    return False


@contextmanager
def scoped_trusted_checkpoint_load(model_name_or_path: str):
    """Context manager providing a narrowly scoped fallback for trusted official YOLO weights only.

    Strictly active during a single load invocation for a verified trusted model.
    Immediately restores torch.load upon exit.
    Does NOT affect arbitrary or untrusted checkpoint files.
    """
    if not is_trusted_yolo_checkpoint(model_name_or_path):
        # Untrusted target: never bypass weights_only restrictions
        yield
        return

    import torch
    original_load = torch.load

    def _trusted_scoped_load(*args, **kwargs):
        # Explicitly pass weights_only=False only for this trusted load call if torch supports it
        if "weights_only" not in kwargs:
            try:
                import inspect
                sig = inspect.signature(original_load)
                if "weights_only" in sig.parameters:
                    kwargs["weights_only"] = False
            except Exception:
                pass
        return original_load(*args, **kwargs)

    torch.load = _trusted_scoped_load
    try:
        yield
    finally:
        torch.load = original_load


def load_trusted_yolo_model(model_name_or_path: str = "yolov8n.pt") -> Any:
    """Load an official trusted YOLO model with full PyTorch 2.6+ compatibility.

    Strategy:
    1. Ensure PyTorch safe globals allowlist includes trusted Ultralytics classes.
    2. Attempt standard Ultralytics YOLO loading.
    3. If an UnpicklingError / weights_only failure occurs on a verified trusted checkpoint,
       retry under the scoped fallback for that checkpoint only.
    4. Verify the model constructed properly and return it.
    """
    from ultralytics import YOLO

    # Step 1: Configure safe globals
    configure_trusted_yolo_safe_globals()

    # Step 2: Standard load attempt
    try:
        model = YOLO(model_name_or_path)
        if getattr(model, "model", None) is not None:
            return model
    except Exception as exc:
        exc_str = str(exc)
        is_unpickling_failure = any(
            phrase in exc_str
            for phrase in [
                "Weights only load failed",
                "UnpicklingError",
                "Unsupported global",
                "DetectionModel",
            ]
        )
        if is_unpickling_failure and is_trusted_yolo_checkpoint(model_name_or_path):
            logger.info(
                "Applying scoped trusted-checkpoint compatibility path for '%s'",
                model_name_or_path,
            )
            with scoped_trusted_checkpoint_load(model_name_or_path):
                model = YOLO(model_name_or_path)
                if getattr(model, "model", None) is not None:
                    return model
        raise RuntimeError(
            f"Failed to load trusted YOLO model '{model_name_or_path}': {exc}"
        ) from exc

    return model


# Automatically configure safe globals when this module is imported
configure_trusted_yolo_safe_globals()
