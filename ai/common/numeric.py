"""
Sentinel Universal Finite Number & Numeric Integrity Module (Phase 15.1)

Guarantees that no NaN, Infinity, -Infinity, or unhandled non-finite values
propagate across detectors, schemas, databases, or API responses.
"""
import math
from typing import Any, Optional, Union, Dict, List


def is_finite_number(val: Any) -> bool:
    """
    Return True if val is an int or float and is neither NaN nor Infinite.
    Booleans are excluded (isinstance(True, int) is True in Python).
    """
    if val is None or isinstance(val, bool):
        return False
    if isinstance(val, (int, float)):
        try:
            return not (math.isnan(val) or math.isinf(val))
        except (TypeError, ValueError):
            return False
    return False


def ensure_finite(val: Any, default: Optional[float] = 0.0) -> Optional[float]:
    """
    Ensure value is a finite float. If not finite, returns the default value.
    If default is None, returns None.
    """
    if is_finite_number(val):
        return float(val)
    return default


def clamp_finite(
    val: Any,
    min_val: float,
    max_val: float,
    default: float = 0.0,
) -> float:
    """
    Ensure val is finite, then clamp it to [min_val, max_val].
    If val is not finite, returns default clamped to [min_val, max_val].
    """
    finite_val = ensure_finite(val, default=default)
    if finite_val is None:
        finite_val = default
    return max(min_val, min(max_val, finite_val))


def sanitize_for_json(obj: Any, non_finite_replacement: Any = None) -> Any:
    """
    Recursively traverse dictionaries, lists, tuples and replace
    any NaN, Infinity, or -Infinity floats with non_finite_replacement (default None).
    Guarantees compliance with RFC 8259 JSON specifications.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        if is_finite_number(obj):
            return obj
        return non_finite_replacement
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v, non_finite_replacement) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item, non_finite_replacement) for item in obj]
    return obj
