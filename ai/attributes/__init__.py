"""
Universal Visual Attributes & Analysis Module for Sentinel (Phase 20.2)
"""
from ai.attributes.color_analyzer import (
    VehicleColorAnalyzer,
    RobustColorExtractor,
    TemporalColorFilter,
)
from ai.attributes.person_analyzer import PersonAttributeAnalyzer

__all__ = [
    "VehicleColorAnalyzer",
    "RobustColorExtractor",
    "TemporalColorFilter",
    "PersonAttributeAnalyzer",
]
