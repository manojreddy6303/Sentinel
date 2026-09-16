"""Event generation and timeline aggregation package."""
from .generator import EventGenerator, create_detection_event
from .repository import EventRepository, JSONEventRepository, get_event_repository

__all__ = [
    "EventGenerator",
    "create_detection_event",
    "EventRepository",
    "JSONEventRepository",
    "get_event_repository",
]
