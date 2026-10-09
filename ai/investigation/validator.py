"""
Structured Investigation Intent Validator & Guardrail Engine

Ensures that LLM-generated query parameters are strictly validated before
hitting the Sentinel database layer:
- Object class normalization and verification
- Temporal window validation (positive floats, start <= end)
- Confidence boundary enforcement [0.0, 1.0]
- Ethical guardrail checks:
    - Identity & facial recognition protection
    - Criminal/threat attribution refusal
    - Vehicle color hallucination prevention
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# Standard surveillance classes supported by YOLO & Sentinel
VALID_SURVEILLANCE_CLASSES = set(settings.SURVEILLANCE_CLASSES) | {"vehicle_group", "dog"}

# Common synonyms mapped to canonical COCO classes
SYNONYM_MAP = {
    "cars": "car",
    "automobile": "car",
    "automobiles": "car",
    "vehicle": "vehicle_group",
    "vehicles": "vehicle_group",
    "people": "person",
    "persons": "person",
    "human": "person",
    "humans": "person",
    "pedestrian": "person",
    "pedestrians": "person",
    "man": "person",
    "men": "person",
    "woman": "person",
    "women": "person",
    "child": "person",
    "children": "person",
    "individual": "person",
    "individuals": "person",
    "bikes": "bicycle",
    "bicycles": "bicycle",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "motorcycles": "motorcycle",
    "trucks": "truck",
    "buses": "bus",
    "bag": "backpack",
    "bags": "backpack",
    "purses": "handbag",
    "purse": "handbag",
    "phone": "cell phone",
    "phones": "cell phone",
    "cellphone": "cell phone",
}

# Ethical / unsupported regex triggers
IDENTITY_PATTERNS = [
    r"\b(who\s+is|who\s+are|who\s+was)\b",
    r"\b(name\s+of|names\s+of|person'?s\s+name|what\s+is\s+(?:the|this|that|his|her|their)\s+(?:person'?s\s+)?name)\b",
    r"\b(facial\s+recognition|face\s+id|identify\s+(?:this|the|a)?\s*(?:person|people|man|woman|human|individual)|identity)\b",
]

CRIMINAL_PATTERNS = [
    r"\b(criminal|criminals|thief|thieves|suspect|suspects|terrorist|guilty)\b",
    r"\bdefinitely\s+(?:a\s+)?(?:theft|crime|robbery|stealing)\b",
    r"\bis\s+this\s+a\s+crime\b",
]

COLOR_ADJECTIVES = {
    "red", "blue", "green", "black", "white", "silver", "gray", "grey", "yellow", "orange", "dark", "bright"
}


class ValidationError(Exception):
    """Raised when structured intent violates schema or guardrails."""
    pass


class StructuredIntentValidator:
    """Validates and sanitizes structured query intent produced by LLMs."""

    @classmethod
    def check_guardrails(cls, text: str) -> Optional[Dict[str, Any]]:
        """
        Check for ethical violations (identity inquiry, criminal labeling).
        Returns a guardrail violation dictionary if triggered, else None.
        """
        t = text.lower()

        # 1. Criminal Attribution Guardrail
        for pat in CRIMINAL_PATTERNS:
            if re.search(pat, t):
                return {
                    "is_supported": False,
                    "guardrail_triggered": "criminal_attribution",
                    "message": (
                        "Sentinel provides observational detection data only. "
                        "It does not assess criminal culpability, identify suspects, or infer illegal intent."
                    ),
                    "intent": "guardrail_rejected",
                }

        # 2. Facial Recognition / Identity Guardrail
        # Exempt anonymous surveillance questions like "who was involved", "who took", "who interacted", "who was present"
        is_tracking_inquiry = bool(
            re.search(r"\bwho\s+(?:was|is)\s+(?:involved|present|there|detected|seen|interacting|moving|active)\b", t)
            or re.search(r"\bwho\s+(?:took|interacted|moved|touched)\b", t)
            or re.search(r"\b(who\s+was\s+in|who\s+all\s+were)\b", t)
        )
        if not is_tracking_inquiry:
            for pat in IDENTITY_PATTERNS:
                if re.search(pat, t):
                    return {
                        "is_supported": False,
                        "guardrail_triggered": "identity",
                        "message": "Sentinel does not perform facial recognition or identity identification.",
                        "intent": "guardrail_rejected",
                    }

        return None

    @classmethod
    def validate_and_sanitize(
        cls, raw_intent: Dict[str, Any], raw_query_text: str = ""
    ) -> Dict[str, Any]:
        """
        Validate and sanitize the structured intent.
        Raises ValidationError if parameters are invalid.
        """
        if not isinstance(raw_intent, dict):
            raise ValidationError("Structured intent must be a JSON dictionary.")

        # Check raw query guardrails first
        guardrail_hit = cls.check_guardrails(raw_query_text)
        if guardrail_hit:
            return guardrail_hit

        intent_type = raw_intent.get("intent", "investigate")
        if raw_intent.get("guardrail_category") == "identity":
            return {
                "is_supported": False,
                "guardrail_triggered": "identity",
                "message": "Sentinel does not perform facial recognition or identity identification.",
                "intent": "guardrail_rejected",
            }

        sanitized: Dict[str, Any] = {
            "intent": intent_type,
            "is_supported": True,
            "object_class": None,
            "start_time": None,
            "end_time": None,
            "min_confidence": None,
            "event_type": raw_intent.get("event_type"),
            "color": raw_intent.get("color"),
            "track_id": raw_intent.get("track_id"),
            "result_type": raw_intent.get("result_type", "detections"),
            "is_summary_request": bool(raw_intent.get("is_summary_request", False)),
            "is_activity_request": bool(raw_intent.get("is_activity_request", False)),
        }

        # Check raw query for clothing color / track inquiry (e.g. "blue colour lady", "lady in blue")
        if raw_query_text:
            q_lower = raw_query_text.lower()
            if not sanitized["color"]:
                for c in ["blue", "red", "black", "white", "green", "yellow", "orange", "silver", "gray", "grey"]:
                    if re.search(rf"\b{c}\b", q_lower):
                        sanitized["color"] = c
                        break
            if any(term in q_lower for term in ["lady", "ladies", "dress", "skirt", "attire", "wearing", "clothing", "coat", "hoodie", "jacket"]):
                if not sanitized["object_class"]:
                    sanitized["object_class"] = "person"
            if sanitized["color"] or sanitized["track_id"] or (sanitized["object_class"] == "person" and any(term in q_lower for term in ["lady", "dress", "wearing", "color", "colour"])):
                if not sanitized["event_type"]:
                    sanitized["result_type"] = "tracks"

            # Specific event prioritization: recognize crowd dispersal inquiries generically
            if re.search(r"\b(?:crowd\s+dispers\w*|did\s+(?:the|a)?\s*crowd\s+disperse|dispers\w*\s+as\s+a\s+crowd|people\s+disperse\s+as\s+a\s+crowd|any\s+crowd\s+dispers\w*|dispersals?|crowd\s+disperse)\b", q_lower):
                sanitized["intent"] = "investigate"
                sanitized["event_type"] = "POTENTIAL_CROWD_DISPERSAL"
                sanitized["category"] = "crowd"
                sanitized["result_type"] = "security_events"
                sanitized["is_activity_request"] = False
                sanitized["is_summary_request"] = False

        # Priority order: specific event intent > generic activity intent
        if sanitized.get("event_type"):
            sanitized["is_activity_request"] = False

        # Validate object classes
        raw_classes = raw_intent.get("object_classes")
        if raw_classes:
            if isinstance(raw_classes, str):
                raw_classes = [raw_classes]
            elif not isinstance(raw_classes, list):
                raise ValidationError("object_classes must be a list of strings or null.")

            valid_class = None
            for c in raw_classes:
                if not isinstance(c, str):
                    continue
                c_clean = c.lower().strip()

                # Prevent color hallucination in class name (e.g. "red car" -> "car")
                parts = c_clean.split()
                clean_parts = [p for p in parts if p not in COLOR_ADJECTIVES]
                c_clean = " ".join(clean_parts) if clean_parts else c_clean

                # Map synonym
                mapped = SYNONYM_MAP.get(c_clean, c_clean)
                if mapped in VALID_SURVEILLANCE_CLASSES:
                    valid_class = mapped
                    break

            sanitized["object_class"] = valid_class

        # Validate start_time and end_time
        start_t = raw_intent.get("start_time")
        end_t = raw_intent.get("end_time")

        if start_t is not None:
            try:
                start_val = float(start_t)
                if start_val < 0.0:
                    raise ValidationError("start_time cannot be negative.")
                sanitized["start_time"] = round(start_val, 2)
            except (ValueError, TypeError):
                raise ValidationError(f"Invalid start_time: {start_t}")

        if end_t is not None:
            try:
                end_val = float(end_t)
                if end_val < 0.0:
                    raise ValidationError("end_time cannot be negative.")
                sanitized["end_time"] = round(end_val, 2)
            except (ValueError, TypeError):
                raise ValidationError(f"Invalid end_time: {end_t}")

        if (
            sanitized["start_time"] is not None
            and sanitized["end_time"] is not None
            and sanitized["start_time"] > sanitized["end_time"]
        ):
            # Invert if start > end
            sanitized["start_time"], sanitized["end_time"] = (
                sanitized["end_time"],
                sanitized["start_time"],
            )

        # Validate confidence
        min_conf = raw_intent.get("min_confidence")
        if min_conf is not None:
            try:
                conf_val = float(min_conf)
                # If given as percentage (e.g. 70), normalize to [0, 1]
                if conf_val > 1.0 and conf_val <= 100.0:
                    conf_val = conf_val / 100.0
                if conf_val < 0.0 or conf_val > 1.0:
                    raise ValidationError(f"Confidence must be between 0.0 and 1.0 (got {conf_val}).")
                sanitized["min_confidence"] = round(conf_val, 4)
            except (ValueError, TypeError):
                raise ValidationError(f"Invalid min_confidence: {min_conf}")

        # Validate result_type
        valid_result_types = [
            "detections", "events", "count", "security_events", "tracks",
            "vehicle_attributes", "faces", "specialized", "correlated_incidents", "evidence",
        ]
        if sanitized["result_type"] not in valid_result_types:
            if sanitized.get("color") or sanitized.get("track_id"):
                sanitized["result_type"] = "tracks"
            else:
                sanitized["result_type"] = "detections" if sanitized["object_class"] else "events"

        return sanitized
