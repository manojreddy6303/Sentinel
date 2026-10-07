"""
Natural-Language Investigation Query Parser & Engine for Sentinel

Translates human investigator queries into structured database filter operations:
- Object classes (singular, plural, synonyms)
- Temporal boundaries ("between X and Y seconds", "after X", "before Y", "at X")
- Confidence thresholds ("confidence above 70%", "confidence > 0.8")
- Query intent / result_type ("detections", "events", "count")
- Ethical guardrails: explicitly rejects unsupported identity/criminal attribution queries.
"""

import re
from typing import Dict, Any, Optional, Tuple, List

# Supported YOLO / surveillance object classes mapped from colloquial synonyms
OBJECT_SYNONYMS = {
    # Person
    "person": "person",
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
    "lady": "person",
    "ladies": "person",
    "guy": "person",
    "guys": "person",
    "girl": "person",
    "girls": "person",
    "dress": "person",
    "dresses": "person",
    "skirt": "person",
    "skirts": "person",
    "coat": "person",
    "coats": "person",
    "jacket": "person",
    "jackets": "person",
    "hoodie": "person",
    "hoodies": "person",
    "shirt": "person",
    "shirts": "person",
    "pants": "person",
    "trousers": "person",
    "clothes": "person",
    "clothing": "person",
    "attire": "person",

    # Vehicles
    "car": "car",
    "cars": "car",
    "automobile": "car",
    "automobiles": "car",
    "vehicle": "vehicle_group",
    "vehicles": "vehicle_group",
    "bus": "bus",
    "buses": "bus",
    "truck": "truck",
    "trucks": "truck",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "bike": "bicycle",
    "bikes": "bicycle",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "train": "train",
    "trains": "train",
    "boat": "boat",
    "boats": "boat",
    "airplane": "airplane",
    "airplanes": "airplane",
    "plane": "airplane",
    "planes": "airplane",

    # Belongings & Common COCO items
    "backpack": "backpack",
    "backpacks": "backpack",
    "bag": "backpack",
    "bags": "backpack",
    "handbag": "handbag",
    "handbags": "handbag",
    "purse": "handbag",
    "purses": "handbag",
    "suitcase": "suitcase",
    "suitcases": "suitcase",
    "luggage": "suitcase",
    "bottle": "bottle",
    "bottles": "bottle",
    "cell phone": "cell phone",
    "cellphone": "cell phone",
    "cellphones": "cell phone",
    "phone": "cell phone",
    "phones": "cell phone",
    "chair": "chair",
    "chairs": "chair",
    "traffic light": "traffic light",
    "traffic lights": "traffic light",
    "stop sign": "stop sign",
    "stop signs": "stop sign",
}

# Ethical / unsupported query patterns (identity recognition, criminal attribution, emotion/threat speculation)
UNSUPPORTED_PATTERNS = [
    r"\b(who\s+is|who\s+are|who\s+was)\b",
    r"\b(name\s+of|names\s+of|person'?s\s+name|what\s+is\s+(?:the|this|that)\s+person'?s\s+name)\b",
    r"\b(criminal|criminals|thief|thieves|suspect|suspects|terrorist|guilty)\b",
    r"\bdefinitely\s+(?:a\s+)?(?:theft|crime|robbery|stealing)\b",
    r"\bis\s+this\s+a\s+crime\b",
    r"\b(facial\s+recognition|face\s+id|identify\s+(?:this|the|a)?\s*person|identify\s+people|identity)\b",
    r"\b(danger|threat|dangerous)\b",
    r"\b(license\s+plate|plate\s+number)\b",
]

# Phase 8: Supported controlled vehicle color vocabulary
SUPPORTED_VEHICLE_COLORS = [
    "black", "white", "grey", "gray", "silver", "red", "blue", "green", "yellow", "orange", "brown"
]


class InvestigationParser:
    """Parses natural-language queries into structured filters for database execution."""

    @staticmethod
    def parse_query(user_query: str) -> Dict[str, Any]:
        cleaned = user_query.strip().lower() if user_query else ""

        if not cleaned:
            return {
                "is_supported": False,
                "message": "Please enter an investigation query.",
                "interpreted_filters": {},
                "result_type": "error",
            }

        # 1. Check for ethical and unsupported inquiries (identity, crime, facial recognition)
        # Exempt non-biometric anonymous surveillance questions like "who was involved", "who took", "who interacted"
        is_tracking_inquiry = bool(
            re.search(r"\bwho\s+(?:was|is)\s+(?:involved|present|there|detected|seen|interacting|moving|active)\b", cleaned)
            or re.search(r"\bwho\s+(?:took|interacted|moved|touched)\b", cleaned)
            or re.search(r"\b(who\s+was\s+in|who\s+all\s+were)\b", cleaned)
        )
        if not is_tracking_inquiry:
            for pat in UNSUPPORTED_PATTERNS:
                if re.search(pat, cleaned):
                    return {
                        "is_supported": False,
                        "message": (
                            "I can currently investigate detected objects, timestamps, confidence levels, and events. "
                            "Identity recognition, facial analysis, and criminal identification are not supported."
                        ),
                        "interpreted_filters": {},
                        "result_type": "unsupported",
                    }

        # 1a. Check for generic Evidence queries (e.g. "show me the evidence", "show the evidence", "view evidence")
        # Ensure it does not hijack domain event queries like "show smoke evidence", "show fire evidence", "show weapon evidence"
        is_domain_event = any(k in cleaned for k in ["smoke", "fire", "weapon", "gun", "knife", "fall", "collision", "crash", "car", "person"])
        if not is_domain_event and re.search(r"\b(?:show\s+(?:me\s+)?(?:the\s+)?(?:relevant\s+)?evidence|evidence\s+records?|view\s+evidence|evidence\s+clips?|evidence\s+snapshots?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "evidence",
                "interpreted_filters": {},
            }

        # 1b. Check for Phase 8 Face Detection queries (strictly visual region detections, no biometric identity)
        if re.search(r"\b(face\s+detections?|faces?\s+detected|show\s+faces?|detected\s+faces?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "faces",
                "message": "Observational visual face regions only. No biometric identity inference or database matching.",
                "interpreted_filters": {},
            }

        # 1c. Check for Phase 8 Vehicle Color Analysis queries
        matched_color = None
        for col in SUPPORTED_VEHICLE_COLORS:
            if re.search(r"\b" + col + r"\b", cleaned):
                matched_color = "grey" if col == "gray" else col
                break

        if matched_color:
            is_vehicle = bool(re.search(r"\b(car|cars|vehicle|vehicles|truck|trucks|bus|buses|motorcycle|motorcycles|automobile|automobiles)\b", cleaned))
            is_person = bool(re.search(
                r"\b(person|people|someone|suspect|individual|individuals|man|men|woman|women|lady|ladies|guy|guys|girl|girls|wearing|hoodie|hoodies|jacket|jackets|shirt|shirts|dress|dresses|skirt|skirts|pants|trousers|coat|coats|clothes|clothing|attire)\b",
                cleaned
            ))
            has_action = bool(re.search(
                r"\b(did|do|doing|done|action|actions|activity|movement|move|moved|moving|walk|walking|run|running|ran|stand|standing|stood|sit|sitting|sat|behavior|happened|events?)\b",
                cleaned
            ))
            if is_vehicle:
                # Extract specific vehicle class if present
                v_class = "vehicle_group"
                for v_syn in ["car", "truck", "bus", "motorcycle"]:
                    if re.search(r"\b" + v_syn + r"\b", cleaned):
                        v_class = v_syn
                        break
                return {
                    "is_supported": True,
                    "result_type": "vehicle_attributes",
                    "interpreted_filters": {
                        "color": matched_color,
                        "object_class": v_class,
                    },
                }
            elif is_person:
                interpreted_filters = {
                    "color": matched_color,
                    "object_class": "person",
                }
                if has_action:
                    interpreted_filters["query_action"] = True
                for descriptor in ["dress", "skirt", "jacket", "hoodie", "shirt", "coat", "pants", "suit"]:
                    if re.search(r"\b" + descriptor + r"\b", cleaned):
                        interpreted_filters["clothing_descriptor"] = descriptor
                        break
                return {
                    "is_supported": True,
                    "result_type": "tracks",
                    "interpreted_filters": interpreted_filters,
                }
            else:
                # Color queried for non-supported category
                return {
                    "is_supported": False,
                    "message": (
                        "Visual color analysis is supported for vehicles and person clothing. "
                        "Sentinel can investigate object colors (e.g., blue person, red truck), classes, timestamps, and events."
                    ),
                    "interpreted_filters": {},
                    "result_type": "unsupported",
                }

        # 1d. Check for Phase 8 Object Tracking queries
        track_id_match = re.search(r"\b(track[-_\s]?\d+)\b", cleaned)
        if track_id_match:
            raw_track_str = track_id_match.group(1).replace("_", "-").replace(" ", "-").upper()
            # Normalize e.g. "TRACK 004" -> "TRACK-004", "TRACK 4" -> "TRACK-004"
            parts = raw_track_str.split("-")
            if len(parts) == 2 and parts[1].isdigit():
                norm_track_id = f"TRACK-{int(parts[1]):03d}"
            else:
                norm_track_id = raw_track_str
            return {
                "is_supported": True,
                "result_type": "tracks",
                "interpreted_filters": {
                    "track_id": norm_track_id,
                },
            }

        if re.search(r"\b(tracked\s+(?:people|person|persons|vehicles?|cars?)|all\s+tracks?|show\s+tracks?)\b", cleaned):
            trk_cls = None
            if re.search(r"\b(people|person|persons)\b", cleaned):
                trk_cls = "person"
            elif re.search(r"\b(vehicles?|cars?|trucks?)\b", cleaned):
                trk_cls = "car"
            return {
                "is_supported": True,
                "result_type": "tracks",
                "interpreted_filters": {
                    "object_class": trk_cls,
                },
            }

        # 1e. Check for Phase 8 Security Intelligence Events queries
        if (
            re.search(r"\b(potential\s+intrusions?|intrusions?)\b", cleaned)
            or (re.search(r"\b(restricted\s+zones?|restricted\s+areas?)\b", cleaned) and not re.search(r"\b(occupan|crowd)", cleaned))
        ):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_INTRUSION",
                },
            }

        if re.search(r"\b(prolonged\s+presence|loitering|lingering)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "PROLONGED_PRESENCE",
                },
            }

        if re.search(r"\b(abandoned\s+objects?|abandoned\s+items?|unattended\s+items?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_ABANDONED_OBJECT",
                },
            }

        if re.search(r"\b(activity\s+peaks?|high\s+activity|elevated\s+activity)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "HIGH_ACTIVITY_PERIOD",
                },
            }

        if re.search(r"\b(observational\s+anomal(?:y|ies)|unusual\s+activity\s+patterns?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "OBSERVATIONAL_ANOMALY",
                },
            }

        # 1f. Check for Phase 8.1 / Phase 13 Property & Object Incident Intelligence queries
        if re.search(r"\b(abandoned\s+objects?|abandoned\s+belongings?|unattended\s+bags?|unattended\s+objects?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_ABANDONED_OBJECT",
                },
            }

        if re.search(r"\b(objects?\s+left\s+behind|left\s+behind\s+objects?|belongings?\s+left\s+behind)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_OBJECT_LEFT_BEHIND",
                },
            }

        if re.search(r"\b(object\s+pickups?|pick(?:ing)?\s+up\s+objects?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_OBJECT_PICKUP",
                },
            }

        if re.search(r"\b(which\s+events?\s+should\s+i\s+review|what\s+should\s+i\s+review|events?\s+to\s+review|review\s+required\s+events?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "validation_decision": "REVIEW_REQUIRED",
                },
            }

        if re.search(r"\b(objects?\s+moved|moved\s+objects?|moving\s+objects?|displaced\s+objects?|objects?\s+displaced|object\s+displacements?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_OBJECT_DISPLACEMENT",
                },
            }

        if re.search(r"\b(property\s+tampering|tampering\s+events?|fixture\s+tampering)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_PROPERTY_TAMPERING",
                },
            }

        if re.search(r"\b(restricted\s+object\s+movements?|objects?\s+in\s+restricted\s+zone)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_RESTRICTED_OBJECT_MOVEMENT",
                },
            }

        if re.search(r"\b(removed\s+objects?|object\s+removals?|object\s+disappearances?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_OBJECT_REMOVAL",
                },
            }

        if re.search(r"\b(theft\s+patterns?|theft\s+events?|possible\s+thefts?|thefts?|stealing|stolen|what\s+was\s+stolen|object\s+takeaways?|takeaway\s+patterns?|takeaways?|burglary|taken|what\s+was\s+taken|was\s+anything\s+taken|did\s+anyone\s+take|anything\s+taken|take\s+anything|took\s+(?:the|an|anything|something)|who\s+took|interacted\s+with\s+(?:the\s+)?object|who\s+interacted|what\s+objects?\s+(?:were|was)\s+moved|objects?\s+moved|displaced\s+objects?)\b", cleaned):
            t_start, t_end, c_min, c_max = InvestigationParser._extract_time_and_conf(cleaned)
            filters: Dict[str, Any] = {"event_type": "POTENTIAL_THEFT"}
            if t_start is not None:
                filters["start_time"] = t_start
            if t_end is not None:
                filters["end_time"] = t_end
            if c_min is not None:
                filters["min_confidence"] = c_min
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": filters,
            }

        if re.search(r"\b(who\s+was\s+involved|who\s+is\s+involved|involved\s+people|who\s+was\s+present|everyone\s+present|people\s+involved)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "tracks",
                "interpreted_filters": {
                    "object_class": "person",
                },
            }

        if re.search(r"\b(what\s+are\s+(?:the\s+)?people\s+doing|people\s+activity|what\s+is\s+everyone\s+doing)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "tracks",
                "interpreted_filters": {
                    "object_class": "person",
                    "query_action": True,
                },
            }

        if re.search(r"\b(when\s+did\s+(?:the\s+)?(?:suspicious\s+activity|suspicious\s+event|incident|it|that)\s+(?:happen|occur|begin|start)|time\s+of\s+(?:the\s+)?incident)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {},
            }

        if re.search(r"\b(property\s+incidents?|property\s+events?|object\s+incidents?|object\s+events?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "category": "property",
                },
            }

        # 1g. Check for Phase 11 Vehicle Incident Intelligence queries
        if re.search(r"\b(vehicle\s+incidents?|vehicle\s+events?|traffic\s+incidents?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "category": "vehicle",
                },
            }

        if re.search(r"\b(potential\s+collisions?|collisions?|crash(?:es)?|vehicle\s+collisions?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_VEHICLE_COLLISION",
                },
            }

        if re.search(r"\b(near\s+collisions?|near\s+miss(?:es)?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_NEAR_COLLISION",
                },
            }

        if re.search(r"\b(sudden\s+stops?|abrupt\s+stops?|rapid\s+decelerations?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_SUDDEN_VEHICLE_STOP",
                },
            }

        if re.search(r"\b(wrong\s+ways?|wrong-ways?|counter\s+flow|opposing\s+traffic|opposing\s+flow)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_WRONG_WAY_VEHICLE",
                },
            }

        if re.search(r"\b(unusual\s+trajector(?:y|ies)|erratic\s+movement|erratic\s+driving|unusual\s+movement)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_UNUSUAL_VEHICLE_TRAJECTORY",
                },
            }

        if re.search(r"\b(stationary\s+vehicles?|stopped\s+vehicles?|vehicle\s+breakdowns?|breakdowns?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_STATIONARY_VEHICLE",
                },
            }

        # 1h. Check for Phase 12 Person Incident Intelligence queries
        if re.search(r"\b(person\s+incidents?|person\s+events?|pedestrian\s+incidents?|people\s+incidents?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "category": "person",
                },
            }

        if re.search(r"\b(potential\s+falls?|person\s+falls?|falls?|falling|tripped|tripping|did\s+(?:that|the|anyone|someone|he|she|this)?\s*(?:person|lady|man)?\s*(?:wearing\s+\w+)?\s*fall)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_PERSON_FALL",
                },
            }

        if re.search(r"\b(person\s+down|people\s+down|lying\s+down|on\s+the\s+ground|collapsed\s+person)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_PERSON_DOWN",
                },
            }

        if re.search(r"\b(panic\s+running|running|panic\s+dispersion|fleeing|sprinting)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_PANIC_RUNNING",
                },
            }

        if re.search(r"\b(unusual\s+rapid\s+movement|rapid\s+person\s+movement|abrupt\s+movement|fast\s+person)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "UNUSUAL_RAPID_PERSON_MOVEMENT",
                },
            }

        if re.search(r"\b(potential\s+altercations?|physical\s+altercations?|altercations?|fights?|fighting|scuffles?|brawls?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_PHYSICAL_ALTERCATION",
                },
            }

        if re.search(r"\b(forced\s+movements?|forced\s+walking|dragging\s+person|dragged)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_FORCED_MOVEMENT",
                },
            }

        if re.search(r"\b(person\s+following|following\s+each\s+other|following|stalking|tailing)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "PERSON_FOLLOWING",
                },
            }

        if re.search(r"\b(coordinated\s+movement|coordinated\s+person\s+movement|synchronized\s+movement|group\s+movement)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "COORDINATED_PERSON_MOVEMENT",
                },
            }

        # Phase 14: Crowd, Density & Zone Intelligence patterns
        if re.search(r"\b(crowd\s+surges?|find\s+crowd\s+surges?|potential\s+crowd\s+surge|surging\s+crowd)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_CROWD_SURGE",
                },
            }

        if re.search(r"\b(crowd\s+dispersals?|dispersal|crowd\s+dispersing|show\s+crowd\s+dispersal)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_CROWD_DISPERSAL",
                },
            }

        if re.search(r"\b(unusual\s+crowd\s+movement|abnormal\s+crowd\s+movement|show\s+unusual\s+crowd\s+movement|unexpected\s+crowd\s+flow)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_UNUSUAL_CROWD_MOVEMENT",
                },
            }

        if re.search(r"\b(restricted\s+zone\s+crowding|restricted\s+zone\s+occupancy|crowding\s+in\s+(?:the\s+)?(?:entrance|restricted|sterile|exit)?\s*(?:gate|zone)|crowding\s+in\s+the\s+entrance\s+gate)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_RESTRICTED_ZONE_CROWDING",
                },
            }

        if re.search(r"\b(unusual\s+zone\s+activity|abnormal\s+zone\s+activity|show\s+unusual\s+zone\s+activity)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_UNUSUAL_ZONE_ACTIVITY",
                },
            }

        if re.search(r"\b(crowded\s+areas?|high\s+density(?:\s+events?)?|crowd\s+density|pedestrian\s+density|high\s+pedestrian\s+density|crowding)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "HIGH_PEDESTRIAN_DENSITY",
                },
            }

        if re.search(r"\b(zone\s+occupancy|which\s+zones\s+had\s+(?:the\s+)?highest\s+occupancy|highest\s+occupancy|show\s+zone\s+occupancy)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "ZONE_OCCUPANCY_OBSERVATION",
                },
            }

        # 1g. Phase 15 Specialized Visual Detection Queries (Fire, Smoke, Weapon, Specialized All)
        t_start, t_end, c_min, c_max = InvestigationParser._extract_time_and_conf(cleaned)

        if re.search(r"\b(fire\s*(?:and|&|\+)\s*smoke|smoke\s*(?:and|&|\+)\s*fire)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_FIRE_SMOKE",
                    "start_time": t_start,
                    "end_time": t_end,
                    "min_confidence": c_min,
                    "max_confidence": c_max,
                },
            }

        if re.search(r"\b(fire\s+events?|flame\s+events?|show\s+fire|find\s+(?:possible\s+)?fire|fire\s+evidence|fire\s+detections?)\b", cleaned) or (re.search(r"\bfire\b", cleaned) and not re.search(r"\b(fire hydrant|fire extinguisher)\b", cleaned)):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_FIRE",
                    "start_time": t_start,
                    "end_time": t_end,
                    "min_confidence": c_min,
                    "max_confidence": c_max,
                },
            }

        if re.search(r"\b(smoke\s+events?|plume\s+events?|show\s+smoke|find\s+(?:possible\s+)?smoke|smoke\s+evidence|smoke\s+detections?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_SMOKE",
                    "start_time": t_start,
                    "end_time": t_end,
                    "min_confidence": c_min,
                    "max_confidence": c_max,
                },
            }

        if re.search(r"\b(weapon\s+detections?|potential\s+weapons?|show\s+potential\s+weapons?|suspicious\s+(?:visual\s+)?objects?|weapon\s+events?|knife\s+detections?|handgun\s+detections?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {
                    "event_type": "POTENTIAL_WEAPON_VISUAL",
                    "start_time": t_start,
                    "end_time": t_end,
                    "min_confidence": c_min,
                    "max_confidence": c_max,
                },
            }

        if re.search(r"\b(all\s+specialized\s+visual|specialized\s+visual\s+events?|specialized\s+events?|specialized\s+visual)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "specialized",
                "interpreted_filters": {
                    "start_time": t_start,
                    "end_time": t_end,
                    "min_confidence": c_min,
                    "max_confidence": c_max,
                },
            }

        # Phase 16: Advanced Incident Correlation, Fusion & Storyline Queries
        if re.search(r"\b(incidents?\s+involv(?:ing|ed)\s+(?:a\s+)?person\s+(?:and|with)\s+(?:an?\s+)?(?:object|item|suitcase|bag|belonging))\b", cleaned) or \
           re.search(r"\b(person\s+(?:and|with)\s+(?:an?\s+)?(?:object|item|suitcase|bag|belonging)\s+incidents?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "correlated_incidents",
                "interpreted_filters": {
                    "object_pair": ["person", "object"],
                },
            }

        if re.search(r"\b(incidents?\s+involv(?:ing|ed)\s+(?:the\s+same\s+)?person\s+(?:and|with)\s+(?:a\s+)?vehicle)\b", cleaned) or \
           re.search(r"\b(person\s+(?:and|with)\s+vehicle\s+incidents?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "correlated_incidents",
                "interpreted_filters": {
                    "object_pair": ["person", "vehicle"],
                },
            }

        if re.search(r"\b(which\s+events\s+happened\s+before\s+the\s+collision|events?\s+before\s+(?:the\s+)?collision|sequence\s+before\s+(?:the\s+)?collision)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "correlated_incidents",
                "interpreted_filters": {
                    "category": "VEHICLE",
                    "query_focus": "pre_collision_sequence",
                },
            }

        if re.search(r"\b(show\s+(?:the\s+)?(?:full\s+)?sequence\s+around\s+(?:the\s+)?(?:potential\s+)?theft|theft\s+sequence|theft\s+storyline)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "correlated_incidents",
                "interpreted_filters": {
                    "category": "PROPERTY",
                    "query_focus": "theft_storyline",
                },
            }

        if re.search(r"\b(what\s+evidence\s+supports\s+this\s+incident|evidence\s+supporting\s+incidents?|incident\s+evidence\s+provenance)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "correlated_incidents",
                "interpreted_filters": {
                    "query_focus": "evidence_support",
                },
            }

        if re.search(r"\b(which\s+incidents\s+were\s+reviewed\s+or\s+rejected|reviewed\s+or\s+rejected\s+incidents?|rejected\s+incidents?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "correlated_incidents",
                "interpreted_filters": {
                    "validation_decision": "REVIEW_REQUIRED",
                },
            }

        if re.search(r"\b(multiple\s+alerts\s+(?:that\s+)?describe\s+(?:the\s+)?same\s+event|duplicate\s+alerts?|fused\s+alerts?|correlated\s+alerts?|correlated\s+incidents?)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "correlated_incidents",
                "interpreted_filters": {
                    "query_focus": "duplicate_fusion",
                },
            }

        if re.search(r"\b(suspicious\s+activity|security\s+events?|events\s+to\s+review|which\s+events\s+should\s+i\s+review|review\s+events?|all\s+security\s+events)\b", cleaned):
            return {
                "is_supported": True,
                "result_type": "security_events",
                "interpreted_filters": {},
            }

        # 2. Determine result type: "count", "events", or "detections"
        result_type = "detections"
        if re.search(r"\b(how\s+many|count|total\s+number)\b", cleaned):
            result_type = "count"
        elif re.search(r"\b(events?|timeline|activities|activity|happened|occurred)\b", cleaned) and not re.search(r"\b(detections?|objects?)\b", cleaned):
            result_type = "events"

        # 3. Parse Object Class
        object_class: Optional[str] = None
        # Check two-word synonyms first
        for phrase in ["cell phone", "traffic light", "stop sign"]:
            if phrase in cleaned:
                object_class = OBJECT_SYNONYMS[phrase]
                break

        if not object_class:
            tokens = re.findall(r"[a-z0-9]+", cleaned)
            for token in tokens:
                if token in OBJECT_SYNONYMS:
                    object_class = OBJECT_SYNONYMS[token]
                    break

        # 4. Parse Time Range
        start_time: Optional[float] = None
        end_time: Optional[float] = None

        # Between X and Y [seconds|s]
        between_match = re.search(
            r"between\s+(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)(?:\s*(?:seconds|secs|s))?",
            cleaned,
        )
        if between_match:
            start_time = float(between_match.group(1))
            end_time = float(between_match.group(2))
        else:
            # From X to Y [seconds|s]
            from_to_match = re.search(
                r"from\s+(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)(?:\s*(?:seconds|secs|s))?",
                cleaned,
            )
            if from_to_match:
                start_time = float(from_to_match.group(1))
                end_time = float(from_to_match.group(2))
            else:
                # After X [seconds|s]
                after_match = re.search(
                    r"(?:after|past|from|>=|>)\s*(\d+(?:\.\d+)?)(?:\s*(?:seconds|secs|s))?",
                    cleaned,
                )
                if after_match:
                    start_time = float(after_match.group(1))

                # Before X [seconds|s]
                before_match = re.search(
                    r"(?:before|prior\s+to|until|<=|<)\s*(\d+(?:\.\d+)?)(?:\s*(?:seconds|secs|s))?",
                    cleaned,
                )
                if before_match:
                    end_time = float(before_match.group(1))

                # At / around X seconds
                at_match = re.search(
                    r"(?:at|around)\s+(\d+(?:\.\d+)?)(?:\s*(?:seconds|secs|s))?",
                    cleaned,
                )
                if at_match and start_time is None and end_time is None:
                    point = float(at_match.group(1))
                    start_time = max(0.0, point - 1.0)
                    end_time = point + 1.0

        # Minutes parsing: e.g. "around 1 minute", "at 2 mins"
        min_match = re.search(r"(?:at|around)\s+(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b", cleaned)
        if min_match and start_time is None and end_time is None:
            point_sec = float(min_match.group(1)) * 60.0
            start_time = max(0.0, point_sec - 5.0)
            end_time = point_sec + 5.0

        # Ensure start_time <= end_time
        if start_time is not None and end_time is not None and start_time > end_time:
            start_time, end_time = end_time, start_time

        # 5. Parse Confidence Filter
        min_confidence: Optional[float] = None
        max_confidence: Optional[float] = None

        # Confidence above / greater than / >= X[%]
        conf_above = re.search(
            r"(?:confidence|conf)\s*(?:above|greater\s+than|over|at\s+least|>=|>)\s*(\d+(?:\.\d+)?)\s*(%?)",
            cleaned,
        )
        if conf_above:
            val = float(conf_above.group(1))
            min_confidence = val / 100.0 if conf_above.group(2) == "%" or val > 1.0 else val

        # Confidence below / less than / <= X[%]
        conf_below = re.search(
            r"(?:confidence|conf)\s*(?:below|less\s+than|under|<=|<)\s*(\d+(?:\.\d+)?)\s*(%?)",
            cleaned,
        )
        if conf_below:
            val = float(conf_below.group(1))
            max_confidence = val / 100.0 if conf_below.group(2) == "%" or val > 1.0 else val

        is_unique = bool(re.search(r"\b(unique|distinct|different)\b", cleaned))

        return {
            "is_supported": True,
            "result_type": result_type,
            "interpreted_filters": {
                "object_class": object_class,
                "start_time": start_time,
                "end_time": end_time,
                "min_confidence": min_confidence,
                "max_confidence": max_confidence,
                "is_unique": is_unique,
            },
        }

    # ------------------------------------------------------------------
    # Phase 17: parse_investigation_query — builds InvestigationQuery
    # ------------------------------------------------------------------

    @staticmethod
    def parse_investigation_query(
        user_query: str,
        video_id: str,
        video_duration_seconds: float = 0.0,
    ) -> "InvestigationQuery":
        """
        Phase 17 entry point: parse natural-language query into a fully
        validated InvestigationQuery object.

        Temporal modes supported:
          around X seconds     → window of ±15s (configurable)
          between X and Y      → exact range [X, Y]
          from X to Y          → exact range [X, Y]
          after X / before Y   → open-ended bounds
          during X–Y           → range (synonym of between)
          overlapping X–Y      → incidents overlapping [X, Y]
          immediately before X → [X-30, X]
          immediately after X  → [X, X+30]
          before the collision → handled via correlated_only + category VEHICLE

        Validation decision filters:
          review required / pending review → REVIEW_REQUIRED
          accepted incidents               → ACCEPTED
          rejected incidents               → REJECTED
          superseded candidates            → SUPERSEDED

        Assessment score filters:
          high score / score above 0.7
          low score / score below 0.5

        Evidence filters:
          with evidence / evidence available → evidence_required=True

        Correlated filters:
          correlated incidents / fused incidents → correlated_only=True
        """
        from backend.app.services.investigation_query import (
            InvestigationQuery, DEFAULT_AROUND_WINDOW_SECONDS
        )

        cleaned = (user_query or "").strip().lower()
        q = InvestigationQuery(
            video_id=video_id,
            raw_query_text=user_query,
            video_duration_seconds=video_duration_seconds if video_duration_seconds > 0.0 else None,
        )

        # ---------------------------------------------------------------
        # 1. Temporal parsing (extended Phase 17)
        # ---------------------------------------------------------------
        def _parse_time_val(val_str: str, unit_str: Optional[str] = None) -> float:
            v = float(val_str)
            if unit_str and any(unit_str.strip().startswith(u) for u in ["minute", "min", "m"]):
                return v * 60.0
            return v

        # ---------------------------------------------------------------
        # 1. Temporal parsing (extended Phase 17: seconds and minutes)
        # ---------------------------------------------------------------
        # immediately before X [min/s]
        imm_before = re.search(
            r"immediately\s+(?:before|prior\s+to|preceding)\s+(\d+(?:\.\d+)?)(?:\s*(minutes?|mins?|m|seconds?|secs?|s))?",
            cleaned,
        )
        if imm_before:
            t = _parse_time_val(imm_before.group(1), imm_before.group(2))
            q.time_start = max(0.0, t - 30.0)
            q.time_end = t
        else:
            # immediately after X [min/s]
            imm_after = re.search(
                r"immediately\s+(?:after|following)\s+(\d+(?:\.\d+)?)(?:\s*(minutes?|mins?|m|seconds?|secs?|s))?",
                cleaned,
            )
            if imm_after:
                t = _parse_time_val(imm_after.group(1), imm_after.group(2))
                q.time_start = t
                q.time_end = t + 30.0
            else:
                # between X [min/s] and Y [min/s] / during / from
                between_m = re.search(
                    r"(?:between|during|from)\s+(\d+(?:\.\d+)?)(?:\s*(minutes?|mins?|m|seconds?|secs?|s))?\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)(?:\s*(minutes?|mins?|m|seconds?|secs?|s))?",
                    cleaned,
                )
                if between_m:
                    unit1 = between_m.group(2)
                    unit2 = between_m.group(4)
                    # If unit2 is minutes and unit1 was not specified, treat both as minutes
                    if not unit1 and unit2 and any(unit2.strip().startswith(u) for u in ["minute", "min", "m"]):
                        unit1 = unit2
                    q.time_start = _parse_time_val(between_m.group(1), unit1)
                    q.time_end = _parse_time_val(between_m.group(3), unit2)
                else:
                    # around X [min/s]
                    around_match = re.search(
                        r"(?:around|approximately|near|at)\s+(\d+(?:\.\d+)?)(?:\s*(minutes?|mins?|m|seconds?|secs?|s))?",
                        cleaned,
                    )
                    if around_match:
                        t = _parse_time_val(around_match.group(1), around_match.group(2))
                        half = DEFAULT_AROUND_WINDOW_SECONDS / 2.0
                        q.time_start = max(0.0, t - half)
                        q.time_end = t + half
                    else:
                        # after X [min/s]
                        after_m = re.search(
                            r"(?:^|\s)(?:after|past|since|from|>)\s+(\d+(?:\.\d+)?)(?:\s*(minutes?|mins?|m|seconds?|secs?|s))?",
                            cleaned,
                        )
                        if after_m:
                            q.time_start = _parse_time_val(after_m.group(1), after_m.group(2))

                        # before X [min/s]
                        before_m = re.search(
                            r"(?:before|prior\s+to|until|<)\s+(\d+(?:\.\d+)?)(?:\s*(minutes?|mins?|m|seconds?|secs?|s))?",
                            cleaned,
                        )
                        if before_m:
                            q.time_end = _parse_time_val(before_m.group(1), before_m.group(2))

        # ---------------------------------------------------------------
        # 2. Incident category detection
        # ---------------------------------------------------------------
        cats = []
        if re.search(r"\b(vehicle|car|truck|collision|crash|near.miss)\b", cleaned):
            cats.append("VEHICLE")
        if re.search(r"\b(theft|takeaway|object|suitcase|bag|property|luggage|abandoned|unattended)\b", cleaned):
            cats.append("PROPERTY")
        if re.search(r"\b(person|people|pedestrian|fall|running|following|altercation)\b", cleaned):
            cats.append("PERSON")
        if re.search(r"\b(crowd|density|surge|dispersal)\b", cleaned):
            cats.append("CROWD")
        if re.search(r"\b(zone|restricted\s+area|restricted\s+zone)\b", cleaned):
            cats.append("ZONE")
        if re.search(r"\b(fire|smoke|weapon|specialized)\b", cleaned):
            cats.append("SPECIALIZED")
        # Only set if narrowing (not adding all categories)
        if len(cats) <= 2:
            q.incident_categories = cats

        # ---------------------------------------------------------------
        # 3. Validation decision filter
        # ---------------------------------------------------------------
        if re.search(r"\b(review.required|pending\s+review|requires?\s+(?:human\s+)?verification|human\s+verification)\b", cleaned):
            q.validation_decisions = ["REVIEW_REQUIRED"]
        elif re.search(r"\b(accepted\s+incidents?|accepted\s+events?|verified\s+incidents?)\b", cleaned):
            q.validation_decisions = ["ACCEPTED"]
        elif re.search(r"\b(rejected\s+incidents?|dismissed\s+incidents?)\b", cleaned):
            q.validation_decisions = ["REJECTED"]
        elif re.search(r"\b(superseded\s+candidates?|superseded\s+incidents?)\b", cleaned):
            q.validation_decisions = ["SUPERSEDED"]

        # ---------------------------------------------------------------
        # 4. Assessment score filter
        # ---------------------------------------------------------------
        score_above = re.search(
            r"(?:score|assessment\s+score)\s*(?:above|greater\s+than|>\s*=?|at\s+least)\s*(\d+(?:\.\d+)?)\s*(%?)",
            cleaned,
        )
        if score_above:
            val = float(score_above.group(1))
            q.min_assessment_score = val / 100.0 if score_above.group(2) == "%" or val > 1.0 else val

        score_below = re.search(
            r"(?:score|assessment\s+score)\s*(?:below|less\s+than|<\s*=?|under)\s*(\d+(?:\.\d+)?)\s*(%?)",
            cleaned,
        )
        if score_below:
            val = float(score_below.group(1))
            q.max_assessment_score = val / 100.0 if score_below.group(2) == "%" or val > 1.0 else val

        if re.search(r"\b(high\s+(?:assessment\s+)?score|high\s+confidence\s+incidents?)\b", cleaned):
            q.min_assessment_score = 0.70
        if re.search(r"\b(low\s+(?:assessment\s+)?score|low\s+confidence\s+incidents?)\b", cleaned):
            q.max_assessment_score = 0.50

        # ---------------------------------------------------------------
        # 5. Reliability filter
        # ---------------------------------------------------------------
        if re.search(r"\b(high\s+reliability)\b", cleaned):
            q.reliability_levels = ["HIGH"]
        elif re.search(r"\b(medium\s+reliability)\b", cleaned):
            q.reliability_levels = ["MEDIUM"]
        elif re.search(r"\b(low\s+reliability)\b", cleaned):
            q.reliability_levels = ["LOW"]

        # 6. Evidence filter
        # ---------------------------------------------------------------
        if re.search(r"\b(with\s+evidence|evidence\s+available|has\s+evidence|show.*evidence|related\s+to.*evidence|high\s+evidence)\b", cleaned):
            q.evidence_required = True

        if re.search(r"\b(abandoned\s+object|abandoned|unattended)\b", cleaned):
            q.search_text = "abandoned"
        elif re.search(r"\b(vehicle\s+stopped|stopped)\b", cleaned):
            q.search_text = "stopped"

        # ---------------------------------------------------------------
        # 7. Correlated / independent filter
        # ---------------------------------------------------------------
        if re.search(r"\b(correlated\s+incidents?|fused\s+incidents?|multi.signal)\b", cleaned):
            q.correlated_only = True

        # ---------------------------------------------------------------
        # 8. Track ID detection (video-scoped)
        # ---------------------------------------------------------------
        track_m = re.findall(r"\btrack[-_\s]?(\d+)\b", cleaned)
        if track_m:
            q.track_ids = [f"TRACK-{int(n):03d}" for n in track_m]

        # ---------------------------------------------------------------
        # 9. Object class detection
        # ---------------------------------------------------------------
        obj_classes = []
        for phrase in ["cell phone", "traffic light", "stop sign"]:
            if phrase in cleaned:
                obj_classes.append(phrase)
        if not obj_classes:
            for token in re.findall(r"[a-z0-9]+", cleaned):
                if token in OBJECT_SYNONYMS:
                    mapped = OBJECT_SYNONYMS[token]
                    if mapped not in obj_classes:
                        obj_classes.append(mapped)
        q.object_classes = obj_classes[:3]  # limit to 3 classes

        return q.validate()

    @staticmethod
    def _extract_time_and_conf(cleaned: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        start_time: Optional[float] = None
        end_time: Optional[float] = None
        between_match = re.search(
            r"between\s+(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)(?:\s*(?:seconds|secs|s))?",
            cleaned,
        )
        if between_match:
            start_time = float(between_match.group(1))
            end_time = float(between_match.group(2))
        else:
            from_to_match = re.search(
                r"from\s+(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)(?:\s*(?:seconds|secs|s))?",
                cleaned,
            )
            if from_to_match:
                start_time = float(from_to_match.group(1))
                end_time = float(from_to_match.group(2))
            else:
                after_match = re.search(
                    r"(?:after|past|from|>=|>)\s*(\d+(?:\.\d+)?)(?:\s*(?:seconds|secs|s))?",
                    cleaned,
                )
                if after_match:
                    start_time = float(after_match.group(1))
                before_match = re.search(
                    r"(?:before|prior\s+to|until|<=|<)\s*(\d+(?:\.\d+)?)(?:\s*(?:seconds|secs|s))?",
                    cleaned,
                )
                if before_match:
                    end_time = float(before_match.group(1))

                at_match = re.search(
                    r"(?:at|around)\s+(\d+(?:\.\d+)?)(?:\s*(?:seconds|secs|s))?",
                    cleaned,
                )
                if at_match and start_time is None and end_time is None:
                    point = float(at_match.group(1))
                    start_time = max(0.0, point - 2.0)
                    end_time = point + 2.0

        if start_time is not None and end_time is not None and start_time > end_time:
            start_time, end_time = end_time, start_time

        min_confidence: Optional[float] = None
        max_confidence: Optional[float] = None
        conf_above = re.search(
            r"(?:confidence|conf|evidence\s+strength)\s*(?:above|greater\s+than|over|at\s+least|>=|>)\s*(\d+(?:\.\d+)?)\s*(%?)",
            cleaned,
        )
        if conf_above:
            val = float(conf_above.group(1))
            min_confidence = val / 100.0 if conf_above.group(2) == "%" or val > 1.0 else val
        elif re.search(r"\b(high\s+evidence\s+strength|high\s+confidence)\b", cleaned):
            min_confidence = 0.70

        return start_time, end_time, min_confidence, max_confidence

