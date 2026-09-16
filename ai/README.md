# Sentinel AI Modular Architecture

This directory houses the modular AI and computer vision subsystems for Sentinel.

## Module Breakdown

```
ai/
├── video/           # Video ingestion, keyframe sampling, and stream handling
├── detection/       # Computer vision models (YOLO, OpenCV) for object/motion detection
├── events/          # Aggregation of detections into temporal, searchable events
├── extraction/      # Sub-clip extraction and evidence artifact generation
└── investigation/   # Generative AI / LLM pipeline for conversational video querying
```

## Core Principles & Safety Boundaries

1. **Strictly No Facial Recognition:**
   - Sentinel operates strictly on generic object, person, and vehicle detection (e.g. YOLO classes: person, car, bicycle, backpack).
   - No biometric analysis, face matching, or facial recognition algorithms are permitted.

2. **No Automated Criminal Identification:**
   - Sentinel is an investigative assistance tool for human security analysts.
   - It extracts timestamps, motion, and visual events; it never labels individuals as suspects or makes legal/criminal determinations.

3. **Probabilistic Nature (No 100% Accuracy Claims):**
   - All computer vision detections output confidence scores (0.0 to 1.0) and are subject to visual occlusion, lighting conditions, and camera resolution.
   - The platform treats AI outputs as investigative leads for human review, not absolute ground truth.

4. **Modular & Pluggable:**
   - Each component is isolated behind clean class interfaces to permit independent testing and model updates (e.g. swapping YOLOv8 to future vision backends).
