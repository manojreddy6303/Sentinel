"""
Incident Dossier PDF Generation Engine for Sentinel (Phase 9)

Produces professional, audit-ready security incident dossiers using ReportLab.
Strictly evidence-grounded and observational, zero LLM dependency.
"""

import os
import io
import html
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    HRFlowable,
)
from reportlab.pdfgen import canvas
from PIL import Image as PILImage

from ai.reporting.schema import (
    ReportDataPayload,
    ReportEvidenceItem,
    ReportSecurityEvent,
    ReportSpecializedEvent,
    ReportCorrelatedIncident,
)

logger = logging.getLogger(__name__)

# Color Palette Definitions
C_NAVY_DARK = colors.HexColor("#0f172a")      # Slate 900
C_NAVY_MID = colors.HexColor("#1e293b")       # Slate 800
C_NAVY_LIGHT = colors.HexColor("#334155")     # Slate 700
C_ACCENT_BLUE = colors.HexColor("#0284c7")    # Sky 600
C_ACCENT_TEAL = colors.HexColor("#0f766e")    # Teal 700
C_TEXT_MAIN = colors.HexColor("#0f172a")      # Dark slate
C_TEXT_MUTED = colors.HexColor("#64748b")     # Slate 500
C_BG_LIGHT = colors.HexColor("#f8fafc")       # Slate 50
C_BG_CARD = colors.HexColor("#f1f5f9")        # Slate 100
C_BORDER_LIGHT = colors.HexColor("#cbd5e1")   # Slate 300
C_BORDER_DARK = colors.HexColor("#94a3b8")    # Slate 400

# Alert / Status Colors
C_ALERT_THEFT_BG = colors.HexColor("#fef2f2")    # Red 50
C_ALERT_THEFT_BORDER = colors.HexColor("#dc2626")# Red 600
C_ALERT_THEFT_TXT = colors.HexColor("#991b1b")   # Red 800

C_ALERT_WARN_BG = colors.HexColor("#fffbeb")     # Amber 50
C_ALERT_WARN_BORDER = colors.HexColor("#d97706") # Amber 600
C_ALERT_WARN_TXT = colors.HexColor("#92400e")    # Amber 800

C_ALERT_INFO_BG = colors.HexColor("#eff6ff")     # Blue 50
C_ALERT_INFO_BORDER = colors.HexColor("#2563eb") # Blue 600
C_ALERT_INFO_TXT = colors.HexColor("#1e40af")    # Blue 800

PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN_LEFT = 40
MARGIN_RIGHT = 40
MARGIN_TOP = 46
MARGIN_BOTTOM = 46
USABLE_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT  # 532 pt


def _esc(val: Any) -> str:
    """Safely convert value to string with HTML escaping for ReportLab Paragraphs."""
    if val is None:
        return "N/A"
    s = str(val)
    # Replace non-standard dashes and quotes with standard ASCII/unicode equivalents
    s = s.replace("\u2014", "--").replace("\u2013", "-")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    return html.escape(s)


def _fmt_ts(seconds: Optional[float]) -> str:
    """Format seconds into MM:SS.s string."""
    if seconds is None:
        return "N/A"
    try:
        sec_f = float(seconds)
        mins = int(sec_f // 60)
        rem_sec = sec_f % 60
        return f"{mins:02d}:{rem_sec:04.1f} ({sec_f:.1f}s)"
    except Exception:
        return f"{seconds}s"


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and stamp total page count
    and professional running headers/footers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
        self.report_id = "SENTINEL-DOSSIER"

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_decorations(num_pages)
            super().showPage()
        super().save()

    def _draw_decorations(self, page_count: int):
        self.saveState()

        # Running Header (Pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(C_NAVY_MID)
            self.drawString(MARGIN_LEFT, PAGE_HEIGHT - 32, "SENTINEL // SECURITY INCIDENT DOSSIER")
            self.setFont("Helvetica", 8)
            self.setFillColor(C_TEXT_MUTED)
            self.drawRightString(PAGE_WIDTH - MARGIN_RIGHT, PAGE_HEIGHT - 32, f"REF: {self.report_id}")
            self.setStrokeColor(C_BORDER_LIGHT)
            self.setLineWidth(0.75)
            self.line(MARGIN_LEFT, PAGE_HEIGHT - 36, PAGE_WIDTH - MARGIN_RIGHT, PAGE_HEIGHT - 36)

        # Running Footer (All Pages)
        self.setStrokeColor(C_BORDER_LIGHT)
        self.setLineWidth(0.75)
        self.line(MARGIN_LEFT, 36, PAGE_WIDTH - MARGIN_RIGHT, 36)

        self.setFont("Helvetica", 7.5)
        self.setFillColor(C_TEXT_MUTED)
        self.drawString(
            MARGIN_LEFT,
            24,
            "CONFIDENTIAL // SECURITY ANALYSIS // DATABASE-GROUNDED // OBSERVATIONAL INTELLIGENCE",
        )
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(C_NAVY_MID)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(PAGE_WIDTH - MARGIN_RIGHT, 24, page_str)

        self.restoreState()


class IncidentDossierPDFGenerator:
    """Generates professional Incident Dossier PDF documents."""

    def __init__(self, data: ReportDataPayload):
        self.data = data
        self.styles = getSampleStyleSheet()
        self._init_custom_styles()

    def _init_custom_styles(self):
        """Define fine-tuned typography styles for the dossier."""
        self.styles.add(ParagraphStyle(
            name="DossierTitle",
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=C_NAVY_DARK,
            alignment=0,
        ))
        self.styles.add(ParagraphStyle(
            name="DossierSubTitle",
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            textColor=C_TEXT_MUTED,
            alignment=0,
        ))
        self.styles.add(ParagraphStyle(
            name="ClassificationBadge",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#ffffff"),
            alignment=1,
        ))
        self.styles.add(ParagraphStyle(
            name="SectionHeading",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=C_NAVY_DARK,
            spaceBefore=14,
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name="SubSectionHeading",
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=C_NAVY_MID,
            spaceBefore=8,
            spaceAfter=4,
        ))
        self.styles.add(ParagraphStyle(
            name="BodyStandard",
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=C_TEXT_MAIN,
        ))
        self.styles.add(ParagraphStyle(
            name="BodyMuted",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=C_TEXT_MUTED,
        ))
        self.styles.add(ParagraphStyle(
            name="TableCell",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=C_TEXT_MAIN,
        ))
        self.styles.add(ParagraphStyle(
            name="TableCellBold",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=C_NAVY_DARK,
        ))
        self.styles.add(ParagraphStyle(
            name="TableHeader",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=colors.white,
        ))
        self.styles.add(ParagraphStyle(
            name="NoticeBoxText",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=C_ALERT_WARN_TXT,
        ))
        self.styles.add(ParagraphStyle(
            name="TheftBoxText",
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=C_ALERT_THEFT_TXT,
        ))
        self.styles.add(ParagraphStyle(
            name="CodeSnippet",
            fontName="Courier",
            fontSize=7.5,
            leading=10,
            textColor=C_NAVY_MID,
        ))

    def generate(self, output_path_or_buffer) -> Tuple[int, int]:
        """
        Build and write the dossier PDF.
        Returns (file_size_bytes, page_count).
        """
        doc = SimpleDocTemplate(
            output_path_or_buffer,
            pagesize=letter,
            leftMargin=MARGIN_LEFT,
            rightMargin=MARGIN_RIGHT,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
        )

        story = []
        self._build_header_banner(story)
        self._build_executive_summary(story)
        self._build_video_sensor_profile(story)
        self._build_detection_statistics(story)
        self._build_security_intelligence(story)
        self._build_theft_deep_dive(story)
        self._build_specialized_visual_section(story)
        self._build_correlated_incidents_section(story)
        self._build_timeline(story)
        self._build_evidence_gallery(story)
        self._build_investigation_findings(story)
        self._build_forensic_investigation_section(story)  # Phase 17: optional forensic context
        self._build_limitations_and_notice(story)
        self._build_metadata_provenance(story)

        # Build document with custom canvas
        canvas_maker = lambda *args, **kwargs: NumberedCanvas(*args, **kwargs)
        # Configure report_id on canvas
        def make_canvas(*args, **kwargs):
            c = NumberedCanvas(*args, **kwargs)
            c.report_id = self.data.report_id
            return c

        doc.build(story, canvasmaker=make_canvas)

        # Calculate final size & page count
        if isinstance(output_path_or_buffer, (str, Path)):
            p = Path(output_path_or_buffer)
            size_bytes = p.stat().st_size
        elif hasattr(output_path_or_buffer, "getvalue"):
            size_bytes = len(output_path_or_buffer.getvalue())
        else:
            size_bytes = 0

        page_count = getattr(doc, "page", 1)
        return size_bytes, page_count

    # -------------------------------------------------------------------------
    # Section Builders
    # -------------------------------------------------------------------------

    def _build_header_banner(self, story: list):
        """Top cover banner with classification, title, and metadata grid."""
        # Top Classification Stripe
        class_table = Table(
            [[Paragraph(f"<b>{_esc(self.data.classification)}</b>", self.styles["ClassificationBadge"])]],
            colWidths=[USABLE_WIDTH],
        )
        class_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_NAVY_DARK),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(class_table)
        story.append(Spacer(1, 8))

        # Title & Reference Bar
        header_data = [
            [
                Paragraph("<b>SENTINEL</b> | INCIDENT DOSSIER", self.styles["DossierTitle"]),
                Paragraph(
                    f"<b>REPORT ID:</b> {_esc(self.data.report_id)}<br/>"
                    f"<b>GENERATED:</b> {_esc(self.data.generated_at_iso[:19].replace('T', ' '))} UTC",
                    self.styles["TableCellBold"],
                ),
            ],
            [
                Paragraph("EVIDENCE-GROUNDED VIDEO SURVEILLANCE INVESTIGATION AUDIT", self.styles["DossierSubTitle"]),
                Paragraph("<b>STATUS:</b> VERIFIED ARCHIVE // DETERMINISTIC", self.styles["TableCell"]),
            ]
        ]
        h_table = Table(header_data, colWidths=[332, 200])
        h_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(h_table)
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1.5, color=C_ACCENT_BLUE, spaceBefore=2, spaceAfter=8))

    def _build_executive_summary(self, story: list):
        """Executive summary card with observed facts vs analysis."""
        story.append(Paragraph("1. EXECUTIVE SUMMARY & OBSERVATIONAL OVERVIEW", self.styles["SectionHeading"]))

        v = self.data.video
        s = self.data.stats

        # Compute major classes string
        sorted_classes = sorted(s.class_counts.items(), key=lambda x: x[1], reverse=True)
        top_classes_str = ", ".join([f"{k} ({v})" for k, v in sorted_classes[:4]]) if sorted_classes else "None recorded"

        summary_text = (
            f"This dossier compiles automated, evidence-grounded computer vision intelligence "
            f"extracted from surveillance recording <b>{_esc(v.original_filename)}</b> "
            f"(Video UUID: <code>{_esc(v.video_id[:8])}...</code>). "
            f"During the <b>{v.duration_seconds or 0:.1f}s</b> recording, Sentinel registered a total of "
            f"<b>{s.total_detections}</b> verified detections across <b>{len(s.class_counts)}</b> object categories, "
            f"yielding <b>{s.total_tracks}</b> individual persistent tracks and <b>{s.total_security_events}</b> "
            f"security behavioral events. <b>{s.total_evidence_items}</b> forensic evidence artifacts "
            f"(including bounding snapshots and bounded clips) were captured and archived."
        )

        theft_callout = ""
        if self.data.theft_events:
            te = self.data.theft_events[0]
            theft_callout = (
                f"<br/><br/><b>CRITICAL OBSERVATIONAL FINDING:</b> An automated behavioral pattern matching "
                f"<b>POTENTIAL_OBJECT_TAKEAWAY</b> was registered at <b>{_fmt_ts(te.timestamp_seconds)}</b> "
                f"involving Track <b>{_esc(te.track_id or 'N/A')}</b> and class <b>{_esc(te.object_class or 'N/A')}</b> "
                f"(Pattern Confidence: {te.confidence * 100:.0f}%). "
                f"Supporting spatial grounded observations and physical evidence were preserved under Evidence Vault ref "
                f"<b>{_esc(te.evidence_id or 'N/A')}</b>."
            )

        card_content = [
            Paragraph(f"<b>OBSERVED FORENSIC FACTS:</b><br/>{summary_text}{theft_callout}", self.styles["BodyStandard"]),
            Spacer(1, 4),
            Paragraph(
                f"<b>MAJOR OBJECT CLASSES:</b> {top_classes_str} | "
                f"<b>TOTAL ANONYMOUS TRACKS:</b> {s.total_tracks} | "
                f"<b>VALIDATED TRACKS:</b> {s.validated_tracks} | "
                f"<b>ANONYMOUS FACE REGIONS:</b> {s.total_face_detections}",
                self.styles["BodyMuted"],
            ),
            Spacer(1, 3),
            Paragraph(
                "<b>SCORE DISAMBIGUATION:</b> "
                "\"Behavior Score\" (Section 4) reflects the behavioral pattern confidence calibrated by the incident validator "
                "(capped at 65% for review-required patterns). "
                "\"Evidence Strength (Optical)\" (Specialized Section) reflects the raw visual sensor confidence "
                "from the specialized detector (fire/smoke/weapon) prior to incident-level validation. "
                "These are independent measurements of different pipeline stages — they are not expected to match.",
                self.styles["BodyMuted"],
            ),
        ]

        card_table = Table([[card_content]], colWidths=[USABLE_WIDTH])
        card_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_BG_LIGHT),
            ("BOX", (0, 0), (-1, -1), 1, C_BORDER_LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(card_table)
        story.append(Spacer(1, 6))

        # Mandatory Human Verification Warning Callout
        notice_data = [[
            Paragraph("<b>HUMAN VERIFICATION REQUIREMENT:</b> " + _esc(self.data.human_verification_notice), self.styles["NoticeBoxText"])
        ]]
        notice_table = Table(notice_data, colWidths=[USABLE_WIDTH])
        notice_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_ALERT_WARN_BG),
            ("BOX", (0, 0), (-1, -1), 1, C_ALERT_WARN_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(notice_table)
        story.append(Spacer(1, 8))

    def _build_video_sensor_profile(self, story: list):
        """Table detailing source video parameters, resolution, codec, and sampling specifications."""
        story.append(Paragraph("2. SOURCE VIDEO & SENSOR SPECIFICATIONS", self.styles["SectionHeading"]))

        v = self.data.video
        size_mb = f"{v.file_size_bytes / (1024 * 1024):.2f} MB" if v.file_size_bytes else "N/A"
        fps_str = f"{v.fps:.2f} fps" if v.fps else "N/A"
        dur_str = f"{v.duration_seconds:.2f}s" if v.duration_seconds else "N/A"
        sampling_str = f"Sampled at {v.sampling_rate_fps or 1.0:.1f} FPS"
        analyzed_frames_str = str(v.analyzed_frames_count or v.frame_count or "N/A")

        spec_data = [
            [
                Paragraph("<b>Parameter</b>", self.styles["TableHeader"]),
                Paragraph("<b>Recorded Specification</b>", self.styles["TableHeader"]),
                Paragraph("<b>Parameter</b>", self.styles["TableHeader"]),
                Paragraph("<b>Recorded Specification</b>", self.styles["TableHeader"]),
            ],
            [
                Paragraph("Source Filename", self.styles["TableCellBold"]),
                Paragraph(_esc(v.original_filename), self.styles["TableCell"]),
                Paragraph("Video UUID", self.styles["TableCellBold"]),
                Paragraph(f"<code>{_esc(v.video_id)}</code>", self.styles["TableCell"]),
            ],
            [
                Paragraph("Duration", self.styles["TableCellBold"]),
                Paragraph(dur_str, self.styles["TableCell"]),
                Paragraph("Source Frame Rate", self.styles["TableCellBold"]),
                Paragraph(fps_str, self.styles["TableCell"]),
            ],
            [
                Paragraph("Video Resolution", self.styles["TableCellBold"]),
                Paragraph(_esc(v.resolution or "Native Surveillance Resolution"), self.styles["TableCell"]),
                Paragraph("Video Codec", self.styles["TableCellBold"]),
                Paragraph(_esc(v.codec or "Standard Surveillance Stream"), self.styles["TableCell"]),
            ],
            [
                Paragraph("Sampling Rate", self.styles["TableCellBold"]),
                Paragraph(sampling_str, self.styles["TableCell"]),
                Paragraph("Analyzed Frames", self.styles["TableCellBold"]),
                Paragraph(analyzed_frames_str, self.styles["TableCell"]),
            ],
            [
                Paragraph("Ingestion Status", self.styles["TableCellBold"]),
                Paragraph(f"<font color='{C_ACCENT_TEAL.hexval()}'><b>{_esc(v.status.upper())}</b></font>", self.styles["TableCell"]),
                Paragraph("Analyzed At", self.styles["TableCellBold"]),
                Paragraph(_esc(v.processed_at or v.uploaded_at or "N/A"), self.styles["TableCell"]),
            ],
        ]

        table = Table(spec_data, colWidths=[110, 156, 110, 156])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_NAVY_MID),
            ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG_LIGHT]),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))

    def _build_detection_statistics(self, story: list):
        """Table detailing raw, validated, uncertain, and rejected observation counts alongside persistent track counts."""
        story.append(Paragraph("3. DETECTION & SPATIAL TRACKING ANALYTICS", self.styles["SectionHeading"]))

        s = self.data.stats

        # 6-column breakdown table: Class | Raw | Validated | Uncertain | Rejected | Tracks
        data = [
            [
                Paragraph("<b>Object Class</b>", self.styles["TableHeader"]),
                Paragraph("<b>Raw Obs.</b>", self.styles["TableHeader"]),
                Paragraph("<b>Validated</b>", self.styles["TableHeader"]),
                Paragraph("<b>Uncertain</b>", self.styles["TableHeader"]),
                Paragraph("<b>Rejected</b>", self.styles["TableHeader"]),
                Paragraph("<b>Anonymous Tracks</b>", self.styles["TableHeader"]),
            ]
        ]

        # Combine all classes that have any raw, validated, uncertain, or track records
        all_class_keys = set(s.class_counts.keys()) | set(s.class_raw_counts.keys()) | set(s.class_track_counts.keys())
        sorted_classes = sorted(all_class_keys, key=lambda x: (s.class_counts.get(x, 0), s.class_raw_counts.get(x, 0)), reverse=True)

        if sorted_classes:
            for cls_name in sorted_classes:
                val_cnt = s.class_counts.get(cls_name, 0)
                raw_cnt = s.class_raw_counts.get(cls_name, val_cnt)
                unc_cnt = s.class_uncertain_counts.get(cls_name, 0)
                rej_cnt = s.class_rejected_counts.get(cls_name, 0)
                trk_cnt = s.class_track_counts.get(cls_name, 0)

                data.append([
                    Paragraph(f"<b>{_esc(cls_name.title())}</b>", self.styles["TableCellBold"]),
                    Paragraph(str(raw_cnt), self.styles["TableCell"]),
                    Paragraph(f"<b>{val_cnt}</b>", self.styles["TableCellBold"]),
                    Paragraph(str(unc_cnt), self.styles["TableCell"]),
                    Paragraph(str(rej_cnt), self.styles["TableCell"]),
                    Paragraph(f"<b>{trk_cnt}</b>", self.styles["TableCell"]),
                ])
        else:
            data.append([
                Paragraph("No detections recorded", self.styles["TableCell"]),
                Paragraph("0", self.styles["TableCell"]),
                Paragraph("0", self.styles["TableCell"]),
                Paragraph("0", self.styles["TableCell"]),
                Paragraph("0", self.styles["TableCell"]),
                Paragraph("0", self.styles["TableCell"]),
            ])

        table = Table(data, colWidths=[112, 80, 85, 75, 75, 105])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_NAVY_MID),
            ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG_LIGHT]),
        ]))
        story.append(table)
        story.append(Spacer(1, 4))

        # Explicit counting semantics callout
        sampling_fps = self.data.video.sampling_rate_fps or 1.0
        callout_text = (
            f"<b>COUNTING METHODOLOGY & TRANSPARENCY NOTICE:</b> "
            f"Detection statistics represent analyzed video frames sampled at <b>{sampling_fps:.1f} FPS</b>. "
            f"<b>Detection Observations</b> quantify localized bounding-box observations across analyzed frames. "
            f"<b>Anonymous Tracks</b> quantify persistent spatial entities tracked over time. "
            f"Multiple detection observations of the same object over time do not represent separate physical objects or individuals."
        )
        callout_table = Table([[Paragraph(callout_text, self.styles["BodyMuted"])]], colWidths=[USABLE_WIDTH])
        callout_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_BG_CARD),
            ("BOX", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(callout_table)
        story.append(Spacer(1, 8))

    def _build_security_intelligence(self, story: list):
        """Security intelligence findings: intrusions, presence, anomalies, anonymous faces."""
        story.append(Paragraph("4. SECURITY INTELLIGENCE & BEHAVIORAL EVENTS", self.styles["SectionHeading"]))

        # Check for anonymous face detections
        if self.data.anonymous_faces_count > 0:
            face_note = (
                f"<b>PRIVACY & IDENTITY SAFEGUARD:</b> <b>{self.data.anonymous_faces_count}</b> anonymous "
                f"visual face regions were localized. Sentinel operates strictly in accordance with privacy safeguards: "
                f"NO facial recognition, identification, database querying, or identity matching is performed. "
                f"Detections represent anonymous bounding coordinates only."
            )
            face_table = Table([[Paragraph(face_note, self.styles["BodyMuted"])]], colWidths=[USABLE_WIDTH])
            face_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), C_BG_CARD),
                ("BOX", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(face_table)
            story.append(Spacer(1, 6))

        # Vehicle Attribute Analysis
        if self.data.vehicle_attributes:
            v_items = []
            for va in self.data.vehicle_attributes[:5]:
                track = str(va.get("track_id") or "N/A")
                vtype = str(va.get("vehicle_type") or va.get("object_class") or "vehicle")
                color = str(va.get("color") or va.get("primary_color") or "unknown")
                raw_conf = va.get("color_confidence") if va.get("color_confidence") is not None else va.get("confidence")
                try:
                    conf = float(raw_conf) if raw_conf is not None else 0.0
                except (ValueError, TypeError):
                    conf = 0.0
                v_items.append(f"Track <b>{_esc(track)}</b>: {_esc(color.title())} {_esc(vtype)} ({conf*100:.0f}% color conf)")

            va_text = "<b>VEHICLE CHROMATIC PROFILES:</b> " + " | ".join(v_items)
            story.append(Paragraph(va_text, self.styles["BodyStandard"]))
            story.append(Spacer(1, 6))

        # Security Events Table
        sec_events = self.data.security_events
        if not sec_events:
            story.append(Paragraph("<i>No abnormal behavioral or security events flagged during automated analysis.</i>", self.styles["BodyMuted"]))
            story.append(Spacer(1, 8))
            return

        data = [
            [
                Paragraph("<b>Time (s)</b>", self.styles["TableHeader"]),
                Paragraph("<b>Event Type</b>", self.styles["TableHeader"]),
                Paragraph("<b>Severity</b>", self.styles["TableHeader"]),
                Paragraph("<b>Track / Object</b>", self.styles["TableHeader"]),
                Paragraph("<b>Behavior Score</b>", self.styles["TableHeader"]),
                Paragraph("<b>Observational Description</b>", self.styles["TableHeader"]),
            ]
        ]

        # Display all security events for complete cross-layer count parity
        for se in sec_events:
            t_str = _fmt_ts(se.timestamp_seconds)
            sev = se.severity or "NORMAL"
            track_obj = f"{se.track_id or '-'} / {se.object_class or '-'}"
            score = f"{se.confidence * 100:.0f}%" if se.confidence else "N/A"

            # Color severity
            if se.event_type in ("POTENTIAL_THEFT", "POTENTIAL_OBJECT_TAKEAWAY"):
                sev_html = f"<font color='{C_ALERT_THEFT_TXT.hexval()}'><b>HIGH</b></font>"
            elif sev == "HIGH":
                sev_html = f"<font color='{C_ALERT_WARN_TXT.hexval()}'><b>HIGH</b></font>"
            else:
                sev_html = f"<font color='{C_TEXT_MUTED.hexval()}'>{_esc(sev)}</font>"

            # Truncate description cleanly if too long
            desc = se.description or ""
            if len(desc) > 120:
                desc = desc[:117] + "..."

            data.append([
                Paragraph(t_str, self.styles["TableCellBold"]),
                Paragraph(_esc(se.event_type.replace('_', ' ')), self.styles["TableCellBold"]),
                Paragraph(sev_html, self.styles["TableCell"]),
                Paragraph(_esc(track_obj), self.styles["TableCell"]),
                Paragraph(score, self.styles["TableCell"]),
                Paragraph(_esc(desc), self.styles["TableCell"]),
            ])

        table = Table(data, colWidths=[65, 105, 52, 90, 40, 180])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_NAVY_MID),
            ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG_LIGHT]),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))

    def _build_theft_deep_dive(self, story: list):
        """Dedicated high-visibility card for potential theft / takeaway incidents."""
        theft_events = self.data.theft_events
        if not theft_events:
            return

        theft_section = [
            Paragraph("5. INCIDENT INVESTIGATION: POTENTIAL OBJECT TAKEAWAY PATTERN", self.styles["SectionHeading"])
        ]

        for te in theft_events:
            signals = te.observable_signals or []
            signals_html = ""
            for sig in signals:
                signals_html += f"&bull; {_esc(sig)}<br/>"

            deep_dive_content = [
                Paragraph(
                    f"<b>PATTERN IDENTIFIER:</b> {_esc(te.id)} | "
                    f"<b>TIMESTAMP:</b> {_fmt_ts(te.timestamp_seconds)} | "
                    f"<b>BEHAVIOR CONFIDENCE:</b> {te.confidence * 100:.0f}%",
                    self.styles["TheftBoxText"],
                ),
                Spacer(1, 3),
                Paragraph(
                    f"<b>SUSPECT BEHAVIOR:</b> Potential object-takeaway pattern observed.<br/>"
                    f"<b>PRIMARY ACTOR TRACK:</b> <b>{_esc(te.track_id or 'N/A')}</b> (Person) | "
                    f"<b>INTERACTION OBJECT:</b> <b>{_esc(te.object_class or 'N/A')}</b><br/>"
                    f"<b>LINKED EVIDENCE VAULT ID:</b> <b>{_esc(te.evidence_id or 'N/A')}</b>",
                    self.styles["BodyStandard"],
                ),
                Spacer(1, 4),
                Paragraph(f"<b>SUPPORTING OBSERVATIONAL SIGNALS:</b><br/>{signals_html or 'Observations matched physical interaction and departure trajectory.'}", self.styles["BodyMuted"]),
                Spacer(1, 4),
                Paragraph(
                    "<b>LEGAL & COMPLIANCE MANDATE:</b> "
                    "This observation indicates an anomalous physical takeaway pattern. It DOES NOT establish "
                    "legal ownership, authorization, or criminal intent. Immediate review by authorized security "
                    "personnel is required prior to enforcement action.",
                    self.styles["NoticeBoxText"],
                ),
            ]

            card_table = Table([[deep_dive_content]], colWidths=[USABLE_WIDTH])
            card_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), C_ALERT_THEFT_BG),
                ("BOX", (0, 0), (-1, -1), 1.5, C_ALERT_THEFT_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]))
            theft_section.append(card_table)
            theft_section.append(Spacer(1, 8))

        story.append(KeepTogether(theft_section))

    def _build_specialized_visual_section(self, story: list):
        """Specialized visual detection events: fire, smoke, weapon, posture."""
        spec_events = getattr(self.data, "specialized_events", [])
        if not spec_events:
            return

        spec_section = [
            Paragraph("SPECIALIZED VISUAL INTELLIGENCE & FORENSIC SIGNALS", self.styles["SectionHeading"])
        ]

        # Callout notice regarding probabilistic visual signatures
        spec_notice = (
            "<b>SPECIALIZED VISUAL DETECTION NOTICE:</b> Specialized visual models operate on localized "
            "chromatic, geometric, and temporal patterns. All detected signals represent observational machine "
            "indications (e.g. potential fire/smoke signatures or potential weapon-like visual silhouettes) "
            "and require mandatory human operator review. Sentinel does NOT establish definitive hazards or criminal status."
        )
        spec_notice_table = Table([[Paragraph(spec_notice, self.styles["BodyMuted"])]], colWidths=[USABLE_WIDTH])
        spec_notice_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_BG_CARD),
            ("BOX", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        spec_section.append(spec_notice_table)
        spec_section.append(Spacer(1, 6))

        data = [
            [
                Paragraph("<b>Time (s)</b>", self.styles["TableHeader"]),
                Paragraph("<b>Visual Event Type</b>", self.styles["TableHeader"]),
                Paragraph("<b>Detector / Model</b>", self.styles["TableHeader"]),
                Paragraph("<b>Evidence Strength (Optical)</b>", self.styles["TableHeader"]),
                Paragraph("<b>Status</b>", self.styles["TableHeader"]),
                Paragraph("<b>Observational Description</b>", self.styles["TableHeader"]),
            ]
        ]

        for se in spec_events[:10]:
            t_str = _fmt_ts(se.timestamp_seconds)
            evt_display = se.event_type.replace('_', ' ')
            strength_str = f"{se.evidence_strength * 100:.0f}%" if se.evidence_strength else "N/A"
            det_str = f"{se.detector_name or 'specialized'}"
            if se.model_name:
                det_str += f"<br/><font color='{C_TEXT_MUTED.hexval()}'>{se.model_name}</font>"

            # Observational wording fallback
            desc = se.description
            if not desc:
                if "FIRE" in se.event_type and "SMOKE" in se.event_type:
                    desc = "Potential fire and smoke visual evidence was detected in localized region."
                elif "FIRE" in se.event_type:
                    desc = "Potential fire visual evidence was detected with sustained chromatic features."
                elif "SMOKE" in se.event_type:
                    desc = "Potential smoke-like visual evidence was detected with diffuse plume dynamics."
                elif "WEAPON" in se.event_type:
                    desc = "Potential weapon-like object visual evidence was detected in frame."
                elif "POSE" in se.event_type:
                    desc = "Postural transition evidence observed across tracking sequence."
                else:
                    desc = f"Specialized visual observation ({se.event_type})."

            status_color = C_ALERT_WARN_TXT if se.review_required else C_ACCENT_TEAL
            status_html = f"<font color='{status_color.hexval()}'><b>{_esc(se.validation_status)}</b></font>"

            data.append([
                Paragraph(t_str, self.styles["TableCellBold"]),
                Paragraph(_esc(evt_display), self.styles["TableCellBold"]),
                Paragraph(det_str, self.styles["TableCell"]),
                Paragraph(strength_str, self.styles["TableCell"]),
                Paragraph(status_html, self.styles["TableCell"]),
                Paragraph(_esc(desc), self.styles["TableCell"]),
            ])

        table = Table(data, colWidths=[65, 110, 85, 45, 65, 162])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_NAVY_MID),
            ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG_LIGHT]),
        ]))
        spec_section.append(table)
        spec_section.append(Spacer(1, 8))
        story.append(KeepTogether(spec_section))

    def _build_correlated_incidents_section(self, story: list):
        """Phase 16: Canonical Correlated Incidents & Multi-Signal Storylines."""
        corr_incidents = getattr(self.data, "correlated_incidents", [])
        if not corr_incidents:
            return

        corr_section = [
            Paragraph("CORRELATED INCIDENT STORYLINES & MULTI-SIGNAL FUSION", self.styles["SectionHeading"])
        ]

        corr_notice = (
            "<b>MULTI-SIGNAL CORRELATION & FUSION NOTICE:</b> Correlated incidents represent automated contextual fusion "
            "of multi-signal observations across temporal and spatial trajectories. Discrete detections, kinetic transitions, "
            "and entity interactions are unified into coherent incident episodes with complete evidence provenance. "
            "Validation decisions (ACCEPTED vs REVIEW_REQUIRED) reflect evidence strength; human verification is required."
        )
        corr_notice_table = Table([[Paragraph(corr_notice, self.styles["BodyMuted"])]], colWidths=[USABLE_WIDTH])
        corr_notice_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_BG_CARD),
            ("BOX", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        corr_section.append(corr_notice_table)
        corr_section.append(Spacer(1, 6))

        for ci in corr_incidents[:6]:
            t_span = f"{_fmt_ts(ci.start_time)} - {_fmt_ts(ci.end_time)}"
            tracks_str = ", ".join(ci.primary_track_ids) if ci.primary_track_ids else "None"
            supp_tracks_str = ", ".join(ci.supporting_track_ids) if ci.supporting_track_ids else "None"
            objs_str = ", ".join(ci.involved_object_classes) if ci.involved_object_classes else "None"
            ev_ids_str = ", ".join(ci.evidence_ids) if ci.evidence_ids else "None"

            score_str = f"{ci.assessment_score * 100:.0f}%"
            val_dec = ci.validation_decision
            if val_dec == "ACCEPTED":
                dec_html = "<font color='#16a34a'><b>ACCEPTED</b></font>"
            else:
                dec_html = f"<font color='{C_ALERT_WARN_TXT.hexval()}'><b>REVIEW REQUIRED</b></font>"

            incident_card = [
                Paragraph(
                    f"<b>INCIDENT ID:</b> {_esc(ci.incident_id)} | "
                    f"<b>CATEGORY:</b> {_esc(ci.incident_category)} | "
                    f"<b>TIME SPAN:</b> {t_span} ({ci.duration:.1f}s)",
                    self.styles["TableCellBold"],
                ),
                Spacer(1, 2),
                Paragraph(
                    f"<b>VALIDATION DECISION:</b> {dec_html} | "
                    f"<b>ASSESSMENT SCORE:</b> <b>{score_str}</b> | "
                    f"<b>RELIABILITY:</b> {_esc(ci.reliability_rating)}",
                    self.styles["BodyStandard"],
                ),
                Spacer(1, 2),
                Paragraph(
                    f"<b>PRIMARY ENTITIES:</b> <b>{_esc(tracks_str)}</b> ({_esc(objs_str)}) | "
                    f"<b>SUPPORTING TRACKS:</b> {_esc(supp_tracks_str)} | "
                    f"<b>EVIDENCE REFS:</b> {_esc(ev_ids_str)}",
                    self.styles["BodyMuted"],
                ),
                Spacer(1, 3),
                Paragraph(f"<b>DETERMINISTIC STORYLINE:</b> {_esc(ci.storyline)}", self.styles["BodyStandard"]),
            ]

            card_table = Table([[incident_card]], colWidths=[USABLE_WIDTH])
            card_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), C_BG_LIGHT),
                ("BOX", (0, 0), (-1, -1), 1.0, C_BORDER_DARK),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))
            corr_section.append(card_table)
            corr_section.append(Spacer(1, 6))

        story.append(KeepTogether(corr_section))

    def _build_timeline(self, story: list):
        """Chronological table of verified investigation events."""
        story.append(Paragraph("6. CHRONOLOGICAL INVESTIGATION TIMELINE", self.styles["SectionHeading"]))

        timeline = sorted(self.data.timeline, key=lambda x: x.timestamp_seconds)
        if not timeline:
            story.append(Paragraph("<i>No timeline events recorded.</i>", self.styles["BodyMuted"]))
            story.append(Spacer(1, 8))
            return

        data = [
            [
                Paragraph("<b>Timestamp</b>", self.styles["TableHeader"]),
                Paragraph("<b>Event Type</b>", self.styles["TableHeader"]),
                Paragraph("<b>Class / Track</b>", self.styles["TableHeader"]),
                Paragraph("<b>Score</b>", self.styles["TableHeader"]),
                Paragraph("<b>Evidence Ref</b>", self.styles["TableHeader"]),
                Paragraph("<b>Notes / Context</b>", self.styles["TableHeader"]),
            ]
        ]

        # Select representative distinct events (max 15 rows)
        seen_keys = set()
        selected = []
        for e in timeline:
            # Dedup key: approximate second + event type + track
            key = (round(e.timestamp_seconds, 1), e.event_type, e.track_id)
            if key not in seen_keys:
                seen_keys.add(key)
                selected.append(e)
            if len(selected) >= 15:
                break

        for item in selected:
            t_str = _fmt_ts(item.timestamp_seconds)
            ct_str = f"{item.object_class or '-'}"
            if item.track_id:
                ct_str += f" [{item.track_id}]"
            conf_str = f"{item.confidence * 100:.0f}%" if item.confidence else "-"
            ev_str = item.evidence_id or "-"
            notes = item.description or f"Observed {item.object_class or 'target'}"
            if len(notes) > 80:
                notes = notes[:77] + "..."

            data.append([
                Paragraph(t_str, self.styles["TableCellBold"]),
                Paragraph(_esc(item.event_type.replace('_', ' ')), self.styles["TableCell"]),
                Paragraph(_esc(ct_str), self.styles["TableCell"]),
                Paragraph(conf_str, self.styles["TableCell"]),
                Paragraph(f"<code>{_esc(ev_str)}</code>", self.styles["TableCell"]),
                Paragraph(_esc(notes), self.styles["TableCell"]),
            ])

        table = Table(data, colWidths=[65, 110, 95, 45, 75, 142])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_NAVY_MID),
            ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG_LIGHT]),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))

    def _build_evidence_gallery(self, story: list):
        """Evidence vault section embedding real snapshots and metadata."""
        story.append(Paragraph("7. FORENSIC EVIDENCE VAULT & VISUAL PROVENANCE", self.styles["SectionHeading"]))

        evidence_items = self.data.evidence
        if not evidence_items:
            story.append(Paragraph("<i>No forensic evidence items stored in vault for this recording.</i>", self.styles["BodyMuted"]))
            story.append(Spacer(1, 8))
            return

        # Up to 4 key evidence items with images
        for idx, ev in enumerate(evidence_items[:4]):
            ev_block = []

            meta_header = (
                f"<b>EVIDENCE REF:</b> <code>{_esc(ev.id)}</code> | "
                f"<b>TIMESTAMP:</b> {_fmt_ts(ev.timestamp_seconds)} | "
                f"<b>TYPE:</b> {_esc(ev.evidence_type)} | "
                f"<b>CLASS:</b> {_esc(ev.object_class or 'N/A')}"
            )
            if ev.confidence:
                meta_header += f" ({ev.confidence * 100:.0f}% conf)"

            ev_block.append(Paragraph(meta_header, self.styles["TableCellBold"]))
            ev_block.append(Spacer(1, 4))

            # Attempt image embed
            img_rendered = False
            if ev.local_image_path and os.path.exists(ev.local_image_path):
                try:
                    with PILImage.open(ev.local_image_path) as pil_img:
                        w, h = pil_img.size
                        aspect = h / max(w, 1)

                    # Scale to fit printable width nicely
                    target_w = 460
                    target_h = target_w * aspect
                    if target_h > 210:
                        target_h = 210
                        target_w = target_h / aspect

                    rl_img = RLImage(ev.local_image_path, width=target_w, height=target_h)
                    ev_block.append(rl_img)
                    img_rendered = True
                except Exception as exc:
                    logger.warning(f"Could not embed image {ev.local_image_path}: {exc}")

            if not img_rendered:
                placeholder = Table([[Paragraph(
                    "<i>Visual frame snapshot unavailable on local storage node or unreadable.</i>",
                    self.styles["BodyMuted"],
                )]], colWidths=[460])
                placeholder.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), C_BG_LIGHT),
                    ("BOX", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("PADDING", (0, 0), (-1, -1), 16),
                ]))
                ev_block.append(placeholder)

            ev_block.append(Spacer(1, 3))
            note_str = ev.notes or (
                "Verified bounding evidence snapshot captured during automated surveillance intelligence pipeline."
            )
            ev_block.append(Paragraph(f"<b>Forensic Notes:</b> {_esc(note_str)}", self.styles["BodyMuted"]))

            card = Table([[ev_block]], colWidths=[USABLE_WIDTH])
            card.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), C_BG_LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(KeepTogether(card))
            story.append(Spacer(1, 6))

    def _build_investigation_findings(self, story: list):
        """Structured Q&A findings from deterministic queries."""
        findings = self.data.investigation_findings
        if not findings:
            return

        story.append(Paragraph("8. INVESTIGATION QUERY FINDINGS", self.styles["SectionHeading"]))

        data = [
            [
                Paragraph("<b>Investigation Query</b>", self.styles["TableHeader"]),
                Paragraph("<b>Grounded Database Finding</b>", self.styles["TableHeader"]),
            ]
        ]

        for item in findings[:5]:
            data.append([
                Paragraph(f"<b>&ldquo;{_esc(item.query)}&rdquo;</b>", self.styles["TableCellBold"]),
                Paragraph(_esc(item.finding), self.styles["TableCell"]),
            ])

        table = Table(data, colWidths=[180, 352])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_NAVY_MID),
            ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG_LIGHT]),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))

    def _build_forensic_investigation_section(self, story: list):
        """
        Phase 17: Optional Forensic Investigation Findings section.

        Rendered only when self.data.forensic_investigation_context is set.
        Content is grounded exclusively in retrieved database records.
        No LLM hallucination is possible since context is pre-executed.
        """
        ctx = getattr(self.data, "forensic_investigation_context", None)
        if not ctx:
            return

        story.append(Paragraph("FORENSIC INVESTIGATION FINDINGS", self.styles["SectionHeading"]))

        # Interpretation
        interpretation = ctx.get("interpretation", "")
        if interpretation:
            story.append(Paragraph(
                f"<b>Query Interpretation:</b> {_esc(interpretation)}",
                self.styles["BodyText"],
            ))
            story.append(Spacer(1, 4))

        # Grounded answer
        grounded = ctx.get("grounded_answer", "")
        if grounded:
            answer_table = Table(
                [[Paragraph(_esc(grounded), self.styles["BodyText"])]],
                colWidths=[USABLE_WIDTH],
            )
            answer_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), C_BG_CARD),
                ("BOX", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(answer_table)
            story.append(Spacer(1, 6))

        # Matched incidents table
        incidents = ctx.get("matched_incidents", [])
        if incidents:
            story.append(Paragraph(
                f"<b>Matched Incidents ({len(incidents)}):</b>",
                self.styles["BodyText"],
            ))
            story.append(Spacer(1, 4))

            inc_data = [[
                Paragraph("<b>Category</b>", self.styles["TableHeader"]),
                Paragraph("<b>Time</b>", self.styles["TableHeader"]),
                Paragraph("<b>Score</b>", self.styles["TableHeader"]),
                Paragraph("<b>Decision</b>", self.styles["TableHeader"]),
                Paragraph("<b>Storyline Summary</b>", self.styles["TableHeader"]),
            ]]
            for inc in incidents[:10]:
                storyline_short = (inc.get("storyline") or "")[:120]
                if len(inc.get("storyline") or "") > 120:
                    storyline_short += "…"
                inc_data.append([
                    Paragraph(_esc(inc.get("incident_category", "")), self.styles["TableCell"]),
                    Paragraph(
                        f"{inc.get('start_time', 0):.1f}s–{inc.get('end_time', 0):.1f}s",
                        self.styles["TableCell"],
                    ),
                    Paragraph(
                        f"{inc.get('assessment_score', 0):.2f}",
                        self.styles["TableCell"],
                    ),
                    Paragraph(_esc(inc.get("validation_decision", "")), self.styles["TableCell"]),
                    Paragraph(_esc(storyline_short), self.styles["TableCell"]),
                ])

            inc_table = Table(inc_data, colWidths=[70, 70, 45, 90, 257])
            inc_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), C_NAVY_MID),
                ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG_LIGHT]),
            ]))
            story.append(inc_table)
            story.append(Spacer(1, 6))

        # Summary counts
        total = ctx.get("total_results", 0)
        events_count = len(ctx.get("matched_events", []))
        tracks_count = len(ctx.get("matched_tracks", []))
        evidence_count = len(ctx.get("matched_evidence", []))
        if total:
            story.append(Paragraph(
                f"<b>Result Summary:</b> {total} total match(es) — "
                f"{events_count} security event(s), {tracks_count} track(s), {evidence_count} evidence item(s). "
                f"{'Results truncated by limit.' if ctx.get('truncated') else ''}",
                self.styles["BodyMuted"],
            ))

        story.append(Spacer(1, 8))

    def _build_limitations_and_notice(self, story: list):

        """Standard security operations limitations and safety language."""
        story.append(Paragraph("9. TECHNICAL LIMITATIONS & AUDIT ADVISORY", self.styles["SectionHeading"]))

        points = [
            "<b>Optical & Environmental Constraints:</b> Detection accuracy is subject to video resolution, compression artifacts, ambient lighting, sensor noise, perspective angle, and object occlusion.",
            "<b>Anonymous Track Association:</b> Trajectory tracking is based purely on spatio-temporal kinematic modeling and anonymous appearance embeddings. It does not identify individuals.",
            "<b>No Facial Recognition:</b> Sentinel explicitly does not execute biometric identity matching or reverse identity searches. Face detections represent bounding region coordinates only.",
            "<b>Behavioral Pattern Nature:</b> Alerts (e.g., potential theft, intrusion, loitering) are observational statistical anomaly patterns. They require independent human investigation before establishing culpability.",
            "<b>Data Integrity:</b> All metrics and timestamps in this dossier reflect stored database records extracted directly from the video stream.",
        ]

        bullet_html = "<br/>".join([f"&bull; {p}" for p in points])
        lim_table = Table([[Paragraph(bullet_html, self.styles["BodyMuted"])]], colWidths=[USABLE_WIDTH])
        lim_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_BG_CARD),
            ("BOX", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(lim_table)
        story.append(Spacer(1, 8))

    def _build_metadata_provenance(self, story: list):
        """Technical provenance block: generator version, report hash, timestamp."""
        story.append(Paragraph("10. DOSSIER METADATA & PROVENANCE", self.styles["SectionHeading"]))

        hash_disp = self.data.sha256_hash or "SHA-256 Computed Post-Render & Stored in Database"
        meta_rows = [
            [
                Paragraph("<b>Dossier Ref ID</b>", self.styles["TableCellBold"]),
                Paragraph(f"<code>{_esc(self.data.report_id)}</code>", self.styles["TableCell"]),
                Paragraph("<b>Platform Version</b>", self.styles["TableCellBold"]),
                Paragraph("Sentinel Security Platform (Universal Master Build)", self.styles["TableCell"]),
            ],
            [
                Paragraph("<b>Source Video ID</b>", self.styles["TableCellBold"]),
                Paragraph(f"<code>{_esc(self.data.video.video_id)}</code>", self.styles["TableCell"]),
                Paragraph("<b>Timestamp (UTC)</b>", self.styles["TableCellBold"]),
                Paragraph(_esc(self.data.generated_at_iso), self.styles["TableCell"]),
            ],
            [
                Paragraph("<b>Pipeline Engine</b>", self.styles["TableCellBold"]),
                Paragraph("Deterministic Relational Database Engine", self.styles["TableCell"]),
                Paragraph("<b>Audit Status</b>", self.styles["TableCellBold"]),
                Paragraph("<font color='#0284c7'><b>DATABASE-GROUNDED // VERIFIED</b></font>", self.styles["TableCell"]),
            ],
            [
                Paragraph("<b>Cryptographic Hash</b>", self.styles["TableCellBold"]),
                Paragraph(f"<font size='6.5'><code>{_esc(hash_disp)}</code></font>", self.styles["TableCell"]),
                Paragraph("<b>Verification</b>", self.styles["TableCellBold"]),
                Paragraph("SHA-256 Forensic Integrity Verification", self.styles["TableCell"]),
            ],
        ]

        table = Table(meta_rows, colWidths=[120, 146, 120, 146])
        table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_BG_LIGHT, colors.white]),
        ]))
        story.append(table)
