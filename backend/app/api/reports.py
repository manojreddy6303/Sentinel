"""
Reports API Router for Sentinel (Phase 9)

Provides endpoints for:
- Generating Incident Dossier PDF reports for processed videos
- Listing historical generated reports for a video
- Retrieving report metadata
- Downloading report PDF files
- Inline viewing of report PDF documents
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse

try:
    from app.services.report_service import (
        ReportService,
        ReportError,
        ReportNotFoundError,
        ReportSecurityError,
    )
except ImportError:
    from backend.app.services.report_service import (
        ReportService,
        ReportError,
        ReportNotFoundError,
        ReportSecurityError,
    )

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Reports"])
report_service = ReportService()


class GenerateReportRequest(BaseModel):
    title: Optional[str] = Field(default="SECURITY INCIDENT DOSSIER", description="Custom dossier title")
    classification: Optional[str] = Field(
        default="CONFIDENTIAL // LAW ENFORCEMENT & SECURITY OPERATIONS",
        description="Security classification label",
    )
    queries: Optional[List[str]] = Field(
        default=None,
        description="Optional specific investigation queries to include in report findings",
    )


@router.post("/videos/{video_id}/reports/generate", status_code=status.HTTP_201_CREATED)
def generate_video_report(video_id: str, req: Optional[GenerateReportRequest] = None) -> Dict[str, Any]:
    """
    Generate a professional, evidence-grounded Incident Dossier PDF report for a processed video.
    """
    title = req.title if req and req.title else "SECURITY INCIDENT DOSSIER"
    classification = req.classification if req and req.classification else "CONFIDENTIAL // LAW ENFORCEMENT & SECURITY OPERATIONS"
    queries = req.queries if req and req.queries else None

    try:
        report = report_service.generate_dossier(
            video_id=video_id,
            title=title,
            classification=classification,
            custom_queries=queries,
        )
        return {
            "status": "success",
            "message": "Incident dossier generated successfully.",
            "report": report,
        }
    except ReportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to generate report for video {video_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate incident dossier. Please verify video data.",
        )


@router.get("/videos/{video_id}/reports")
def list_video_reports(video_id: str) -> Dict[str, Any]:
    """
    List all historical reports generated for a given video.
    """
    try:
        reports = report_service.list_reports(video_id=video_id)
        return {
            "status": "success",
            "video_id": video_id,
            "count": len(reports),
            "reports": reports,
        }
    except Exception as exc:
        logger.error(f"Failed to list reports for video {video_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list reports for video.",
        )


@router.get("/reports", summary="List all generated incident dossier reports")
def list_all_reports() -> Dict[str, Any]:
    """List all generated incident dossiers across Sentinel."""
    try:
        reports = report_service.list_reports(video_id=None)
        return {
            "status": "success",
            "count": len(reports),
            "reports": reports,
        }
    except Exception as exc:
        logger.error(f"Failed to list reports: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list reports.",
        )


@router.get("/reports/{report_id}")
def get_report_details(report_id: str) -> Dict[str, Any]:
    """
    Retrieve report metadata and execution summary by public report_id.
    """
    try:
        report = report_service.get_report(report_id)
        return {
            "status": "success",
            "report": report,
        }
    except ReportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to retrieve report {report_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve report details.",
        )


@router.get("/reports/{report_id}/download")
def download_report_pdf(report_id: str):
    """
    Download generated report PDF file as an attachment.
    """
    try:
        file_path, filename = report_service.get_report_file_path(report_id)
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/pdf",
            content_disposition_type="attachment",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "application/pdf",
            },
        )
    except ReportSecurityError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except ReportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to download report {report_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download report file.",
        )


@router.get("/reports/{report_id}/view")
def view_report_pdf(report_id: str):
    """
    View generated report PDF file inline in browser.
    """
    try:
        file_path, filename = report_service.get_report_file_path(report_id)
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/pdf",
            content_disposition_type="inline",
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Content-Type": "application/pdf",
            },
        )
    except ReportSecurityError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except ReportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to view report {report_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to view report file.",
        )
