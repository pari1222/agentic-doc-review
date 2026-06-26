"""
FastAPI Routes — exposes the Document Review System via HTTP.

Endpoints:
  POST /generate-document  — Trigger the agentic workflow
  GET  /document/{id}      — Retrieve a stored document
  POST /approve            — Human-in-the-loop approval
  GET  /health             — Health check
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from app.schemas.document_schema import (
    ApproveDocumentRequest,
    ApproveDocumentResponse,
    DocumentRecord,
    DocumentStatus,
    GenerateDocumentRequest,
    GenerateDocumentResponse,
    ReviewStatus,
)
from app.services.storage_service import (
    get_document,
    save_document,
    update_document_approval,
)
from app.graph.workflow import run_document_workflow

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /generate-document
# ---------------------------------------------------------------------------

@router.post(
    "/generate-document",
    response_model=GenerateDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a structured professional document from raw input",
    tags=["Documents"],
)
async def generate_document(request: GenerateDocumentRequest) -> GenerateDocumentResponse:
    """
    Accepts raw user input (notes, transcripts, bullet points) and runs the
    full Writer → Critic → (Revision)* agentic workflow.

    Returns the final document along with its review status and quality score.
    """
    logger.info(
        "POST /generate-document | content_length=%d | doc_type=%s",
        len(request.content),
        request.document_type,
    )

    try:
        # ── Run the LangGraph workflow ──────────────────────────────────────
        final_state = await run_document_workflow(
            user_input=request.content,
            document_type=request.document_type.value if request.document_type else None,
        )

        document_id: str = final_state["document_id"]
        review_result: dict = final_state.get("review_result", {})
        approved: bool = final_state.get("approved", False)
        human_required: bool = final_state.get("human_review_required", False)

        # ── Determine final status ──────────────────────────────────────────
        if approved:
            doc_status = DocumentStatus.APPROVED
        elif human_required:
            doc_status = DocumentStatus.NEEDS_HUMAN_REVIEW
        else:
            doc_status = DocumentStatus.PROCESSING

        score: int = review_result.get("score", 0)
        missing: list[str] = review_result.get("missing_sections", [])
        feedback: list[str] = review_result.get("feedback", [])

        # ── Persist to storage ──────────────────────────────────────────────
        record = DocumentRecord(
            document_id=document_id,
            user_input=request.content,
            document=final_state.get("draft_document", ""),
            document_type=final_state.get("document_type", "general"),
            status=doc_status,
            score=score,
            iteration_count=final_state.get("iteration_count", 0),
            missing_sections=missing,
            feedback=feedback,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        await save_document(record)
        logger.info("Document %s saved | status=%s | score=%d", document_id, doc_status, score)

        return GenerateDocumentResponse(
            document_id=document_id,
            document=record.document,
            document_type=record.document_type,
            status=doc_status,
            score=score,
            iteration_count=record.iteration_count,
            missing_sections=missing,
            feedback=feedback,
        )

    except Exception as exc:
        logger.exception("Workflow failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document generation failed: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /document/{id}
# ---------------------------------------------------------------------------

@router.get(
    "/document/{document_id}",
    response_model=DocumentRecord,
    summary="Retrieve a stored document by ID",
    tags=["Documents"],
)
async def get_document_by_id(document_id: str) -> DocumentRecord:
    """Return a previously generated document by its unique ID."""
    logger.info("GET /document/%s", document_id)

    record = await get_document(document_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return record


# ---------------------------------------------------------------------------
# POST /approve
# ---------------------------------------------------------------------------

@router.post(
    "/approve",
    response_model=ApproveDocumentResponse,
    summary="Human approval endpoint for documents in review queue",
    tags=["Human Review"],
)
async def approve_document(request: ApproveDocumentRequest) -> ApproveDocumentResponse:
    """
    Human-in-the-loop approval gate.

    Documents that exhausted AI revision attempts are routed here.
    A human reviewer submits their decision (approved=true/false).
    """
    logger.info(
        "POST /approve | document_id=%s | approved=%s",
        request.document_id,
        request.approved,
    )

    record = await get_document(request.document_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{request.document_id}' not found.",
        )

    # Documents already approved/rejected do not need re-processing
    if record.status in (DocumentStatus.APPROVED, DocumentStatus.REJECTED):
        return ApproveDocumentResponse(
            document_id=request.document_id,
            status=record.status,
            message=f"Document was already {record.status.value}.",
        )

    updated = await update_document_approval(
        document_id=request.document_id,
        approved=request.approved,
        reviewer_notes=request.reviewer_notes,
    )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document approval status.",
        )

    final_status = DocumentStatus.APPROVED if request.approved else DocumentStatus.REJECTED
    msg = (
        "Document approved by human reviewer."
        if request.approved
        else "Document rejected by human reviewer."
    )

    logger.info("Document %s human decision: %s", request.document_id, final_status)
    return ApproveDocumentResponse(
        document_id=request.document_id,
        status=final_status,
        message=msg,
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    summary="Health check",
    tags=["System"],
)
async def health_check() -> dict:
    """Returns service health status."""
    return {"status": "healthy", "service": "agentic-document-review"}
