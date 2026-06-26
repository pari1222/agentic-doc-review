"""
Pydantic models and TypedDict schemas for the Document Review System.
Defines all data structures used across agents, workflow, and API.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    PRD = "prd"
    COMPLIANCE_REPORT = "compliance_report"
    CONSULTING_MEMO = "consulting_memo"
    LEGAL_SUMMARY = "legal_summary"
    GENERAL = "general"


class ReviewStatus(str, Enum):
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    PENDING_HUMAN = "pending_human"
    REJECTED = "rejected"


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    APPROVED = "approved"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class DocumentState(TypedDict):
    """
    Central state object passed through the LangGraph workflow nodes.
    Every field is optional to allow partial updates at each node.
    """
    document_id: str          # Unique identifier for this document run
    user_input: str           # Raw unstructured input from the user
    document_type: str        # Detected or inferred document type
    draft_document: str       # Latest generated Markdown document
    review_result: dict       # Latest Critic Agent review output
    iteration_count: int      # How many write→review cycles have run
    approved: bool            # True once Critic approves or human approves
    human_review_required: bool  # Routed to human queue after max iterations
    error: str                # Any error message for observability


# ---------------------------------------------------------------------------
# Critic Agent Output
# ---------------------------------------------------------------------------

class ReviewResult(BaseModel):
    """Structured output from the Critic Agent."""
    status: ReviewStatus = Field(..., description="approved | needs_revision")
    score: int = Field(..., ge=0, le=100, description="Quality score 0-100")
    missing_sections: list[str] = Field(
        default_factory=list,
        description="List of required sections that are absent"
    )
    feedback: list[str] = Field(
        default_factory=list,
        description="Actionable revision instructions"
    )


# ---------------------------------------------------------------------------
# API Request / Response Models
# ---------------------------------------------------------------------------

class GenerateDocumentRequest(BaseModel):
    """Payload for POST /generate-document."""
    content: str = Field(
        ...,
        min_length=10,
        description="Raw notes, bullet points, or meeting transcript"
    )
    document_type: DocumentType | None = Field(
        default=None,
        description="Optional hint. If omitted the Writer Agent auto-detects."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "content": "Meeting notes: We need a mobile app for tracking expenses...",
            "document_type": "prd"
        }
    }}


class GenerateDocumentResponse(BaseModel):
    """Response for POST /generate-document."""
    document_id: str
    document: str
    document_type: str
    status: DocumentStatus
    score: int
    iteration_count: int
    missing_sections: list[str] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentRecord(BaseModel):
    """Full document record stored in Redis / returned by GET /document/{id}."""
    document_id: str
    user_input: str
    document: str
    document_type: str
    status: DocumentStatus
    score: int
    iteration_count: int
    missing_sections: list[str] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    approved_by_human: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ApproveDocumentRequest(BaseModel):
    """Payload for POST /approve."""
    document_id: str = Field(..., description="ID of the document to approve/reject")
    approved: bool = Field(..., description="True to approve, False to reject")
    reviewer_notes: str | None = Field(
        default=None,
        description="Optional human reviewer comments"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "document_id": "123",
            "approved": True,
            "reviewer_notes": "Looks good, minor formatting tweaks accepted."
        }
    }}


class ApproveDocumentResponse(BaseModel):
    """Response for POST /approve."""
    document_id: str
    status: DocumentStatus
    message: str


class ErrorResponse(BaseModel):
    """Standard error envelope."""
    error: str
    detail: str | None = None
    document_id: str | None = None


# ---------------------------------------------------------------------------
# Template Definitions (used by Writer Agent)
# ---------------------------------------------------------------------------

PRD_REQUIRED_SECTIONS: list[str] = [
    "Product Overview",
    "Problem Statement",
    "Business Objectives",
    "Success Metrics",
    "User Personas",
    "Functional Requirements",
    "Non-Functional Requirements",
    "Edge Cases",
    "Technical Specifications",
    "Dependencies",
    "Risks",
    "Acceptance Criteria",
    "Release Plan",
]

COMPLIANCE_REQUIRED_SECTIONS: list[str] = [
    "Executive Summary",
    "Regulatory Framework",
    "Scope",
    "Findings",
    "Risk Assessment",
    "Remediation Plan",
    "Conclusion",
]

CONSULTING_REQUIRED_SECTIONS: list[str] = [
    "Executive Summary",
    "Background",
    "Current State Analysis",
    "Key Findings",
    "Recommendations",
    "Implementation Roadmap",
    "Risk & Mitigations",
    "Conclusion",
]

LEGAL_REQUIRED_SECTIONS: list[str] = [
    "Executive Summary",
    "Parties Involved",
    "Key Terms & Definitions",
    "Obligations",
    "Liabilities",
    "Dispute Resolution",
    "Governing Law",
    "Conclusion",
]

REQUIRED_SECTIONS_MAP: dict[str, list[str]] = {
    DocumentType.PRD: PRD_REQUIRED_SECTIONS,
    DocumentType.COMPLIANCE_REPORT: COMPLIANCE_REQUIRED_SECTIONS,
    DocumentType.CONSULTING_MEMO: CONSULTING_REQUIRED_SECTIONS,
    DocumentType.LEGAL_SUMMARY: LEGAL_REQUIRED_SECTIONS,
    DocumentType.GENERAL: [],
}
