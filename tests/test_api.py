"""
Integration tests for FastAPI routes.

Uses TestClient with mocked workflow and storage to avoid real LLM calls.
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.document_schema import (
    DocumentRecord,
    DocumentStatus,
    ReviewStatus,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_state(approved=True, human_required=False):
    return {
        "document_id": "mock-doc-id-123",
        "user_input": "test input",
        "document_type": "prd",
        "draft_document": "# Product Overview\n\nMock document content.",
        "review_result": {
            "status": "approved" if approved else "needs_revision",
            "score": 90 if approved else 55,
            "missing_sections": [],
            "feedback": [],
        },
        "iteration_count": 1,
        "approved": approved,
        "human_review_required": human_required,
        "error": "",
    }


def _make_mock_record(doc_id="mock-doc-id-123", status=DocumentStatus.APPROVED):
    return DocumentRecord(
        document_id=doc_id,
        user_input="test input",
        document="# Product Overview\n\nMock document content.",
        document_type="prd",
        status=status,
        score=90,
        iteration_count=1,
        missing_sections=[],
        feedback=[],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# POST /api/v1/generate-document
# ---------------------------------------------------------------------------

class TestGenerateDocument:
    def test_successful_generation(self):
        mock_state = _make_mock_state(approved=True)
        mock_record = _make_mock_record()

        with (
            patch(
                "app.api.routes.run_document_workflow",
                new_callable=AsyncMock,
                return_value=mock_state,
            ),
            patch(
                "app.api.routes.save_document",
                new_callable=AsyncMock,
            ),
        ):
            response = client.post(
                "/api/v1/generate-document",
                json={"content": "We need a mobile expense tracking app for SMBs."},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "mock-doc-id-123"
        assert data["status"] == "approved"
        assert data["score"] == 90
        assert "document" in data

    def test_content_too_short_returns_422(self):
        response = client.post(
            "/api/v1/generate-document",
            json={"content": "short"},
        )
        assert response.status_code == 422

    def test_missing_content_returns_422(self):
        response = client.post("/api/v1/generate-document", json={})
        assert response.status_code == 422

    def test_human_review_status_on_max_iterations(self):
        mock_state = _make_mock_state(approved=False, human_required=True)

        with (
            patch(
                "app.api.routes.run_document_workflow",
                new_callable=AsyncMock,
                return_value=mock_state,
            ),
            patch("app.api.routes.save_document", new_callable=AsyncMock),
        ):
            response = client.post(
                "/api/v1/generate-document",
                json={"content": "Compliance report for GDPR Q3 audit findings."},
            )

        assert response.status_code == 200
        assert response.json()["status"] == "needs_human_review"


# ---------------------------------------------------------------------------
# GET /api/v1/document/{id}
# ---------------------------------------------------------------------------

class TestGetDocument:
    def test_returns_document(self):
        mock_record = _make_mock_record()

        with patch(
            "app.api.routes.get_document",
            new_callable=AsyncMock,
            return_value=mock_record,
        ):
            response = client.get("/api/v1/document/mock-doc-id-123")

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "mock-doc-id-123"

    def test_returns_404_when_not_found(self):
        with patch(
            "app.api.routes.get_document",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.get("/api/v1/document/nonexistent")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v1/approve
# ---------------------------------------------------------------------------

class TestApproveDocument:
    def test_approve_document(self):
        mock_record = _make_mock_record(status=DocumentStatus.NEEDS_HUMAN_REVIEW)
        updated_record = _make_mock_record(status=DocumentStatus.APPROVED)

        with (
            patch(
                "app.api.routes.get_document",
                new_callable=AsyncMock,
                return_value=mock_record,
            ),
            patch(
                "app.api.routes.update_document_approval",
                new_callable=AsyncMock,
                return_value=updated_record,
            ),
        ):
            response = client.post(
                "/api/v1/approve",
                json={"document_id": "mock-doc-id-123", "approved": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"

    def test_reject_document(self):
        mock_record = _make_mock_record(status=DocumentStatus.NEEDS_HUMAN_REVIEW)
        rejected_record = _make_mock_record(status=DocumentStatus.REJECTED)

        with (
            patch(
                "app.api.routes.get_document",
                new_callable=AsyncMock,
                return_value=mock_record,
            ),
            patch(
                "app.api.routes.update_document_approval",
                new_callable=AsyncMock,
                return_value=rejected_record,
            ),
        ):
            response = client.post(
                "/api/v1/approve",
                json={
                    "document_id": "mock-doc-id-123",
                    "approved": False,
                    "reviewer_notes": "Lacks sufficient detail.",
                },
            )

        assert response.status_code == 200
        assert response.json()["status"] == "rejected"

    def test_approve_nonexistent_document_returns_404(self):
        with patch(
            "app.api.routes.get_document",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.post(
                "/api/v1/approve",
                json={"document_id": "ghost-id", "approved": True},
            )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_check(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
