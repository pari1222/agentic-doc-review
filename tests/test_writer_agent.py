"""
Unit tests for the Writer Agent.

Tests cover:
  - Document type detection
  - Node output structure
  - Error handling
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class TestWriterAgentNode:
    @pytest.mark.asyncio
    async def test_returns_draft_document(self):
        from app.agents.writer_agent import writer_agent_node

        fake_doc = "# Product Overview\n\nThis is the overview.\n## Problem Statement\n\nThe problem."

        with (
            patch(
                "app.agents.writer_agent._detect_document_type",
                new_callable=AsyncMock,
                return_value="prd",
            ),
            patch(
                "app.agents.writer_agent.get_llm",
                return_value=MagicMock(
                    ainvoke=AsyncMock(return_value=FakeLLMResponse(fake_doc))
                ),
            ),
        ):
            state = {
                "document_id": "test-001",
                "user_input": "We need a mobile app for expense tracking",
                "document_type": "",
                "draft_document": "",
                "review_result": {},
                "iteration_count": 0,
                "approved": False,
                "human_review_required": False,
                "error": "",
            }
            result = await writer_agent_node(state)

        assert result["draft_document"] == fake_doc
        assert result["document_type"] == "prd"
        assert result["error"] == ""

    @pytest.mark.asyncio
    async def test_preserves_document_type_on_revision(self):
        from app.agents.writer_agent import writer_agent_node

        fake_doc = "# Executive Summary\n\nRevised content."

        with patch(
            "app.agents.writer_agent.get_llm",
            return_value=MagicMock(
                ainvoke=AsyncMock(return_value=FakeLLMResponse(fake_doc))
            ),
        ):
            state = {
                "document_id": "test-002",
                "user_input": "Compliance review for Q3",
                "document_type": "compliance_report",  # Already set
                "draft_document": "# Draft v1",
                "review_result": {
                    "status": "needs_revision",
                    "score": 60,
                    "missing_sections": ["Risk Assessment"],
                    "feedback": ["Add risk matrix"],
                },
                "iteration_count": 1,
                "approved": False,
                "human_review_required": False,
                "error": "",
            }
            result = await writer_agent_node(state)

        # Type should be preserved, not re-detected
        assert result["document_type"] == "compliance_report"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self):
        from app.agents.writer_agent import writer_agent_node

        with patch(
            "app.agents.writer_agent._detect_document_type",
            new_callable=AsyncMock,
            return_value="general",
        ), patch(
            "app.agents.writer_agent.get_llm",
            return_value=MagicMock(
                ainvoke=AsyncMock(side_effect=Exception("LLM timeout"))
            ),
        ):
            state = {
                "document_id": "test-003",
                "user_input": "Some input",
                "document_type": "",
                "draft_document": "",
                "review_result": {},
                "iteration_count": 0,
                "approved": False,
                "human_review_required": False,
                "error": "",
            }
            result = await writer_agent_node(state)

        assert "error" in result
        assert "LLM timeout" in result["error"]
