"""
Unit tests for the Critic Agent.

Tests cover:
  - Section detection from Markdown
  - Missing section identification
  - Hard rule enforcement
  - Node output structure
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.critic_agent import (
    _detect_present_sections,
    _find_missing_sections,
    _enforce_hard_rules,
    critic_agent_node,
)
from app.schemas.document_schema import ReviewResult, ReviewStatus


# ---------------------------------------------------------------------------
# _detect_present_sections
# ---------------------------------------------------------------------------

class TestDetectPresentSections:
    def test_detects_atx_headings(self):
        doc = "# Product Overview\n## Problem Statement\n### User Personas"
        result = _detect_present_sections(doc)
        assert "product overview" in result
        assert "problem statement" in result
        assert "user personas" in result

    def test_ignores_non_headings(self):
        doc = "Some text\nAnother line\n- bullet point"
        result = _detect_present_sections(doc)
        assert len(result) == 0

    def test_handles_empty_document(self):
        result = _detect_present_sections("")
        assert result == set()

    def test_strips_trailing_colon(self):
        doc = "## Success Metrics:"
        result = _detect_present_sections(doc)
        assert "success metrics" in result


# ---------------------------------------------------------------------------
# _find_missing_sections
# ---------------------------------------------------------------------------

class TestFindMissingSections:
    def test_finds_missing_sections_in_prd(self):
        doc = "# Product Overview\n## Problem Statement"
        missing = _find_missing_sections(doc, "prd")
        # PRD has 13 required sections; we only have 2
        assert "Success Metrics" in missing
        assert "Edge Cases" in missing
        assert "Product Overview" not in missing

    def test_no_missing_when_all_present(self):
        sections = [
            "Product Overview", "Problem Statement", "Business Objectives",
            "Success Metrics", "User Personas", "Functional Requirements",
            "Non-Functional Requirements", "Edge Cases",
            "Technical Specifications", "Dependencies", "Risks",
            "Acceptance Criteria", "Release Plan",
        ]
        doc = "\n".join(f"## {s}" for s in sections)
        missing = _find_missing_sections(doc, "prd")
        assert missing == []

    def test_returns_empty_for_unknown_type(self):
        doc = "# Some Section"
        missing = _find_missing_sections(doc, "unknown_type")
        assert missing == []

    def test_general_type_has_no_required_sections(self):
        missing = _find_missing_sections("# Anything", "general")
        assert missing == []


# ---------------------------------------------------------------------------
# _enforce_hard_rules
# ---------------------------------------------------------------------------

class TestEnforceHardRules:
    def test_overrides_approved_when_sections_missing(self):
        result = ReviewResult(
            status=ReviewStatus.APPROVED,
            score=90,
            missing_sections=[],
            feedback=[],
        )
        updated = _enforce_hard_rules(result, ["Edge Cases"], threshold=80)
        assert updated.status == ReviewStatus.NEEDS_REVISION
        assert "Edge Cases" in updated.missing_sections
        assert updated.score < 80

    def test_overrides_approved_when_score_below_threshold(self):
        result = ReviewResult(
            status=ReviewStatus.APPROVED,
            score=70,
            missing_sections=[],
            feedback=[],
        )
        updated = _enforce_hard_rules(result, [], threshold=80)
        assert updated.status == ReviewStatus.NEEDS_REVISION

    def test_keeps_approved_when_all_conditions_met(self):
        result = ReviewResult(
            status=ReviewStatus.APPROVED,
            score=85,
            missing_sections=[],
            feedback=[],
        )
        updated = _enforce_hard_rules(result, [], threshold=80)
        assert updated.status == ReviewStatus.APPROVED

    def test_merges_missing_sections(self):
        result = ReviewResult(
            status=ReviewStatus.NEEDS_REVISION,
            score=60,
            missing_sections=["Risks"],
            feedback=[],
        )
        updated = _enforce_hard_rules(result, ["Edge Cases"], threshold=80)
        assert "Risks" in updated.missing_sections
        assert "Edge Cases" in updated.missing_sections


# ---------------------------------------------------------------------------
# critic_agent_node
# ---------------------------------------------------------------------------

class TestCriticAgentNode:
    @pytest.mark.asyncio
    async def test_returns_error_on_empty_draft(self):
        state = {
            "document_id": "test-123",
            "draft_document": "",
            "document_type": "prd",
            "iteration_count": 0,
            "review_result": {},
            "approved": False,
            "human_review_required": False,
            "error": "",
        }
        result = await critic_agent_node(state)
        assert result["approved"] is False
        assert result["review_result"]["score"] == 0
        assert result["iteration_count"] == 1

    @pytest.mark.asyncio
    async def test_increments_iteration_count(self):
        mock_result = ReviewResult(
            status=ReviewStatus.APPROVED,
            score=90,
            missing_sections=[],
            feedback=[],
        )
        state = {
            "document_id": "test-456",
            "draft_document": "# Product Overview\n...",
            "document_type": "general",
            "iteration_count": 1,
            "review_result": {},
            "approved": False,
            "human_review_required": False,
            "error": "",
        }
        with patch(
            "app.agents.critic_agent._llm_review",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await critic_agent_node(state)

        assert result["iteration_count"] == 2

    @pytest.mark.asyncio
    async def test_approved_on_high_score_no_missing(self):
        mock_result = ReviewResult(
            status=ReviewStatus.APPROVED,
            score=92,
            missing_sections=[],
            feedback=[],
        )
        state = {
            "document_id": "test-789",
            "draft_document": "# Product Overview\n...",
            "document_type": "general",
            "iteration_count": 0,
            "review_result": {},
            "approved": False,
            "human_review_required": False,
            "error": "",
        }
        with patch(
            "app.agents.critic_agent._llm_review",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await critic_agent_node(state)

        assert result["approved"] is True
        assert result["review_result"]["status"] == "approved"
