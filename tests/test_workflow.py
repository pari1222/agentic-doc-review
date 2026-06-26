"""
Unit tests for the LangGraph workflow routing logic.

Tests cover:
  - route_after_critic conditional routing
  - human_review_queue_node output
  - Workflow state transitions
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from app.graph.workflow import route_after_critic, human_review_queue_node
from langgraph.graph import END


class TestRouteAfterCritic:
    def _state(self, approved=False, iteration_count=1, error="", human_required=False):
        return {
            "document_id": "test",
            "user_input": "input",
            "document_type": "prd",
            "draft_document": "doc",
            "review_result": {},
            "iteration_count": iteration_count,
            "approved": approved,
            "human_review_required": human_required,
            "error": error,
        }

    def test_routes_to_end_when_approved(self):
        state = self._state(approved=True, iteration_count=1)
        with patch(
            "app.graph.workflow.get_llm_settings",
            return_value=type("S", (), {"max_revision_iterations": 3})(),
        ):
            result = route_after_critic(state)
        assert result == END

    def test_routes_to_human_queue_at_max_iterations(self):
        state = self._state(approved=False, iteration_count=3)
        with patch(
            "app.graph.workflow.get_llm_settings",
            return_value=type("S", (), {"max_revision_iterations": 3})(),
        ):
            result = route_after_critic(state)
        assert result == "human_review_queue"

    def test_routes_to_writer_for_revision(self):
        state = self._state(approved=False, iteration_count=1)
        with patch(
            "app.graph.workflow.get_llm_settings",
            return_value=type("S", (), {"max_revision_iterations": 3})(),
        ):
            result = route_after_critic(state)
        assert result == "writer_agent"

    def test_routes_to_human_queue_on_error(self):
        state = self._state(approved=False, iteration_count=1, error="Something broke")
        with patch(
            "app.graph.workflow.get_llm_settings",
            return_value=type("S", (), {"max_revision_iterations": 3})(),
        ):
            result = route_after_critic(state)
        assert result == "human_review_queue"


class TestHumanReviewQueueNode:
    @pytest.mark.asyncio
    async def test_sets_human_review_required(self):
        state = {
            "document_id": "test",
            "user_input": "input",
            "document_type": "prd",
            "draft_document": "doc",
            "review_result": {},
            "iteration_count": 3,
            "approved": False,
            "human_review_required": False,
            "error": "",
        }
        result = await human_review_queue_node(state)
        assert result["human_review_required"] is True
        assert result["approved"] is False
