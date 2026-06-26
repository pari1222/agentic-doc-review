"""
LangGraph Workflow — orchestrates the Writer → Critic → (Revision)* loop.

Graph topology:
  START
    │
    ▼
  writer_agent
    │
    ▼
  critic_agent
    │
    ├─ approved ──────────────────────────► END
    │
    ├─ needs_revision (iteration < max) ──► writer_agent (revision loop)
    │
    └─ max iterations reached ───────────► human_review_queue ──► END
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.critic_agent import critic_agent_node
from app.agents.writer_agent import writer_agent_node
from app.schemas.document_schema import DocumentState
from app.services.llm_service import get_llm_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Human Review Queue node
# ---------------------------------------------------------------------------

async def human_review_queue_node(state: DocumentState) -> dict:
    """
    Reached when MAX_REVISION_ITERATIONS is exhausted without approval.
    Flags the document for human review and halts autonomous processing.
    """
    logger.warning(
        "Document %s routed to Human Review Queue after %d iterations",
        state.get("document_id"),
        state.get("iteration_count"),
    )
    return {
        "human_review_required": True,
        "approved": False,
    }


# ---------------------------------------------------------------------------
# Conditional edge: what to do after the Critic
# ---------------------------------------------------------------------------

def route_after_critic(state: DocumentState) -> str:
    """
    Routing logic after the Critic Agent runs.

    Returns the name of the next node to execute.
    """
    settings = get_llm_settings()
    max_iterations = settings.max_revision_iterations

    approved: bool = state.get("approved", False)
    iteration_count: int = state.get("iteration_count", 0)
    error: str = state.get("error", "")

    if error:
        # Propagate errors to human review
        logger.error("Error detected, routing to human_review_queue: %s", error)
        return "human_review_queue"

    if approved:
        logger.info(
            "Document approved at iteration %d — routing to END", iteration_count
        )
        return END  # type: ignore[return-value]

    if iteration_count >= max_iterations:
        logger.warning(
            "Max iterations (%d) reached — routing to human_review_queue",
            max_iterations,
        )
        return "human_review_queue"

    logger.info(
        "Document needs revision (iteration %d / %d) — routing to writer_agent",
        iteration_count,
        max_iterations,
    )
    return "writer_agent"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_workflow() -> Any:
    """
    Construct and compile the LangGraph StateGraph.

    Returns:
        A compiled LangGraph runnable (supports .ainvoke()).
    """
    graph = StateGraph(DocumentState)

    # ── Register nodes ───────────────────────────────────────────────────────
    graph.add_node("writer_agent", writer_agent_node)
    graph.add_node("critic_agent", critic_agent_node)
    graph.add_node("human_review_queue", human_review_queue_node)

    # ── Edges ────────────────────────────────────────────────────────────────
    graph.add_edge(START, "writer_agent")
    graph.add_edge("writer_agent", "critic_agent")

    # Conditional routing after Critic
    graph.add_conditional_edges(
        "critic_agent",
        route_after_critic,
        {
            "writer_agent": "writer_agent",          # Revision loop
            "human_review_queue": "human_review_queue",
            END: END,
        },
    )

    graph.add_edge("human_review_queue", END)

    return graph.compile()


# Singleton compiled workflow
_workflow = None


def get_workflow() -> Any:
    """Return the cached compiled workflow (lazy initialisation)."""
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
        logger.info("LangGraph workflow compiled successfully")
    return _workflow


# ---------------------------------------------------------------------------
# High-level run helper
# ---------------------------------------------------------------------------

async def run_document_workflow(
    user_input: str,
    document_type: str | None = None,
) -> DocumentState:
    """
    Execute the full document generation workflow.

    Args:
        user_input:    Raw text from the user.
        document_type: Optional type hint (auto-detected if None).

    Returns:
        Final DocumentState after the workflow completes.
    """
    document_id = str(uuid.uuid4())
    logger.info("Starting workflow | document_id=%s", document_id)

    initial_state: DocumentState = {
        "document_id": document_id,
        "user_input": user_input,
        "document_type": document_type or "",
        "draft_document": "",
        "review_result": {},
        "iteration_count": 0,
        "approved": False,
        "human_review_required": False,
        "error": "",
    }

    workflow = get_workflow()
    final_state: DocumentState = await workflow.ainvoke(initial_state)

    logger.info(
        "Workflow complete | document_id=%s | approved=%s | iterations=%d",
        document_id,
        final_state.get("approved"),
        final_state.get("iteration_count"),
    )
    return final_state
