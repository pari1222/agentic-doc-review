"""
Writer Agent — generates structured professional documents from raw input.

Responsibilities:
  1. Detect the document type from user input (unless already set).
  2. Select the appropriate company template.
  3. Call the LLM to produce a well-structured Markdown document.
  4. On revision cycles, incorporate Critic feedback into the rewrite.
"""

from __future__ import annotations

import logging
from textwrap import dedent

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.document_schema import (
    REQUIRED_SECTIONS_MAP,
    DocumentType,
    DocumentState,
)
from app.services.llm_service import get_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

WRITER_SYSTEM_PROMPT = dedent("""
You are a Senior Technical Writer and Subject Matter Expert specialised in
producing professional business documents.

Your responsibilities:
- Analyse the user's raw input (notes, bullet points, transcripts, etc.).
- Identify the document type if not specified.
- Generate a comprehensive, well-structured Markdown document.
- Follow the required section template exactly.
- Use professional, precise, and concise language.
- Populate every section with substantive content derived from the input.
- If information for a section is not explicitly provided, make reasonable
  professional inferences and mark them with *(inferred)*.
- Output ONLY the Markdown document — no preamble or explanation.
""").strip()


REVISION_SYSTEM_PROMPT = dedent("""
You are a Senior Technical Writer performing a targeted revision.

You will receive:
1. The original document draft.
2. Critic feedback identifying missing sections and quality issues.

Your responsibilities:
- Add ALL missing sections with substantive content.
- Address every feedback point specifically.
- Preserve all existing correct sections unchanged.
- Improve clarity, completeness, and professionalism.
- Output ONLY the revised complete Markdown document.
""").strip()


# ---------------------------------------------------------------------------
# Template selector
# ---------------------------------------------------------------------------

def _get_template_instructions(doc_type: str) -> str:
    """Return a bullet list of required sections for the given document type."""
    sections = REQUIRED_SECTIONS_MAP.get(doc_type, [])
    if not sections:
        return "Structure the document professionally with appropriate sections."
    section_list = "\n".join(f"  - {s}" for s in sections)
    return f"The document MUST contain these sections in order:\n{section_list}"


# ---------------------------------------------------------------------------
# Type detection prompt
# ---------------------------------------------------------------------------

TYPE_DETECTION_PROMPT = dedent("""
Analyse the following user input and identify the most appropriate document type.
Respond with ONLY one of these exact strings (no quotes, no extra text):
  prd
  compliance_report
  consulting_memo
  legal_summary
  general

User input:
{input}
""").strip()


async def _detect_document_type(user_input: str) -> str:
    """Ask the LLM to classify the document type from the raw input."""
    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content="You are a document classification expert."),
        HumanMessage(content=TYPE_DETECTION_PROMPT.format(input=user_input[:2000])),
    ]
    response = await llm.ainvoke(messages)
    detected = response.content.strip().lower()

    valid_types = {dt.value for dt in DocumentType}
    if detected not in valid_types:
        logger.warning(
            "Unexpected document type '%s' — defaulting to 'general'", detected
        )
        detected = DocumentType.GENERAL.value

    logger.info("Detected document type: %s", detected)
    return detected


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

async def _generate_document(
    user_input: str,
    doc_type: str,
    iteration: int,
    review_result: dict,
) -> str:
    """
    Generate (or revise) a document via the LLM.

    On iteration 0 this is a fresh generation.
    On subsequent iterations it is a targeted revision using Critic feedback.
    """
    llm = get_llm()
    template_instructions = _get_template_instructions(doc_type)

    if iteration == 0:
        # ── Initial generation ──────────────────────────────────────────────
        user_message = dedent(f"""
            Document Type: {doc_type.upper().replace('_', ' ')}

            {template_instructions}

            Raw Input:
            {user_input}

            Generate the complete professional document now.
        """).strip()

        messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    else:
        # ── Revision pass ───────────────────────────────────────────────────
        missing = review_result.get("missing_sections", [])
        feedback = review_result.get("feedback", [])
        score = review_result.get("score", 0)

        missing_str = "\n".join(f"  - {s}" for s in missing) if missing else "  None"
        feedback_str = "\n".join(f"  - {f}" for f in feedback) if feedback else "  None"

        user_message = dedent(f"""
            REVISION REQUEST (Iteration {iteration})

            Current quality score: {score}/100
            {template_instructions}

            Missing sections that MUST be added:
            {missing_str}

            Critic feedback to address:
            {feedback_str}

            Original user input:
            {user_input}

            Produce the fully revised document addressing all issues above.
        """).strip()

        messages = [
            SystemMessage(content=REVISION_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

    logger.info(
        "Writer Agent calling LLM (doc_type=%s, iteration=%d)", doc_type, iteration
    )
    response = await llm.ainvoke(messages)
    return response.content.strip()


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

async def writer_agent_node(state: DocumentState) -> dict:
    """
    LangGraph node for the Writer Agent.

    Reads from state, writes back:
      - document_type  (set on first pass)
      - draft_document (updated every pass)
      - error          (set on failure)
    """
    logger.info(
        "Writer Agent invoked | document_id=%s | iteration=%d",
        state.get("document_id"),
        state.get("iteration_count", 0),
    )

    try:
        user_input: str = state["user_input"]
        iteration: int = state.get("iteration_count", 0)
        review_result: dict = state.get("review_result", {})

        # Detect type on first pass; preserve on revisions
        doc_type: str = state.get("document_type") or ""
        if not doc_type:
            doc_type = await _detect_document_type(user_input)

        draft = await _generate_document(user_input, doc_type, iteration, review_result)

        logger.info(
            "Writer Agent produced document (%d chars) | iteration=%d",
            len(draft),
            iteration,
        )
        return {
            "document_type": doc_type,
            "draft_document": draft,
            "error": "",
        }

    except Exception as exc:
        logger.exception("Writer Agent failed: %s", exc)
        return {"error": f"Writer Agent error: {exc}"}
