"""
Writer Agent — generates structured professional documents from raw input.
"""

from __future__ import annotations

import logging
from textwrap import dedent

from langchain_core.messages import HumanMessage

from app.schemas.document_schema import (
    REQUIRED_SECTIONS_MAP,
    DocumentType,
    DocumentState,
)
from app.services.llm_service import get_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

WRITER_PERSONA = dedent("""
You are a Senior Technical Writer and Subject Matter Expert specialised in
producing professional business documents. Your output is ONLY the complete
Markdown document — no preamble, no explanation, no commentary.
""").strip()

REVISION_PERSONA = dedent("""
You are a Senior Technical Writer performing a targeted revision.
Add ALL missing sections and address every feedback point.
Preserve existing correct sections.
Output ONLY the complete revised Markdown document — nothing else.
""").strip()


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------

TYPE_DETECTION_PROMPT = dedent("""
Analyse the following user input and identify the most appropriate document type.
Respond with ONLY one of these exact strings (no quotes, no punctuation):
  prd
  compliance_report
  consulting_memo
  legal_summary
  general

User input:
{input}
""").strip()


async def _detect_document_type(user_input: str) -> str:
    llm = get_llm(temperature=0.0)
    msg = TYPE_DETECTION_PROMPT.format(input=user_input[:2000])
    response = await llm.ainvoke([HumanMessage(content=msg)])
    detected = response.content.strip().lower()

    valid = {dt.value for dt in DocumentType}
    if detected not in valid:
        logger.warning("Unexpected doc type '%s' — using 'general'", detected)
        return DocumentType.GENERAL.value

    logger.info("Detected document type: %s", detected)
    return detected


# ---------------------------------------------------------------------------
# Template helper
# ---------------------------------------------------------------------------

def _template_instructions(doc_type: str) -> str:
    sections = REQUIRED_SECTIONS_MAP.get(doc_type, [])
    if not sections:
        return "Structure the document professionally with appropriate sections."
    section_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(sections))
    return f"The document MUST contain ALL of these sections in order:\n{section_list}"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def _generate_document(
    user_input: str,
    doc_type: str,
    iteration: int,
    review_result: dict,
) -> str:
    llm = get_llm()
    template = _template_instructions(doc_type)

    if iteration == 0:
        prompt = dedent(f"""
            {WRITER_PERSONA}

            Task: Generate a complete professional {doc_type.upper().replace('_', ' ')} document.

            {template}

            Instructions:
            - Use proper Markdown headings (## for each section)
            - Write substantive content for every section
            - If information is not explicit, make professional inferences marked as *(inferred)*
            - Be specific, measurable, and professional throughout

            Raw Input from user:
            {user_input}

            Generate the complete document now:
        """).strip()

    else:
        missing  = review_result.get("missing_sections", [])
        feedback = review_result.get("feedback", [])
        score    = review_result.get("score", 0)

        missing_str  = "\n".join(f"  - {s}" for s in missing)  if missing  else "  None"
        feedback_str = "\n".join(f"  - {f}" for f in feedback) if feedback else "  None"

        prompt = dedent(f"""
            {REVISION_PERSONA}

            Revision #{iteration} — Current score: {score}/100

            {template}

            REQUIRED: Add these missing sections:
            {missing_str}

            REQUIRED: Address this feedback:
            {feedback_str}

            Original user input:
            {user_input}

            Output the complete revised document:
        """).strip()

    logger.info("Writer Agent calling LLM | doc_type=%s | iteration=%d", doc_type, iteration)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

async def writer_agent_node(state: DocumentState) -> dict:
    logger.info(
        "Writer Agent | doc_id=%s | iteration=%d",
        state.get("document_id"), state.get("iteration_count", 0),
    )

    try:
        user_input    : str  = state["user_input"]
        iteration     : int  = state.get("iteration_count", 0)
        review_result : dict = state.get("review_result", {})

        doc_type: str = state.get("document_type") or ""
        if not doc_type:
            doc_type = await _detect_document_type(user_input)

        draft = await _generate_document(user_input, doc_type, iteration, review_result)

        logger.info("Writer produced %d chars | iteration=%d", len(draft), iteration)
        return {
            "document_type": doc_type,
            "draft_document": draft,
            "error": "",
        }

    except Exception as exc:
        import traceback
        logger.error("Writer Agent FAILED:\n%s", traceback.format_exc())
        return {
            "draft_document": "",
            "error": f"Writer Agent error: {exc}",
        }
