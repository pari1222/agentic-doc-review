"""
Critic Agent — reviews generated documents against company standards.

Responsibilities:
  1. Parse the Markdown document and check for required sections.
  2. Ask the LLM to score quality and produce actionable feedback.
  3. Return a structured ReviewResult (approved / needs_revision).
  4. Hard-check for missing required sections before relying on LLM score.
"""

from __future__ import annotations

import json
import logging
import re
from textwrap import dedent

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.document_schema import (
    REQUIRED_SECTIONS_MAP,
    DocumentState,
    ReviewResult,
    ReviewStatus,
)
from app.services.llm_service import get_llm, get_llm_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = dedent("""
You are a ruthless but fair Senior Quality Assurance Reviewer for professional documents.

Your job is to evaluate a document and return ONLY a valid JSON object — nothing else.

Evaluation criteria:
1. Are all required sections present and non-empty?
2. Is the content substantive (not just placeholder text)?
3. Is the language professional, clear, and precise?
4. Are requirements specific and measurable (not vague)?
5. Is the document internally consistent?

Return this exact JSON structure:
{
  "status": "approved" | "needs_revision",
  "score": <integer 0-100>,
  "missing_sections": [<list of missing section names>],
  "feedback": [<list of specific actionable revision instructions>]
}

Rules:
- "approved" only when score >= threshold AND missing_sections is empty.
- Score reflects overall quality (completeness, clarity, professionalism).
- Each feedback item must be specific and actionable.
- Output ONLY the JSON — no markdown fences, no explanation.
""").strip()


# ---------------------------------------------------------------------------
# Section presence checker
# ---------------------------------------------------------------------------

def _detect_present_sections(document: str) -> set[str]:
    """
    Extract heading names from a Markdown document.
    Supports ATX headings (# Heading) at any level.
    """
    headings = re.findall(r"^#{1,6}\s+(.+)$", document, re.MULTILINE)
    # Normalise: strip trailing punctuation & lowercase for comparison
    return {h.strip().rstrip(":").lower() for h in headings}


def _find_missing_sections(document: str, doc_type: str) -> list[str]:
    """Return required sections absent from the document."""
    required = REQUIRED_SECTIONS_MAP.get(doc_type, [])
    if not required:
        return []

    present = _detect_present_sections(document)
    missing = []
    for section in required:
        if section.lower() not in present:
            missing.append(section)
    return missing


# ---------------------------------------------------------------------------
# LLM-powered quality review
# ---------------------------------------------------------------------------

async def _llm_review(
    document: str,
    doc_type: str,
    pre_detected_missing: list[str],
    threshold: int,
) -> ReviewResult:
    """Call the LLM to score quality and generate feedback."""
    llm = get_llm(temperature=0.1)

    required_sections = REQUIRED_SECTIONS_MAP.get(doc_type, [])
    sections_str = "\n".join(f"  - {s}" for s in required_sections) or "  (none specified)"
    missing_str = (
        "\n".join(f"  - {s}" for s in pre_detected_missing)
        if pre_detected_missing
        else "  None detected by pre-check"
    )

    user_message = dedent(f"""
        Document Type: {doc_type.upper().replace('_', ' ')}
        Approval threshold: {threshold}/100

        Required sections:
        {sections_str}

        Pre-check detected these missing sections:
        {missing_str}

        --- DOCUMENT START ---
        {document}
        --- DOCUMENT END ---

        Evaluate the document. Return only the JSON object.
    """).strip()

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    logger.info("Critic Agent calling LLM for quality review")
    response = await llm.ainvoke(messages)

    # ── Parse JSON response ──────────────────────────────────────────────────
    raw = response.content.strip()
    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

    try:
        data = json.loads(raw)
        return ReviewResult.model_validate(data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to parse Critic LLM response: %s\nRaw: %s", exc, raw)
        # Fallback: construct a conservative result forcing revision
        return ReviewResult(
            status=ReviewStatus.NEEDS_REVISION,
            score=40,
            missing_sections=pre_detected_missing,
            feedback=["Critic could not parse LLM response. Manual review recommended."],
        )


# ---------------------------------------------------------------------------
# Post-processing: enforce hard rules
# ---------------------------------------------------------------------------

def _enforce_hard_rules(
    result: ReviewResult,
    pre_detected_missing: list[str],
    threshold: int,
) -> ReviewResult:
    """
    Override LLM result with deterministic hard rules:
    - If any required section is missing → needs_revision.
    - If score < threshold → needs_revision.
    - Merge pre-detected missing sections with LLM-detected ones.
    """
    # Merge missing sections
    all_missing = list(
        {*result.missing_sections, *pre_detected_missing}
    )
    result.missing_sections = all_missing

    # Hard override conditions
    if all_missing or result.score < threshold:
        result.status = ReviewStatus.NEEDS_REVISION
        if result.score >= threshold and all_missing:
            # LLM approved but we found missing sections — penalise score
            result.score = min(result.score, threshold - 1)
    else:
        result.status = ReviewStatus.APPROVED

    return result


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

async def critic_agent_node(state: DocumentState) -> dict:
    """
    LangGraph node for the Critic Agent.

    Reads from state, writes back:
      - review_result   (structured dict from ReviewResult)
      - approved        (bool)
      - iteration_count (incremented)
      - error           (set on failure)
    """
    logger.info(
        "Critic Agent invoked | document_id=%s | iteration=%d",
        state.get("document_id"),
        state.get("iteration_count", 0),
    )

    try:
        draft: str = state.get("draft_document", "")
        doc_type: str = state.get("document_type", "general")
        iteration: int = state.get("iteration_count", 0)

        if not draft:
            return {
                "review_result": {
                    "status": "needs_revision",
                    "score": 0,
                    "missing_sections": [],
                    "feedback": ["No document was generated."],
                },
                "approved": False,
                "iteration_count": iteration + 1,
                "error": "Empty draft received by Critic Agent",
            }

        settings = get_llm_settings()
        threshold = settings.quality_score_threshold

        # Step 1: Hard structural check
        pre_detected_missing = _find_missing_sections(draft, doc_type)
        logger.info(
            "Pre-check missing sections: %s", pre_detected_missing or "none"
        )

        # Step 2: LLM quality review
        result = await _llm_review(draft, doc_type, pre_detected_missing, threshold)

        # Step 3: Enforce hard rules over LLM result
        result = _enforce_hard_rules(result, pre_detected_missing, threshold)

        approved = result.status == ReviewStatus.APPROVED
        logger.info(
            "Critic result | status=%s | score=%d | missing=%s",
            result.status,
            result.score,
            result.missing_sections,
        )

        return {
            "review_result": result.model_dump(),
            "approved": approved,
            "iteration_count": iteration + 1,
            "error": "",
        }

    except Exception as exc:
        logger.exception("Critic Agent failed: %s", exc)
        return {
            "review_result": {
                "status": "needs_revision",
                "score": 0,
                "missing_sections": [],
                "feedback": [f"Critic Agent error: {exc}"],
            },
            "approved": False,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "error": f"Critic Agent error: {exc}",
        }
