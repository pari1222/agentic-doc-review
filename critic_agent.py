"""
Critic Agent — reviews generated documents against company standards.
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
You are a Senior Quality Reviewer for professional documents.

Evaluate the document and return ONLY a valid JSON object — no markdown fences, no extra text.

JSON structure (return exactly this):
{
  "status": "approved",
  "score": 85,
  "missing_sections": [],
  "feedback": []
}

Rules:
- status must be "approved" or "needs_revision"
- score is an integer 0-100
- missing_sections is a list of section name strings
- feedback is a list of actionable string instructions
- Output ONLY the raw JSON — nothing else
""").strip()


# ---------------------------------------------------------------------------
# Section checker
# ---------------------------------------------------------------------------

def _detect_present_sections(document: str) -> set[str]:
    headings = re.findall(r"^#{1,6}\s+(.+)$", document, re.MULTILINE)
    return {h.strip().rstrip(":").lower() for h in headings}


def _find_missing_sections(document: str, doc_type: str) -> list[str]:
    required = REQUIRED_SECTIONS_MAP.get(doc_type, [])
    if not required:
        return []
    present = _detect_present_sections(document)
    return [s for s in required if s.lower() not in present]


# ---------------------------------------------------------------------------
# LLM review
# ---------------------------------------------------------------------------

async def _llm_review(
    document: str,
    doc_type: str,
    pre_detected_missing: list[str],
    threshold: int,
) -> ReviewResult:
    llm = get_llm(temperature=0.1)

    required_sections = REQUIRED_SECTIONS_MAP.get(doc_type, [])
    sections_str = "\n".join(f"  - {s}" for s in required_sections) or "  (none specified)"
    missing_str  = "\n".join(f"  - {s}" for s in pre_detected_missing) if pre_detected_missing else "  None"

    # Truncate very long documents to avoid token limits
    doc_preview = document[:6000] + "\n...[truncated]" if len(document) > 6000 else document

    user_message = dedent(f"""
        Document Type: {doc_type.upper().replace('_', ' ')}
        Quality threshold: {threshold}/100

        Required sections:
        {sections_str}

        Pre-check found these missing sections:
        {missing_str}

        --- DOCUMENT ---
        {doc_preview}
        --- END ---

        Return ONLY the JSON evaluation object.
    """).strip()

    # For Gemini: merge system prompt into human message
    combined = f"{CRITIC_SYSTEM_PROMPT}\n\n{user_message}"
    messages = [HumanMessage(content=combined)]

    logger.info("Critic Agent calling LLM")
    response = await llm.ainvoke(messages)

    raw = response.content.strip()
    # Remove markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Extract JSON object if surrounded by extra text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        data = json.loads(raw)
        return ReviewResult.model_validate(data)
    except Exception as exc:
        logger.error("Critic JSON parse failed: %s | raw=%s", exc, raw[:300])
        return ReviewResult(
            status=ReviewStatus.NEEDS_REVISION,
            score=50,
            missing_sections=pre_detected_missing,
            feedback=["Could not parse review output. Please check document structure."],
        )


# ---------------------------------------------------------------------------
# Hard rules
# ---------------------------------------------------------------------------

def _enforce_hard_rules(
    result: ReviewResult,
    pre_detected_missing: list[str],
    threshold: int,
) -> ReviewResult:
    all_missing = list({*result.missing_sections, *pre_detected_missing})
    result.missing_sections = all_missing

    if all_missing or result.score < threshold:
        result.status = ReviewStatus.NEEDS_REVISION
        if result.score >= threshold and all_missing:
            result.score = min(result.score, threshold - 1)
    else:
        result.status = ReviewStatus.APPROVED

    return result


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

async def critic_agent_node(state: DocumentState) -> dict:
    logger.info(
        "Critic Agent | doc_id=%s | iteration=%d",
        state.get("document_id"), state.get("iteration_count", 0),
    )

    try:
        draft     : str  = state.get("draft_document", "")
        doc_type  : str  = state.get("document_type", "general")
        iteration : int  = state.get("iteration_count", 0)

        if not draft or not draft.strip():
            return {
                "review_result": {
                    "status": "needs_revision",
                    "score": 0,
                    "missing_sections": [],
                    "feedback": ["No document was generated. Check LLM API key and model name."],
                },
                "approved": False,
                "iteration_count": iteration + 1,
                "error": "Empty draft",
            }

        s = get_llm_settings()
        threshold = s.quality_score_threshold

        pre_missing = _find_missing_sections(draft, doc_type)
        logger.info("Pre-check missing: %s", pre_missing or "none")

        result  = await _llm_review(draft, doc_type, pre_missing, threshold)
        result  = _enforce_hard_rules(result, pre_missing, threshold)
        approved = result.status == ReviewStatus.APPROVED

        logger.info(
            "Critic result | status=%s | score=%d | missing=%s",
            result.status, result.score, result.missing_sections,
        )

        return {
            "review_result": result.model_dump(),
            "approved": approved,
            "iteration_count": iteration + 1,
            "error": "",
        }

    except Exception as exc:
        import traceback
        logger.error("Critic Agent error:\n%s", traceback.format_exc())
        return {
            "review_result": {
                "status": "needs_revision",
                "score": 0,
                "missing_sections": [],
                "feedback": [f"Critic Agent error: {exc}"],
            },
            "approved": False,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "error": str(exc),
        }
