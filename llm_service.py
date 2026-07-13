"""
LLM Service — Multi-provider LangChain model factory.

IMPORTANT — Why Google AQ keys fail (ACCESS_TOKEN_TYPE_UNSUPPORTED):
─────────────────────────────────────────────────────────────────────
AQ.xxx tokens are short-lived OAuth 2.0 user access tokens (~1hr TTL).
The Gemini REST API (generativelanguage.googleapis.com) only accepts:
  - Static API keys (AIza... format) via ?key= or x-goog-api-key header
  - Service account credentials via OAuth with correct token type

AQ tokens carry token_type=ACCESS_TOKEN which the API rejects as
ACCESS_TOKEN_TYPE_UNSUPPORTED. No endpoint, SDK version, or header
combination resolves this — it is enforced server-side by Google.

SOLUTION OPTIONS:
  1. OpenRouter (free) — set LLM_PROVIDER=openai, OPENAI_BASE_URL=https://openrouter.ai/api/v1
  2. Fresh Google account — generates AIza keys that work
  3. OpenAI directly — set LLM_PROVIDER=openai, OPENAI_API_KEY=sk-...
  4. Anthropic — set LLM_PROVIDER=anthropic, ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load .env file into os.environ. Searches upward from cwd."""
    try:
        from dotenv import load_dotenv, find_dotenv
        path = find_dotenv(usecwd=True)
        if path:
            load_dotenv(path, override=True)
            logger.debug("Loaded .env: %s", path)
            return
        # Manual fallback
        for candidate in [
            pathlib.Path.cwd() / ".env",
            pathlib.Path(__file__).parents[2] / ".env",
        ]:
            if candidate.exists():
                load_dotenv(str(candidate), override=True)
                logger.debug("Loaded .env: %s", candidate)
                return
    except Exception as exc:
        logger.warning("dotenv load failed: %s", exc)


_load_dotenv()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def get_llm_settings():
    """Read LLM config live from environment (no caching to avoid stale reads)."""
    class S:
        llm_provider            = os.environ.get("LLM_PROVIDER", "openai").strip()
        # Google Gemini (AIza keys only — AQ tokens are not supported by the API)
        google_api_key          = os.environ.get("GOOGLE_API_KEY", "").strip()
        google_model            = os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash").strip()
        # OpenAI / OpenRouter / any OpenAI-compatible API
        openai_api_key          = os.environ.get("OPENAI_API_KEY", "").strip()
        openai_model            = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
        openai_base_url         = os.environ.get("OPENAI_BASE_URL", "").strip()
        # Anthropic
        anthropic_api_key       = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        anthropic_model         = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip()
        # Shared
        temperature             = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
        max_tokens              = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
        max_revision_iterations = int(os.environ.get("MAX_REVISION_ITERATIONS", "3"))
        quality_score_threshold = int(os.environ.get("QUALITY_SCORE_THRESHOLD", "80"))
    return S()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_llm(
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """
    Return a configured LangChain chat model.

    Supported providers: google, openai, anthropic.
    For OpenRouter, set LLM_PROVIDER=openai and OPENAI_BASE_URL=https://openrouter.ai/api/v1
    """
    s        = get_llm_settings()
    temp     = temperature if temperature is not None else s.temperature
    tokens   = max_tokens  if max_tokens  is not None else s.max_tokens
    provider = s.llm_provider.lower()

    logger.info("Building LLM | provider=%s", provider)

    # ── Google Gemini (requires AIza... API key) ─────────────────────────────
    if provider == "google":
        key = s.google_api_key
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY not set in .env\n"
                "Get a key at: https://aistudio.google.com/app/apikeys\n"
                "Note: AQ... tokens are NOT supported. You need an AIza... API key.\n"
                "If your account only generates AQ tokens, use OpenRouter instead:\n"
                "  LLM_PROVIDER=openai\n"
                "  OPENAI_BASE_URL=https://openrouter.ai/api/v1\n"
                "  OPENAI_API_KEY=sk-or-v1-... (free at openrouter.ai)"
            )

        if key.startswith("AQ.") or key.startswith("AQ "):
            raise ValueError(
                "Your Google API key starts with 'AQ.' — this is an OAuth access token,\n"
                "NOT a static API key. The Gemini REST API rejects it with 401 ACCESS_TOKEN_TYPE_UNSUPPORTED.\n\n"
                "WHY: AQ tokens are short-lived (~1hr) OAuth user tokens. The Gemini API only\n"
                "accepts static API keys (AIza... format) or service account credentials.\n\n"
                "YOUR OPTIONS:\n"
                "1. Use OpenRouter (FREE, works now):\n"
                "   LLM_PROVIDER=openai\n"
                "   OPENAI_BASE_URL=https://openrouter.ai/api/v1\n"
                "   OPENAI_API_KEY=sk-or-v1-xxx  (get free key at openrouter.ai)\n"
                "   OPENAI_MODEL=google/gemini-2.0-flash-exp:free\n\n"
                "2. Create fresh Google account -> aistudio.google.com/app/apikeys -> AIza key\n\n"
                "3. Use OpenAI: LLM_PROVIDER=openai, OPENAI_API_KEY=sk-...\n\n"
                "4. Use Anthropic: LLM_PROVIDER=anthropic, ANTHROPIC_API_KEY=sk-ant-..."
            )

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Run: pip install langchain-google-genai")

        logger.info("Gemini model: %s", s.google_model)
        return ChatGoogleGenerativeAI(
            model=s.google_model,
            google_api_key=key,
            temperature=temp,
            max_output_tokens=tokens,
            convert_system_message_to_human=True,
        )

    # ── OpenAI / OpenRouter / any OpenAI-compatible provider ─────────────────
    if provider == "openai":
        key = s.openai_api_key
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not set in .env\n"
                "For OpenRouter (free Gemini access): get key at openrouter.ai\n"
                "For OpenAI: get key at platform.openai.com/api-keys"
            )

        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Run: pip install langchain-openai")

        kwargs: dict[str, Any] = dict(
            model=s.openai_model,
            api_key=key,
            temperature=temp,
            max_tokens=tokens,
        )
        if s.openai_base_url:
            kwargs["base_url"] = s.openai_base_url
            logger.info("OpenAI-compatible base_url: %s | model: %s", s.openai_base_url, s.openai_model)
        else:
            logger.info("OpenAI model: %s", s.openai_model)

        return ChatOpenAI(**kwargs)

    # ── Anthropic ────────────────────────────────────────────────────────────
    if provider == "anthropic":
        key = s.anthropic_api_key
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set in .env\n"
                "Get key at: console.anthropic.com"
            )

        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Run: pip install langchain-anthropic")

        logger.info("Anthropic model: %s", s.anthropic_model)
        return ChatAnthropic(
            model=s.anthropic_model,
            api_key=key,
            temperature=temp,
            max_tokens=tokens,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER='{s.llm_provider}'. "
        "Choose: 'google' (AIza keys only), 'openai', or 'anthropic'."
    )


def get_llm_with_structured_output(schema: Any) -> Any:
    return get_llm().with_structured_output(schema)
