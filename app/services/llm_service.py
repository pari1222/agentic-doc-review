"""
LLM Service — centralised factory for LangChain chat models.

Reads LLM_PROVIDER from environment and returns the appropriate
ChatOpenAI or ChatAnthropic instance. Both support structured output
via .with_structured_output() so callers are provider-agnostic.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class LLMSettings(BaseSettings):
    """Reads from environment variables / .env file."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "openai"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Shared
    temperature: float = 0.2          # Low temp for structured professional docs
    max_tokens: int = 8192
    max_revision_iterations: int = 3
    quality_score_threshold: int = 80


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    """Cached settings singleton."""
    return LLMSettings()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_llm(
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """
    Return a configured LangChain chat model based on LLM_PROVIDER env var.

    Args:
        temperature: Override default temperature if provided.
        max_tokens:  Override default max_tokens if provided.

    Returns:
        A LangChain BaseChatModel instance (OpenAI or Anthropic).

    Raises:
        ValueError: If the provider is unsupported or API key is missing.
    """
    settings = get_llm_settings()
    temp = temperature if temperature is not None else settings.temperature
    tokens = max_tokens if max_tokens is not None else settings.max_tokens

    provider = settings.llm_provider.lower()
    logger.info("Initialising LLM provider=%s", provider)

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        from langchain_openai import ChatOpenAI  # lazy import
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=temp,
            max_tokens=tokens,
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        from langchain_anthropic import ChatAnthropic  # lazy import
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=temp,
            max_tokens=tokens,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. "
        "Choose 'openai' or 'anthropic'."
    )


def get_llm_with_structured_output(schema: Any) -> Any:
    """
    Return an LLM chain that enforces structured JSON output matching `schema`.

    Usage:
        chain = get_llm_with_structured_output(ReviewResult)
        result: ReviewResult = chain.invoke(messages)
    """
    llm = get_llm()
    return llm.with_structured_output(schema)
