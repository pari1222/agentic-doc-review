"""
Storage Service — Redis-backed document persistence.

Stores DocumentRecord objects as JSON. Falls back to an in-memory dict
when Redis is unavailable (useful for local dev without Docker).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import redis.asyncio as aioredis
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.document_schema import DocumentRecord, DocumentStatus

logger = logging.getLogger(__name__)

# TTL for documents in Redis — 7 days
DOC_TTL_SECONDS = 60 * 60 * 24 * 7


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class StorageSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    redis_url: str = "redis://localhost:6379/0"


# ---------------------------------------------------------------------------
# In-Memory fallback store
# ---------------------------------------------------------------------------
_memory_store: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Redis client (lazy singleton)
# ---------------------------------------------------------------------------
_redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> Optional[aioredis.Redis]:
    """Return a connected Redis client, or None if unavailable."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        settings = StorageSettings()
        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        _redis_client = client
        logger.info("Connected to Redis at %s", settings.redis_url)
        return _redis_client
    except Exception as exc:
        logger.warning("Redis unavailable (%s). Using in-memory store.", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def save_document(record: DocumentRecord) -> None:
    """Persist a DocumentRecord to Redis (or memory fallback)."""
    key = f"doc:{record.document_id}"
    payload = record.model_dump_json()

    redis = await get_redis()
    if redis:
        await redis.setex(key, DOC_TTL_SECONDS, payload)
        logger.debug("Saved document %s to Redis", record.document_id)
    else:
        _memory_store[key] = payload
        logger.debug("Saved document %s to memory", record.document_id)


async def get_document(document_id: str) -> Optional[DocumentRecord]:
    """Retrieve a DocumentRecord by ID. Returns None if not found."""
    key = f"doc:{document_id}"

    redis = await get_redis()
    if redis:
        raw = await redis.get(key)
    else:
        raw = _memory_store.get(key)

    if not raw:
        logger.debug("Document %s not found", document_id)
        return None

    try:
        return DocumentRecord.model_validate_json(raw)
    except Exception as exc:
        logger.error("Failed to deserialise document %s: %s", document_id, exc)
        return None


async def update_document_approval(
    document_id: str,
    approved: bool,
    reviewer_notes: Optional[str] = None,
) -> Optional[DocumentRecord]:
    """
    Set human approval status on an existing document.
    Returns the updated record, or None if document not found.
    """
    record = await get_document(document_id)
    if not record:
        return None

    record.approved_by_human = approved
    record.status = (
        DocumentStatus.APPROVED if approved else DocumentStatus.REJECTED
    )
    record.updated_at = datetime.utcnow()
    if reviewer_notes:
        record.feedback = record.feedback + [f"[Human Reviewer]: {reviewer_notes}"]

    await save_document(record)
    logger.info(
        "Document %s human approval=%s", document_id, approved
    )
    return record
