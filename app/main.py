"""
FastAPI Application Entry Point

Configures:
  - Structured logging
  - CORS middleware
  - API router
  - Startup / shutdown lifecycle events
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.api.routes import router
from app.graph.workflow import get_workflow


# ---------------------------------------------------------------------------
# App Settings
# ---------------------------------------------------------------------------

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    allowed_origins: str = "http://localhost:3000,http://localhost:8080"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


settings = AppSettings()


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure structlog for JSON logging in production, pretty in dev."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.app_env == "production":
        processors = [*shared_processors, structlog.processors.JSONRenderer()]
    else:
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so LangChain/FastAPI logs flow through
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )


configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the LangGraph workflow on startup."""
    logger.info("Starting Agentic Document Review System (env=%s)", settings.app_env)
    try:
        get_workflow()  # Compile the graph eagerly
        logger.info("LangGraph workflow ready")
    except Exception as exc:
        logger.warning("Workflow pre-warm failed: %s", exc)
    yield
    logger.info("Shutting down Agentic Document Review System")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic AI Document Review System",
    description=(
        "Autonomous multi-agent workflow that converts rough notes and "
        "unstructured text into professional documents (PRDs, Compliance "
        "Reports, Consulting Memos, Legal Summaries) using Writer and "
        "Critic AI agents powered by LangGraph."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service": "Agentic AI Document Review System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
