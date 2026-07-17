"""
FastAPI Application Entry Point
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
    allowed_origins: str = "*"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


settings = AppSettings()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging() -> None:
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
        processors = [*shared_processors, structlog.dev.ConsoleRenderer(colors=True)]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)


configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Agentic Document Review System (env=%s)", settings.app_env)
    try:
        get_workflow()
        logger.info("LangGraph workflow ready")
    except Exception as exc:
        logger.warning("Workflow pre-warm failed: %s", exc)
    yield
    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic AI Document Review System",
    description="Multi-agent document generation workflow using LangGraph.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routes ────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")

# ── Frontend static files ─────────────────────────────────────────────────────
_frontend_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)

if os.path.isdir(_frontend_dir):
    app.mount("/assets", StaticFiles(directory=_frontend_dir), name="assets")
    logger.info("Frontend dir mounted: %s", _frontend_dir)


@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = os.path.join(_frontend_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"service": "Agentic AI Document Review System", "docs": "/docs"}


@app.get("/style.css", include_in_schema=False)
async def serve_css():
    return FileResponse(os.path.join(_frontend_dir, "style.css"), media_type="text/css")


@app.get("/app.js", include_in_schema=False)
async def serve_js():
    return FileResponse(os.path.join(_frontend_dir, "app.js"), media_type="application/javascript")
