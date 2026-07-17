<<<<<<< HEAD
# Agentic AI Document Review System

A production-ready autonomous multi-agent workflow that converts rough notes, bullet points, meeting transcripts, or unstructured text into highly structured professional documents.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Agent Workflow](#agent-workflow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Human-in-the-Loop](#human-in-the-loop)
- [Supported Document Types](#supported-document-types)
- [Running Tests](#running-tests)
- [Docker Deployment](#docker-deployment)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI HTTP Layer                           │
│  POST /generate-document │ GET /document/{id} │ POST /approve  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LangGraph Workflow Engine                      │
│                                                                 │
│   ┌──────────┐    ┌──────────────┐    ┌───────────────────┐    │
│   │  Writer  │───►│    Critic    │───►│  Route Decision   │    │
│   │  Agent   │    │    Agent     │    │                   │    │
│   │  (GPT-4o)│    │  (GPT-4o)   │    │ approved → END    │    │
│   └──────────┘    └──────────────┘    │ revision → Writer │    │
│        ▲                              │ max iter → Human  │    │
│        └──────── revision loop ───────┘                   │    │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Storage Layer (Redis / In-Memory)                  │
│         DocumentRecord │ TTL: 7 days │ JSON serialised          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Workflow

```
START
  │
  ▼
┌─────────────────────────────────────────────┐
│              Writer Agent                   │
│  • Detect document type from input          │
│  • Select company template                  │
│  • Generate structured Markdown document    │
│  • On revisions: incorporate Critic notes   │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│              Critic Agent                   │
│  • Hard-check: are all sections present?    │
│  • LLM quality review (0-100 score)         │
│  • Produce structured JSON feedback         │
│  • Enforce: score ≥ threshold + no missing  │
└──────────────────────┬──────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
   approved?                   needs_revision?
   score ≥ 80                  score < 80 OR
   no missing sections         missing sections
          │                         │
          ▼                    iteration < 3?
        END                         │
                           ┌────────┴────────┐
                           │                 │
                          YES                NO
                           │                 │
                           ▼                 ▼
                      Writer Agent    Human Review Queue
                      (revision)           │
                           │               ▼
                           │       POST /approve
                           │       (human decision)
                           └──────────────►END
```

### State Schema

```python
class DocumentState(TypedDict):
    document_id: str           # UUID for this run
    user_input: str            # Raw input text
    document_type: str         # Detected: prd | compliance_report | etc.
    draft_document: str        # Current Markdown document
    review_result: dict        # Latest Critic output
    iteration_count: int       # Write→Review cycles completed
    approved: bool             # True when Critic or human approves
    human_review_required: bool # True after max iterations
    error: str                 # Error details if any
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI |
| Agent Framework | LangGraph 0.1+ |
| LLM | GPT-4o (OpenAI) or Claude 3.5 (Anthropic) |
| Data Validation | Pydantic v2 |
| Structured Output | JSON Schema via `.with_structured_output()` |
| Storage | Redis 7.2 (in-memory fallback for dev) |
| Logging | structlog |
| Deployment | Docker + Docker Compose |
| Testing | pytest + pytest-asyncio |

---

## Project Structure

```
agentic-document-review/
│
├── app/
│   ├── agents/
│   │   ├── writer_agent.py      # Writer Agent: generates documents
│   │   └── critic_agent.py      # Critic Agent: reviews & scores
│   │
│   ├── graph/
│   │   └── workflow.py          # LangGraph StateGraph + routing logic
│   │
│   ├── api/
│   │   └── routes.py            # FastAPI endpoints
│   │
│   ├── schemas/
│   │   └── document_schema.py   # Pydantic models + TypedDict state
│   │
│   ├── services/
│   │   ├── llm_service.py       # LLM factory (OpenAI / Anthropic)
│   │   └── storage_service.py   # Redis persistence
│   │
│   └── main.py                  # FastAPI app + middleware + lifespan
│
├── tests/
│   ├── test_critic_agent.py
│   ├── test_writer_agent.py
│   ├── test_workflow.py
│   └── test_api.py
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Quick Start

### 1. Clone and configure

```bash
cd agentic-document-review
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### 2. Run with Docker Compose (recommended)

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 3. Run locally (without Docker)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Anthropic model name |
| `MAX_REVISION_ITERATIONS` | `3` | Max Writer→Critic cycles before human review |
| `QUALITY_SCORE_THRESHOLD` | `80` | Minimum score (0-100) to approve |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `APP_ENV` | `development` | `development` or `production` |

---

## API Reference

### `POST /api/v1/generate-document`

Trigger the full agentic workflow.

**Request:**
```json
{
  "content": "Meeting notes: We need a mobile expense tracking app...",
  "document_type": "prd"
}
```
`document_type` is optional — the Writer Agent auto-detects it.

**Response:**
```json
{
  "document_id": "3f7a1b2c-...",
  "document": "# Product Overview\n\n...",
  "document_type": "prd",
  "status": "approved",
  "score": 92,
  "iteration_count": 2,
  "missing_sections": [],
  "feedback": [],
  "created_at": "2025-01-15T10:30:00Z"
}
```

**Status values:**
- `approved` — Critic approved the document
- `needs_human_review` — Max iterations exhausted, awaiting human approval
- `processing` — Still in progress (rare in sync mode)

---

### `GET /api/v1/document/{document_id}`

Retrieve a previously generated document.

**Response:** Full `DocumentRecord` including all review metadata.

---

### `POST /api/v1/approve`

Human-in-the-loop approval for documents in the review queue.

**Request:**
```json
{
  "document_id": "3f7a1b2c-...",
  "approved": true,
  "reviewer_notes": "Looks good, approved with minor notes."
}
```

**Response:**
```json
{
  "document_id": "3f7a1b2c-...",
  "status": "approved",
  "message": "Document approved by human reviewer."
}
```

---

### `GET /api/v1/health`

```json
{"status": "healthy", "service": "agentic-document-review"}
```

---

## Human-in-the-Loop

When the Critic Agent cannot approve a document after `MAX_REVISION_ITERATIONS` cycles, the document is flagged with `status: needs_human_review`.

**Workflow:**
1. `POST /generate-document` returns `status: needs_human_review`
2. Human reviewer retrieves document via `GET /document/{id}`
3. Reviewer submits decision via `POST /approve`
4. Document status updates to `approved` or `rejected`

This pattern supports audit trails and compliance requirements where automated approval alone is insufficient.

---

## Supported Document Types

| Type | Value | Required Sections |
|------|-------|-------------------|
| Product Requirements Doc | `prd` | 13 sections (Overview → Release Plan) |
| Compliance Report | `compliance_report` | 7 sections |
| Consulting Memo | `consulting_memo` | 8 sections |
| Legal Summary | `legal_summary` | 8 sections |
| General | `general` | None enforced |

---

## Critic Agent Output Format

```json
{
  "status": "approved",
  "score": 92,
  "missing_sections": [],
  "feedback": []
}
```

or when revision is needed:

```json
{
  "status": "needs_revision",
  "score": 65,
  "missing_sections": [
    "Success Metrics",
    "Edge Cases"
  ],
  "feedback": [
    "Add measurable KPIs with baseline values and targets",
    "Include at least 5 failure scenarios with fallback behaviour"
  ]
}
```

---

## Running Tests

```bash
# Install test dependencies (included in requirements.txt)
pip install -r requirements.txt

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_critic_agent.py -v

# Run with coverage
pip install pytest-cov
pytest --cov=app --cov-report=term-missing
```

---

## Docker Deployment

```bash
# Build and start all services
docker compose up --build -d

# View logs
docker compose logs -f app

# Stop
docker compose down

# Stop and remove volumes
docker compose down -v
```

### Production Notes

- The `Dockerfile` uses a two-stage build to keep the runtime image small (~200MB).
- The app runs as a non-root user (`appuser`) for security.
- Redis persists data via an `appendonly` AOF file in a named volume.
- The HEALTHCHECK endpoint ensures the container is replaced if the app becomes unresponsive.
- Set `APP_ENV=production` to enable JSON-formatted structured logging.
=======
# agentic-doc-review
Agentic AI Document Review System is a multi-agent AI application that converts raw notes and drafts into professional documents. Using Writer and Critic agents with LangGraph and FastAPI, it automates document creation, review, revision, and approval while ensuring quality, consistency, and accuracy.
>>>>>>> e9eb40e4131989212a21e725660fd5a794b83d44
