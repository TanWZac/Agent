# How-To Guide

A practical guide for users who want to run, integrate, and customize the **LangGraph Notepad RAG Agent**.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Agent (CLI)](#running-the-agent-cli)
5. [Running as an HTTP Service](#running-as-an-http-service)
6. [Using the API](#using-the-api)
7. [Uploading Files](#uploading-files)
8. [Connecting via MCP](#connecting-via-mcp)
9. [Choosing a Storage Backend](#choosing-a-storage-backend)
10. [Using Agent Personas](#using-agent-personas)
11. [Programmatic Integration](#programmatic-integration)
12. [Running with Docker](#running-with-docker)
13. [Database Migrations (Alembic)](#database-migrations-alembic)
14. [Running Tests](#running-tests)
15. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.10 or later
- An OpenAI API key (or a local HuggingFace GGUF model)
- (Optional) Docker for containerised deployment

---

## Installation

```bash
# Clone the repository
git clone https://github.com/TanWZac/Agent.git
cd Agent

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

---

## Configuration

All settings are controlled via environment variables or a `.env` file in the project root.

### Minimal `.env` file

```dotenv
OPENAI_API_KEY=sk-...your-key-here...
```

### Full list of environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model to use |
| `OPENAI_TEMPERATURE` | `0` | LLM temperature (0–2) |
| `LLM_PROVIDER` | `openai` | LLM provider: `openai` or `huggingface` |
| `STORE_BACKEND` | `file` | Storage backend: `file`, `chroma`, or `sqlite` |
| `NOTE_FILE` | `data/notepad.txt` | Path to note file (file backend) |
| `SQLITE_DB_URL` | `sqlite:///data/notepad.db` | SQLite connection URL (sqlite backend) |
| `CHROMA_COLLECTION` | `notepad` | ChromaDB collection name |
| `CHROMA_PERSIST_DIR` | `data/chroma_db` | ChromaDB persistence directory |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `MAX_NOTE_FILE_SIZE_MB` | `10` | Max note file size |
| `RETRIEVAL_TOP_K` | `3` | Number of notes to retrieve |
| `WEB_SEARCH_MAX_RESULTS` | `5` | Max web search results |
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server port |
| `API_KEY` | *(empty — auth disabled)* | If set, clients must send `X-API-Key` header |
| `MAX_UPLOAD_FILE_SIZE_MB` | `50` | Maximum file upload size |
| `FILE_CHUNK_SIZE` | `1000` | Characters per chunk for file ingestion |
| `FILE_CHUNK_OVERLAP` | `200` | Overlap between chunks for context continuity |

---

## Running the Agent (CLI)

```bash
# Start interactive conversation
notepad-agent

# Use a custom note file
notepad-agent --note-file my_notes.txt

# Use a specific persona
notepad-agent --persona research_assistant

# List all available personas
notepad-agent --list-personas

# Enable debug logging
notepad-agent --log-level DEBUG
```

Once running, type messages at the `You:` prompt. Type `exit` to quit.

---

## Running as an HTTP Service

```bash
# Via CLI flag
notepad-agent --serve

# Or the dedicated entry point
notepad-agent-serve
```

The API starts on `http://0.0.0.0:8000` by default.

---

## Using the API

### Health check

```bash
curl http://localhost:8000/health
```

### Send a chat message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?"}'
```

### Send a chat message with a specific persona

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Review this code snippet...", "persona": "code_reviewer"}'
```

### List available personas

```bash
curl http://localhost:8000/personas
```

### Use an existing session (multi-turn)

```bash
# The first /chat response returns a session_id — pass it in subsequent requests
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Remember my name is Alex", "session_id": "YOUR_SESSION_ID"}'
```

### Authentication

If `API_KEY` is set in your environment, include the key in every mutating request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"message": "Hello"}'
```

### API endpoints summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/personas` | List available personas |
| `POST` | `/chat` | Send a message |
| `POST` | `/chat/upload` | Upload a file and chat about its content |
| `GET` | `/sessions/{id}` | Session info |
| `DELETE` | `/sessions/{id}` | Delete session |
| `POST` | `/sessions/{id}/reset` | Reset session history |
| `GET` | `/mcp` | MCP SSE endpoint |
| `POST` | `/mcp` | MCP message endpoint |

---

## Uploading Files

The agent supports file upload via [Microsoft MarkItDown](https://github.com/microsoft/markitdown). Uploaded files are converted to Markdown, chunked, and persisted in the vector store for cross-turn RAG retrieval.

### Supported file types

PDF, DOCX, DOC, XLSX, XLS, PPTX, PPT, HTML, TXT, MD, CSV, TSV, JSON, XML, RTF, EPUB, IPYNB, WAV, MP3, JPG, PNG, GIF, BMP, TIFF, WEBP, ZIP.

### Upload via CLI

```bash
# Analyze a file
notepad-agent --file report.pdf

# Then ask follow-up questions about it in the same session
You: What were the key revenue figures mentioned in the report?
```

### Upload via API

```bash
# Upload a file with a question
curl -X POST http://localhost:8000/chat/upload \
  -F "file=@quarterly_report.docx" \
  -F "message=What are the top 3 action items?"

# Upload without a message (auto-summarizes)
curl -X POST http://localhost:8000/chat/upload \
  -F "file=@presentation.pptx"

# Upload into an existing session
curl -X POST http://localhost:8000/chat/upload \
  -F "file=@data.xlsx" \
  -F "message=Compare this with the previous data" \
  -F "session_id=YOUR_SESSION_ID"
```

### Upload via Python

```python
from src.agent import AgentSession

session = AgentSession()

# Ingest a file into the session's vector store
num_chunks = session.ingest_file("path/to/report.pdf")
print(f"Stored {num_chunks} chunks")

# Ask questions — the agent retrieves relevant chunks automatically
response = session.chat("What are the main conclusions in the report?")
```

### How it works

1. The file is converted to Markdown via MarkItDown
2. The Markdown is split into overlapping chunks (configurable via `FILE_CHUNK_SIZE` and `FILE_CHUNK_OVERLAP`)
3. Each chunk is stored in the note store with metadata: `[File: filename | Chunk N/M]`
4. The agent uses `retrieve_notes` to find relevant chunks when answering questions
5. File content persists across conversation turns within the session

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_UPLOAD_FILE_SIZE_MB` | `50` | Maximum allowed file size |
| `FILE_CHUNK_SIZE` | `1000` | Target chunk size in characters |
| `FILE_CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |

---

## Connecting via MCP

The agent exposes an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server at the `/mcp` endpoint.

### From Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "notepad-agent": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### From VS Code (Copilot MCP)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "notepad-agent": {
      "type": "sse",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### From Python code

```python
from mcp import ClientSession
from mcp.client.sse import sse_client

async with sse_client("http://localhost:8000/mcp") as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("search_web", {"query": "latest AI news"})
        print(result)
```

### MCP tools exposed

| Tool | Description |
|------|-------------|
| `search_web` | Search the web via DuckDuckGo |
| `save_note` | Save a note to persistent memory |
| `retrieve_notes` | RAG retrieval over saved notes |
| `ingest_file` | Convert and ingest uploaded files via MarkItDown |
| `list_notes` | List all saved notes |
| `clear_notes` | Clear all notes |

---

## Choosing a Storage Backend

### File (default)

Stores notes in a plain text file. Zero dependencies, good for single-user local use.

```dotenv
STORE_BACKEND=file
NOTE_FILE=data/notepad.txt
```

### ChromaDB (vector search)

Stores notes with semantic embeddings for better retrieval quality.

```dotenv
STORE_BACKEND=chroma
CHROMA_COLLECTION=notepad
CHROMA_PERSIST_DIR=data/chroma_db
```

### SQLite (relational)

Stores notes in a SQLite database via SQLAlchemy ORM. Good for structured queries and when you need migrations.

```dotenv
STORE_BACKEND=sqlite
SQLITE_DB_URL=sqlite:///data/notepad.db
```

Run Alembic to create tables:

```bash
alembic upgrade head
```

---

## Using Agent Personas

Personas change the agent's system prompt — its duty, responsibility, tone, and domain knowledge — without modifying code.

### Built-in personas

| Name | Purpose |
|------|---------|
| `default` | General-purpose assistant |
| `research_assistant` | Academic research with citations |
| `customer_support` | Friendly, solution-oriented support |
| `code_reviewer` | Senior code reviewer providing actionable feedback |
| `technical_writer` | Clear documentation and technical writing |
| `data_analyst` | Data interpretation and statistical analysis |

### Using personas from the CLI

```bash
notepad-agent --persona code_reviewer
```

### Using personas from the API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain quicksort", "persona": "research_assistant"}'
```

### Creating a custom persona (Python)

```python
from src.agent.prompts import Persona, register_persona

my_persona = Persona(
    name="legal_advisor",
    display_name="a legal advisor",
    description="You provide general legal information and guidance.",
    duty="Answer legal questions with references to relevant principles and statutes.",
    responsibility="Always clarify you are not a licensed attorney and cannot provide legal advice.",
    tone="formal, precise, and cautious",
    tool_instructions="Use web search to find up-to-date legal references.",
    domain_knowledge="Focus on common law jurisdictions. Cite sources when possible.",
    custom_rules=[
        "Never guarantee legal outcomes.",
        "Recommend consulting a qualified attorney for specific situations.",
    ],
)
register_persona(my_persona)
```

Then use it by name: `--persona legal_advisor` or `"persona": "legal_advisor"` in API requests.

---

## Programmatic Integration

```python
from src.agent import AgentSession
from src.config import get_settings

# Create a session with custom settings and persona
settings = get_settings(note_file="my_notes.txt")
session = AgentSession(settings=settings, persona="research_assistant")

# Chat — maintains history within the session
response = session.chat("What are the latest papers on transformer architectures?")
print(response)

# The session remembers context
response = session.chat("Summarize the top 3 results")
print(response)
```

---

## Running with Docker

### Build the image

```bash
docker build -f deploy/Dockerfile -t notepad-agent .
```

### Run the container

```bash
docker run -d \
  --name notepad-agent \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e API_KEY=my-secret-key \
  notepad-agent
```

### Verify it's running

```bash
curl http://localhost:8000/health
# {"status":"healthy","active_sessions":0}
```

### Docker Compose (example)

```yaml
version: "3.9"
services:
  agent:
    build:
      context: .
      dockerfile: deploy/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - API_KEY=${API_KEY}
      - STORE_BACKEND=sqlite
      - SQLITE_DB_URL=sqlite:///data/notepad.db
    volumes:
      - agent-data:/app/data
volumes:
  agent-data:
```

---

## Database Migrations (Alembic)

Only applicable when using `STORE_BACKEND=sqlite`.

```bash
# Apply all pending migrations
alembic upgrade head

# Check current migration version
alembic current

# Create a new migration after modifying models in src/store/db.py
alembic revision --autogenerate -m "add column X to notes"

# Rollback one migration
alembic downgrade -1
```

---

## Running Tests

```bash
# Run all tests
pytest -q

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_api.py -v
```

---

## Responsible AI Features

The agent includes a layered Responsible AI system that runs automatically on every interaction. You can also use individual components programmatically.

### Installing RAI dependencies

Some RAI features (NeMo Guardrails, Fairlearn) require optional dependencies:

```bash
pip install -e ".[rai]"
```

### Validating a Test Set Before Evaluation

If you have a dataset of (prompt, output, reference) triples for bias/fairness testing, validate coherence first:

```python
from src.responsible_ai import TestSetValidator

validator = TestSetValidator(threshold=0.4, max_invalid_ratio=0.2)

prompts = ["What is Python?", "Explain gravity", ...]
outputs = ["Python is a programming language.", "Gravity pulls objects together.", ...]
references = ["Python is a versatile language.", "Gravity is a fundamental force.", ...]

report = validator.validate(prompts, outputs, references)

if report.is_valid:
    print("Test set is coherent — proceed with evaluation.")
else:
    print(report.summary)
    print(f"Invalid row indices: {report.invalid_rows}")

    # Optionally filter and keep only valid rows:
    f_prompts, f_outputs, f_refs, report = validator.filter_valid_rows(
        prompts, outputs, references
    )
    print(f"Filtered to {len(f_prompts)} valid rows.")
```

**Configuration options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.4` | Minimum cosine similarity for a (column A, column B) pair to be considered "matching" |
| `max_invalid_ratio` | `0.2` | Maximum fraction of invalid rows before the dataset is rejected (20%) |

### Using NeMo Guardrails

```python
from src.responsible_ai.nemo_rails import NemoContentRail

rail = NemoContentRail()

# Check user input
result = rail.check_input("How do I hack into a system?")
if not result.is_safe:
    print(f"Blocked: {result.reason}")

# Check model output
result = rail.check_output("Here's how to bypass security...")
```

### Running Fairness Evaluation

```python
from src.responsible_ai.fairness import FairnessEvaluator

evaluator = FairnessEvaluator()

# Evaluate response quality across demographic groups
report = evaluator.evaluate_response_quality(
    responses=["Good answer", "Poor answer", "Great answer", "Ok answer"],
    sensitive_features=["group_a", "group_b", "group_a", "group_b"],
    quality_scores=[0.9, 0.3, 0.85, 0.6],
)

print(f"Fair: {report.is_fair}")
print(report.recommendations)
```

---

## Troubleshooting

### "Configuration error: OPENAI_API_KEY is required"

Set the `OPENAI_API_KEY` environment variable or add it to your `.env` file.

### "ModuleNotFoundError: No module named 'src'"

Make sure you installed the project in editable mode (`pip install -e .`) or set `PYTHONPATH` to the project root:

```bash
export PYTHONPATH=$(pwd)
```

### API returns 403 Forbidden

You have `API_KEY` set but are not sending the `X-API-Key` header. Either:
- Add `-H "X-API-Key: your-key"` to your request, or
- Unset `API_KEY` to disable authentication (dev only).

### ChromaDB import errors

Install the full dependencies:

```bash
pip install -e ".[dev]"
```

### "Persona 'xyz' not found"

Run `notepad-agent --list-personas` to see available personas. Custom personas must be registered before use (see [Creating a custom persona](#creating-a-custom-persona-python)).

### Agent is slow on first run

The sentence-transformers model is downloaded on first use (~90 MB). Subsequent runs use the cached model.

### Docker health check failing

Ensure the container can reach port 8000 internally. Check logs:

```bash
docker logs notepad-agent
```
