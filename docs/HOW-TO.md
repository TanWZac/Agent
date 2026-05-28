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
7. [Connecting via MCP](#connecting-via-mcp)
8. [Choosing a Storage Backend](#choosing-a-storage-backend)
9. [Using Agent Personas](#using-agent-personas)
10. [Programmatic Integration](#programmatic-integration)
11. [Running with Docker](#running-with-docker)
12. [Database Migrations (Alembic)](#database-migrations-alembic)
13. [Running Tests](#running-tests)
14. [Troubleshooting](#troubleshooting)

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
| `GET` | `/sessions/{id}` | Session info |
| `DELETE` | `/sessions/{id}` | Delete session |
| `POST` | `/sessions/{id}/reset` | Reset session history |
| `GET` | `/mcp` | MCP SSE endpoint |
| `POST` | `/mcp` | MCP message endpoint |

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
