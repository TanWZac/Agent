# LangGraph Notepad RAG Agent

A LangGraph-powered conversational agent with web search and persistent notepad memory. Designed for **interactive CLI** use, **HTTP service integration**, and **MCP (Model Context Protocol) server** exposure.

## Features

- LangGraph agent with tool-calling (GPT-4o-mini default, configurable)
- Web search via DuckDuckGo
- Persistent notepad with lightweight RAG retrieval (Jaccard + stop-word filtering)
- **File upload & ingestion** via [Microsoft MarkItDown](https://github.com/microsoft/markitdown) — supports PDF, DOCX, XLSX, PPTX, HTML, images, and more
- Multi-turn conversation memory within sessions
- **FastAPI HTTP service** for integration into microservice architectures
- **MCP server** (SSE transport) — expose tools to Claude Desktop, VS Code Copilot, or any MCP client
- Centralized configuration via environment variables
- Structured logging for observability
- Graceful error handling with custom exception hierarchy

## Architecture

```
src/
├── cli.py                  # CLI entry point
├── agent/                  # LangGraph agent core
│   ├── file_ingest.py      # File upload & MarkItDown conversion
│   ├── graph.py            # State graph with RAI guardrail nodes
│   ├── llm.py              # LLM factory (OpenAI / HuggingFace)
│   ├── session.py          # AgentSession — orchestrates conversation
│   └── tools.py            # Tool definitions (search, save, retrieve, ingest_file)
├── config/                 # Configuration management
│   ├── __init__.py         # Settings dataclass + loaders
│   └── config.json         # Default configuration values
├── core/                   # Shared utilities
│   ├── embeddings.py       # Sentence-transformer embeddings
│   ├── exceptions.py       # Custom exception hierarchy
│   └── logging.py          # Structured logging setup
├── responsible_ai/         # Responsible AI guardrails
│   ├── bias_evaluator.py   # Bias & fairness detection
│   ├── config.py           # RAI configuration
│   ├── content_filter.py   # Harmful content + jailbreak detection
│   ├── fairness.py         # Fairlearn statistical fairness metrics
│   ├── guardrails.py       # Orchestrator (input/output checks)
│   ├── nemo_rails.py       # NeMo Guardrails content safety
│   ├── pii_detector.py     # PII detection & redaction
│   ├── testset_validator.py# Test set relevancy validation
│   └── transparency.py     # Privacy-preserving audit logger
├── server/                 # Service layer
│   ├── api.py              # FastAPI HTTP service
│   ├── mcp.py              # MCP server (SSE transport)
│   └── registry.py         # MCP tool registry
└── store/                  # Storage backends
    ├── chroma_store.py     # ChromaDB vector store
    ├── factory.py          # Store factory
  ├── file_store.py       # File-based notepad store
  ├── sql_store.py        # SQLite note store (SQLAlchemy)
  └── db.py               # SQLAlchemy ORM models
```

## Responsible AI

The agent ships with a comprehensive **Responsible AI** layer that wraps every interaction:

| Component | Description |
|-----------|-------------|
| **Content Filter** | Regex-based detection of harmful content, jailbreak attempts, and prompt injection |
| **PII Detector** | Identifies and redacts emails, phone numbers, SSNs, credit cards, and IP addresses |
| **Bias Evaluator** | Keyword-based bias/stereotype detection across protected categories |
| **NeMo Guardrails** | LLM-as-judge content safety via NVIDIA NeMo Guardrails (Colang flows) |
| **Fairlearn Integration** | Statistical fairness metrics (demographic parity, selection rate) for model outputs |
| **Test Set Validator** | Embedding-based coherence check for evaluation datasets before bias/fairness testing |
| **Audit Logger** | Privacy-preserving JSONL audit trail with configurable retention |

### Test Set Validation

Before running bias or fairness evaluation on a user-provided test set, validate that the (prompt, output, reference) triples are semantically coherent:

```python
from src.responsible_ai import TestSetValidator

validator = TestSetValidator(threshold=0.4, max_invalid_ratio=0.2)
report = validator.validate(prompts, outputs, references)

if not report.is_valid:
    print(report.summary)
    print(f"Invalid rows: {report.invalid_rows}")

# Or filter and proceed with only valid rows:
f_prompts, f_outputs, f_refs, report = validator.filter_valid_rows(
    prompts, outputs, references
)
```

Install optional RAI dependencies:

```bash
pip install -e ".[rai]"
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

## Usage

### CLI Mode

```bash
notepad-agent
# or with custom note file:
notepad-agent --note-file my_notes.txt
```

### HTTP Service Mode

```bash
# Via CLI flag:
notepad-agent --serve

# Or directly:
notepad-agent-serve
```

The API runs on `http://0.0.0.0:8000` by default (configurable via `API_HOST`/`API_PORT`).

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Send message, get response |
| `POST` | `/chat/upload` | Upload a file and chat about its content |
| `GET` | `/sessions/{id}` | Session info |
| `DELETE` | `/sessions/{id}` | Delete session |
| `POST` | `/sessions/{id}/reset` | Reset session history |
| `GET` | `/mcp` | MCP SSE connection endpoint |
| `POST` | `/mcp` | MCP message handling endpoint |

**Example:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Berlin?"}'
```

**File upload example:**

```bash
curl -X POST http://localhost:8000/chat/upload \
  -F "file=@report.pdf" \
  -F "message=Summarize the key findings"
```

### Programmatic Integration

```python
from src.agent import AgentSession
from src.config import get_settings

settings = get_settings(note_file="custom_notes.txt")
session = AgentSession(settings=settings)

response = session.chat("Remember that my name is Alex")
response = session.chat("What is my name?")  # Uses both session history and RAG
```

## Configuration

All settings are loaded from environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model to use |
| `OPENAI_TEMPERATURE` | `0` | LLM temperature |
| `STORE_BACKEND` | `file` | Storage backend: `file`, `chroma`, or `sqlite` |
| `NOTE_FILE` | `data/notepad.txt` | Notepad file path |
| `SQLITE_DB_URL` | `sqlite:///data/notepad.db` | SQLite connection URL |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_NOTE_FILE_SIZE_MB` | `10` | Max notepad file size |
| `RETRIEVAL_TOP_K` | `3` | Number of notes to retrieve |
| `WEB_SEARCH_MAX_RESULTS` | `5` | Max web search results |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `MAX_UPLOAD_FILE_SIZE_MB` | `50` | Max upload file size |
| `FILE_CHUNK_SIZE` | `1000` | Chunk size for file ingestion (chars) |
| `FILE_CHUNK_OVERLAP` | `200` | Overlap between chunks (chars) |

## Alembic Migrations

If you use `STORE_BACKEND=sqlite`, you can manage schema changes with Alembic:

```bash
# apply all migrations
alembic upgrade head

# create a new migration after model changes
alembic revision --autogenerate -m "describe change"

# rollback one migration
alembic downgrade -1
```

## Testing

```bash
pytest -q
```

## MCP Server

The agent exposes an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server via SSE transport at `/mcp`. Any MCP-compatible client can connect and use the agent's tools.

### MCP Tools Exposed

| Tool | Description |
|------|-------------|
| `search_web` | Search the web via DuckDuckGo |
| `save_note` | Save a note to persistent memory |
| `retrieve_notes` | RAG retrieval over saved notes |
| `list_notes` | List all saved notes |
| `clear_notes` | Clear all notes |

### Connecting from Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "notepad-agent": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### Connecting from VS Code (MCP client)

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

### Connecting programmatically

```python
from mcp import ClientSession
from mcp.client.sse import sse_client

async with sse_client("http://localhost:8000/mcp") as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("search_web", {"query": "latest AI news"})
```

## Scaling Considerations

- **Session store**: Currently in-memory. Replace `_sessions` dict in `api.py` with Redis or a database for horizontal scaling.
- **Notepad storage**: File-based. For multi-instance deployments, swap `NotepadRAG` backing store to a shared database or object storage.
- **Rate limiting**: Add middleware (e.g., `slowapi`) for production deployments.
- **Auth**: Add API key or OAuth middleware before exposing publicly.
- **MCP**: The MCP server shares the same FastAPI process. For high-throughput MCP usage, consider running a dedicated instance.
