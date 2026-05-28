# LangGraph Notepad RAG Agent

A LangGraph-powered conversational agent with web search and persistent notepad memory. Designed for **interactive CLI** use, **HTTP service integration**, and **MCP (Model Context Protocol) server** exposure.

## Features

- LangGraph agent with tool-calling (GPT-4o-mini default, configurable)
- Web search via DuckDuckGo
- Persistent notepad with lightweight RAG retrieval (Jaccard + stop-word filtering)
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
│   ├── graph.py            # State graph with RAI guardrail nodes
│   ├── llm.py              # LLM factory (OpenAI / HuggingFace)
│   ├── session.py          # AgentSession — orchestrates conversation
│   └── tools.py            # Tool definitions (search, save, retrieve)
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
│   ├── content_filter.py   # Harmful content detection
│   ├── guardrails.py       # Orchestrator (input/output checks)
│   ├── pii_detector.py     # PII detection & redaction
│   └── transparency.py     # Privacy-preserving audit logger
├── server/                 # Service layer
│   ├── api.py              # FastAPI HTTP service
│   ├── mcp.py              # MCP server (SSE transport)
│   └── registry.py         # MCP tool registry
└── store/                  # Storage backends
    ├── chroma_store.py     # ChromaDB vector store
    ├── factory.py          # Store factory
    └── file_store.py       # File-based notepad store
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
| `NOTE_FILE` | `data/notepad.txt` | Notepad file path |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_NOTE_FILE_SIZE_MB` | `10` | Max notepad file size |
| `RETRIEVAL_TOP_K` | `3` | Number of notes to retrieve |
| `WEB_SEARCH_MAX_RESULTS` | `5` | Max web search results |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |

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
