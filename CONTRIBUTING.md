# Contributing to LangGraph Notepad RAG Agent

Thanks for considering contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/TanWZac/Agent.git
cd Agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API key
```

## Running Tests

```bash
pytest -q
```

## Code Style

This project uses:
- **ruff** for linting and formatting
- **mypy** (strict mode) for type checking

Run locally before pushing:
```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/ --ignore-missing-imports
```

## Branch Strategy

- `main` — stable releases
- `feature` — active development
- `responsible-ai` — RAI guardrails feature branch
- Feature branches: branch off `main` or the relevant parent

## Pull Requests

1. Fork and create a feature branch
2. Make changes with tests
3. Ensure CI passes (tests + lint + type-check)
4. Open a PR with a clear description

## Project Structure

```
src/
├── agent/           # LangGraph agent (graph, LLM, tools, session)
├── config/          # Settings and config.json
├── core/            # Shared utilities (logging, exceptions, embeddings)
├── responsible_ai/  # Content filter, PII, bias eval, guardrails, audit
├── server/          # FastAPI + MCP server
└── store/           # Note storage backends (file, ChromaDB)
```

## Reporting Issues

Open an issue with:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
