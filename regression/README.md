# Agent Regression Suite

This directory contains a lightweight regression test harness for running
end-to-end agent interactions and collecting traces for debugging and
regression verification.

Usage (API mode):

1. Start the API server (see project README). Export `API_KEY` if required.

```bash
export API_KEY="yourkey"
python -m uvicorn src.server.api:app --reload --host 127.0.0.1 --port 8000
```

2. Run the regression runner against the API:

```bash
python regression/run_regression.py --api-url http://127.0.0.1:8000 --dataset regression/cases.json --output regression/report.json --api-key "$API_KEY"
```

Usage (local mode):

- If you have the repo dependencies installed (LLM/chain libraries), run the harness without `--api-url` to instantiate `AgentSession` locally. The harness will attempt to import `src.agent.AgentSession` and `src.observability` to capture traces.

```bash
python regression/run_regression.py --dataset regression/cases.json --output regression/report.json
```

Files:
- `cases.json`: example regression cases
- `run_regression.py`: CLI runner (HTTP or local)

Configuration:
- `regression/config.json`: optional per-category `semantic_threshold` values and a `default_threshold` used when a case does not define `semantic_threshold` explicitly.

CLI options:
- `--config`: path to a JSON config file (default: `regression/config.json`).

Example `regression/config.json`:
```json
{
	"default_threshold": 0.5,
	"category_thresholds": {"factual": 0.9, "summarization": 0.6}
}
```
