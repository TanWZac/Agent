#!/usr/bin/env python3
"""Run a small agent regression suite against the HTTP API or locally.

The runner supports two modes:
- HTTP mode: provide `--api-url` or set `AGENT_API_URL` to target a running API server.
- Local mode: no API URL provided; the runner attempts to import `AgentSession`.

Outputs a JSON report with per-case results and optional traces (if available).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_threshold(case: dict[str, Any], config: dict[str, Any]) -> float:
    """Resolve the semantic threshold for a case using case override -> category -> default."""
    if case.get("semantic_threshold") is not None:
        try:
            return float(case.get("semantic_threshold"))
        except Exception:
            pass
    cat = case.get("category")
    if cat and isinstance(config.get("category_thresholds"), dict) and cat in config.get("category_thresholds", {}):
        try:
            return float(config["category_thresholds"][cat])
        except Exception:
            pass
    return float(config.get("default_threshold", 0.5))


def run_case_http(case: dict[str, Any], api_url: str, api_key: str | None, config: dict[str, Any]) -> dict:
    """Run a single case against the HTTP API and return result dict."""
    import requests

    session_id = case.get("session_id") or str(uuid.uuid4())
    payload = {"message": case["prompt"], "session_id": session_id}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    result: dict = {"id": case.get("id"), "session_id": session_id}
    try:
        r = requests.post(f"{api_url.rstrip('/')}/chat", json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        resp = r.json()
        result["response"] = resp.get("response", "")
    except Exception as e:
        result["error"] = f"chat_error: {e}"
        result["response"] = ""

    # Try to fetch traces for this session
    try:
        r2 = requests.get(f"{api_url.rstrip('/')}/traces", params={"session_id": session_id}, headers=headers, timeout=20)
        if r2.ok:
            traces = r2.json().get("traces", [])
            if traces:
                trace_id = traces[0].get("trace_id")
                r3 = requests.get(f"{api_url.rstrip('/')}/traces/{trace_id}", headers=headers, timeout=20)
                if r3.ok:
                    result["trace"] = r3.json()
                else:
                    result["trace"] = None
            else:
                result["trace"] = None
        else:
            result["trace"] = None
    except Exception:
        result["trace"] = None

    # Semantic/expected matching
    expected = case.get("expected")
    threshold = _resolve_threshold(case, config)
    result["semantic_similarity"] = None
    result["semantic_passed"] = None
    result["semantic_threshold"] = threshold
    if expected:
        try:
            from src.core.semantic import semantic_score

            best_score, best = semantic_score(result.get("response", ""), expected)
            result["semantic_similarity"] = best_score
            result["semantic_passed"] = best_score >= threshold
            result["semantic_best_match"] = best
        except Exception:
            # If local semantic module isn't available, fallback to substring
            should = expected[0]
            passed = should.lower() in (result.get("response") or "").lower()
            result["semantic_similarity"] = 1.0 if passed else 0.0
            result["semantic_passed"] = passed
            result["semantic_best_match"] = should
        result["passed"] = result["semantic_passed"]
        result["details"] = f"semantic match (threshold={threshold})"
    else:
        result["passed"] = True
        result["details"] = "no assertion"

    return result


def run_case_local(case: dict[str, Any], config: dict[str, Any]) -> dict:
    try:
        from src.agent import AgentSession
        from src.observability import core as observ
    except Exception as e:
        # Try adding repository root to sys.path and retry (useful when running from repo)
        try:
            repo_root = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(repo_root))
            from src.agent import AgentSession  # type: ignore
            from src.observability import core as observ  # type: ignore
        except Exception as e2:
            return {"id": case.get("id"), "error": f"local_import_failed: {e2}", "passed": False}

    session_id = case.get("session_id") or str(uuid.uuid4())
    sess = AgentSession(session_id=session_id)
    resp = sess.chat(case["prompt"])

    # Fetch traces from observability store
    traces = observ.list_traces(session_id=session_id)
    trace_detail = None
    if traces:
        trace_id = traces[0]["trace_id"]
        trace_detail = observ.get_trace(trace_id)

    result = {"id": case.get("id"), "session_id": session_id, "response": resp, "trace": trace_detail}
    expected = case.get("expected")
    threshold = _resolve_threshold(case, config)
    result["semantic_similarity"] = None
    result["semantic_passed"] = None
    result["semantic_threshold"] = threshold
    if expected:
        try:
            from src.core.semantic import semantic_score

            best_score, best = semantic_score(resp or "", expected)
            result["semantic_similarity"] = best_score
            result["semantic_passed"] = best_score >= threshold
            result["semantic_best_match"] = best
        except Exception as e:
            # Fallback to token jaccard
            def tok_j(a, b):
                return len(set((a or "").lower().split()) & set((b or "").lower().split())) / max(1, len(set((a or "").lower().split()) | set((b or "").lower().split())))

            best_score = 0.0
            best = ""
            for c in expected:
                s = tok_j(resp, c)
                if s > best_score:
                    best_score = s
                    best = c
            result["semantic_similarity"] = best_score
            result["semantic_passed"] = best_score >= threshold
            result["semantic_best_match"] = best
        result["passed"] = result["semantic_passed"]
        result["details"] = f"semantic match (threshold={threshold})"
    else:
        result["passed"] = True
        result["details"] = "no assertion"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent regression suite")
    parser.add_argument("--api-url", type=str, default=os.getenv("AGENT_API_URL"), help="Agent API base URL (HTTP mode)")
    parser.add_argument("--api-key", type=str, default=os.getenv("API_KEY"), help="X-API-Key for API mode")
    parser.add_argument("--dataset", type=str, default="regression/cases.json", help="Path to cases JSON")
    parser.add_argument("--output", type=str, default="regression/report.json", help="Output report JSON")
    parser.add_argument("--config", type=str, default="regression/config.json", help="Path to regression config JSON")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print("Dataset not found:", dataset_path)
        sys.exit(2)

    cases = load_cases(dataset_path)
    results = []

    # Load optional config for per-category thresholds
    cfg_path = Path(args.config)
    config: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as fh:
                config = json.load(fh)
        except Exception:
            config = {}

    mode = "http" if args.api_url else "local"
    print(f"Running regression suite ({mode} mode) - {len(cases)} cases")

    for case in cases:
        print(f"- Case: {case.get('id')} ...", end=" ")
        start = time.time()
        try:
            if args.api_url:
                res = run_case_http(case, args.api_url, args.api_key, config)
            else:
                res = run_case_local(case, config)
        except Exception as e:
            res = {"id": case.get("id"), "error": str(e), "passed": False}
        duration = time.time() - start
        res["duration_seconds"] = round(duration, 2)
        results.append(res)
        print("OK" if res.get("passed") else "FAIL")

    passed_count = sum(1 for r in results if r.get("passed"))
    total = len(results)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_cases": total,
        "passed_cases": passed_count,
        "pass_rate": passed_count / total if total else 0.0,
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # Persist run artifact for the regression dashboard (runs directory)
    try:
        runs_dir = Path(__file__).resolve().parent / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        run_file = runs_dir / f"{run_id}.json"
        # Save full report under a run id
        with run_file.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

        # Update index for quick listing
        index_file = runs_dir / "index.json"
        try:
            if index_file.exists():
                with index_file.open("r", encoding="utf-8") as fh:
                    index = json.load(fh)
            else:
                index = []
        except Exception:
            index = []

        index_item = {
            "run_id": run_id,
            "timestamp": report.get("timestamp"),
            "pass_rate": report.get("pass_rate"),
            "total_cases": report.get("total_cases"),
            "passed_cases": report.get("passed_cases"),
            "file": str(run_file.resolve()),
        }
        # prepend and keep most recent 200 runs
        index.insert(0, index_item)
        index = index[:200]
        with index_file.open("w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2)

        print(f"Saved run artifact: {run_file}")
    except Exception as e:
        print("Warning: failed to persist run artifact:", e)

    print(f"\nSummary: {passed_count}/{total} passed ({report['pass_rate']:.0%})")
    if report["pass_rate"] < 1.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
