"""Lightweight trace store with optional file persistence.

Designed for development/PoC usage: stores traces in-memory and
persists each trace as a JSON file under a configurable directory.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional


class TraceStore:
    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self._persist_dir = persist_dir or os.getenv("TRACE_PERSIST_DIR", "data/traces")
        os.makedirs(self._persist_dir, exist_ok=True)
        self._traces: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_trace(self, session_id: Optional[str] = None, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        trace_id = str(uuid.uuid4())
        now = time.time()
        trace = {
            "trace_id": trace_id,
            "session_id": session_id,
            "name": name,
            "start_time": now,
            "end_time": None,
            "events": [],
            "metadata": metadata or {},
        }
        with self._lock:
            self._traces[trace_id] = trace
            self._persist(trace_id)
        return trace_id

    def add_event(self, trace_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        now = time.time()
        event = {"ts": now, "type": event_type, "payload": payload or {}}
        with self._lock:
            trace = self._traces.get(trace_id)
            if not trace:
                return
            trace["events"].append(event)
            # update last-seen time
            trace["end_time"] = now
            self._persist(trace_id)

    def end_trace(self, trace_id: str, status: Optional[str] = None) -> None:
        now = time.time()
        with self._lock:
            trace = self._traces.get(trace_id)
            if not trace:
                return
            trace["end_time"] = now
            if status is not None:
                trace.setdefault("metadata", {})["status"] = status
            self._persist(trace_id)

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._traces.get(trace_id)

    def list_traces(self, session_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            traces = list(self._traces.values())
        if session_id:
            traces = [t for t in traces if t.get("session_id") == session_id]
        # sort by start_time desc
        traces.sort(key=lambda t: t.get("start_time", 0), reverse=True)
        return traces[:limit]

    def _persist(self, trace_id: str) -> None:
        trace = self._traces.get(trace_id)
        if not trace:
            return
        path = os.path.join(self._persist_dir, f"{trace_id}.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(trace, fh, indent=2, default=str)
        except Exception:
            # best-effort persistence — do not fail on write errors
            pass
