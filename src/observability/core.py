"""Core observability API used by the agent and server.

Provides a simple trace lifecycle API: start_trace, add_event, end_trace,
and helpers to retrieve traces. Integrates with the project's existing
OpenTelemetry helper when available.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from src.observability.trace_store import TraceStore

logger = logging.getLogger("observability.core")

_store = TraceStore()


def start_trace(session_id: Optional[str] = None, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Create a new trace and return the trace id."""
    trace_id = _store.create_trace(session_id=session_id, name=name, metadata=metadata)
    logger.debug("Started trace %s for session %s", trace_id, session_id)
    return trace_id


def add_event(trace_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Append an event to a trace and emit an OTel span (best-effort)."""
    _store.add_event(trace_id, event_type, payload)
    # Best-effort OTel integration
    try:
        from src.core.tracing import get_tracer

        tracer = get_tracer("observability")
        # create a short-lived span for the event
        with tracer.start_as_current_span(event_type) as span:
            if payload:
                for k, v in payload.items():
                    try:
                        span.set_attribute(f"obs.{k}", v)
                    except Exception:
                        # ignore attributes that can't be serialized
                        pass
    except Exception:
        # If tracing package isn't installed or any error occurs, continue silently
        pass


def end_trace(trace_id: str, status: Optional[str] = None) -> None:
    """Mark trace as finished and record an optional status."""
    _store.end_trace(trace_id, status=status)
    logger.debug("Ended trace %s status=%s", trace_id, status)


def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    return _store.get_trace(trace_id)


def list_traces(session_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    return _store.list_traces(session_id=session_id, limit=limit)
