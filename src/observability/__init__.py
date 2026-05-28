"""Simple observability facade for traces and events.

This module exposes a lightweight trace API used by the agent and server
to record step-by-step execution events for trace-replay and basic
observability features.
"""

from .core import (
    start_trace,
    end_trace,
    add_event,
    get_trace,
    list_traces,
)

__all__ = ["start_trace", "end_trace", "add_event", "get_trace", "list_traces"]
