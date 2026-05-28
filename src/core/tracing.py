"""OpenTelemetry tracing — spans for LLM calls, tools, and guardrails.

Initializes a tracer provider with configurable exporters (console, OTLP, Azure Monitor).
Each component creates spans with semantic attributes for latency analysis and debugging.

Configuration (env vars):
    OTEL_ENABLED: "true" to enable tracing (default: "false")
    OTEL_SERVICE_NAME: service name (default: "notepad-rag-agent")
    OTEL_EXPORTER: "console", "otlp", or "azure" (default: "console")
    OTEL_ENDPOINT: OTLP endpoint URL (default: "http://localhost:4317")
    APPLICATIONINSIGHTS_CONNECTION_STRING: Azure Monitor connection string

Usage::

    from src.core.tracing import get_tracer, trace_span

    tracer = get_tracer("agent.graph")
    with tracer.start_as_current_span("llm_call") as span:
        span.set_attribute("llm.model", "gpt-4o-mini")
        result = llm.invoke(messages)
        span.set_attribute("llm.tokens", result.usage_metadata.get("total_tokens", 0))
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Generator

from src.core.logging import get_logger

logger = get_logger("core.tracing")

_OTEL_ENABLED = os.getenv("OTEL_ENABLED", "false").lower() == "true"
_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "notepad-rag-agent")
_EXPORTER = os.getenv("OTEL_EXPORTER", "console")
_OTLP_ENDPOINT = os.getenv("OTEL_ENDPOINT", "http://localhost:4317")


class _NoOpSpan:
    """No-op span when tracing is disabled — zero overhead."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoOpTracer:
    """No-op tracer when OTel is disabled."""

    def start_as_current_span(self, name: str, **kwargs) -> _NoOpSpan:
        return _NoOpSpan()


@lru_cache(maxsize=1)
def _init_tracer_provider():
    """Initialize the OTel TracerProvider once."""
    if not _OTEL_ENABLED:
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        resource = Resource.create({"service.name": _SERVICE_NAME})
        provider = TracerProvider(resource=resource)

        if _EXPORTER == "console":
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        elif _EXPORTER == "otlp":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=_OTLP_ENDPOINT)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        elif _EXPORTER == "azure":
            conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
            if conn_str:
                from azure.monitor.opentelemetry.exporter import (
                    AzureMonitorTraceExporter,
                )

                exporter = AzureMonitorTraceExporter(connection_string=conn_str)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            else:
                logger.warning("OTEL_EXPORTER=azure but APPLICATIONINSIGHTS_CONNECTION_STRING not set")
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        logger.info("OpenTelemetry tracing enabled: exporter=%s, service=%s", _EXPORTER, _SERVICE_NAME)
        return provider

    except ImportError as e:
        logger.warning("OpenTelemetry packages not installed, tracing disabled: %s", e)
        return None


def get_tracer(name: str):
    """Get a tracer instance by name.

    Returns a real OTel tracer if enabled, otherwise a no-op tracer with zero overhead.

    :param name: Tracer name (e.g., "agent.graph", "rai.guardrails").
    :return: Tracer instance.
    """
    provider = _init_tracer_provider()
    if provider is None:
        return _NoOpTracer()

    from opentelemetry import trace

    return trace.get_tracer(name)


@contextmanager
def trace_span(
    tracer_name: str,
    span_name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator:
    """Convenience context manager to create a span with attributes.

    :param tracer_name: Module/component name for the tracer.
    :param span_name: Name of the span.
    :param attributes: Initial span attributes.
    """
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span
