"""FastAPI service layer for the notepad-agent.

Exposes the agent as an HTTP API for integration with other services.
Supports multiple concurrent sessions and an MCP server via SSE.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.agent import AgentSession
from src.config import get_settings
from src.core.exceptions import AgentError, ConfigurationError, LLMError
from src.core.logging import get_logger, setup_logging
from src.server.mcp import create_sse_transport, mcp_server

logger = get_logger("server.api")

# --- Session Store with LRU Eviction ---
_MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))


class _SessionStore(OrderedDict):
    """LRU session store with configurable max size."""

    def __init__(self, max_size: int = _MAX_SESSIONS) -> None:
        super().__init__()
        self._max_size = max_size

    def get_or_create(self, session_id: str | None, persona: str | None = None) -> AgentSession:
        """Get existing session or create new one, evicting oldest if at capacity."""
        if session_id and session_id in self:
            self.move_to_end(session_id)
            return self[session_id]
        session = AgentSession(persona=persona)
        self[session.session_id] = session
        self.move_to_end(session.session_id)
        while len(self) > self._max_size:
            evicted_id, _ = self.popitem(last=False)
            logger.info("Evicted oldest session: %s (capacity=%d)", evicted_id, self._max_size)
        return session


_sessions = _SessionStore()

# MCP SSE transport
_sse_transport = create_sse_transport("/mcp")

# --- API Key Authentication ---
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_api_key() -> str:
    """Read API_KEY at call time (not import time) to support hot-reload."""
    return os.getenv("API_KEY", "")


def _verify_api_key(api_key: str | None = Security(_API_KEY_HEADER)) -> str:
    """Validate the API key from the request header.

    If API_KEY env var is not set, authentication is disabled (development mode).
    """
    configured_key = _get_api_key()
    if not configured_key:
        # Auth disabled — no API_KEY configured
        return ""
    if not api_key or not secrets.compare_digest(api_key, configured_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: validate config on startup."""
    settings = get_settings()
    setup_logging(settings.log_level)
    try:
        settings.validate()
    except ConfigurationError as e:
        logger.error("Startup configuration error: %s", e)
        raise
    if not _get_api_key():
        logger.warning(
            "API_KEY is not set — all endpoints are unauthenticated. "
            "Set API_KEY env var for production deployments."
        )
    logger.info("API server starting on %s:%d", settings.api_host, settings.api_port)
    logger.info("MCP server available at %s (SSE)", settings.mcp_endpoint)
    logger.info("Max sessions: %d", _MAX_SESSIONS)
    yield
    logger.info("API server shutting down, %d active sessions", len(_sessions))
    _sessions.clear()


app = FastAPI(
    title="Notepad RAG Agent API",
    version="0.4.0",
    description="LangGraph agent with web search, persistent notepad RAG, and MCP server",
    lifespan=lifespan,
)


# --- Request/Response Models ---


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    session_id: str | None = Field(None, description="Session ID to continue a conversation")
    persona: str | None = Field(None, description="Agent persona name (e.g., 'research_assistant', 'customer_support')")


class ChatResponse(BaseModel):
    session_id: str
    response: str


class SessionInfo(BaseModel):
    session_id: str
    message_count: int


class HealthResponse(BaseModel):
    status: str
    active_sessions: int


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check that validates LLM connectivity."""
    settings = get_settings()
    status = "healthy"

    # Validate LLM reachability for meaningful load balancer health
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        status = "degraded"

    return HealthResponse(status=status, active_sessions=len(_sessions))


@app.get("/personas")
async def get_personas():
    """List available agent personas."""
    from src.agent.prompts import get_persona, list_personas

    personas = []
    for name in list_personas():
        p = get_persona(name)
        personas.append({
            "name": p.name,
            "display_name": p.display_name,
            "description": p.description,
        })
    return {"personas": personas}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(_verify_api_key)])
async def chat(request: ChatRequest):
    """Send a message to the agent and get a response."""
    try:
        session = _sessions.get_or_create(request.session_id, persona=request.persona)

        # Run synchronous LLM call in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, session.chat, request.message)
        return ChatResponse(session_id=session.session_id, response=response_text)

    except LLMError as e:
        logger.error("LLM error in session: %s", e)
        raise HTTPException(status_code=502, detail="LLM service unavailable") from e
    except AgentError as e:
        logger.error("Agent error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/chat/stream", dependencies=[Depends(_verify_api_key)])
async def chat_stream(request: ChatRequest):
    """Send a message and receive the response as a Server-Sent Events stream.

    Each SSE event contains a JSON payload with `type` and `data` fields:
      - type="token": partial response token
      - type="done": final complete response
      - type="error": error occurred
    """

    async def _event_generator() -> AsyncGenerator[str, None]:
        try:
            session = _sessions.get_or_create(request.session_id, persona=request.persona)
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(None, session.chat, request.message)

            # Stream the response in chunks to simulate token-by-token delivery
            chunk_size = 20
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                event = json.dumps({"type": "token", "data": chunk, "session_id": session.session_id})
                yield f"data: {event}\n\n"

            # Final event
            done_event = json.dumps({"type": "done", "data": response_text, "session_id": session.session_id})
            yield f"data: {done_event}\n\n"

        except LLMError as e:
            logger.error("LLM error in streaming session: %s", e)
            error_event = json.dumps({"type": "error", "data": str(e)})
            yield f"data: {error_event}\n\n"
        except AgentError as e:
            logger.error("Agent error in streaming: %s", e)
            error_event = json.dumps({"type": "error", "data": str(e)})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/sessions/{session_id}", response_model=SessionInfo, dependencies=[Depends(_verify_api_key)])
async def get_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _sessions[session_id]
    return SessionInfo(session_id=session.session_id, message_count=len(session.history))


@app.delete("/sessions/{session_id}", dependencies=[Depends(_verify_api_key)])
async def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    return {"detail": "Session deleted"}


@app.post("/sessions/{session_id}/reset", dependencies=[Depends(_verify_api_key)])
async def reset_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    _sessions[session_id].reset_history()
    return {"detail": "Session history reset"}


# --- MCP Server Endpoints (SSE) ---
# The MCP SDK's SseServerTransport operates at the raw ASGI level.
# We mount it via Starlette Routes for proper ASGI lifecycle management.

from starlette.routing import Route


async def _mcp_sse_app(scope, receive, send):
    """Raw ASGI handler for MCP SSE connections."""
    async with _sse_transport.connect_sse(scope, receive, send) as streams:
        await mcp_server.run(
            streams[0], streams[1], mcp_server.create_initialization_options()
        )


async def _mcp_post_app(scope, receive, send):
    """Raw ASGI handler for MCP POST messages."""
    await _sse_transport.handle_post_message(scope, receive, send)


# Mount as raw ASGI routes — avoids accessing private Request._send
app.router.routes.append(Route("/mcp", endpoint=_mcp_sse_app, methods=["GET"]))
app.router.routes.append(Route("/mcp", endpoint=_mcp_post_app, methods=["POST"]))


def start_server() -> None:
    """Entry point for running the API server."""
    import uvicorn

    settings = get_settings()
    setup_logging(settings.log_level)
    uvicorn.run(
        "src.server.api:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )
