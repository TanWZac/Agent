"""FastAPI service layer for the notepad-agent.

Exposes the agent as an HTTP API for integration with other services.
Supports multiple concurrent sessions and an MCP server via SSE.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.agent import AgentSession
from src.config import get_settings
from src.core.exceptions import AgentError, ConfigurationError, LLMError
from src.core.logging import get_logger, setup_logging
from src.server.mcp import create_sse_transport, mcp_server

logger = get_logger("server.api")

# In-memory session store. Replace with Redis/DB for horizontal scaling.
_sessions: Dict[str, AgentSession] = {}

# MCP SSE transport
_sse_transport = create_sse_transport("/mcp")


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
    logger.info("API server starting on %s:%d", settings.api_host, settings.api_port)
    logger.info("MCP server available at %s (SSE)", settings.mcp_endpoint)
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
    return HealthResponse(status="healthy", active_sessions=len(_sessions))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the agent and get a response."""
    try:
        if request.session_id and request.session_id in _sessions:
            session = _sessions[request.session_id]
        else:
            session = AgentSession()
            _sessions[session.session_id] = session

        response_text = session.chat(request.message)
        return ChatResponse(session_id=session.session_id, response=response_text)

    except LLMError as e:
        logger.error("LLM error in session: %s", e)
        raise HTTPException(status_code=502, detail="LLM service unavailable") from e
    except AgentError as e:
        logger.error("Agent error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _sessions[session_id]
    return SessionInfo(session_id=session.session_id, message_count=len(session.history))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    return {"detail": "Session deleted"}


@app.post("/sessions/{session_id}/reset")
async def reset_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    _sessions[session_id].reset_history()
    return {"detail": "Session history reset"}


# --- MCP Server Endpoints (SSE) ---


@app.get("/mcp")
async def mcp_sse_endpoint(request: Request):
    """SSE endpoint for MCP clients to connect to."""
    async with _sse_transport.connect_sse(
        request.scope, request.receive, request._send  # type: ignore[attr-defined]
    ) as streams:
        await mcp_server.run(
            streams[0], streams[1], mcp_server.create_initialization_options()
        )


@app.post("/mcp")
async def mcp_post_endpoint(request: Request):
    """POST endpoint for MCP client messages."""
    await _sse_transport.handle_post_message(
        request.scope, request.receive, request._send  # type: ignore[attr-defined]
    )


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
