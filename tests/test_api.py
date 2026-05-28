"""Tests for the API service layer."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from langchain_core.messages import AIMessage
from src.config import get_settings


@pytest.fixture
def mock_settings():
    """Settings with a fake API key for testing."""
    return get_settings(openai_api_key="test-key-not-real")


@pytest.fixture
def client(mock_settings, tmp_path):
    """Create a test client with mocked LLM."""
    test_settings = get_settings(
        openai_api_key="test-key-not-real",
        note_file=str(tmp_path / "test_notes.txt"),
    )

    with patch("src.agent.llm.create_llm") as mock_create_llm, \
         patch("src.agent.tools.DuckDuckGoSearchResults") as mock_ddg_cls, \
         patch("src.agent.session.get_settings", return_value=test_settings), \
         patch("src.config.get_settings", return_value=test_settings):
        # Mock the LLM to return a proper AIMessage
        mock_llm = MagicMock()
        mock_bound = MagicMock()

        mock_response = AIMessage(content="Hello from mock assistant")
        mock_bound.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_bound
        mock_create_llm.return_value = mock_llm

        from src.server.api import app
        yield TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_sessions" in data


def test_chat_creates_session(client):
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "response" in data


def test_chat_missing_message(client):
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422  # Validation error


def test_session_not_found(client):
    response = client.get("/sessions/nonexistent-id")
    assert response.status_code == 404


def test_delete_session_not_found(client):
    response = client.delete("/sessions/nonexistent-id")
    assert response.status_code == 404


def test_chat_stream_returns_sse(client):
    """Test that /chat/stream returns a valid SSE response."""
    import json

    response = client.post("/chat/stream", json={"message": "Hello"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Parse SSE events
    events = []
    for line in response.text.strip().split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # Should have at least one token event and one done event
    assert len(events) >= 2
    assert events[-1]["type"] == "done"
    assert "session_id" in events[-1]
    # The done event contains the full response text
    assert len(events[-1]["data"]) > 0
