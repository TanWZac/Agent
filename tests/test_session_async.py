"""Tests for AgentSession asynchronous APIs."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from src.agent import AgentSession
from src.config import get_settings


def _build_session(tmp_path):
    settings = get_settings(
        openai_api_key="test-key-not-real",
        note_file=str(tmp_path / "notes.txt"),
    )

    with patch("src.agent.llm.create_llm") as mock_create_llm, \
         patch("src.agent.tools.DuckDuckGoSearchResults"):
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = AIMessage(content="Hello from mock assistant")
        mock_bound.ainvoke = AsyncMock(return_value=AIMessage(content="Hello from mock assistant"))
        mock_llm.bind_tools.return_value = mock_bound
        mock_create_llm.return_value = mock_llm
        return AgentSession(settings=settings)


def test_chat_async_returns_response(tmp_path):
    session = _build_session(tmp_path)

    response = asyncio.run(session.chat_async("Hello"))

    assert isinstance(response, str)
    assert response.strip()
    assert len(session.history) == 2


def test_ingest_file_bytes_async_calls_ingestor(tmp_path):
    session = _build_session(tmp_path)

    with patch("src.agent.file_ingest.ingest_bytes_to_store", return_value=3) as mock_ingest:
        chunks = asyncio.run(session.ingest_file_bytes_async(b"abc", "sample.txt"))

    assert chunks == 3
    mock_ingest.assert_called_once_with(b"abc", "sample.txt", session.notepad)


def test_ingest_file_async_calls_ingestor(tmp_path):
    session = _build_session(tmp_path)

    with patch("src.agent.file_ingest.ingest_file_to_store", return_value=2) as mock_ingest:
        chunks = asyncio.run(session.ingest_file_async("/tmp/sample.txt"))

    assert chunks == 2
    mock_ingest.assert_called_once_with("/tmp/sample.txt", session.notepad)
