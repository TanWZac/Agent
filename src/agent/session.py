"""
Agent session — orchestrates tools, graph, and conversation history.

:mod:`session` manages a single conversational session, including message history and tool orchestration.
"""

from __future__ import annotations

import os
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from src.agent.graph import build_graph
from src.agent.tools import create_tools
from src.config import Settings, get_rai_config, get_settings
from src.core.logging import get_logger
from src.responsible_ai.config import RAIConfig
from src.store import NoteStore
from src.store.factory import create_note_store

logger = get_logger("agent.session")

# Conversation summarization threshold (configurable)
_MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
_SUMMARIZE_KEEP_RECENT = int(os.getenv("SUMMARIZE_KEEP_RECENT", "6"))


class AgentSession:
    """
    Encapsulates a single conversational session with history.

    Designed for both CLI usage and integration into web services.
    Each session maintains its own message history and can be serialized.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        session_id: str | None = None,
        store: NoteStore | None = None,
        rai_config: RAIConfig | None = None,
        persona: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._settings.validate()
        self.session_id = session_id or str(uuid4())
        self._history: list[AnyMessage] = []
        self._rai_config = rai_config or get_rai_config()
        self._persona = persona
        self._store = store or create_note_store(
            backend=self._settings.store_backend,
            note_file=self._settings.note_file,
            max_size_mb=self._settings.max_note_file_size_mb,
            collection_name=self._settings.chroma_collection,
            persist_directory=self._settings.chroma_persist_dir,
            db_url=self._settings.sqlite_db_url,
        )
        tools = create_tools(self._store, self._settings)
        self._graph = build_graph(self._settings, tools, rai_config=self._rai_config, persona=self._persona)
        # Prevent direct graph access — guardrails can only be bypassed by subclassing
        self.__graph_private = self._graph
        logger.info(
            "Session created: id=%s, model=%s, store=%s, persona=%s",
            self.session_id, self._settings.openai_model,
            self._settings.store_backend, self._persona or "default",
        )

    @property
    def graph(self):
        """Read-only access to the compiled graph. Use .chat() for guarded invocation."""
        raise AttributeError(
            "Direct graph access is not permitted. Use session.chat() which "
            "applies Responsible AI guardrails on all input/output."
        )

    @property
    def notepad(self) -> NoteStore:
        return self._store

    @property
    def history(self) -> list[AnyMessage]:
        """
        Read-only access to the conversation history.

        :return: List of conversation messages.
        """
        return list(self._history)

    def chat(self, user_message: str) -> str:
        """
        Send a message and get the assistant's response.

        Maintains conversation history across calls within this session.

        :param user_message: The user's input text.
        :return: The assistant's response text.
        """
        human_msg = HumanMessage(content=user_message)
        self._history.append(human_msg)

        logger.info("Session %s: user message (%d chars)", self.session_id, len(user_message))

        result = self._graph.invoke({
            "messages": list(self._history),
            "session_id": self.session_id,
        })

        messages = result.get("messages", [])
        if not messages:
            logger.error("Session %s: graph returned empty messages", self.session_id)
            return "I'm sorry, I couldn't generate a response. Please try again."

        # Walk backwards to find the last AI message (skip tool messages)
        assistant_msg = None
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai" and not getattr(msg, "tool_calls", None):
                assistant_msg = msg
                break
        if assistant_msg is None:
            assistant_msg = messages[-1]

        assistant_text = getattr(assistant_msg, "content", str(assistant_msg))

        self._history.append(assistant_msg)

        # Auto-summarize if history exceeds threshold
        if len(self._history) > _MAX_HISTORY_MESSAGES:
            self._summarize_history()

        logger.info("Session %s: assistant response (%d chars)", self.session_id, len(assistant_text))
        return assistant_text

    def _summarize_history(self) -> None:
        """Compress older messages into a summary to stay within token limits.

        Keeps the most recent messages intact and replaces older ones with a
        summary SystemMessage.
        """
        if len(self._history) <= _SUMMARIZE_KEEP_RECENT:
            return

        # Split: old messages to summarize, recent messages to keep
        to_summarize = self._history[:-_SUMMARIZE_KEEP_RECENT]
        recent = self._history[-_SUMMARIZE_KEEP_RECENT:]

        # Build summary from old messages
        summary_parts = []
        for msg in to_summarize:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            if content and role in ("human", "ai"):
                # Truncate long messages in summary
                truncated = content[:200] + "..." if len(content) > 200 else content
                prefix = "User" if role == "human" else "Assistant"
                summary_parts.append(f"{prefix}: {truncated}")

        if summary_parts:
            summary_text = (
                "[Conversation summary of earlier messages]\n"
                + "\n".join(summary_parts[-10:])  # Keep last 10 summarized exchanges
            )
            summary_msg = SystemMessage(content=summary_text)
            self._history = [summary_msg] + recent
            logger.info(
                "Session %s: summarized %d messages into 1 summary + %d recent",
                self.session_id, len(to_summarize), len(recent),
            )

    def reset_history(self) -> None:
        """
        Clear session history (notepad persists).
        """
        self._history.clear()
        logger.info("Session %s: history reset", self.session_id)
