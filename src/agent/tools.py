"""Agent tools — each tool is a standalone function for use in LangGraph."""

from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool

from src.config import Settings
from src.core.exceptions import ToolExecutionError
from src.core.logging import get_logger
from src.store import NoteStore

logger = get_logger("agent.tools")


def create_tools(store: NoteStore, settings: Settings) -> list:
    """Build the list of LangGraph-compatible tools bound to a store instance.

    Args:
        store: The note store for save/retrieve operations.
        settings: Application settings for search config.

    Returns:
        List of @tool-decorated callables.
    """
    web_search = DuckDuckGoSearchResults(max_results=settings.web_search_max_results)

    @tool
    def search_web(query: str) -> str:
        """Search the web for recent information."""
        try:
            return web_search.run(query)
        except Exception as e:
            logger.error("Web search failed: %s", e)
            raise ToolExecutionError(f"Web search failed: {e}") from e

    @tool
    def save_note(note: str) -> str:
        """Save an important fact into persistent notepad memory."""
        store.append(note)
        return "Saved to notepad."

    @tool
    def retrieve_notes(question: str) -> str:
        """Retrieve relevant notes from the persistent notepad memory."""
        hits = store.retrieve(question, k=settings.retrieval_top_k)
        if not hits:
            return "No relevant notes found."
        return "\n".join([f"- {h.text} (score={h.score:.3f})" for h in hits])

    return [search_web, save_note, retrieve_notes]
