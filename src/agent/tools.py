"""
Agent tools — each tool is a standalone function for use in LangGraph.

:mod:`tools` defines callable tools for the agent, including web search, persistent notepad operations,
and file ingestion via MarkItDown.
"""

from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool

from src.agent.file_ingest import convert_file_to_markdown
from src.config import Settings
from src.core.exceptions import ToolExecutionError
from src.core.logging import get_logger
from src.store import NoteStore

logger = get_logger("agent.tools")


def create_tools(store: NoteStore, settings: Settings) -> list:
    """
    Build the list of LangGraph-compatible tools bound to a store instance.

    :param store: The note store for save/retrieve operations.
    :param settings: Application settings for search config.
    :return: List of @tool-decorated callables.
    """
    web_search = DuckDuckGoSearchResults(max_results=settings.web_search_max_results)

    @tool
    def search_web(query: str) -> str:
        """
        Search the web for recent information.

        :param query: Search query string.
        :return: Search results as a string.
        """
        try:
            return web_search.run(query)
        except Exception as e:
            logger.error("Web search failed: %s", e)
            raise ToolExecutionError(f"Web search failed: {e}") from e

    @tool
    def save_note(note: str) -> str:
        """
        Save an important fact into persistent notepad memory.

        :param note: The note to save.
        :return: Confirmation message.
        """
        store.append(note)
        return "Saved to notepad."

    @tool
    def retrieve_notes(question: str) -> str:
        """
        Retrieve relevant notes from the persistent notepad memory.

        :param question: Query for relevant notes.
        :return: Relevant notes as a string.
        """
        hits = store.retrieve(question, k=settings.retrieval_top_k)
        if not hits:
            return "No relevant notes found."
        return "\n".join([f"- {h.text} (score={h.score:.3f})" for h in hits])

    @tool
    def ingest_file(file_path: str) -> str:
        """
        Convert an uploaded file to Markdown for analysis.

        Supports PDF, DOCX, XLSX, PPTX, HTML, images, and many other formats.
        Use this tool when the user uploads a file and wants to discuss its content.

        :param file_path: Path to the uploaded file.
        :return: The file content converted to Markdown.
        """
        try:
            content = convert_file_to_markdown(file_path)
            # Truncate very large documents to avoid token limits
            max_chars = 50000
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n... [truncated — file too large to display in full]"
            return content
        except ToolExecutionError:
            raise
        except Exception as e:
            logger.error("File ingestion failed: %s", e)
            raise ToolExecutionError(f"File ingestion failed: {e}") from e

    return [search_web, save_note, retrieve_notes, ingest_file]
