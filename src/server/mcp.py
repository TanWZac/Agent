"""MCP (Model Context Protocol) server — tool handlers and protocol wiring."""

from __future__ import annotations

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool

from src.config import get_settings
from src.core.logging import get_logger
from src.server.registry import ToolRegistry
from src.store import NoteStore
from src.store.factory import create_note_store

logger = get_logger("server.mcp")


# ---------------------------------------------------------------------------
# Global instances
# ---------------------------------------------------------------------------

mcp_server = Server("notepad-rag-agent")
registry = ToolRegistry()

_store: NoteStore | None = None


def _get_store() -> NoteStore:
    """Lazy-initialize the note store for MCP tools."""
    global _store
    if _store is None:
        settings = get_settings()
        _store = create_note_store(
            backend=settings.store_backend,
            note_file=settings.note_file,
            max_size_mb=settings.max_note_file_size_mb,
            collection_name=settings.chroma_collection,
            persist_directory=settings.chroma_persist_dir,
        )
    return _store


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def _handle_search_tools(arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    if not query:
        return [TextContent(type="text", text="Error: query is required.")]

    k = arguments.get("k", 5)
    results = registry.search(query, k=k)

    if not results:
        return [TextContent(type="text", text="No matching tools found.")]

    lines = []
    for tool, score in results:
        lines.append(f"- **{tool.name}** (relevance: {score:.3f})\n  {tool.description}")
    return [TextContent(type="text", text=f"Found {len(results)} relevant tools:\n\n" + "\n".join(lines))]


async def _handle_search_web(arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    if not query:
        return [TextContent(type="text", text="Error: query is required.")]

    try:
        from langchain_community.tools import DuckDuckGoSearchResults

        settings = get_settings()
        search = DuckDuckGoSearchResults(max_results=settings.web_search_max_results)
        results = search.run(query)
        logger.info("Web search completed for: %s", query)
        return [TextContent(type="text", text=results)]
    except Exception as e:
        logger.error("Web search failed: %s", e)
        return [TextContent(type="text", text=f"Search failed: {e}")]


async def _handle_save_note(arguments: dict) -> list[TextContent]:
    note = arguments.get("note", "")
    if not note:
        return [TextContent(type="text", text="Error: note is required.")]

    try:
        store = _get_store()
        store.append(note)
        return [TextContent(type="text", text="Note saved successfully.")]
    except Exception as e:
        logger.error("Save note failed: %s", e)
        return [TextContent(type="text", text=f"Failed to save note: {e}")]


async def _handle_retrieve_notes(arguments: dict) -> list[TextContent]:
    question = arguments.get("question", "")
    if not question:
        return [TextContent(type="text", text="Error: question is required.")]

    k = arguments.get("k", 3)
    try:
        store = _get_store()
        hits = store.retrieve(question, k=k)
        if not hits:
            return [TextContent(type="text", text="No relevant notes found.")]
        result = "\n".join([f"- {h.text} (relevance: {h.score:.3f})" for h in hits])
        return [TextContent(type="text", text=result)]
    except Exception as e:
        logger.error("Retrieve notes failed: %s", e)
        return [TextContent(type="text", text=f"Retrieval failed: {e}")]


async def _handle_list_notes(arguments: dict) -> list[TextContent]:
    try:
        store = _get_store()
        notes = store.load_notes()
        if not notes:
            return [TextContent(type="text", text="Notepad is empty.")]
        result = "\n".join([f"{i+1}. {note}" for i, note in enumerate(notes)])
        return [TextContent(type="text", text=f"Notes ({len(notes)} total):\n{result}")]
    except Exception as e:
        logger.error("List notes failed: %s", e)
        return [TextContent(type="text", text=f"Failed to list notes: {e}")]


async def _handle_clear_notes(arguments: dict) -> list[TextContent]:
    try:
        store = _get_store()
        store.clear()
        return [TextContent(type="text", text="All notes cleared.")]
    except Exception as e:
        logger.error("Clear notes failed: %s", e)
        return [TextContent(type="text", text=f"Failed to clear notes: {e}")]


# ---------------------------------------------------------------------------
# Register all tools
# ---------------------------------------------------------------------------

registry.register(
    name="search_tools",
    description="Search available tools by describing what you need. Call this first to discover "
    "which tools are relevant for your task before using them.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language description of what you need."},
            "k": {"type": "integer", "description": "Maximum number of tools to return.", "default": 5},
        },
        "required": ["query"],
    },
    handler=_handle_search_tools,
    tags=["meta", "discovery", "search", "find", "tools", "help"],
)

registry.register(
    name="search_web",
    description="Search the web for recent information using DuckDuckGo.",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query."}},
        "required": ["query"],
    },
    handler=_handle_search_web,
    tags=["web", "internet", "search", "lookup", "news", "information", "duckduckgo"],
)

registry.register(
    name="save_note",
    description="Save an important fact or note into persistent notepad memory.",
    input_schema={
        "type": "object",
        "properties": {"note": {"type": "string", "description": "The note text to save."}},
        "required": ["note"],
    },
    handler=_handle_save_note,
    tags=["memory", "save", "store", "persist", "write", "notepad", "remember"],
)

registry.register(
    name="retrieve_notes",
    description="Retrieve relevant notes from the persistent notepad memory using semantic search.",
    input_schema={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question or topic to search notes for."},
            "k": {"type": "integer", "description": "Maximum number of notes to return.", "default": 3},
        },
        "required": ["question"],
    },
    handler=_handle_retrieve_notes,
    tags=["memory", "retrieve", "recall", "search", "notes", "context", "rag"],
)

registry.register(
    name="list_notes",
    description="List all saved notes from the persistent notepad.",
    input_schema={"type": "object", "properties": {}},
    handler=_handle_list_notes,
    tags=["memory", "list", "notes", "view", "all", "notepad"],
)

registry.register(
    name="clear_notes",
    description="Clear all notes from the persistent notepad. Use with caution.",
    input_schema={"type": "object", "properties": {}},
    handler=_handle_clear_notes,
    tags=["memory", "clear", "delete", "reset", "notepad"],
)


# ---------------------------------------------------------------------------
# MCP protocol handlers
# ---------------------------------------------------------------------------


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [t.to_mcp_tool() for t in registry.all_tools()]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info("MCP tool called: %s", name)
    tool = registry.get(name)
    if tool is None:
        return [TextContent(type="text", text=f"Unknown tool: {name}. Use search_tools to discover available tools.")]
    return await tool.handler(arguments)


def create_sse_transport(endpoint: str = "/mcp") -> SseServerTransport:
    """Create the SSE transport for the MCP server."""
    return SseServerTransport(endpoint)
