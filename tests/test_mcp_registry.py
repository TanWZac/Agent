"""Tests for the MCP tool registry and semantic search."""

import pytest

from src.server.registry import ToolRegistry
from src.server.mcp import registry
from mcp.types import TextContent


@pytest.fixture
def sample_registry():
    """Create a registry with test tools."""
    reg = ToolRegistry()

    async def noop(args):
        return [TextContent(type="text", text="ok")]

    reg.register(
        name="web_search",
        description="Search the internet for information using a web browser.",
        input_schema={"type": "object", "properties": {}},
        handler=noop,
        tags=["web", "internet", "search", "lookup", "online", "browse"],
    )
    reg.register(
        name="save_data",
        description="Persist data to long-term storage and memory.",
        input_schema={"type": "object", "properties": {}},
        handler=noop,
        tags=["save", "persist", "storage", "memory", "write", "store"],
    )
    reg.register(
        name="compute_math",
        description="Perform mathematical calculations and arithmetic operations.",
        input_schema={"type": "object", "properties": {}},
        handler=noop,
        tags=["math", "calculate", "arithmetic", "numbers", "computation"],
    )
    return reg


def test_registry_search_finds_relevant_tool(sample_registry):
    results = sample_registry.search("find something on the web", threshold=0.1)
    assert results
    assert results[0][0].name == "web_search"


def test_registry_search_memory_tools(sample_registry):
    results = sample_registry.search("save and persist data to memory", threshold=0.1)
    assert results
    assert results[0][0].name == "save_data"


def test_registry_search_math_tools(sample_registry):
    results = sample_registry.search("do a mathematical calculation", threshold=0.1)
    assert results
    assert results[0][0].name == "compute_math"


def test_registry_search_empty_query(sample_registry):
    results = sample_registry.search("")
    assert results == []


def test_registry_search_no_match(sample_registry):
    # With a very high threshold, nothing should match
    results = sample_registry.search("xyzzy foobar baz", threshold=0.99)
    assert results == []


def test_registry_get_existing_tool(sample_registry):
    tool = sample_registry.get("web_search")
    assert tool is not None
    assert tool.name == "web_search"


def test_registry_get_missing_tool(sample_registry):
    assert sample_registry.get("nonexistent") is None


def test_registry_count(sample_registry):
    assert sample_registry.count == 3


def test_registry_to_mcp_tool(sample_registry):
    tool = sample_registry.get("web_search")
    mcp_tool = tool.to_mcp_tool()
    assert mcp_tool.name == "web_search"
    assert "internet" in mcp_tool.description


def test_global_registry_has_search_tools():
    """The global registry should always include search_tools for discovery."""
    tool = registry.get("search_tools")
    assert tool is not None
    assert "discover" in tool.description.lower() or "search" in tool.description.lower()


def test_global_registry_routes_all_tools():
    """All registered tools should be discoverable."""
    assert registry.count >= 6  # search_tools + 5 domain tools
