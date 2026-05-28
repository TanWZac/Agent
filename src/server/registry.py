"""MCP ToolRegistry with semantic search for tool discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np
from numpy.typing import NDArray

from mcp.types import TextContent, Tool

from src.core.embeddings import cosine_similarity, embed
from src.core.logging import get_logger

logger = get_logger("server.registry")


@dataclass
class RegisteredTool:
    """A tool entry in the registry with metadata for search."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict], Awaitable[list[TextContent]]]
    tags: list[str] = field(default_factory=list)
    _embedding: NDArray[np.float32] | None = field(default=None, repr=False)

    @property
    def searchable_text(self) -> str:
        return f"{self.name}: {self.description}. Tags: {', '.join(self.tags)}"

    def to_mcp_tool(self) -> Tool:
        return Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema,
        )


class ToolRegistry:
    """Registry of MCP tools with semantic search for discovery.

    Uses embeddings and cosine similarity so clients can search for
    relevant tools before calling them.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._embeddings_dirty: bool = True
        self._corpus_embeddings: NDArray[np.float32] | None = None
        self._tool_order: list[str] = []

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[[dict], Awaitable[list[TextContent]]],
        tags: list[str] | None = None,
    ) -> None:
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            tags=tags or [],
        )
        self._embeddings_dirty = True

    def _ensure_embeddings(self) -> None:
        if not self._embeddings_dirty and self._corpus_embeddings is not None:
            return
        if not self._tools:
            self._corpus_embeddings = np.array([], dtype=np.float32).reshape(0, 768)
            self._tool_order = []
            self._embeddings_dirty = False
            return

        self._tool_order = list(self._tools.keys())
        texts = [self._tools[name].searchable_text for name in self._tool_order]
        self._corpus_embeddings = embed(texts)
        self._embeddings_dirty = False
        logger.debug("Tool embeddings computed for %d tools", len(self._tool_order))

    def search(self, query: str, k: int = 5, threshold: float = 0.3) -> list[tuple[RegisteredTool, float]]:
        """Semantic search over registered tools."""
        if not query.strip() or not self._tools:
            return []

        self._ensure_embeddings()

        query_embedding = embed([query])[0]
        scores = cosine_similarity(query_embedding, self._corpus_embeddings)

        scored: list[tuple[RegisteredTool, float]] = []
        for i, score in enumerate(scores):
            if score >= threshold:
                tool_name = self._tool_order[i]
                scored.append((self._tools[tool_name], float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    @property
    def count(self) -> int:
        return len(self._tools)
