"""Optional ChromaDB helper (lightweight placeholder).

This module attempts to import `chromadb`. If available, a concrete
integration can be implemented here. For now it provides a helpful
exception when used on systems without ChromaDB installed.
"""

from __future__ import annotations

try:
    import chromadb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    chromadb = None  # type: ignore


class ChromaNotAvailable(Exception):
    pass


def ensure_chromadb() -> None:
    if chromadb is None:
        raise ChromaNotAvailable("chromadb is not installed in this environment")


__all__ = ["ensure_chromadb", "ChromaNotAvailable"]
