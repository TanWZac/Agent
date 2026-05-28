"""Factory for creating NoteStore instances based on configuration."""

from __future__ import annotations

from src.core.logging import get_logger
from src.store import NoteStore

logger = get_logger("store.factory")


def create_note_store(backend: str = "file", **kwargs) -> NoteStore:
    """Create a NoteStore instance for the specified backend.

    Args:
        backend: One of "file" or "chroma".
        **kwargs: Backend-specific configuration passed to the constructor.

    Raises:
        ValueError: If the backend is not recognized.
    """
    if backend == "file":
        from src.store.file_store import FileNoteStore

        return FileNoteStore(
            note_file=kwargs.get("note_file", "data/notepad.txt"),
            max_size_mb=kwargs.get("max_size_mb", 10),
        )
    elif backend == "chroma":
        from src.store.chroma_store import ChromaNoteStore

        return ChromaNoteStore(
            collection_name=kwargs.get("collection_name", "notepad"),
            persist_directory=kwargs.get("persist_directory", "data/chroma_db"),
        )
    else:
        raise ValueError(
            f"Unknown store backend: '{backend}'. Supported: 'file', 'chroma'."
        )
