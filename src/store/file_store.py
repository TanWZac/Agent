"""File-based note store — zero-dependency default backend."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from src.core.embeddings import cosine_similarity, embed
from src.core.exceptions import NotepadError, NotepadFullError, RetrievalError
from src.core.logging import get_logger
from src.store import NoteStore, RetrievedNote

logger = get_logger("store.file")


class FileNoteStore(NoteStore):
    """File-based note store with semantic retrieval.

    Notes are stored as lines in a plain text file.
    Retrieval uses cosine similarity over embeddings.
    Embeddings are cached and incrementally updated on append.
    """

    def __init__(self, note_file: str = "data/notepad.txt", max_size_mb: int = 10) -> None:
        self.note_file = Path(note_file)
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self.note_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.note_file.exists():
            self.note_file.write_text("", encoding="utf-8")
        # Embedding cache: keeps pre-computed embeddings in sync with notes
        self._cached_notes: list[str] = []
        self._cached_embeddings: np.ndarray | None = None
        logger.info("FileNoteStore initialized: file=%s, max_size=%dMB", self.note_file, max_size_mb)

    def _check_size(self) -> None:
        try:
            size = self.note_file.stat().st_size
        except OSError as e:
            raise NotepadError(f"Cannot stat note file: {e}") from e
        if size >= self._max_size_bytes:
            raise NotepadFullError(
                f"Notepad file exceeds {self._max_size_bytes // (1024*1024)}MB limit. "
                "Consider archiving old notes."
            )

    def append(self, note: str) -> None:
        normalized = note.strip()
        if not normalized:
            return
        self._check_size()
        try:
            with self.note_file.open("a", encoding="utf-8") as f:
                f.write(normalized + "\n")
            logger.debug("Appended note (%d chars)", len(normalized))
        except OSError as e:
            raise NotepadError(f"Failed to write note: {e}") from e

    def load_notes(self) -> List[str]:
        try:
            content = self.note_file.read_text(encoding="utf-8")
        except OSError as e:
            raise NotepadError(f"Failed to read notes: {e}") from e
        return [line.strip() for line in content.splitlines() if line.strip()]

    def retrieve(self, query: str, k: int = 3, threshold: float = 0.1) -> List[RetrievedNote]:
        try:
            notes = self.load_notes()
        except NotepadError:
            logger.exception("Retrieval failed during note loading")
            raise RetrievalError("Could not load notes for retrieval.")

        if not notes or not query.strip():
            return []

        # Use cached embeddings; only re-embed new notes
        note_embeddings = self._get_or_update_embeddings(notes)

        query_embedding = embed([query])
        scores = cosine_similarity(query_embedding[0], note_embeddings)

        scored: List[RetrievedNote] = []
        for i, score in enumerate(scores):
            if score >= threshold:
                scored.append(RetrievedNote(text=notes[i], score=float(score)))

        scored.sort(key=lambda item: item.score, reverse=True)
        logger.debug("Retrieved %d candidates, returning top %d", len(scored), k)
        return scored[:k]

    def _get_or_update_embeddings(self, notes: list[str]) -> np.ndarray:
        """Return embeddings for all notes, using cache for previously seen notes."""
        if self._cached_embeddings is not None and self._cached_notes == notes[:len(self._cached_notes)]:
            # Notes only grew (append-only file) — embed just the new ones
            new_notes = notes[len(self._cached_notes):]
            if not new_notes:
                return self._cached_embeddings
            new_embeddings = embed(new_notes)
            self._cached_embeddings = np.vstack([self._cached_embeddings, new_embeddings])
            self._cached_notes = list(notes)
            return self._cached_embeddings

        # Full re-embed (file was modified externally or first call)
        self._cached_embeddings = embed(notes)
        self._cached_notes = list(notes)
        return self._cached_embeddings

    def clear(self) -> None:
        self.note_file.write_text("", encoding="utf-8")
        self._cached_notes = []
        self._cached_embeddings = None
        logger.info("FileNoteStore cleared")
