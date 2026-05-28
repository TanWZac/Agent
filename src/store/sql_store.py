"""SQLAlchemy-backed note store with semantic retrieval."""

from __future__ import annotations

from typing import List

from sqlalchemy import delete, select
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.embeddings import cosine_similarity, embed
from src.core.exceptions import NotepadError, RetrievalError
from src.core.logging import get_logger
from src.store import NoteStore, RetrievedNote
from src.store.db import Base, NoteRecord

logger = get_logger("store.sqlite")


class SqlNoteStore(NoteStore):
    """SQLite-backed note store with semantic retrieval.

    Notes are persisted in a SQL table.
    Retrieval uses application-side embedding similarity.
    """

    def __init__(self, db_url: str = "sqlite:///data/notepad.db") -> None:
        self._engine = create_engine(db_url, future=True)
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False, class_=Session)
        logger.info("SqlNoteStore initialized: db_url=%s", db_url)

    def append(self, note: str) -> None:
        normalized = note.strip()
        if not normalized:
            return

        try:
            with self._session_factory() as session:
                session.add(NoteRecord(text=normalized))
                session.commit()
            logger.debug("Appended note to sqlite (%d chars)", len(normalized))
        except Exception as e:
            raise NotepadError(f"Failed to write note to sqlite: {e}") from e

    def load_notes(self) -> List[str]:
        try:
            with self._session_factory() as session:
                rows = session.execute(
                    select(NoteRecord.text).order_by(NoteRecord.id.asc())
                ).all()
            return [row[0] for row in rows]
        except Exception as e:
            raise NotepadError(f"Failed to read notes from sqlite: {e}") from e

    def retrieve(self, query: str, k: int = 3, threshold: float = 0.1) -> List[RetrievedNote]:
        if not query.strip():
            return []

        try:
            notes = self.load_notes()
        except NotepadError:
            logger.exception("Retrieval failed during sqlite note loading")
            raise RetrievalError("Could not load notes for retrieval.")

        if not notes:
            return []

        query_embedding = embed([query])
        note_embeddings = embed(notes)
        scores = cosine_similarity(query_embedding[0], note_embeddings)

        scored: List[RetrievedNote] = []
        for i, score in enumerate(scores):
            if score >= threshold:
                scored.append(RetrievedNote(text=notes[i], score=float(score)))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]

    def clear(self) -> None:
        try:
            with self._session_factory() as session:
                session.execute(delete(NoteRecord))
                session.commit()
            logger.info("SqlNoteStore cleared")
        except Exception as e:
            raise NotepadError(f"Failed to clear sqlite notes: {e}") from e
