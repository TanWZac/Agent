"""SQLAlchemy-backed note store with semantic retrieval."""

from __future__ import annotations

import asyncio
from typing import List

from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
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

    @staticmethod
    def _to_sync_sqlite_url(db_url: str) -> str:
        """Normalize sqlite URL for synchronous SQLAlchemy engine creation."""
        if db_url.startswith("sqlite+aiosqlite://"):
            return db_url.replace("sqlite+aiosqlite://", "sqlite://", 1)
        return db_url

    @staticmethod
    def _to_async_sqlite_url(db_url: str) -> str:
        """Normalize sqlite URL for asynchronous SQLAlchemy engine creation."""
        if db_url.startswith("sqlite://") and not db_url.startswith("sqlite+aiosqlite://"):
            return db_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
        return db_url

    def __init__(self, db_url: str = "sqlite:///data/notepad.db") -> None:
        sync_db_url = self._to_sync_sqlite_url(db_url)
        async_db_url = self._to_async_sqlite_url(db_url)

        self._engine = create_engine(sync_db_url, future=True)
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False, class_=Session)

        self._async_engine = create_async_engine(async_db_url, future=True)
        self._async_session_factory = async_sessionmaker(
            self._async_engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
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

    async def append_async(self, note: str) -> None:
        normalized = note.strip()
        if not normalized:
            return

        try:
            async with self._async_session_factory() as session:
                session.add(NoteRecord(text=normalized))
                await session.commit()
            logger.debug("Appended note to sqlite async (%d chars)", len(normalized))
        except Exception as e:
            raise NotepadError(f"Failed to write note to sqlite async: {e}") from e

    async def load_notes_async(self) -> List[str]:
        try:
            async with self._async_session_factory() as session:
                rows = await session.execute(
                    select(NoteRecord.text).order_by(NoteRecord.id.asc())
                )
            return [row[0] for row in rows.all()]
        except Exception as e:
            raise NotepadError(f"Failed to read notes from sqlite async: {e}") from e

    async def retrieve_async(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.1,
    ) -> List[RetrievedNote]:
        if not query.strip():
            return []

        try:
            notes = await self.load_notes_async()
        except NotepadError:
            logger.exception("Async retrieval failed during sqlite note loading")
            raise RetrievalError("Could not load notes for retrieval.")

        if not notes:
            return []

        # Embedding generation is CPU-bound; run it off the event loop.
        query_embedding = await asyncio.to_thread(embed, [query])
        note_embeddings = await asyncio.to_thread(embed, notes)
        scores = await asyncio.to_thread(cosine_similarity, query_embedding[0], note_embeddings)

        scored: List[RetrievedNote] = []
        for i, score in enumerate(scores):
            if score >= threshold:
                scored.append(RetrievedNote(text=notes[i], score=float(score)))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]

    async def clear_async(self) -> None:
        try:
            async with self._async_session_factory() as session:
                await session.execute(delete(NoteRecord))
                await session.commit()
            logger.info("SqlNoteStore cleared async")
        except Exception as e:
            raise NotepadError(f"Failed to clear sqlite notes async: {e}") from e

    async def count_async(self) -> int:
        try:
            async with self._async_session_factory() as session:
                result = await session.execute(select(func.count()).select_from(NoteRecord))
            return int(result.scalar_one())
        except Exception as e:
            raise NotepadError(f"Failed to count sqlite notes async: {e}") from e
