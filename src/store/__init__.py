"""Abstract storage interface for note persistence and retrieval."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class RetrievedNote:
    """A note returned from retrieval with its relevance score."""

    text: str
    score: float


class NoteStore(ABC):
    """Abstract interface for note storage and retrieval.

    All backends must implement this interface to be usable by the agent.
    """

    @abstractmethod
    def append(self, note: str) -> None:
        """Persist a note."""

    @abstractmethod
    def load_notes(self) -> List[str]:
        """Load all stored notes."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 3, threshold: float = 0.1) -> List[RetrievedNote]:
        """Retrieve top-k relevant notes for a query."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all stored notes."""

    @property
    def count(self) -> int:
        """Return the number of stored notes."""
        return len(self.load_notes())

    async def append_async(self, note: str) -> None:
        """Asynchronously persist a note.

        The default implementation offloads the synchronous backend call to a
        worker thread. Backends can override this with native async I/O.
        """
        await asyncio.to_thread(self.append, note)

    async def load_notes_async(self) -> List[str]:
        """Asynchronously load all stored notes.

        The default implementation offloads the synchronous backend call to a
        worker thread. Backends can override this with native async I/O.
        """
        return await asyncio.to_thread(self.load_notes)

    async def retrieve_async(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.1,
    ) -> List[RetrievedNote]:
        """Asynchronously retrieve top-k relevant notes for a query.

        The default implementation offloads the synchronous backend call to a
        worker thread. Backends can override this with native async I/O.
        """
        return await asyncio.to_thread(self.retrieve, query, k, threshold)

    async def clear_async(self) -> None:
        """Asynchronously delete all stored notes.

        The default implementation offloads the synchronous backend call to a
        worker thread. Backends can override this with native async I/O.
        """
        await asyncio.to_thread(self.clear)

    async def count_async(self) -> int:
        """Return the number of stored notes asynchronously."""
        notes = await self.load_notes_async()
        return len(notes)
