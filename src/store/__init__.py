"""Abstract storage interface for note persistence and retrieval."""

from __future__ import annotations

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
