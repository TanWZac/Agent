"""ChromaDB-based note store — persistent vector database backend."""

from __future__ import annotations

from typing import List
from uuid import uuid4

from src.core.embeddings import embed
from src.core.exceptions import NotepadError, RetrievalError
from src.core.logging import get_logger
from src.store import NoteStore, RetrievedNote

logger = get_logger("store.chroma")


class ChromaNoteStore(NoteStore):
    """ChromaDB-backed note store with built-in vector indexing.

    Notes are stored with pre-computed embeddings.
    Retrieval is handled by Chroma's native similarity search.
    """

    def __init__(
        self,
        collection_name: str = "notepad",
        persist_directory: str = "data/chroma_db",
    ) -> None:
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "ChromaDB is required for the chroma backend. "
                "Install it with: pip install chromadb"
            ) from e

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaNoteStore initialized: collection=%s, persist_dir=%s, docs=%d",
            collection_name, persist_directory, self._collection.count(),
        )

    def append(self, note: str) -> None:
        normalized = note.strip()
        if not normalized:
            return
        try:
            embedding = embed([normalized])[0].tolist()
            self._collection.add(
                ids=[str(uuid4())],
                documents=[normalized],
                embeddings=[embedding],
            )
            logger.debug("Appended note to ChromaDB (%d chars)", len(normalized))
        except Exception as e:
            raise NotepadError(f"Failed to store note in ChromaDB: {e}") from e

    def load_notes(self) -> List[str]:
        try:
            result = self._collection.get()
            return result["documents"] or []
        except Exception as e:
            raise NotepadError(f"Failed to load notes from ChromaDB: {e}") from e

    def retrieve(self, query: str, k: int = 3, threshold: float = 0.1) -> List[RetrievedNote]:
        if not query.strip():
            return []

        try:
            query_embedding = embed([query])[0].tolist()
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self._collection.count()) or 1,
            )
        except Exception as e:
            logger.error("ChromaDB retrieval failed: %s", e)
            raise RetrievalError(f"ChromaDB retrieval failed: {e}") from e

        if not results["documents"] or not results["documents"][0]:
            return []

        scored: List[RetrievedNote] = []
        documents = results["documents"][0]
        distances = results["distances"][0] if results["distances"] else []

        for i, doc in enumerate(documents):
            score = 1.0 - distances[i] if i < len(distances) else 0.0
            if score >= threshold:
                scored.append(RetrievedNote(text=doc, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored

    def clear(self) -> None:
        try:
            all_ids = self._collection.get()["ids"]
            if all_ids:
                self._collection.delete(ids=all_ids)
            logger.info("ChromaNoteStore cleared")
        except Exception as e:
            raise NotepadError(f"Failed to clear ChromaDB: {e}") from e
