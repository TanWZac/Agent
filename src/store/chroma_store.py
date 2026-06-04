"""ChromaDB-based note store — persistent vector database backend."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, List
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
        use_async_http: bool = False,
        host: str = "localhost",
        port: int = 8000,
        ssl: bool = False,
        tenant: str = "default_tenant",
        database: str = "default_database",
    ) -> None:
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "ChromaDB is required for the chroma backend. "
                "Install it with: pip install chromadb"
            ) from e

        self._chroma = chromadb
        self._collection_name = collection_name
        self._use_async_http = use_async_http
        self._async_client: Any | None = None
        self._async_collection: Any | None = None

        if use_async_http:
            # Native async mode uses AsyncHttpClient for async methods and
            # HttpClient for sync methods to preserve backward compatibility.
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=ssl,
                tenant=tenant,
                database=database,
            )
            self._async_client_kwargs = {
                "host": host,
                "port": port,
                "ssl": ssl,
                "tenant": tenant,
                "database": database,
            }
            log_details = f"mode=async_http, host={host}, port={port}, ssl={ssl}"
        else:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                tenant=tenant,
                database=database,
            )
            self._async_client_kwargs = None
            log_details = f"mode=persistent, persist_dir={persist_directory}"

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaNoteStore initialized: collection=%s, %s, docs=%d",
            collection_name, log_details, self._collection.count(),
        )

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _ensure_async_collection(self):
        if not self._use_async_http:
            return None
        if self._async_collection is not None:
            return self._async_collection

        if self._async_client_kwargs is None:
            raise NotepadError("Async Chroma client configuration is missing.")

        async_client = await self._chroma.AsyncHttpClient(**self._async_client_kwargs)
        self._async_client = async_client
        self._async_collection = await self._maybe_await(
            async_client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        )
        return self._async_collection

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

    async def append_async(self, note: str) -> None:
        if not self._use_async_http:
            await super().append_async(note)
            return

        normalized = note.strip()
        if not normalized:
            return

        try:
            embedding = await asyncio.to_thread(embed, [normalized])
            collection = await self._ensure_async_collection()
            await self._maybe_await(
                collection.add(
                    ids=[str(uuid4())],
                    documents=[normalized],
                    embeddings=[embedding[0].tolist()],
                )
            )
            logger.debug("Appended note to ChromaDB async (%d chars)", len(normalized))
        except Exception as e:
            raise NotepadError(f"Failed to store note in ChromaDB async: {e}") from e

    async def load_notes_async(self) -> List[str]:
        if not self._use_async_http:
            return await super().load_notes_async()

        try:
            collection = await self._ensure_async_collection()
            result = await self._maybe_await(collection.get())
            return result["documents"] or []
        except Exception as e:
            raise NotepadError(f"Failed to load notes from ChromaDB async: {e}") from e

    async def retrieve_async(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.1,
    ) -> List[RetrievedNote]:
        if not self._use_async_http:
            return await super().retrieve_async(query, k, threshold)
        if not query.strip():
            return []

        try:
            query_embedding = await asyncio.to_thread(embed, [query])
            collection = await self._ensure_async_collection()
            total = await self._maybe_await(collection.count())
            results = await self._maybe_await(
                collection.query(
                    query_embeddings=[query_embedding[0].tolist()],
                    n_results=min(k, total) or 1,
                )
            )
        except Exception as e:
            logger.error("ChromaDB async retrieval failed: %s", e)
            raise RetrievalError(f"ChromaDB async retrieval failed: {e}") from e

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

    async def clear_async(self) -> None:
        if not self._use_async_http:
            await super().clear_async()
            return

        try:
            collection = await self._ensure_async_collection()
            payload = await self._maybe_await(collection.get())
            all_ids = payload["ids"]
            if all_ids:
                await self._maybe_await(collection.delete(ids=all_ids))
            logger.info("ChromaNoteStore cleared async")
        except Exception as e:
            raise NotepadError(f"Failed to clear ChromaDB async: {e}") from e

    async def count_async(self) -> int:
        if not self._use_async_http:
            return await super().count_async()

        try:
            collection = await self._ensure_async_collection()
            total = await self._maybe_await(collection.count())
            return int(total)
        except Exception as e:
            raise NotepadError(f"Failed to count ChromaDB async notes: {e}") from e

    def close(self) -> None:
        # Chroma clients do not currently guarantee a close hook across APIs.
        client_close = getattr(self._client, "close", None)
        if callable(client_close):
            client_close()
        self._collection = None

    async def close_async(self) -> None:
        if self._use_async_http and self._async_client is not None:
            async_client_close = getattr(self._async_client, "aclose", None)
            if callable(async_client_close):
                await self._maybe_await(async_client_close())
            else:
                maybe_close = getattr(self._async_client, "close", None)
                if callable(maybe_close):
                    await asyncio.to_thread(maybe_close)
        await super().close_async()
        self._async_collection = None
        self._async_client = None
