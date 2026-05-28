"""Persistent session store — Redis-backed with in-memory fallback.

Enables horizontal scaling and crash recovery for agent sessions.
Sessions are serialized as JSON and stored with TTL-based expiration.

Configuration (via config.json / Settings):
    session.store_backend: "memory" or "redis" (default: "memory")
    session.redis_url: Redis connection URL (default: "redis://localhost:6379/0")
    session.ttl_hours: Session time-to-live in hours (default: 24)
    server.max_sessions: Maximum stored sessions (default: 1000)
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

from src.config import Settings, get_settings
from src.core.logging import get_logger

logger = get_logger("server.session_store")


class SessionData:
    """Serializable session metadata (not the full AgentSession object)."""

    def __init__(
        self,
        session_id: str,
        persona: str | None = None,
        message_count: int = 0,
        created_at: float | None = None,
        last_active: float | None = None,
    ) -> None:
        self.session_id = session_id
        self.persona = persona
        self.message_count = message_count
        self.created_at = created_at or time.time()
        self.last_active = last_active or time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "persona": self.persona,
            "message_count": self.message_count,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionData":
        return cls(**data)


class SessionStore(ABC):
    """Abstract session store interface."""

    @abstractmethod
    def get(self, session_id: str) -> SessionData | None:
        """Retrieve session metadata."""
        ...

    @abstractmethod
    def put(self, data: SessionData) -> None:
        """Store or update session metadata."""
        ...

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        ...

    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored sessions."""
        ...

    @abstractmethod
    def touch(self, session_id: str) -> None:
        """Update last_active timestamp."""
        ...


class MemorySessionStore(SessionStore):
    """In-memory LRU session store with size limit (default, no external deps)."""

    def __init__(self, max_size: int | None = None, settings: Settings | None = None) -> None:
        self._store: OrderedDict[str, SessionData] = OrderedDict()
        _settings = settings or get_settings()
        self._max_size = max_size if max_size is not None else _settings.max_sessions

    def get(self, session_id: str) -> SessionData | None:
        if session_id in self._store:
            self._store.move_to_end(session_id)
            return self._store[session_id]
        return None

    def put(self, data: SessionData) -> None:
        self._store[data.session_id] = data
        self._store.move_to_end(data.session_id)
        while len(self._store) > self._max_size:
            evicted_id, _ = self._store.popitem(last=False)
            logger.info("Evicted session: %s (capacity=%d)", evicted_id, self._max_size)

    def delete(self, session_id: str) -> bool:
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False

    def exists(self, session_id: str) -> bool:
        return session_id in self._store

    def count(self) -> int:
        return len(self._store)

    def touch(self, session_id: str) -> None:
        if session_id in self._store:
            self._store[session_id].last_active = time.time()
            self._store.move_to_end(session_id)


class RedisSessionStore(SessionStore):
    """Redis-backed session store for horizontal scaling and persistence.

    Requires: ``redis`` package.
    Sessions are stored as JSON with TTL-based expiration.
    """

    def __init__(self, redis_url: str | None = None, ttl_hours: int | None = None, settings: Settings | None = None) -> None:
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis package required for Redis session store. "
                "Install with: pip install redis"
            )

        _settings = settings or get_settings()
        _redis_url = redis_url or _settings.redis_url
        _ttl_hours = ttl_hours if ttl_hours is not None else _settings.session_ttl_hours

        self._client = redis.from_url(_redis_url, decode_responses=True)
        self._ttl_seconds = _ttl_hours * 3600
        self._prefix = "agent:session:"

        # Verify connection
        try:
            self._client.ping()
            logger.info("Redis session store connected: %s", _redis_url)
        except redis.ConnectionError as e:
            logger.error("Redis connection failed: %s", e)
            raise

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def get(self, session_id: str) -> SessionData | None:
        data = self._client.get(self._key(session_id))
        if data:
            return SessionData.from_dict(json.loads(data))
        return None

    def put(self, data: SessionData) -> None:
        key = self._key(data.session_id)
        self._client.setex(key, self._ttl_seconds, json.dumps(data.to_dict()))

    def delete(self, session_id: str) -> bool:
        return bool(self._client.delete(self._key(session_id)))

    def exists(self, session_id: str) -> bool:
        return bool(self._client.exists(self._key(session_id)))

    def count(self) -> int:
        keys = self._client.keys(f"{self._prefix}*")
        return len(keys)

    def touch(self, session_id: str) -> None:
        data = self.get(session_id)
        if data:
            data.last_active = time.time()
            self.put(data)


def create_session_store(settings: Settings | None = None) -> SessionStore:
    """Factory: create the session store based on configuration."""
    _settings = settings or get_settings()
    if _settings.session_store_backend == "redis":
        try:
            return RedisSessionStore(settings=_settings)
        except (ImportError, Exception) as e:
            logger.warning("Redis unavailable, falling back to memory store: %s", e)
            return MemorySessionStore(settings=_settings)
    return MemorySessionStore(settings=_settings)
