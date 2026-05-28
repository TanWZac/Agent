"""Tests for the session store and plugin registry."""

import time
import pytest


class TestMemorySessionStore:
    """Tests for the in-memory LRU session store."""

    def test_put_and_get(self):
        from src.server.session_store import MemorySessionStore, SessionData

        store = MemorySessionStore(max_size=10)
        data = SessionData(session_id="s1", persona="default", message_count=5)
        store.put(data)
        retrieved = store.get("s1")
        assert retrieved is not None
        assert retrieved.session_id == "s1"
        assert retrieved.persona == "default"
        assert retrieved.message_count == 5

    def test_get_nonexistent(self):
        from src.server.session_store import MemorySessionStore

        store = MemorySessionStore()
        assert store.get("nonexistent") is None

    def test_lru_eviction(self):
        from src.server.session_store import MemorySessionStore, SessionData

        store = MemorySessionStore(max_size=3)
        for i in range(5):
            store.put(SessionData(session_id=f"s{i}"))
        # Only the last 3 should remain
        assert store.count() == 3
        assert store.exists("s2") is True
        assert store.exists("s3") is True
        assert store.exists("s4") is True
        assert store.exists("s0") is False
        assert store.exists("s1") is False

    def test_delete(self):
        from src.server.session_store import MemorySessionStore, SessionData

        store = MemorySessionStore()
        store.put(SessionData(session_id="s1"))
        assert store.delete("s1") is True
        assert store.delete("s1") is False
        assert store.exists("s1") is False

    def test_touch_updates_timestamp(self):
        from src.server.session_store import MemorySessionStore, SessionData

        store = MemorySessionStore()
        data = SessionData(session_id="s1", last_active=100.0)
        store.put(data)
        store.touch("s1")
        updated = store.get("s1")
        assert updated.last_active > 100.0

    def test_count(self):
        from src.server.session_store import MemorySessionStore, SessionData

        store = MemorySessionStore()
        assert store.count() == 0
        store.put(SessionData(session_id="s1"))
        store.put(SessionData(session_id="s2"))
        assert store.count() == 2


class TestSessionData:
    """Tests for SessionData serialization."""

    def test_to_dict_and_back(self):
        from src.server.session_store import SessionData

        data = SessionData(session_id="test-id", persona="helper", message_count=10)
        d = data.to_dict()
        restored = SessionData.from_dict(d)
        assert restored.session_id == "test-id"
        assert restored.persona == "helper"
        assert restored.message_count == 10

    def test_defaults(self):
        from src.server.session_store import SessionData

        data = SessionData(session_id="x")
        assert data.persona is None
        assert data.message_count == 0
        assert data.created_at > 0
        assert data.last_active > 0


class TestCreateSessionStore:
    """Tests for the factory function."""

    def test_defaults_to_memory(self):
        from src.server.session_store import create_session_store, MemorySessionStore
        from src.config import get_settings

        settings = get_settings(session_store_backend="memory")
        store = create_session_store(settings=settings)
        assert isinstance(store, MemorySessionStore)

    def test_redis_fallback_to_memory(self):
        """When Redis is configured but unavailable, falls back to memory."""
        from src.server.session_store import create_session_store, MemorySessionStore
        from src.config import get_settings

        settings = get_settings(
            session_store_backend="redis",
            redis_url="redis://nonexistent-host:9999/0",
        )
        store = create_session_store(settings=settings)
        assert isinstance(store, MemorySessionStore)


class TestPluginRegistry:
    """Tests for the plugin tool registry."""

    def test_no_config_file(self, tmp_path):
        from src.agent.plugins import PluginToolRegistry

        registry = PluginToolRegistry(str(tmp_path / "nonexistent.yml"))
        tools = registry.load()
        assert tools == []

    def test_load_math_tool(self, tmp_path):
        config_file = tmp_path / "tools.yml"
        config_file.write_text(
            "tools:\n"
            "  - name: calculator\n"
            "    description: Do math\n"
            "    type: python_eval\n"
        )
        from src.agent.plugins import PluginToolRegistry

        registry = PluginToolRegistry(str(config_file))
        tools = registry.load()
        assert len(tools) == 1
        assert tools[0].name == "calculator"

    def test_safe_math_eval(self):
        from src.agent.plugins import _safe_math_eval

        assert _safe_math_eval("2 + 3") == "5"
        assert _safe_math_eval("10 / 2") == "5.0"
        assert _safe_math_eval("2 ^ 10") == "1024"
        assert "Error" in _safe_math_eval("import os")
        assert "Error" in _safe_math_eval("__import__('os')")

    def test_static_response_tool(self, tmp_path):
        config_file = tmp_path / "tools.yml"
        config_file.write_text(
            "tools:\n"
            "  - name: info\n"
            "    description: Get info\n"
            "    type: static_response\n"
            "    config:\n"
            "      response: 'Hello!'\n"
        )
        from src.agent.plugins import PluginToolRegistry

        registry = PluginToolRegistry(str(config_file))
        tools = registry.load()
        assert len(tools) == 1
        result = tools[0].invoke({"query": "anything"})
        assert result == "Hello!"
