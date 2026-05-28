"""Configuration management — loads from JSON file with env var overrides.

Priority order (highest wins):
  1. Keyword arguments passed to get_settings()
  2. Environment variables (OPENAI_API_KEY, STORE_BACKEND, etc.)
  3. Values in config/config.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _load_env() -> None:
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)


_load_env()

_CONFIG_PATH = Path(__file__).parent / "config.json"


def _load_json_config() -> dict[str, Any]:
    """Load the JSON config file."""
    if _CONFIG_PATH.exists():
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


_json_cfg = _load_json_config()


def _cfg(section: str, key: str, env_var: str, default: Any) -> Any:
    """Resolve a config value: env var > JSON > default."""
    env_val = os.getenv(env_var)
    if env_val is not None:
        return env_val
    return _json_cfg.get(section, {}).get(key, default)


@dataclass(frozen=True)
class Settings:
    """Centralized, immutable application settings."""

    # LLM
    openai_api_key: str = field(repr=False, default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _cfg("llm", "model", "OPENAI_MODEL", "gpt-4o-mini"))
    openai_temperature: float = field(
        default_factory=lambda: float(_cfg("llm", "temperature", "OPENAI_TEMPERATURE", "0"))
    )

    # Store
    store_backend: str = field(default_factory=lambda: _cfg("store", "backend", "STORE_BACKEND", "file"))
    note_file: str = field(
        default_factory=lambda: _cfg("store", "path", "NOTE_FILE",
                                     _json_cfg.get("store", {}).get("file", {}).get("path", "data/notepad.txt"))
    )
    max_note_file_size_mb: int = field(
        default_factory=lambda: int(_cfg("store", "max_size_mb", "MAX_NOTE_FILE_SIZE_MB",
                                         _json_cfg.get("store", {}).get("file", {}).get("max_size_mb", 10)))
    )
    chroma_collection: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION",
                                          _json_cfg.get("store", {}).get("chroma", {}).get("collection", "notepad"))
    )
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR",
                                          _json_cfg.get("store", {}).get("chroma", {}).get("persist_dir", "data/chroma_db"))
    )

    # Retrieval
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K",
                                              _json_cfg.get("retrieval", {}).get("top_k", 3)))
    )
    embedding_model: str = field(
        default_factory=lambda: _json_cfg.get("retrieval", {}).get("embedding_model", "all-mpnet-base-v2")
    )

    # Web search
    web_search_max_results: int = field(
        default_factory=lambda: int(os.getenv("WEB_SEARCH_MAX_RESULTS",
                                              _json_cfg.get("search", {}).get("max_results", 5)))
    )

    # Server
    api_host: str = field(
        default_factory=lambda: os.getenv("API_HOST",
                                          _json_cfg.get("server", {}).get("host", "0.0.0.0"))
    )
    api_port: int = field(
        default_factory=lambda: int(os.getenv("API_PORT",
                                              _json_cfg.get("server", {}).get("port", 8000)))
    )
    mcp_endpoint: str = field(
        default_factory=lambda: _json_cfg.get("server", {}).get("mcp_endpoint", "/mcp")
    )

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL",
                                                             _json_cfg.get("logging", {}).get("level", "INFO")))

    def validate(self) -> None:
        """Raise if required settings are missing or invalid."""
        from src.core.exceptions import ConfigurationError

        if not self.openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Export it or add to .env file."
            )
        if self.openai_temperature < 0 or self.openai_temperature > 2:
            raise ConfigurationError("OPENAI_TEMPERATURE must be between 0 and 2.")
        if self.max_note_file_size_mb < 1:
            raise ConfigurationError("MAX_NOTE_FILE_SIZE_MB must be >= 1.")
        if self.store_backend not in ("file", "chroma"):
            raise ConfigurationError(
                f"STORE_BACKEND must be 'file' or 'chroma', got '{self.store_backend}'."
            )


def get_settings(**overrides: object) -> Settings:
    """Factory that creates Settings with optional overrides (useful for testing)."""
    defaults = Settings()
    if not overrides:
        return defaults
    return Settings(
        **{
            **{f.name: getattr(defaults, f.name) for f in defaults.__dataclass_fields__.values()},
            **overrides,
        }  # type: ignore[arg-type]
    )
