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

_env_loaded = False


def _load_env() -> None:
    """Load .env file only once, and only outside test environments."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    # Skip .env loading when running under pytest (PYTEST_CURRENT_TEST is set by pytest)
    if os.getenv("PYTEST_CURRENT_TEST"):
        return
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)

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
    llm_provider: str = field(default_factory=lambda: _cfg("llm", "provider", "LLM_PROVIDER", "openai"))
    openai_api_key: str = field(repr=False, default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _cfg("llm", "model", "OPENAI_MODEL", "gpt-4o-mini"))
    openai_temperature: float = field(
        default_factory=lambda: float(_cfg("llm", "temperature", "OPENAI_TEMPERATURE", "0"))
    )

    # HuggingFace local model
    hf_model_id: str = field(
        default_factory=lambda: _json_cfg.get("llm", {}).get("huggingface", {}).get(
            "model_id", "microsoft/Phi-3-mini-4k-instruct-gguf")
    )
    hf_model_file: str = field(
        default_factory=lambda: _json_cfg.get("llm", {}).get("huggingface", {}).get(
            "model_file", "Phi-3-mini-4k-instruct-q4.gguf")
    )
    hf_n_ctx: int = field(
        default_factory=lambda: int(_json_cfg.get("llm", {}).get("huggingface", {}).get("n_ctx", 4096))
    )
    hf_n_threads: int = field(
        default_factory=lambda: int(_json_cfg.get("llm", {}).get("huggingface", {}).get("n_threads", 4))
    )
    hf_max_tokens: int = field(
        default_factory=lambda: int(_json_cfg.get("llm", {}).get("huggingface", {}).get("max_tokens", 1024))
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
    sqlite_db_url: str = field(
        default_factory=lambda: os.getenv(
            "SQLITE_DB_URL",
            _json_cfg.get("store", {}).get("sqlite", {}).get("db_url", "sqlite:///data/notepad.db"),
        )
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
    max_sessions: int = field(
        default_factory=lambda: int(os.getenv("MAX_SESSIONS",
                                              _json_cfg.get("server", {}).get("max_sessions", 1000)))
    )

    # Session management
    session_store_backend: str = field(
        default_factory=lambda: os.getenv("SESSION_STORE_BACKEND",
                                          _json_cfg.get("session", {}).get("store_backend", "memory"))
    )
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL",
                                          _json_cfg.get("session", {}).get("redis_url", "redis://localhost:6379/0"))
    )
    session_ttl_hours: int = field(
        default_factory=lambda: int(os.getenv("SESSION_TTL_HOURS",
                                              _json_cfg.get("session", {}).get("ttl_hours", 24)))
    )
    max_history_messages: int = field(
        default_factory=lambda: int(os.getenv("MAX_HISTORY_MESSAGES",
                                              _json_cfg.get("session", {}).get("max_history_messages", 20)))
    )
    summarize_keep_recent: int = field(
        default_factory=lambda: int(os.getenv("SUMMARIZE_KEEP_RECENT",
                                              _json_cfg.get("session", {}).get("summarize_keep_recent", 6)))
    )

    # LLM fallback
    llm_fallback_models: list[str] = field(
        default_factory=lambda: [
            m.strip()
            for m in os.getenv("LLM_FALLBACK_MODELS", ",".join(
                _json_cfg.get("llm", {}).get("fallback_models", [])
            )).split(",")
            if m.strip()
        ]
    )

    # Plugins
    plugin_tools_file: str = field(
        default_factory=lambda: os.getenv("PLUGIN_TOOLS_FILE",
                                          _json_cfg.get("plugins", {}).get("tools_file", "config/tools.yml"))
    )

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL",
                                                             _json_cfg.get("logging", {}).get("level", "INFO")))

    def validate(self) -> None:
        """Raise if required settings are missing or invalid."""
        from src.core.exceptions import ConfigurationError

        if self.llm_provider not in ("openai", "huggingface"):
            raise ConfigurationError(
                f"LLM_PROVIDER must be 'openai' or 'huggingface', got '{self.llm_provider}'."
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Export it or add to .env file."
            )
        if self.openai_temperature < 0 or self.openai_temperature > 2:
            raise ConfigurationError("OPENAI_TEMPERATURE must be between 0 and 2.")
        if self.max_note_file_size_mb < 1:
            raise ConfigurationError("MAX_NOTE_FILE_SIZE_MB must be >= 1.")
        if self.store_backend not in ("file", "chroma", "sqlite"):
            raise ConfigurationError(
                f"STORE_BACKEND must be 'file', 'chroma', or 'sqlite', got '{self.store_backend}'."
            )

    # --- Secret protection ---
    _SECRET_FIELDS = frozenset({"openai_api_key"})

    def __getstate__(self) -> dict:
        """Exclude secret fields from pickling/serialization."""
        state = {k: v for k, v in self.__dict__.items() if k not in self._SECRET_FIELDS}
        for k in self._SECRET_FIELDS:
            state[k] = "***"
        return state

    def to_safe_dict(self) -> dict:
        """Return a dict with secrets masked — safe for logging or serialization."""
        return {
            f.name: ("***" if f.name in self._SECRET_FIELDS else getattr(self, f.name))
            for f in self.__dataclass_fields__.values()
        }


def get_settings(**overrides: object) -> Settings:
    """Factory that creates Settings with optional overrides (useful for testing)."""
    _load_env()
    defaults = Settings()
    if not overrides:
        return defaults
    return Settings(
        **{
            **{f.name: getattr(defaults, f.name) for f in defaults.__dataclass_fields__.values()},
            **overrides,
        }  # type: ignore[arg-type]
    )


def get_rai_config():
    """Load Responsible AI configuration from the JSON config file."""
    from src.responsible_ai.config import RAIConfig

    rai_section = _json_cfg.get("responsible_ai", {})
    if not rai_section:
        return RAIConfig()

    return RAIConfig(
        enabled=rai_section.get("enabled", True),
        content_filter_enabled=rai_section.get("content_filter_enabled", True),
        blocked_categories=rai_section.get("blocked_categories", [
            "violence", "self_harm", "hate_speech", "sexual_content", "illegal_activity",
        ]),
        pii_detection_enabled=rai_section.get("pii_detection_enabled", True),
        pii_redact_output=rai_section.get("pii_redact_output", True),
        pii_block_input=rai_section.get("pii_block_input", False),
        bias_evaluation_enabled=rai_section.get("bias_evaluation_enabled", True),
        bias_severity_threshold=rai_section.get("bias_severity_threshold", "medium"),
        bias_monitored_attributes=rai_section.get("bias_monitored_attributes", [
            "race", "gender", "religion", "age", "disability",
            "sexual_orientation", "nationality", "socioeconomic",
        ]),
        bias_block_on_high_severity=rai_section.get("bias_block_on_high_severity", True),
        bias_warn_on_output=rai_section.get("bias_warn_on_output", True),
        audit_enabled=rai_section.get("audit_enabled", True),
        audit_log_file=rai_section.get("audit_log_file", "data/rai_audit.jsonl"),
        audit_backend=rai_section.get("audit_backend", "local"),
        audit_azure_connection_string=os.getenv("AUDIT_AZURE_CONNECTION_STRING", ""),
        audit_azure_container=rai_section.get("audit_azure_container", "rai-audit"),
        audit_retention_days=rai_section.get("audit_retention_days", 90),
        audit_store_content=rai_section.get("audit_store_content", False),
        rate_limit_enabled=rai_section.get("rate_limit_enabled", True),
        max_messages_per_minute=rai_section.get("max_messages_per_minute", 20),
        max_messages_per_session=rai_section.get("max_messages_per_session", 200),
        max_output_length=rai_section.get("max_output_length", 10000),
        disclaimer_enabled=rai_section.get("disclaimer_enabled", True),
        disclaimer_topics=rai_section.get("disclaimer_topics", ["medical", "legal", "financial"]),
    )
