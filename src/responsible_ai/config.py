"""Responsible AI configuration — defines safety thresholds and policy settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RAIConfig:
    """Configuration for Responsible AI guardrails."""

    # Master toggle
    enabled: bool = True

    # Content filtering
    content_filter_enabled: bool = True
    blocked_categories: list[str] = field(default_factory=lambda: [
        "violence",
        "self_harm",
        "hate_speech",
        "sexual_content",
        "illegal_activity",
    ])

    # PII detection
    pii_detection_enabled: bool = True
    pii_redact_output: bool = True
    pii_block_input: bool = False  # If True, block messages containing PII rather than redacting

    # Transparency & audit
    audit_enabled: bool = True
    audit_log_file: str = "data/rai_audit.jsonl"
    audit_backend: str = "local"  # "local" or "azure_blob"
    audit_azure_connection_string: str = ""  # Set via env var AUDIT_AZURE_CONNECTION_STRING
    audit_azure_container: str = "rai-audit"
    audit_retention_days: int = 90  # Auto-purge after N days (0 = keep forever)
    audit_store_content: bool = False  # NEVER store raw content by default (ethical)

    # Bias & fairness
    bias_evaluation_enabled: bool = True
    bias_severity_threshold: str = "medium"  # "low", "medium", "high"
    bias_monitored_attributes: list[str] = field(default_factory=lambda: [
        "race",
        "gender",
        "religion",
        "age",
        "disability",
        "sexual_orientation",
        "nationality",
        "socioeconomic",
    ])
    bias_block_on_high_severity: bool = True
    bias_warn_on_output: bool = True

    # Rate limiting (per session)
    rate_limit_enabled: bool = True
    max_messages_per_minute: int = 20
    max_messages_per_session: int = 200

    # Output safety
    max_output_length: int = 10000
    disclaimer_enabled: bool = True
    disclaimer_topics: list[str] = field(default_factory=lambda: [
        "medical",
        "legal",
        "financial",
    ])


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean from environment variable."""
    val = os.environ.get(key, "")
    if not val:
        return default
    return val.lower() in ("1", "true", "yes")


def _env_int(key: str, default: int) -> int:
    """Read an integer from environment variable."""
    val = os.environ.get(key, "")
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def rai_config_from_env(**overrides: object) -> RAIConfig:
    """Create RAIConfig with environment variable overrides.

    Environment variables (all optional, fall back to dataclass defaults):
        RAI_ENABLED, RAI_CONTENT_FILTER_ENABLED, RAI_PII_DETECTION_ENABLED,
        RAI_PII_REDACT_OUTPUT, RAI_PII_BLOCK_INPUT, RAI_AUDIT_ENABLED,
        RAI_AUDIT_BACKEND, RAI_AUDIT_AZURE_CONNECTION_STRING,
        RAI_AUDIT_AZURE_CONTAINER, RAI_AUDIT_RETENTION_DAYS,
        RAI_AUDIT_STORE_CONTENT, RAI_BIAS_EVALUATION_ENABLED,
        RAI_BIAS_SEVERITY_THRESHOLD, RAI_BIAS_BLOCK_ON_HIGH_SEVERITY,
        RAI_BIAS_WARN_ON_OUTPUT, RAI_RATE_LIMIT_ENABLED,
        RAI_MAX_MESSAGES_PER_MINUTE, RAI_MAX_MESSAGES_PER_SESSION,
        RAI_MAX_OUTPUT_LENGTH, RAI_DISCLAIMER_ENABLED.
    """
    defaults = RAIConfig()
    kwargs: dict[str, object] = {
        "enabled": _env_bool("RAI_ENABLED", defaults.enabled),
        "content_filter_enabled": _env_bool("RAI_CONTENT_FILTER_ENABLED", defaults.content_filter_enabled),
        "pii_detection_enabled": _env_bool("RAI_PII_DETECTION_ENABLED", defaults.pii_detection_enabled),
        "pii_redact_output": _env_bool("RAI_PII_REDACT_OUTPUT", defaults.pii_redact_output),
        "pii_block_input": _env_bool("RAI_PII_BLOCK_INPUT", defaults.pii_block_input),
        "audit_enabled": _env_bool("RAI_AUDIT_ENABLED", defaults.audit_enabled),
        "audit_backend": os.environ.get("RAI_AUDIT_BACKEND", defaults.audit_backend),
        "audit_azure_connection_string": os.environ.get(
            "RAI_AUDIT_AZURE_CONNECTION_STRING", defaults.audit_azure_connection_string
        ),
        "audit_azure_container": os.environ.get("RAI_AUDIT_AZURE_CONTAINER", defaults.audit_azure_container),
        "audit_retention_days": _env_int("RAI_AUDIT_RETENTION_DAYS", defaults.audit_retention_days),
        "audit_store_content": _env_bool("RAI_AUDIT_STORE_CONTENT", defaults.audit_store_content),
        "bias_evaluation_enabled": _env_bool("RAI_BIAS_EVALUATION_ENABLED", defaults.bias_evaluation_enabled),
        "bias_severity_threshold": os.environ.get(
            "RAI_BIAS_SEVERITY_THRESHOLD", defaults.bias_severity_threshold
        ),
        "bias_block_on_high_severity": _env_bool(
            "RAI_BIAS_BLOCK_ON_HIGH_SEVERITY", defaults.bias_block_on_high_severity
        ),
        "bias_warn_on_output": _env_bool("RAI_BIAS_WARN_ON_OUTPUT", defaults.bias_warn_on_output),
        "rate_limit_enabled": _env_bool("RAI_RATE_LIMIT_ENABLED", defaults.rate_limit_enabled),
        "max_messages_per_minute": _env_int("RAI_MAX_MESSAGES_PER_MINUTE", defaults.max_messages_per_minute),
        "max_messages_per_session": _env_int("RAI_MAX_MESSAGES_PER_SESSION", defaults.max_messages_per_session),
        "max_output_length": _env_int("RAI_MAX_OUTPUT_LENGTH", defaults.max_output_length),
        "disclaimer_enabled": _env_bool("RAI_DISCLAIMER_ENABLED", defaults.disclaimer_enabled),
    }
    kwargs.update(overrides)
    return RAIConfig(**kwargs)  # type: ignore[arg-type]
