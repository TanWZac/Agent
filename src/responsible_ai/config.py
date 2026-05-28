"""Responsible AI configuration — defines safety thresholds and policy settings."""

from __future__ import annotations

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
