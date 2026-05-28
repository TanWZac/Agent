"""Transparency & Audit — logs all agent interactions for accountability."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.core.logging import get_logger

logger = get_logger("rai.transparency")


@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    event_type: str = ""  # "input_check", "output_check", "pii_detected", "content_blocked", "rate_limited"
    input_text: str = ""
    output_text: str = ""
    flagged: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """Append-only audit logger for Responsible AI events."""

    def __init__(self, log_file: str = "data/rai_audit.jsonl") -> None:
        self._log_path = Path(log_file)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: AuditEntry) -> None:
        """Append an audit entry to the log file."""
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except OSError as e:
            logger.error("Failed to write audit log: %s", e)

    def log_input_check(
        self, session_id: str, text: str, is_safe: bool, details: dict[str, Any] | None = None
    ) -> None:
        """Log an input safety check."""
        self.log(AuditEntry(
            session_id=session_id,
            event_type="input_check",
            input_text=text[:500],  # Truncate for storage
            flagged=not is_safe,
            details=details or {},
        ))

    def log_output_check(
        self, session_id: str, text: str, is_safe: bool, details: dict[str, Any] | None = None
    ) -> None:
        """Log an output safety check."""
        self.log(AuditEntry(
            session_id=session_id,
            event_type="output_check",
            output_text=text[:500],
            flagged=not is_safe,
            details=details or {},
        ))

    def log_pii_detected(
        self, session_id: str, pii_types: list[str], direction: str = "input"
    ) -> None:
        """Log PII detection event."""
        self.log(AuditEntry(
            session_id=session_id,
            event_type="pii_detected",
            flagged=True,
            details={"pii_types": pii_types, "direction": direction},
        ))

    def log_content_blocked(
        self, session_id: str, categories: list[str], direction: str = "input"
    ) -> None:
        """Log a content block event."""
        self.log(AuditEntry(
            session_id=session_id,
            event_type="content_blocked",
            flagged=True,
            details={"categories": categories, "direction": direction},
        ))

    def log_bias_detected(
        self, session_id: str, flags: list[dict], direction: str = "input"
    ) -> None:
        """Log a bias/fairness detection event."""
        self.log(AuditEntry(
            session_id=session_id,
            event_type="bias_detected",
            flagged=True,
            details={"flags": flags, "direction": direction},
        ))

    def log_rate_limited(self, session_id: str) -> None:
        """Log a rate limiting event."""
        self.log(AuditEntry(
            session_id=session_id,
            event_type="rate_limited",
            flagged=True,
            details={"reason": "Rate limit exceeded"},
        ))

    def get_recent_entries(self, n: int = 50) -> list[dict[str, Any]]:
        """Read the last N audit entries (for monitoring dashboards)."""
        if not self._log_path.exists():
            return []
        entries = []
        try:
            with self._log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to read audit log: %s", e)
        return entries[-n:]
