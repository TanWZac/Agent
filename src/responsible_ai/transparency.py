"""
Transparency & Audit — privacy-preserving audit trail for RAI accountability.

Design principles:

- NO raw user content is stored — only hashed fingerprints and metadata.
- Session IDs are anonymized via one-way hashing.
- Configurable retention policy auto-purges old entries.
- Supports local JSONL and Azure Blob Storage backends.
- Generates compliance reports for stakeholder evidence.

:mod:`transparency` provides audit logging and compliance reporting for Responsible AI.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from src.core.logging import get_logger

logger = get_logger("rai.transparency")


def _hash_value(value: str, salt: str = "rai-audit") -> str:
    """
    One-way hash for anonymizing identifiers. Not reversible.

    :param value: The value to hash.
    :param salt: Salt for hashing.
    :return: 16-character hex digest.
    """
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:16]


def _content_fingerprint(text: str) -> str:
    """
    Generate a non-reversible fingerprint of content (for deduplication only).

    :param text: The text to fingerprint.
    :return: 12-character hex digest.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:12]


@dataclass
class AuditEntry:
    """
    A single audit log entry (stores NO raw user content).

    :ivar timestamp: Time of entry (epoch seconds)
    :ivar session_hash: Anonymized session ID
    :ivar event_type: Event type (e.g. "input_check")
    :ivar direction: "input" or "output"
    :ivar flagged: Whether the entry was flagged
    :ivar content_length: Length of content (not content itself)
    :ivar content_fingerprint: Non-reversible hash for deduplication
    :ivar details: Additional metadata
    """

    timestamp: float = field(default_factory=time.time)
    session_hash: str = ""  # Anonymized session ID
    event_type: str = ""
    direction: str = ""  # "input" or "output"
    flagged: bool = False
    content_length: int = 0  # Length only, not content
    content_fingerprint: str = ""  # Non-reversible hash for dedup
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the audit entry to a dictionary.

        :return: Dictionary representation of the entry.
        """
        return asdict(self)


class AuditBackend(Protocol):
    """
    Protocol for audit storage backends.

    Implementations must provide write, read_entries, and purge_before.
    """

    def write(self, entry: dict[str, Any]) -> None: ...
    def read_entries(self, since: float | None = None, limit: int = 1000) -> list[dict[str, Any]]: ...
    def purge_before(self, cutoff: float) -> int: ...


class LocalFileBackend:
    """
    Local JSONL file backend for audit storage.

    :param log_file: Path to the JSONL log file.
    """

    def __init__(self, log_file: str = "data/rai_audit.jsonl") -> None:
        self._log_path = Path(log_file)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, entry: dict[str, Any]) -> None:
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.error("Failed to write audit log: %s", e)

    def read_entries(self, since: float | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        if not self._log_path.exists():
            return []
        entries = []
        try:
            with self._log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if since and entry.get("timestamp", 0) < since:
                        continue
                    entries.append(entry)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to read audit log: %s", e)
        return entries[-limit:]

    def purge_before(self, cutoff: float) -> int:
        """
        Remove entries older than cutoff timestamp.

        :param cutoff: Epoch seconds; entries older than this are purged.
        :return: Number of entries purged.
        """
        if not self._log_path.exists():
            return 0
        kept = []
        purged = 0
        try:
            with self._log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("timestamp", 0) >= cutoff:
                        kept.append(line)
                    else:
                        purged += 1
            with self._log_path.open("w", encoding="utf-8") as f:
                for line in kept:
                    f.write(line + "\n")
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to purge audit log: %s", e)
        return purged


class AzureBlobBackend:
    """
    Azure Blob Storage backend for production audit storage.

    Stores audit entries as append blobs, partitioned by date.
    Requires: ``azure-storage-blob`` package.

    :param connection_string: Azure Blob Storage connection string.
    :param container_name: Name of the container to use.
    """

    def __init__(
        self,
        connection_string: str,
        container_name: str = "rai-audit",
    ) -> None:
        try:
            from azure.storage.blob import BlobServiceClient, ContainerClient
        except ImportError:
            raise ImportError(
                "azure-storage-blob is required for Azure audit backend. "
                "Install with: pip install azure-storage-blob"
            )

        self._blob_service = BlobServiceClient.from_connection_string(connection_string)
        self._container_name = container_name
        self._container: ContainerClient = self._blob_service.get_container_client(container_name)
        # Create container if it doesn't exist
        try:
            self._container.get_container_properties()
        except Exception:
            self._container.create_container()
        logger.info("Azure Blob audit backend initialized: container=%s", container_name)

    def _blob_name(self, timestamp: float | None = None) -> str:
        """
        Get blob name partitioned by date.

        :param timestamp: Epoch seconds (optional).
        :return: Blob name in format audit/YYYY-MM-DD.jsonl
        """
        ts = timestamp or time.time()
        date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        return f"audit/{date_str}.jsonl"

    def write(self, entry: dict[str, Any]) -> None:
        blob_name = self._blob_name(entry.get("timestamp"))
        blob_client = self._container.get_blob_client(blob_name)
        line = json.dumps(entry) + "\n"
        try:
            # Try to append to existing blob
            blob_client.upload_blob(
                line.encode("utf-8"),
                blob_type="AppendBlob",
                overwrite=False,
            )
        except Exception:
            try:
                # Create new append blob
                blob_client.create_append_blob()
                blob_client.append_block(line.encode("utf-8"))
            except Exception as e:
                logger.error("Failed to write to Azure Blob: %s", e)

    def read_entries(self, since: float | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        try:
            blobs = self._container.list_blobs(name_starts_with="audit/")
            for blob in sorted(blobs, key=lambda b: b.name, reverse=True):
                blob_client = self._container.get_blob_client(blob.name)
                content = blob_client.download_blob().readall().decode("utf-8")
                for line in content.strip().split("\n"):
                    if not line:
                        continue
                    entry = json.loads(line)
                    if since and entry.get("timestamp", 0) < since:
                        continue
                    entries.append(entry)
                if len(entries) >= limit:
                    break
        except Exception as e:
            logger.error("Failed to read from Azure Blob: %s", e)
        return entries[-limit:]

    def purge_before(self, cutoff: float) -> int:
        """Delete blobs older than cutoff date."""
        cutoff_date = datetime.fromtimestamp(cutoff, tz=timezone.utc).strftime("%Y-%m-%d")
        purged = 0
        try:
            blobs = self._container.list_blobs(name_starts_with="audit/")
            for blob in blobs:
                # Extract date from blob name: audit/YYYY-MM-DD.jsonl
                blob_date = blob.name.replace("audit/", "").replace(".jsonl", "")
                if blob_date < cutoff_date:
                    self._container.delete_blob(blob.name)
                    purged += 1
        except Exception as e:
            logger.error("Failed to purge Azure Blob audit: %s", e)
        return purged


class AuditLogger:
    """
    Privacy-preserving audit logger for Responsible AI events.

    Key ethical design decisions:
        - Session IDs are hashed (not stored in plaintext)
        - User content is NEVER stored — only length + non-reversible fingerprint
        - Configurable retention with auto-purge
        - Supports local and cloud backends
    """

    def __init__(
        self,
        log_file: str = "data/rai_audit.jsonl",
        backend: AuditBackend | None = None,
        retention_days: int = 90,
        store_content: bool = False,
    ) -> None:
        """
        Initialize the audit logger.

        :param log_file: Path for local file backend (used if no backend provided).
        :param backend: Custom storage backend (Azure Blob, etc.)
        :param retention_days: How long to keep entries (0 = forever).
        :param store_content: If True, stores truncated content. Default False for privacy.
        """
        self._backend: AuditBackend = backend or LocalFileBackend(log_file)
        self._retention_days = retention_days
        self._store_content = store_content

    def log(self, entry: AuditEntry) -> None:
        """
        Append an audit entry via the configured backend.

        :param entry: The AuditEntry to log.
        """
        self._backend.write(entry.to_dict())

    def log_input_check(
        self, session_id: str, text: str, is_safe: bool, details: dict[str, Any] | None = None
    ) -> None:
        """
        Log an input safety check.

        :param session_id: Session identifier.
        :param text: Input text.
        :param is_safe: Whether the input was deemed safe.
        :param details: Additional details.
        """
        entry_details = details or {}
        if self._store_content:
            entry_details["content_preview"] = text[:200]
        self.log(AuditEntry(
            session_hash=_hash_value(session_id),
            event_type="input_check",
            direction="input",
            flagged=not is_safe,
            content_length=len(text),
            content_fingerprint=_content_fingerprint(text),
            details=entry_details,
        ))

    def log_output_check(
        self, session_id: str, text: str, is_safe: bool, details: dict[str, Any] | None = None
    ) -> None:
        """
        Log an output safety check.

        :param session_id: Session identifier.
        :param text: Output text.
        :param is_safe: Whether the output was deemed safe.
        :param details: Additional details.
        """
        entry_details = details or {}
        if self._store_content:
            entry_details["content_preview"] = text[:200]
        self.log(AuditEntry(
            session_hash=_hash_value(session_id),
            event_type="output_check",
            direction="output",
            flagged=not is_safe,
            content_length=len(text),
            content_fingerprint=_content_fingerprint(text),
            details=entry_details,
        ))

    def log_pii_detected(
        self, session_id: str, pii_types: list[str], direction: str = "input"
    ) -> None:
        """
        Log PII detection event (stores type counts only, never the PII itself).

        :param session_id: Session identifier.
        :param pii_types: List of detected PII types.
        :param direction: "input" or "output".
        """
        self.log(AuditEntry(
            session_hash=_hash_value(session_id),
            event_type="pii_detected",
            direction=direction,
            flagged=True,
            details={"pii_types": pii_types, "pii_count": len(pii_types)},
        ))

    def log_content_blocked(
        self, session_id: str, categories: list[str], direction: str = "input"
    ) -> None:
        """
        Log a content block event.

        :param session_id: Session identifier.
        :param categories: List of blocked content categories.
        :param direction: "input" or "output".
        """
        self.log(AuditEntry(
            session_hash=_hash_value(session_id),
            event_type="content_blocked",
            direction=direction,
            flagged=True,
            details={"categories": categories},
        ))

    def log_bias_detected(
        self, session_id: str, flags: list[dict], direction: str = "input"
    ) -> None:
        """
        Log a bias/fairness detection event.

        :param session_id: Session identifier.
        :param flags: List of bias/fairness flags.
        :param direction: "input" or "output".
        """
        self.log(AuditEntry(
            session_hash=_hash_value(session_id),
            event_type="bias_detected",
            direction=direction,
            flagged=True,
            details={"flags": flags, "flag_count": len(flags)},
        ))

    def log_rate_limited(self, session_id: str) -> None:
        """
        Log a rate limiting event.

        :param session_id: Session identifier.
        """
        self.log(AuditEntry(
            session_hash=_hash_value(session_id),
            event_type="rate_limited",
            direction="input",
            flagged=True,
            details={"reason": "Rate limit exceeded"},
        ))

    def get_recent_entries(self, n: int = 50) -> list[dict[str, Any]]:
        """
        Read the last N audit entries.

        :param n: Number of entries to retrieve.
        :return: List of audit entry dicts.
        """
        return self._backend.read_entries(limit=n)

    def apply_retention_policy(self) -> int:
        """
        Purge entries older than the retention period.

        :return: Number of entries purged.
        """
        if self._retention_days <= 0:
            return 0
        cutoff = time.time() - (self._retention_days * 86400)
        purged = self._backend.purge_before(cutoff)
        if purged:
            logger.info("Retention policy applied: purged %d entries older than %d days", purged, self._retention_days)
        return purged

    def generate_compliance_report(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Generate a compliance report for stakeholder evidence.

        Returns aggregate statistics — no individual user data exposed.

        :param days: Report period in days (default 30).
        :return: Dict with compliance metrics suitable for sharing with auditors.
        """
        since = time.time() - (days * 86400)
        entries = self._backend.read_entries(since=since, limit=100000)

        total_checks = len(entries)
        flagged_entries = [e for e in entries if e.get("flagged")]
        event_types = Counter(e.get("event_type", "unknown") for e in entries)
        flagged_by_type = Counter(e.get("event_type", "unknown") for e in flagged_entries)

        # Bias breakdown
        bias_entries = [e for e in entries if e.get("event_type") == "bias_detected"]
        bias_attributes = Counter()
        for entry in bias_entries:
            for flag in entry.get("details", {}).get("flags", []):
                bias_attributes[flag.get("attribute", "unknown")] += 1

        # PII breakdown
        pii_entries = [e for e in entries if e.get("event_type") == "pii_detected"]
        pii_types_seen = Counter()
        for entry in pii_entries:
            for pii_type in entry.get("details", {}).get("pii_types", []):
                pii_types_seen[pii_type] += 1

        # Content blocked breakdown
        blocked_entries = [e for e in entries if e.get("event_type") == "content_blocked"]
        blocked_categories = Counter()
        for entry in blocked_entries:
            for cat in entry.get("details", {}).get("categories", []):
                blocked_categories[cat] += 1

        report_period_start = datetime.fromtimestamp(since, tz=timezone.utc).isoformat()
        report_period_end = datetime.now(tz=timezone.utc).isoformat()

        return {
            "report_metadata": {
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                "period_start": report_period_start,
                "period_end": report_period_end,
                "period_days": days,
                "privacy_note": "This report contains aggregate statistics only. "
                                "No individual user content or identifiable information is included.",
            },
            "summary": {
                "total_interactions_checked": total_checks,
                "total_flags_raised": len(flagged_entries),
                "flag_rate_percent": round(len(flagged_entries) / max(total_checks, 1) * 100, 2),
                "unique_sessions": len({e.get("session_hash") for e in entries}),
            },
            "events_by_type": dict(event_types),
            "flags_by_type": dict(flagged_by_type),
            "content_safety": {
                "total_blocked": len(blocked_entries),
                "categories_breakdown": dict(blocked_categories),
            },
            "bias_and_fairness": {
                "total_bias_flags": len(bias_entries),
                "attributes_flagged": dict(bias_attributes),
            },
            "privacy_protection": {
                "pii_detections": len(pii_entries),
                "pii_types_breakdown": dict(pii_types_seen),
            },
            "rate_limiting": {
                "total_rate_limited": event_types.get("rate_limited", 0),
            },
            "retention_policy": {
                "retention_days": self._retention_days,
                "policy": "Entries automatically purged after retention period. "
                          "No raw user content stored at any time.",
            },
        }

