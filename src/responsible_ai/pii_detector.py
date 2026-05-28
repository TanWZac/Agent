"""PII Detector — identifies and redacts personally identifiable information."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.core.logging import get_logger

logger = get_logger("rai.pii_detector")


class PIIType(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    ID_NUMBER = "id_number"


@dataclass
class PIIMatch:
    """A single PII detection match."""

    pii_type: PIIType
    start: int
    end: int
    original: str


@dataclass
class PIIResult:
    """Result of PII detection on text."""

    has_pii: bool
    matches: list[PIIMatch]
    redacted_text: str


# Patterns for common PII types
_PII_PATTERNS: dict[PIIType, re.Pattern] = {
    PIIType.EMAIL: re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    PIIType.PHONE: re.compile(
        r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
    ),
    PIIType.SSN: re.compile(
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
    ),
    PIIType.CREDIT_CARD: re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),
    PIIType.IP_ADDRESS: re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    PIIType.ID_NUMBER: re.compile(
        r"\b[A-Z]{1,2}\d{6,9}\b"
    ),
}

# Redaction placeholder per type
_REDACTION_MAP: dict[PIIType, str] = {
    PIIType.EMAIL: "[EMAIL REDACTED]",
    PIIType.PHONE: "[PHONE REDACTED]",
    PIIType.SSN: "[SSN REDACTED]",
    PIIType.CREDIT_CARD: "[CARD REDACTED]",
    PIIType.IP_ADDRESS: "[IP REDACTED]",
    PIIType.ID_NUMBER: "[ID REDACTED]",
}


class PIIDetector:
    """Detects and redacts PII from text."""

    def __init__(self, detect_types: list[PIIType] | None = None) -> None:
        self._types = detect_types or list(PIIType)

    def detect(self, text: str) -> PIIResult:
        """Scan text for PII and produce a redacted version.

        Args:
            text: Input text to scan.

        Returns:
            PIIResult with detection matches and redacted text.
        """
        matches: list[PIIMatch] = []

        for pii_type in self._types:
            pattern = _PII_PATTERNS[pii_type]
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    start=match.start(),
                    end=match.end(),
                    original=match.group(),
                ))

        # Sort by position (descending) to redact from end to start
        matches.sort(key=lambda m: m.start, reverse=True)
        redacted = text
        for m in matches:
            placeholder = _REDACTION_MAP[m.pii_type]
            redacted = redacted[:m.start] + placeholder + redacted[m.end:]

        # Re-sort ascending for the result
        matches.sort(key=lambda m: m.start)

        if matches:
            logger.info(
                "PII detected: %d instance(s) of types %s",
                len(matches),
                list({m.pii_type.value for m in matches}),
            )

        return PIIResult(
            has_pii=bool(matches),
            matches=matches,
            redacted_text=redacted,
        )

    def redact(self, text: str) -> str:
        """Convenience method: return redacted text."""
        return self.detect(text).redacted_text

    def has_pii(self, text: str) -> bool:
        """Quick boolean check for PII presence."""
        return self.detect(text).has_pii
