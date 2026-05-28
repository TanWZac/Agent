"""Guardrails orchestrator — coordinates all RAI checks on input/output."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.core.logging import get_logger
from src.responsible_ai.bias_evaluator import BiasEvaluator
from src.responsible_ai.config import RAIConfig
from src.responsible_ai.content_filter import ContentFilter
from src.responsible_ai.pii_detector import PIIDetector
from src.responsible_ai.transparency import AuditLogger

logger = get_logger("rai.guardrails")

# Disclaimers appended when sensitive topics are discussed
_TOPIC_DISCLAIMERS: dict[str, str] = {
    "medical": (
        "\n\n⚕️ *Disclaimer: This is not medical advice. "
        "Please consult a qualified healthcare professional for medical concerns.*"
    ),
    "legal": (
        "\n\n⚖️ *Disclaimer: This is not legal advice. "
        "Please consult a qualified legal professional for legal matters.*"
    ),
    "financial": (
        "\n\n💰 *Disclaimer: This is not financial advice. "
        "Please consult a qualified financial advisor for investment or financial decisions.*"
    ),
}

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "medical": ["diagnosis", "symptom", "treatment", "medication", "prescri", "disease", "medical condition"],
    "legal": ["lawsuit", "legal action", "attorney", "court", "liability", "contract law", "legal advice"],
    "financial": ["invest", "stock", "portfolio", "financial plan", "retirement fund", "trading strategy"],
}


@dataclass
class GuardrailResult:
    """Result of guardrail processing."""

    allowed: bool
    original_text: str
    processed_text: str
    blocked_reason: str = ""
    pii_detected: bool = False
    content_flagged: bool = False
    bias_detected: bool = False
    rate_limited: bool = False
    disclaimers_added: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Guardrails:
    """Orchestrates all Responsible AI checks for the agent.

    Applies content filtering, PII detection, bias evaluation, rate limiting,
    and disclaimer injection to both inputs and outputs.
    """

    def __init__(self, config: RAIConfig | None = None) -> None:
        self._config = config or RAIConfig()
        self._content_filter = ContentFilter(
            blocked_categories=self._config.blocked_categories
        ) if self._config.content_filter_enabled else None
        self._pii_detector = PIIDetector() if self._config.pii_detection_enabled else None
        self._bias_evaluator = BiasEvaluator(
            monitored_attributes=self._config.bias_monitored_attributes,
            severity_threshold=self._config.bias_severity_threshold,
        ) if self._config.bias_evaluation_enabled else None
        self._audit = self._create_audit_logger() if self._config.audit_enabled else None

        # Rate limiting state: session_id -> list of timestamps
        self._message_timestamps: dict[str, list[float]] = defaultdict(list)
        self._message_counts: dict[str, int] = defaultdict(int)
        self._last_cleanup: float = time.time()
        self._cleanup_interval: float = 300.0  # Cleanup stale sessions every 5 minutes

        logger.info("Guardrails initialized: config=%s", self._config)

    def _create_audit_logger(self) -> AuditLogger:
        """Create the appropriate audit logger backend based on config."""
        from src.responsible_ai.transparency import AzureBlobBackend, LocalFileBackend

        backend = None
        if self._config.audit_backend == "azure_blob" and self._config.audit_azure_connection_string:
            backend = AzureBlobBackend(
                connection_string=self._config.audit_azure_connection_string,
                container_name=self._config.audit_azure_container,
            )
        else:
            backend = LocalFileBackend(log_file=self._config.audit_log_file)

        return AuditLogger(
            log_file=self._config.audit_log_file,
            backend=backend,
            retention_days=self._config.audit_retention_days,
            store_content=self._config.audit_store_content,
        )

    @property
    def config(self) -> RAIConfig:
        return self._config

    @property
    def audit_logger(self) -> AuditLogger | None:
        return self._audit

    def check_input(self, text: str, session_id: str = "") -> GuardrailResult:
        """Run all input guardrails on user message.

        Args:
            text: The user's input message.
            session_id: Session identifier for audit and rate limiting.

        Returns:
            GuardrailResult with processed text or block indication.
        """
        if not self._config.enabled:
            return GuardrailResult(allowed=True, original_text=text, processed_text=text)

        # Rate limiting
        if self._config.rate_limit_enabled and self._is_rate_limited(session_id):
            if self._audit:
                self._audit.log_rate_limited(session_id)
            return GuardrailResult(
                allowed=False,
                original_text=text,
                processed_text="",
                blocked_reason="Rate limit exceeded. Please wait before sending more messages.",
                rate_limited=True,
            )

        processed_text = text
        pii_detected = False
        content_flagged = False
        bias_detected = False

        # Content filtering
        if self._content_filter:
            filter_result = self._content_filter.check(text)
            if not filter_result.is_safe:
                content_flagged = True
                if self._audit:
                    self._audit.log_content_blocked(
                        session_id,
                        [c.value for c in filter_result.categories_flagged],
                        direction="input",
                    )
                return GuardrailResult(
                    allowed=False,
                    original_text=text,
                    processed_text="",
                    blocked_reason=(
                        "I'm unable to process this request as it may violate content safety policies. "
                        "If you believe this is an error, please rephrase your question."
                    ),
                    content_flagged=True,
                )

        # PII detection
        if self._pii_detector:
            pii_result = self._pii_detector.detect(text)
            if pii_result.has_pii:
                pii_detected = True
                pii_types = [m.pii_type.value for m in pii_result.matches]
                if self._audit:
                    self._audit.log_pii_detected(session_id, pii_types, direction="input")

                if self._config.pii_block_input:
                    return GuardrailResult(
                        allowed=False,
                        original_text=text,
                        processed_text="",
                        blocked_reason=(
                            "Your message appears to contain personal information "
                            f"({', '.join(pii_types)}). "
                            "Please remove sensitive data and try again."
                        ),
                        pii_detected=True,
                    )
                # Redact PII before passing to LLM
                processed_text = pii_result.redacted_text

        # Bias evaluation on input
        if self._bias_evaluator:
            bias_result = self._bias_evaluator.evaluate(text)
            if bias_result.has_bias:
                bias_detected = True
                if self._audit:
                    self._audit.log_bias_detected(
                        session_id,
                        flags=[{
                            "type": f.bias_type.value,
                            "attribute": f.protected_attribute.value,
                            "severity": f.severity,
                        } for f in bias_result.flags],
                        direction="input",
                    )
                # Block on high severity if configured
                if (
                    self._config.bias_block_on_high_severity
                    and bias_result.overall_severity == "high"
                ):
                    return GuardrailResult(
                        allowed=False,
                        original_text=text,
                        processed_text="",
                        blocked_reason=(
                            "Your message contains language that may be discriminatory or biased. "
                            "Please rephrase using respectful, inclusive language."
                        ),
                        bias_detected=True,
                    )

        # Record rate-limit timestamp
        if self._config.rate_limit_enabled:
            self._record_message(session_id)

        # Audit log
        if self._audit:
            self._audit.log_input_check(session_id, text, is_safe=True)

        return GuardrailResult(
            allowed=True,
            original_text=text,
            processed_text=processed_text,
            pii_detected=pii_detected,
            content_flagged=content_flagged,
            bias_detected=bias_detected,
        )

    def check_output(self, text: str, session_id: str = "") -> GuardrailResult:
        """Run all output guardrails on assistant response.

        Args:
            text: The assistant's response text.
            session_id: Session identifier for audit.

        Returns:
            GuardrailResult with processed text (potentially with disclaimers).
        """
        if not self._config.enabled:
            return GuardrailResult(allowed=True, original_text=text, processed_text=text)

        processed_text = text
        pii_detected = False
        content_flagged = False
        bias_detected = False
        disclaimers_added: list[str] = []

        # Content filtering on output
        if self._content_filter:
            filter_result = self._content_filter.check(text)
            if not filter_result.is_safe:
                content_flagged = True
                if self._audit:
                    self._audit.log_content_blocked(
                        session_id,
                        [c.value for c in filter_result.categories_flagged],
                        direction="output",
                    )
                return GuardrailResult(
                    allowed=False,
                    original_text=text,
                    processed_text=(
                        "I apologize, but I'm unable to provide that response "
                        "as it may contain unsafe content."
                    ),
                    content_flagged=True,
                )

        # PII redaction on output
        if self._pii_detector and self._config.pii_redact_output:
            pii_result = self._pii_detector.detect(text)
            if pii_result.has_pii:
                pii_detected = True
                processed_text = pii_result.redacted_text
                if self._audit:
                    pii_types = [m.pii_type.value for m in pii_result.matches]
                    self._audit.log_pii_detected(session_id, pii_types, direction="output")

        # Output length cap
        if len(processed_text) > self._config.max_output_length:
            processed_text = processed_text[:self._config.max_output_length] + "\n\n[Response truncated for safety.]"

        # Bias evaluation on output
        if self._bias_evaluator and self._config.bias_warn_on_output:
            bias_result = self._bias_evaluator.evaluate(processed_text)
            if bias_result.has_bias:
                bias_detected = True
                if self._audit:
                    self._audit.log_bias_detected(
                        session_id,
                        flags=[{
                            "type": f.bias_type.value,
                            "attribute": f.protected_attribute.value,
                            "severity": f.severity,
                        } for f in bias_result.flags],
                        direction="output",
                    )
                # For high-severity bias in output, replace the response
                if bias_result.overall_severity == "high":
                    return GuardrailResult(
                        allowed=False,
                        original_text=text,
                        processed_text=(
                            "I need to rephrase my response to ensure it's fair and unbiased. "
                            "Could you please rephrase your question so I can provide "
                            "a more balanced answer?"
                        ),
                        bias_detected=True,
                    )
                # For medium severity, append a fairness notice
                if bias_result.overall_severity == "medium":
                    processed_text += (
                        "\n\n⚠️ *Note: This response may contain generalizations. "
                        "Individual experiences vary widely and stereotypes do not "
                        "reflect the diversity within any group.*"
                    )

        # Topic disclaimers
        if self._config.disclaimer_enabled:
            for topic in self._config.disclaimer_topics:
                keywords = _TOPIC_KEYWORDS.get(topic, [])
                if any(kw.lower() in processed_text.lower() for kw in keywords):
                    disclaimer = _TOPIC_DISCLAIMERS.get(topic, "")
                    if disclaimer and disclaimer not in processed_text:
                        processed_text += disclaimer
                        disclaimers_added.append(topic)

        # Audit log
        if self._audit:
            self._audit.log_output_check(session_id, text, is_safe=True)

        return GuardrailResult(
            allowed=True,
            original_text=text,
            processed_text=processed_text,
            pii_detected=pii_detected,
            content_flagged=content_flagged,
            bias_detected=bias_detected,
            disclaimers_added=disclaimers_added,
        )

    def _is_rate_limited(self, session_id: str) -> bool:
        """Check if session has exceeded rate limits."""
        now = time.time()
        self._cleanup_stale_sessions(now)

        # Per-minute check
        timestamps = self._message_timestamps[session_id]
        recent = [t for t in timestamps if now - t < 60]
        self._message_timestamps[session_id] = recent

        if len(recent) >= self._config.max_messages_per_minute:
            return True

        # Per-session check
        if self._message_counts[session_id] >= self._config.max_messages_per_session:
            return True

        return False

    def _cleanup_stale_sessions(self, now: float) -> None:
        """Remove sessions with no recent activity to prevent unbounded memory growth."""
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        stale_cutoff = now - 3600  # Sessions inactive for 1 hour
        stale_sessions = [
            sid for sid, timestamps in self._message_timestamps.items()
            if not timestamps or timestamps[-1] < stale_cutoff
        ]
        for sid in stale_sessions:
            del self._message_timestamps[sid]
            self._message_counts.pop(sid, None)
        if stale_sessions:
            logger.debug("Cleaned up %d stale rate-limit sessions", len(stale_sessions))

    def _record_message(self, session_id: str) -> None:
        """Record a message timestamp for rate limiting."""
        self._message_timestamps[session_id].append(time.time())
        self._message_counts[session_id] += 1
