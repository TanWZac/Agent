"""Tests for Responsible AI modules."""

import json
import tempfile
from pathlib import Path

import pytest

from src.responsible_ai.bias_evaluator import BiasEvaluator, BiasType, ProtectedAttribute
from src.responsible_ai.config import RAIConfig
from src.responsible_ai.content_filter import ContentCategory, ContentFilter
from src.responsible_ai.guardrails import GuardrailResult, Guardrails
from src.responsible_ai.pii_detector import PIIDetector, PIIType
from src.responsible_ai.transparency import AuditEntry, AuditLogger


# ─── Content Filter Tests ───────────────────────────────────────────────────


class TestContentFilter:
    def test_safe_content_passes(self):
        cf = ContentFilter()
        result = cf.check("What is the weather like today?")
        assert result.is_safe is True
        assert result.categories_flagged == []

    def test_violence_detected(self):
        cf = ContentFilter()
        result = cf.check("how to make a bomb step by step instructions")
        assert result.is_safe is False
        assert ContentCategory.VIOLENCE in result.categories_flagged

    def test_self_harm_detected(self):
        cf = ContentFilter()
        result = cf.check("methods of suicide how to end my life")
        assert result.is_safe is False
        assert ContentCategory.SELF_HARM in result.categories_flagged

    def test_illegal_activity_detected(self):
        cf = ContentFilter()
        result = cf.check("how to hack into a system and bypass security")
        assert result.is_safe is False
        assert ContentCategory.ILLEGAL_ACTIVITY in result.categories_flagged

    def test_unblocked_category_not_flagged(self):
        # Only block violence, not illegal activity
        cf = ContentFilter(blocked_categories=["violence"])
        result = cf.check("how to hack into a system and bypass security")
        assert result.is_safe is True

    def test_is_safe_shorthand(self):
        cf = ContentFilter()
        assert cf.is_safe("Hello, how are you?") is True
        assert cf.is_safe("how to kill someone step by step guide") is False


# ─── PII Detector Tests ─────────────────────────────────────────────────────


class TestPIIDetector:
    def test_no_pii(self):
        detector = PIIDetector()
        result = detector.detect("Hello, how are you doing today?")
        assert result.has_pii is False
        assert result.matches == []
        assert result.redacted_text == "Hello, how are you doing today?"

    def test_email_detected(self):
        detector = PIIDetector()
        result = detector.detect("My email is john.doe@example.com please contact me")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.EMAIL for m in result.matches)
        assert "[EMAIL REDACTED]" in result.redacted_text
        assert "john.doe@example.com" not in result.redacted_text

    def test_phone_detected(self):
        detector = PIIDetector()
        result = detector.detect("Call me at 555-123-4567")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.PHONE for m in result.matches)
        assert "[PHONE REDACTED]" in result.redacted_text

    def test_ssn_detected(self):
        detector = PIIDetector()
        result = detector.detect("My SSN is 123-45-6789")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.SSN for m in result.matches)
        assert "[SSN REDACTED]" in result.redacted_text

    def test_credit_card_detected(self):
        detector = PIIDetector()
        result = detector.detect("Card number: 4111 1111 1111 1111")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.CREDIT_CARD for m in result.matches)
        assert "[CARD REDACTED]" in result.redacted_text

    def test_multiple_pii_redacted(self):
        detector = PIIDetector()
        text = "Email: test@test.com, Phone: 555-123-4567"
        result = detector.detect(text)
        assert result.has_pii is True
        assert len(result.matches) >= 2
        assert "test@test.com" not in result.redacted_text
        assert "555-123-4567" not in result.redacted_text

    def test_redact_shorthand(self):
        detector = PIIDetector()
        redacted = detector.redact("Contact: user@mail.com")
        assert "user@mail.com" not in redacted
        assert "[EMAIL REDACTED]" in redacted

    def test_has_pii_shorthand(self):
        detector = PIIDetector()
        assert detector.has_pii("my email is a@b.com") is True
        assert detector.has_pii("nothing here") is False


# ─── Audit Logger Tests ──────────────────────────────────────────────────────


class TestAuditLogger:
    def test_log_creates_file_and_writes_entry(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_file=log_file)
        audit.log_input_check("session-1", "hello world", is_safe=True)

        entries = audit.get_recent_entries()
        assert len(entries) == 1
        assert entries[0]["session_id"] == "session-1"
        assert entries[0]["event_type"] == "input_check"
        assert entries[0]["flagged"] is False

    def test_log_content_blocked(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_file=log_file)
        audit.log_content_blocked("session-2", ["violence"], direction="input")

        entries = audit.get_recent_entries()
        assert len(entries) == 1
        assert entries[0]["flagged"] is True
        assert entries[0]["details"]["categories"] == ["violence"]

    def test_log_pii_detected(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_file=log_file)
        audit.log_pii_detected("session-3", ["email", "phone"], direction="output")

        entries = audit.get_recent_entries()
        assert len(entries) == 1
        assert entries[0]["details"]["pii_types"] == ["email", "phone"]

    def test_get_recent_entries_limit(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_file=log_file)
        for i in range(10):
            audit.log_input_check(f"session-{i}", f"msg-{i}", is_safe=True)

        entries = audit.get_recent_entries(n=5)
        assert len(entries) == 5


# ─── Guardrails Integration Tests ───────────────────────────────────────────


class TestGuardrails:
    def _make_guardrails(self, tmp_path, **overrides):
        config = RAIConfig(
            audit_log_file=str(tmp_path / "audit.jsonl"),
            **overrides,
        )
        return Guardrails(config=config)

    def test_safe_input_passes(self, tmp_path):
        g = self._make_guardrails(tmp_path)
        result = g.check_input("What is the capital of France?", session_id="s1")
        assert result.allowed is True
        assert result.processed_text == "What is the capital of France?"

    def test_harmful_input_blocked(self, tmp_path):
        g = self._make_guardrails(tmp_path)
        result = g.check_input("how to make a bomb step by step instructions", session_id="s1")
        assert result.allowed is False
        assert result.content_flagged is True
        assert "content safety" in result.blocked_reason.lower()

    def test_pii_input_redacted(self, tmp_path):
        g = self._make_guardrails(tmp_path, pii_block_input=False)
        result = g.check_input("My email is secret@company.com", session_id="s1")
        assert result.allowed is True
        assert result.pii_detected is True
        assert "secret@company.com" not in result.processed_text
        assert "[EMAIL REDACTED]" in result.processed_text

    def test_pii_input_blocked_when_configured(self, tmp_path):
        g = self._make_guardrails(tmp_path, pii_block_input=True)
        result = g.check_input("My SSN is 123-45-6789", session_id="s1")
        assert result.allowed is False
        assert result.pii_detected is True

    def test_safe_output_passes(self, tmp_path):
        g = self._make_guardrails(tmp_path)
        result = g.check_output("The capital of France is Paris.", session_id="s1")
        assert result.allowed is True
        assert result.processed_text == "The capital of France is Paris."

    def test_output_pii_redacted(self, tmp_path):
        g = self._make_guardrails(tmp_path)
        result = g.check_output("Contact them at user@email.com", session_id="s1")
        assert result.allowed is True
        assert "user@email.com" not in result.processed_text
        assert "[EMAIL REDACTED]" in result.processed_text

    def test_output_truncated_when_too_long(self, tmp_path):
        g = self._make_guardrails(tmp_path, max_output_length=100)
        long_text = "x" * 200
        result = g.check_output(long_text, session_id="s1")
        assert result.allowed is True
        assert len(result.processed_text) < 200
        assert "[Response truncated for safety.]" in result.processed_text

    def test_medical_disclaimer_added(self, tmp_path):
        g = self._make_guardrails(tmp_path)
        result = g.check_output(
            "Based on the symptoms and diagnosis, the treatment involves medication.",
            session_id="s1",
        )
        assert result.allowed is True
        assert "medical" in result.disclaimers_added
        assert "not medical advice" in result.processed_text.lower()

    def test_rate_limiting(self, tmp_path):
        g = self._make_guardrails(tmp_path, max_messages_per_minute=3)
        for i in range(3):
            result = g.check_input(f"msg {i}", session_id="s1")
            assert result.allowed is True

        result = g.check_input("msg 4", session_id="s1")
        assert result.allowed is False
        assert result.rate_limited is True

    def test_disabled_guardrails_pass_everything(self, tmp_path):
        g = self._make_guardrails(tmp_path, enabled=False)
        result = g.check_input("how to make a bomb step by step instructions", session_id="s1")
        assert result.allowed is True

    def test_different_sessions_independent_rate_limits(self, tmp_path):
        g = self._make_guardrails(tmp_path, max_messages_per_minute=2)
        # Fill up session s1
        g.check_input("msg1", session_id="s1")
        g.check_input("msg2", session_id="s1")
        # s1 should be rate limited
        result = g.check_input("msg3", session_id="s1")
        assert result.allowed is False
        # s2 should still be fine
        result = g.check_input("msg1", session_id="s2")
        assert result.allowed is True

    def test_bias_high_severity_input_blocked(self, tmp_path):
        g = self._make_guardrails(tmp_path, bias_block_on_high_severity=True)
        result = g.check_input(
            "All immigrants are ruining this country and should be eliminated",
            session_id="s1",
        )
        assert result.allowed is False
        assert result.bias_detected is True
        assert "discriminatory" in result.blocked_reason.lower() or "biased" in result.blocked_reason.lower()

    def test_bias_medium_severity_input_allowed(self, tmp_path):
        g = self._make_guardrails(tmp_path, bias_block_on_high_severity=True)
        result = g.check_input(
            "All women are naturally better at caregiving",
            session_id="s1",
        )
        # Medium severity should still be allowed through (just flagged)
        assert result.allowed is True
        assert result.bias_detected is True

    def test_bias_output_high_severity_replaced(self, tmp_path):
        g = self._make_guardrails(tmp_path, bias_warn_on_output=True)
        result = g.check_output(
            "All immigrants are inferior and should be removed from society",
            session_id="s1",
        )
        assert result.allowed is False
        assert result.bias_detected is True

    def test_bias_output_medium_severity_adds_notice(self, tmp_path):
        g = self._make_guardrails(tmp_path, bias_warn_on_output=True)
        result = g.check_output(
            "Women are naturally better at multitasking than men",
            session_id="s1",
        )
        assert result.allowed is True
        assert result.bias_detected is True
        assert "generalizations" in result.processed_text.lower()


# ─── Bias Evaluator Unit Tests ───────────────────────────────────────────────


class TestBiasEvaluator:
    def test_no_bias_in_neutral_text(self):
        evaluator = BiasEvaluator()
        result = evaluator.evaluate("The weather is nice today.")
        assert result.has_bias is False
        assert result.flags == []
        assert result.overall_severity == "none"

    def test_sweeping_generalization_detected(self):
        evaluator = BiasEvaluator()
        result = evaluator.evaluate("All women are bad at math")
        assert result.has_bias is True
        assert any(f.bias_type == BiasType.SWEEPING_GENERALIZATION for f in result.flags)
        assert any(f.protected_attribute == ProtectedAttribute.GENDER for f in result.flags)

    def test_derogatory_language_high_severity(self):
        evaluator = BiasEvaluator()
        result = evaluator.evaluate("Indigenous people are primitive savages")
        assert result.has_bias is True
        assert result.overall_severity == "high"
        assert any(f.bias_type == BiasType.DEROGATORY_LANGUAGE for f in result.flags)

    def test_racial_stereotyping_detected(self):
        evaluator = BiasEvaluator()
        result = evaluator.evaluate("Asian people are naturally good at math")
        assert result.has_bias is True
        assert any(f.protected_attribute == ProtectedAttribute.RACE for f in result.flags)

    def test_age_bias_detected(self):
        evaluator = BiasEvaluator()
        result = evaluator.evaluate("Old people are always bad with technology")
        assert result.has_bias is True
        assert any(f.protected_attribute == ProtectedAttribute.AGE for f in result.flags)

    def test_religion_bias_detected(self):
        evaluator = BiasEvaluator()
        result = evaluator.evaluate("All muslim people are dangerous")
        assert result.has_bias is True
        assert any(f.protected_attribute == ProtectedAttribute.RELIGION for f in result.flags)

    def test_specific_attributes_only(self):
        # Only monitor gender, not race
        evaluator = BiasEvaluator(monitored_attributes=["gender"])
        result = evaluator.evaluate("All Asian people are naturally smarter")
        assert result.has_bias is False  # Race not monitored

        result = evaluator.evaluate("All women are bad drivers")
        assert result.has_bias is True  # Gender is monitored

    def test_severity_threshold_filters(self):
        # High threshold: only high severity flags
        evaluator = BiasEvaluator(severity_threshold="high")
        result = evaluator.evaluate("All women are naturally better at caregiving")
        # This is a generalization (medium severity), should not be flagged at high threshold
        assert result.has_bias is False

    def test_has_bias_shorthand(self):
        evaluator = BiasEvaluator()
        assert evaluator.has_bias("Hello, how are you?") is False
        assert evaluator.has_bias("All elderly people are always confused") is True

    def test_recommendation_for_high_severity(self):
        evaluator = BiasEvaluator()
        result = evaluator.evaluate("All immigrants are inferior and should be eliminated")
        assert result.has_bias is True
        assert "derogatory" in result.recommendation.lower() or "revised" in result.recommendation.lower()

    def test_recommendation_for_medium_severity(self):
        evaluator = BiasEvaluator()
        result = evaluator.evaluate("All women always tend to be more emotional")
        assert result.has_bias is True
        assert "generalization" in result.recommendation.lower() or "rephras" in result.recommendation.lower()

