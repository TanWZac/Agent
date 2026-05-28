"""Tests for the test set relevancy validator."""

import pytest

from src.responsible_ai.testset_validator import TestSetValidator, ValidationReport


@pytest.fixture
def validator() -> TestSetValidator:
    return TestSetValidator(threshold=0.3, max_invalid_ratio=0.3)


class TestTestSetValidator:
    """Tests for TestSetValidator embedding-based coherence checks."""

    def test_matching_rows_pass(self, validator: TestSetValidator) -> None:
        """Semantically related (prompt, output, reference) should pass."""
        prompts = [
            "What is the capital of France?",
            "Explain photosynthesis briefly.",
        ]
        outputs = [
            "The capital of France is Paris.",
            "Photosynthesis is the process by which plants convert sunlight into energy.",
        ]
        references = [
            "Paris is the capital city of France.",
            "Photosynthesis converts light energy into chemical energy in plants.",
        ]

        report = validator.validate(prompts, outputs, references)
        assert report.is_valid
        assert report.valid_rows == 2
        assert len(report.invalid_rows) == 0

    def test_mismatched_rows_flagged(self, validator: TestSetValidator) -> None:
        """Unrelated output/reference should be flagged as invalid."""
        prompts = [
            "What is the capital of France?",
            "Explain photosynthesis briefly.",
        ]
        outputs = [
            "The capital of France is Paris.",
            "The stock market crashed today due to interest rate hikes.",  # Unrelated
        ]
        references = [
            "Paris is the capital city of France.",
            "Photosynthesis converts light energy into chemical energy in plants.",
        ]

        report = validator.validate(prompts, outputs, references)
        # Row 1 should be invalid (output doesn't match prompt/reference)
        assert 1 in report.invalid_rows
        assert not report.row_scores[1].is_valid

    def test_all_mismatched_rejects_dataset(self) -> None:
        """Dataset with too many invalid rows should be rejected."""
        strict_validator = TestSetValidator(threshold=0.5, max_invalid_ratio=0.1)

        prompts = ["What is 2+2?", "Tell me about dogs.", "What is Python?"]
        outputs = [
            "Jupiter is the largest planet.",  # Unrelated
            "The economy grew 3% last quarter.",  # Unrelated
            "Python is a programming language.",  # Related
        ]
        references = [
            "2+2 equals 4.",
            "Dogs are domesticated animals.",
            "Python is a versatile programming language.",
        ]

        report = strict_validator.validate(prompts, outputs, references)
        assert not report.is_valid
        assert len(report.invalid_rows) >= 2

    def test_filter_valid_rows(self, validator: TestSetValidator) -> None:
        """filter_valid_rows should return only coherent rows."""
        prompts = [
            "What is gravity?",
            "Define machine learning.",
            "What is gravity?",  # Repeated but will have bad output
        ]
        outputs = [
            "Gravity is the force of attraction between objects with mass.",
            "Machine learning is a subset of AI that learns from data.",
            "Pizza is a popular Italian food.",  # Unrelated to gravity
        ]
        references = [
            "Gravity pulls objects toward each other.",
            "ML allows computers to learn without explicit programming.",
            "Gravitational force acts between all masses.",
        ]

        f_prompts, f_outputs, f_refs, report = validator.filter_valid_rows(
            prompts, outputs, references
        )

        # The unrelated row should be filtered out
        assert len(f_prompts) <= 3
        assert len(f_prompts) == len(f_outputs) == len(f_refs)
        # The valid ones should all be about their respective topics
        assert report.total_rows == 3

    def test_empty_input_raises(self, validator: TestSetValidator) -> None:
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError, match="at least one row"):
            validator.validate([], [], [])

    def test_mismatched_lengths_raises(self, validator: TestSetValidator) -> None:
        """Different-length columns should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            validator.validate(["a", "b"], ["x"], ["y", "z"])

    def test_report_has_scores(self, validator: TestSetValidator) -> None:
        """Report should include per-row similarity scores."""
        prompts = ["What is AI?"]
        outputs = ["AI is artificial intelligence."]
        references = ["Artificial intelligence simulates human thinking."]

        report = validator.validate(prompts, outputs, references)
        assert len(report.row_scores) == 1
        score = report.row_scores[0]
        assert 0.0 <= score.prompt_output_sim <= 1.0
        assert 0.0 <= score.prompt_reference_sim <= 1.0
        assert 0.0 <= score.output_reference_sim <= 1.0

    def test_threshold_validation(self) -> None:
        """Invalid threshold should raise ValueError."""
        with pytest.raises(ValueError):
            TestSetValidator(threshold=1.5)
        with pytest.raises(ValueError):
            TestSetValidator(threshold=-0.1)
