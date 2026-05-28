"""Test set relevancy validator — checks coherence of evaluation datasets.

Before running bias/fairness evaluation on a user-provided test set, this module
validates that the (prompt, output, reference) triples are semantically coherent.
It uses the project's embedding model to compute pairwise cosine similarities
and flags rows where columns are not "matching".

Usage::

    from src.responsible_ai.testset_validator import TestSetValidator

    validator = TestSetValidator(threshold=0.4)
    report = validator.validate(prompts, outputs, references)
    if not report.is_valid:
        print(report.summary)
        print(report.invalid_rows)  # indices of mismatched rows
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.embeddings import cosine_similarity, embed
from src.core.logging import get_logger

logger = get_logger("rai.testset_validator")


@dataclass
class RowScore:
    """Similarity scores for a single test set row.

    :ivar index: Row index in the dataset.
    :ivar prompt_output_sim: Cosine similarity between prompt and output.
    :ivar prompt_reference_sim: Cosine similarity between prompt and reference.
    :ivar output_reference_sim: Cosine similarity between output and reference.
    :ivar is_valid: Whether all pairs meet the threshold.
    :ivar reason: Explanation if the row is invalid.
    """

    index: int
    prompt_output_sim: float
    prompt_reference_sim: float
    output_reference_sim: float
    is_valid: bool = True
    reason: str = ""


@dataclass
class ValidationReport:
    """Full validation report for a test set.

    :ivar is_valid: Whether the entire test set passes validation.
    :ivar total_rows: Total number of rows evaluated.
    :ivar valid_rows: Number of valid rows.
    :ivar invalid_rows: Indices of rows that fail validation.
    :ivar row_scores: Detailed scores for each row.
    :ivar mean_prompt_output_sim: Mean similarity between prompts and outputs.
    :ivar mean_prompt_reference_sim: Mean similarity between prompts and references.
    :ivar mean_output_reference_sim: Mean similarity between outputs and references.
    :ivar threshold: Threshold used for validation.
    :ivar summary: Human-readable summary.
    """

    is_valid: bool = True
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: list[int] = field(default_factory=list)
    row_scores: list[RowScore] = field(default_factory=list)
    mean_prompt_output_sim: float = 0.0
    mean_prompt_reference_sim: float = 0.0
    mean_output_reference_sim: float = 0.0
    threshold: float = 0.0
    summary: str = ""


class TestSetValidator:
    """Validates that a test set's (prompt, output, reference) triples are semantically coherent.

    Uses embedding-based cosine similarity to ensure:
    - The output is relevant to the prompt
    - The reference is relevant to the prompt
    - The output and reference are discussing the same topic

    Rows that fail any pairwise check are flagged, and the full dataset is
    rejected if too many rows are invalid.

    :param threshold: Minimum cosine similarity for a pair to be considered
        "matching". Default 0.4 (relatively permissive for diverse topics).
    :param max_invalid_ratio: Maximum fraction of invalid rows before the
        entire dataset is rejected. Default 0.2 (20%).
    """

    def __init__(
        self,
        threshold: float = 0.4,
        max_invalid_ratio: float = 0.2,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        if not 0.0 <= max_invalid_ratio <= 1.0:
            raise ValueError(f"max_invalid_ratio must be between 0 and 1, got {max_invalid_ratio}")
        self._threshold = threshold
        self._max_invalid_ratio = max_invalid_ratio

    @property
    def threshold(self) -> float:
        return self._threshold

    def validate(
        self,
        prompts: list[str],
        outputs: list[str],
        references: list[str],
    ) -> ValidationReport:
        """Validate a test set for semantic coherence.

        :param prompts: List of user prompts / inputs.
        :param outputs: List of model outputs to evaluate.
        :param references: List of reference (ground truth) answers.
        :return: ValidationReport with per-row and aggregate results.
        :raises ValueError: If input lists have different lengths or are empty.
        """
        n = len(prompts)
        if n == 0:
            raise ValueError("Test set must contain at least one row.")
        if len(outputs) != n or len(references) != n:
            raise ValueError(
                f"All columns must have the same length. "
                f"Got prompts={n}, outputs={len(outputs)}, references={len(references)}."
            )

        logger.info("Validating test set: %d rows, threshold=%.2f", n, self._threshold)

        # Compute embeddings for all three columns
        prompt_embeddings = embed(prompts)
        output_embeddings = embed(outputs)
        reference_embeddings = embed(references)

        # Compute pairwise similarities row by row
        row_scores: list[RowScore] = []
        invalid_indices: list[int] = []

        for i in range(n):
            p_emb = prompt_embeddings[i]
            o_emb = output_embeddings[i]
            r_emb = reference_embeddings[i]

            po_sim = float(cosine_similarity(p_emb, o_emb.reshape(1, -1))[0])
            pr_sim = float(cosine_similarity(p_emb, r_emb.reshape(1, -1))[0])
            or_sim = float(cosine_similarity(o_emb, r_emb.reshape(1, -1))[0])

            # Determine which pairs fail
            failures = []
            if po_sim < self._threshold:
                failures.append(f"prompt↔output ({po_sim:.3f})")
            if pr_sim < self._threshold:
                failures.append(f"prompt↔reference ({pr_sim:.3f})")
            if or_sim < self._threshold:
                failures.append(f"output↔reference ({or_sim:.3f})")

            is_valid = len(failures) == 0
            reason = ""
            if not is_valid:
                reason = f"Low similarity: {'; '.join(failures)}"
                invalid_indices.append(i)

            row_scores.append(RowScore(
                index=i,
                prompt_output_sim=po_sim,
                prompt_reference_sim=pr_sim,
                output_reference_sim=or_sim,
                is_valid=is_valid,
                reason=reason,
            ))

        # Compute aggregate statistics
        mean_po = float(np.mean([s.prompt_output_sim for s in row_scores]))
        mean_pr = float(np.mean([s.prompt_reference_sim for s in row_scores]))
        mean_or = float(np.mean([s.output_reference_sim for s in row_scores]))

        valid_count = n - len(invalid_indices)
        invalid_ratio = len(invalid_indices) / n
        dataset_valid = invalid_ratio <= self._max_invalid_ratio

        summary_parts = [
            f"Test set validation: {valid_count}/{n} rows valid ({100 * valid_count / n:.0f}%)",
            f"Threshold: {self._threshold}",
            f"Mean similarities — prompt↔output: {mean_po:.3f}, prompt↔ref: {mean_pr:.3f}, output↔ref: {mean_or:.3f}",
        ]
        if not dataset_valid:
            summary_parts.append(
                f"REJECTED: {len(invalid_indices)} invalid rows ({100 * invalid_ratio:.0f}%) "
                f"exceeds max allowed ratio ({100 * self._max_invalid_ratio:.0f}%)."
            )
        else:
            summary_parts.append("PASSED: Test set is suitable for evaluation.")

        summary = "\n".join(summary_parts)
        logger.info("Validation result: valid=%s, %d/%d rows valid", dataset_valid, valid_count, n)

        return ValidationReport(
            is_valid=dataset_valid,
            total_rows=n,
            valid_rows=valid_count,
            invalid_rows=invalid_indices,
            row_scores=row_scores,
            mean_prompt_output_sim=mean_po,
            mean_prompt_reference_sim=mean_pr,
            mean_output_reference_sim=mean_or,
            threshold=self._threshold,
            summary=summary,
        )

    def filter_valid_rows(
        self,
        prompts: list[str],
        outputs: list[str],
        references: list[str],
    ) -> tuple[list[str], list[str], list[str], ValidationReport]:
        """Validate and return only valid rows for downstream evaluation.

        Convenience method that validates the test set and returns filtered
        columns containing only rows that pass the coherence check.

        :param prompts: List of user prompts.
        :param outputs: List of model outputs.
        :param references: List of reference answers.
        :return: Tuple of (filtered_prompts, filtered_outputs, filtered_references, report).
        """
        report = self.validate(prompts, outputs, references)

        valid_set = set(range(report.total_rows)) - set(report.invalid_rows)
        valid_indices = sorted(valid_set)

        filtered_prompts = [prompts[i] for i in valid_indices]
        filtered_outputs = [outputs[i] for i in valid_indices]
        filtered_references = [references[i] for i in valid_indices]

        return filtered_prompts, filtered_outputs, filtered_references, report
