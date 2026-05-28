"""Fairlearn integration — statistical fairness metrics for LLM outputs.

This module uses Fairlearn's MetricFrame to compute group-level fairness metrics
across demographic cohorts, providing quantitative bias assessment beyond
keyword pattern matching.

Requires: ``pip install fairlearn`` (included in the ``[rai]`` extra).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.logging import get_logger

logger = get_logger("rai.fairness")


@dataclass
class FairnessMetrics:
    """Computed fairness metrics across demographic groups.

    :ivar metric_name: Name of the metric computed.
    :ivar overall: Overall metric value across all samples.
    :ivar by_group: Metric broken down by sensitive group.
    :ivar difference: Max difference between any two groups.
    :ivar ratio: Min ratio between any two groups (1.0 = perfectly fair).
    :ivar is_fair: Whether the metric meets the fairness threshold.
    :ivar threshold: The threshold used for the fairness determination.
    """

    metric_name: str
    overall: float
    by_group: dict[str, float] = field(default_factory=dict)
    difference: float = 0.0
    ratio: float = 1.0
    is_fair: bool = True
    threshold: float = 0.8


@dataclass
class FairnessReport:
    """Complete fairness assessment report.

    :ivar metrics: List of computed fairness metrics.
    :ivar is_fair: Whether all metrics pass their thresholds.
    :ivar summary: Human-readable summary.
    :ivar recommendations: Suggested actions if unfairness detected.
    """

    metrics: list[FairnessMetrics] = field(default_factory=list)
    is_fair: bool = True
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)


class FairnessEvaluator:
    """Evaluates fairness of LLM outputs using Fairlearn.

    Computes demographic parity and other fairness metrics across sensitive
    groups to detect systematic bias in model responses.

    :param ratio_threshold: Minimum acceptable ratio between groups (0-1).
        Default 0.8 means the worst-off group must receive at least 80%
        of the benefit of the best-off group.
    """

    def __init__(self, ratio_threshold: float = 0.8) -> None:
        try:
            import fairlearn  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "fairlearn is required for fairness evaluation. "
                "Install with: pip install fairlearn"
            ) from e
        self._ratio_threshold = ratio_threshold

    def evaluate_response_quality(
        self,
        responses: list[str],
        sensitive_features: list[str],
        *,
        quality_scores: list[float] | None = None,
    ) -> FairnessReport:
        """Evaluate fairness of response quality across demographic groups.

        If quality_scores are not provided, uses response length as a proxy
        metric (longer responses may indicate more helpful answers).

        :param responses: List of LLM responses to evaluate.
        :param sensitive_features: Corresponding sensitive group labels
            (e.g., ["male", "female", "male", ...]).
        :param quality_scores: Optional pre-computed quality scores (0-1).
        :return: FairnessReport with computed metrics.
        """
        from fairlearn.metrics import MetricFrame

        import numpy as np

        if len(responses) != len(sensitive_features):
            raise ValueError(
                f"responses ({len(responses)}) and sensitive_features "
                f"({len(sensitive_features)}) must have the same length."
            )

        if not responses:
            return FairnessReport(is_fair=True, summary="No data to evaluate.")

        # Use provided scores or compute length-based proxy
        if quality_scores is not None:
            scores = np.array(quality_scores, dtype=float)
        else:
            scores = np.array([len(r) for r in responses], dtype=float)
            # Normalize to 0-1 range
            max_score = scores.max()
            if max_score > 0:
                scores = scores / max_score

        sf = np.array(sensitive_features)

        # Compute MetricFrame for mean quality score
        metric_frame = MetricFrame(
            metrics={"mean_quality": np.mean},
            y_true=scores,  # MetricFrame needs y_true; we use scores
            y_pred=scores,  # Same as y_true since we're not comparing predictions
            sensitive_features=sf,
        )

        by_group = metric_frame.by_group["mean_quality"].to_dict()
        overall = float(metric_frame.overall["mean_quality"])
        difference = float(metric_frame.difference()["mean_quality"])
        ratio = float(metric_frame.ratio()["mean_quality"])

        quality_metric = FairnessMetrics(
            metric_name="response_quality_parity",
            overall=overall,
            by_group={str(k): float(v) for k, v in by_group.items()},
            difference=difference,
            ratio=ratio,
            is_fair=ratio >= self._ratio_threshold,
            threshold=self._ratio_threshold,
        )

        metrics = [quality_metric]
        is_fair = all(m.is_fair for m in metrics)

        recommendations = []
        if not is_fair:
            worst_group = min(by_group, key=by_group.get)  # type: ignore[arg-type]
            best_group = max(by_group, key=by_group.get)  # type: ignore[arg-type]
            recommendations.append(
                f"Response quality is lower for '{worst_group}' compared to '{best_group}' "
                f"(ratio: {ratio:.2f}, threshold: {self._ratio_threshold})."
            )
            recommendations.append(
                "Consider reviewing prompts and system instructions for implicit "
                "bias that may affect response quality for certain groups."
            )

        summary = (
            f"Fairness evaluation across {len(set(sensitive_features))} groups: "
            f"{'PASS' if is_fair else 'FAIL'} "
            f"(quality ratio={ratio:.2f}, threshold={self._ratio_threshold})"
        )

        logger.info("Fairness report: %s", summary)

        return FairnessReport(
            metrics=metrics,
            is_fair=is_fair,
            summary=summary,
            recommendations=recommendations,
        )

    def evaluate_selection_rate(
        self,
        selected: list[bool],
        sensitive_features: list[str],
    ) -> FairnessReport:
        """Evaluate whether selection/approval rates are fair across groups.

        Useful for evaluating if the agent approves/rejects requests
        differently based on demographic attributes.

        :param selected: Boolean list indicating selection (True=approved).
        :param sensitive_features: Corresponding sensitive group labels.
        :return: FairnessReport with selection rate metrics.
        """
        from fairlearn.metrics import MetricFrame, selection_rate

        import numpy as np

        if len(selected) != len(sensitive_features):
            raise ValueError("selected and sensitive_features must have the same length.")

        if not selected:
            return FairnessReport(is_fair=True, summary="No data to evaluate.")

        y = np.array(selected, dtype=int)
        sf = np.array(sensitive_features)

        metric_frame = MetricFrame(
            metrics=selection_rate,
            y_true=y,
            y_pred=y,
            sensitive_features=sf,
        )

        by_group = metric_frame.by_group.to_dict()
        overall = float(metric_frame.overall)
        difference = float(metric_frame.difference())
        ratio = float(metric_frame.ratio())

        rate_metric = FairnessMetrics(
            metric_name="selection_rate_parity",
            overall=overall,
            by_group={str(k): float(v) for k, v in by_group.items()},
            difference=difference,
            ratio=ratio,
            is_fair=ratio >= self._ratio_threshold,
            threshold=self._ratio_threshold,
        )

        metrics = [rate_metric]
        is_fair = all(m.is_fair for m in metrics)

        recommendations = []
        if not is_fair:
            worst_group = min(by_group, key=by_group.get)  # type: ignore[arg-type]
            recommendations.append(
                f"Selection rate is significantly lower for '{worst_group}' "
                f"(ratio: {ratio:.2f}). Review approval criteria for potential bias."
            )

        summary = (
            f"Selection rate parity: {'PASS' if is_fair else 'FAIL'} "
            f"(ratio={ratio:.2f}, threshold={self._ratio_threshold})"
        )

        return FairnessReport(
            metrics=metrics,
            is_fair=is_fair,
            summary=summary,
            recommendations=recommendations,
        )
