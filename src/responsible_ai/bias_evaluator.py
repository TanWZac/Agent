"""Bias & Fairness Evaluator — detects stereotyping, demographic bias, and unfair generalizations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from src.core.logging import get_logger

logger = get_logger("rai.bias_evaluator")


class ProtectedAttribute(str, Enum):
    """Protected demographic attributes monitored for bias."""

    RACE = "race"
    GENDER = "gender"
    RELIGION = "religion"
    AGE = "age"
    DISABILITY = "disability"
    SEXUAL_ORIENTATION = "sexual_orientation"
    NATIONALITY = "nationality"
    SOCIOECONOMIC = "socioeconomic"


class BiasType(str, Enum):
    """Types of bias detected."""

    STEREOTYPING = "stereotyping"
    EXCLUSION = "exclusion"
    SWEEPING_GENERALIZATION = "sweeping_generalization"
    DEROGATORY_LANGUAGE = "derogatory_language"
    UNEQUAL_REPRESENTATION = "unequal_representation"


@dataclass
class BiasFlag:
    """A single bias detection flag."""

    bias_type: BiasType
    protected_attribute: ProtectedAttribute
    matched_pattern: str
    explanation: str
    severity: str = "medium"  # "low", "medium", "high"


@dataclass
class BiasEvalResult:
    """Result of bias and fairness evaluation."""

    has_bias: bool
    flags: list[BiasFlag] = field(default_factory=list)
    overall_severity: str = "none"  # "none", "low", "medium", "high"
    recommendation: str = ""


# ─── Protected group terms ───────────────────────────────────────────────────

_RACE_TERMS = (
    r"(?:black|white|asian|hispanic|latino|latina|latinx|african[- ]american|"
    r"caucasian|indigenous|native[- ]american|arab|pacific[- ]islander|"
    r"middle[- ]eastern|south[- ]asian|east[- ]asian)"
)

_GENDER_TERMS = (
    r"(?:women|woman|men|man|female|male|transgender|trans|non[- ]binary|"
    r"girls?|boys?|ladies|gentlemen)"
)

_RELIGION_TERMS = (
    r"(?:muslim|christian|jewish|hindu|buddhist|sikh|atheist|"
    r"catholic|protestant|mormon|evangelical)"
)

_AGE_TERMS = (
    r"(?:old people|elderly|seniors|boomers|millennials|gen[- ]z|"
    r"young people|teenagers|kids|youth|older workers|aged)"
)

_DISABILITY_TERMS = (
    r"(?:disabled|handicapped|wheelchair|blind|deaf|autistic|"
    r"mentally ill|crazy|insane|retarded|crippled|lame)"
)

_ORIENTATION_TERMS = (
    r"(?:gay|lesbian|homosexual|bisexual|queer|lgbtq|straight|heterosexual)"
)

_NATIONALITY_TERMS = (
    r"(?:americans?|chinese|indians?|mexicans?|russians?|japanese|"
    r"germans?|french|british|africans?|europeans?|immigrants?|foreigners?)"
)

_SOCIOECONOMIC_TERMS = (
    r"(?:poor people|rich people|homeless|welfare|lower class|upper class|"
    r"working class|uneducated|illiterate|ghetto|trailer trash|redneck)"
)

_ATTRIBUTE_TERMS: dict[ProtectedAttribute, str] = {
    ProtectedAttribute.RACE: _RACE_TERMS,
    ProtectedAttribute.GENDER: _GENDER_TERMS,
    ProtectedAttribute.RELIGION: _RELIGION_TERMS,
    ProtectedAttribute.AGE: _AGE_TERMS,
    ProtectedAttribute.DISABILITY: _DISABILITY_TERMS,
    ProtectedAttribute.SEXUAL_ORIENTATION: _ORIENTATION_TERMS,
    ProtectedAttribute.NATIONALITY: _NATIONALITY_TERMS,
    ProtectedAttribute.SOCIOECONOMIC: _SOCIOECONOMIC_TERMS,
}

# ─── Bias pattern templates ─────────────────────────────────────────────────

# Sweeping generalizations: "All [group] are..." / "[group] always..."
_GENERALIZATION_TEMPLATES = [
    r"\b(?:all|every|no)\s+{group}(?:\s+\w+){{0,2}}\s+(?:are|is|always|never|can't|cannot|don't|do not|will never|should)\b",
    r"\b{group}(?:\s+\w+){{0,2}}\s+(?:always|never|can't|cannot|don't|do not|will never|tend to|are known for)\b",
    r"\b(?:typical|typical of)\s+{group}\b",
]

# Stereotyping patterns: "[group] are [trait]"
_STEREOTYPE_TEMPLATES = [
    r"\b{group}(?:\s+\w+){{0,2}}\s+(?:are|is)\s+(?:naturally|inherently|genetically|biologically)\s+\w+",
    r"\b{group}(?:\s+\w+){{0,2}}\s+(?:are|is)\s+(?:good|bad|better|worse|smarter|dumber|lazier|more violent|less capable)\b",
    r"\b{group}(?:\s+\w+){{0,2}}\s+(?:belong|should stay|should go back|don't belong)\b",
]

# Derogatory/exclusionary patterns
_DEROGATORY_TEMPLATES = [
    r"\b{group}(?:\s+\w+){{0,2}}\s+(?:are|is)\s+(?:inferior|superior|subhuman|primitive|uncivilized|savage[s]?)\b",
    r"\b(?:get rid of|eliminate|remove|ban)\s+(?:all\s+)?{group}\b",
    r"\b{group}(?:\s+\w+){{0,2}}\s+(?:are ruining|are destroying|are a problem|are the problem|are to blame)\b",
    r"\b(?:those|these|all)\s+(?:\w+\s+)?(?:are|is)\s+(?:inferior|subhuman|primitive|savage[s]?)\b",
]


@dataclass
class _PatternSet:
    """Pre-compiled pattern set for a bias type + attribute combination."""

    bias_type: BiasType
    attribute: ProtectedAttribute
    patterns: list[re.Pattern]


def _compile_patterns() -> list[_PatternSet]:
    """Compile all bias detection patterns for each attribute."""
    result: list[_PatternSet] = []

    templates_by_type: dict[BiasType, list[str]] = {
        BiasType.SWEEPING_GENERALIZATION: _GENERALIZATION_TEMPLATES,
        BiasType.STEREOTYPING: _STEREOTYPE_TEMPLATES,
        BiasType.DEROGATORY_LANGUAGE: _DEROGATORY_TEMPLATES,
    }

    for attribute, group_regex in _ATTRIBUTE_TERMS.items():
        for bias_type, templates in templates_by_type.items():
            compiled = []
            for template in templates:
                pattern_str = template.format(group=group_regex)
                compiled.append(re.compile(pattern_str, re.IGNORECASE))
            result.append(_PatternSet(
                bias_type=bias_type,
                attribute=attribute,
                patterns=compiled,
            ))

    return result


_ALL_PATTERNS = _compile_patterns()


class BiasEvaluator:
    """Evaluates text for demographic bias, stereotyping, and fairness issues."""

    def __init__(
        self,
        monitored_attributes: list[str] | None = None,
        severity_threshold: str = "medium",
    ) -> None:
        """Initialize the bias evaluator.

        Args:
            monitored_attributes: List of ProtectedAttribute values to monitor.
                If None, all attributes are monitored.
            severity_threshold: Minimum severity to flag ("low", "medium", "high").
                Flags below this threshold are still detected but not included in results.
        """
        if monitored_attributes:
            self._monitored = {ProtectedAttribute(a) for a in monitored_attributes}
        else:
            self._monitored = set(ProtectedAttribute)

        self._severity_threshold = severity_threshold
        self._severity_order = {"low": 0, "medium": 1, "high": 2}

    def evaluate(self, text: str) -> BiasEvalResult:
        """Evaluate text for bias and fairness issues.

        Args:
            text: The text to evaluate.

        Returns:
            BiasEvalResult with detected bias flags and recommendations.
        """
        flags: list[BiasFlag] = []

        for pattern_set in _ALL_PATTERNS:
            if pattern_set.attribute not in self._monitored:
                continue

            for pattern in pattern_set.patterns:
                match = pattern.search(text)
                if match:
                    severity = self._determine_severity(pattern_set.bias_type)
                    if self._meets_threshold(severity):
                        flags.append(BiasFlag(
                            bias_type=pattern_set.bias_type,
                            protected_attribute=pattern_set.attribute,
                            matched_pattern=match.group(),
                            explanation=self._explain(pattern_set.bias_type, pattern_set.attribute),
                            severity=severity,
                        ))
                    break  # One match per pattern set is enough

        if not flags:
            return BiasEvalResult(has_bias=False, overall_severity="none")

        overall_severity = max(
            (f.severity for f in flags),
            key=lambda s: self._severity_order.get(s, 0),
        )

        recommendation = self._generate_recommendation(flags)

        logger.warning(
            "Bias detected: %d flag(s), severity=%s, attributes=%s",
            len(flags),
            overall_severity,
            list({f.protected_attribute.value for f in flags}),
        )

        return BiasEvalResult(
            has_bias=True,
            flags=flags,
            overall_severity=overall_severity,
            recommendation=recommendation,
        )

    def has_bias(self, text: str) -> bool:
        """Quick boolean check for bias presence."""
        return self.evaluate(text).has_bias

    def _determine_severity(self, bias_type: BiasType) -> str:
        """Map bias type to severity level."""
        severity_map = {
            BiasType.SWEEPING_GENERALIZATION: "medium",
            BiasType.STEREOTYPING: "medium",
            BiasType.EXCLUSION: "medium",
            BiasType.UNEQUAL_REPRESENTATION: "low",
            BiasType.DEROGATORY_LANGUAGE: "high",
        }
        return severity_map.get(bias_type, "medium")

    def _meets_threshold(self, severity: str) -> bool:
        """Check if severity meets the configured threshold."""
        return self._severity_order.get(severity, 0) >= self._severity_order.get(
            self._severity_threshold, 0
        )

    def _explain(self, bias_type: BiasType, attribute: ProtectedAttribute) -> str:
        """Generate a human-readable explanation for a bias flag."""
        explanations = {
            BiasType.SWEEPING_GENERALIZATION: (
                f"Contains a sweeping generalization about a group based on {attribute.value}. "
                "Such statements oversimplify and can perpetuate harmful stereotypes."
            ),
            BiasType.STEREOTYPING: (
                f"Contains stereotyping language related to {attribute.value}. "
                "Attributing fixed traits to a group based on demographics is unfair."
            ),
            BiasType.DEROGATORY_LANGUAGE: (
                f"Contains derogatory or exclusionary language targeting {attribute.value}. "
                "This type of language can cause harm and perpetuate discrimination."
            ),
            BiasType.EXCLUSION: (
                f"Contains exclusionary framing related to {attribute.value}. "
                "Content should be inclusive and avoid marginalizing groups."
            ),
            BiasType.UNEQUAL_REPRESENTATION: (
                f"Shows potential unequal representation regarding {attribute.value}. "
                "Consider whether all relevant perspectives are fairly represented."
            ),
        }
        return explanations.get(bias_type, f"Potential bias detected related to {attribute.value}.")

    def _generate_recommendation(self, flags: list[BiasFlag]) -> str:
        """Generate an actionable recommendation based on detected flags."""
        high_severity = [f for f in flags if f.severity == "high"]

        if high_severity:
            return (
                "This content contains high-severity bias indicators and should be revised. "
                "Consider removing derogatory language and ensuring respectful, "
                "evidence-based statements about all groups."
            )

        return (
            "This content contains potential bias. Consider rephrasing to avoid "
            "generalizations about demographic groups and ensure fair representation."
        )
