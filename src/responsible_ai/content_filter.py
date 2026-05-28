"""Content filter — detects harmful, unsafe, or policy-violating content."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from src.core.logging import get_logger

logger = get_logger("rai.content_filter")


class ContentCategory(str, Enum):
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    HATE_SPEECH = "hate_speech"
    SEXUAL_CONTENT = "sexual_content"
    ILLEGAL_ACTIVITY = "illegal_activity"
    SAFE = "safe"


@dataclass
class FilterResult:
    """Result of content filtering."""

    is_safe: bool
    categories_flagged: list[ContentCategory]
    explanation: str


# Keyword-based patterns for each category (kept intentionally broad;
# production systems should use ML-based classifiers).
_CATEGORY_PATTERNS: dict[ContentCategory, list[str]] = {
    ContentCategory.VIOLENCE: [
        r"\b(kill|murder|attack|bomb|shoot|stab|assault|terroris[mt])\b.*\b(how|instructions|steps|guide|make)\b",
        r"\b(how to|guide to|steps to)\b.*\b(kill|murder|attack|bomb|shoot|stab)\b",
    ],
    ContentCategory.SELF_HARM: [
        r"\b(how to|ways to|methods? (to|of))\b.*\b(suicide|kill (myself|yourself)|self[- ]harm|end (my|your) life)\b",
        r"\b(suicide|self[- ]harm)\b.*\b(method|instruction|guide|plan)\b",
    ],
    ContentCategory.HATE_SPEECH: [
        r"\b(all|every)\s+\w+\s+(are|should)\s+(die|be killed|be eliminated|be removed)\b",
        r"\b(inferior|subhuman|vermin)\b.*\b(race|ethnic|people|group)\b",
    ],
    ContentCategory.SEXUAL_CONTENT: [
        r"\b(generate|write|create)\b.*\b(explicit|pornograph|erotic)\b.*\b(content|story|image)\b",
        r"\b(child|minor|underage)\b.*\b(sexual|nude|explicit)\b",
    ],
    ContentCategory.ILLEGAL_ACTIVITY: [
        r"\b(how to|guide to|steps to)\b.*\b(hack|crack|exploit|bypass|breach)\b.*\b(system|account|password|security)\b",
        r"\b(how to|steps to|guide to)\b.*\b(make|create|synthesize|cook)\b.*\b(drug|meth|cocaine|explosive|weapon)\b",
        r"\b(how to|ways to)\b.*\b(launder|counterfeit|forge)\b",
    ],
}

# Compiled patterns for efficiency
_COMPILED_PATTERNS: dict[ContentCategory, list[re.Pattern]] = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in _CATEGORY_PATTERNS.items()
}

# Basic leetspeak substitutions
_LEET_MAP: dict[str, str] = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "7": "t", "@": "a", "$": "s", "!": "i",
}


def _normalize_text(text: str) -> str:
    """Normalize Unicode (NFKD) and apply basic leetspeak reversal."""
    # Unicode normalization — collapses fullwidth, accented, and homoglyph chars
    normalized = unicodedata.normalize("NFKD", text)
    # Strip combining marks (accents) to get base characters
    normalized = "".join(
        ch for ch in normalized if not unicodedata.combining(ch)
    )
    # Leetspeak reversal
    normalized = "".join(_LEET_MAP.get(ch, ch) for ch in normalized)
    return normalized


class ContentFilter:
    """Filters content for harmful or policy-violating material."""

    def __init__(self, blocked_categories: list[str] | None = None) -> None:
        if blocked_categories:
            self._blocked = {ContentCategory(c) for c in blocked_categories}
        else:
            self._blocked = set(ContentCategory) - {ContentCategory.SAFE}

    def check(self, text: str) -> FilterResult:
        """Check text against content policies.

        Args:
            text: The text to evaluate.

        Returns:
            FilterResult indicating safety status and flagged categories.
        """
        flagged: list[ContentCategory] = []
        normalized = _normalize_text(text)

        for category, patterns in _COMPILED_PATTERNS.items():
            if category not in self._blocked:
                continue
            for pattern in patterns:
                if pattern.search(text) or pattern.search(normalized):
                    flagged.append(category)
                    break

        if flagged:
            explanation = (
                f"Content flagged for: {', '.join(c.value for c in flagged)}. "
                "This request may violate content safety policies."
            )
            logger.warning("Content filter triggered: categories=%s", flagged)
            return FilterResult(is_safe=False, categories_flagged=flagged, explanation=explanation)

        return FilterResult(is_safe=True, categories_flagged=[], explanation="Content is within policy.")

    def is_safe(self, text: str) -> bool:
        """Quick boolean check for content safety."""
        return self.check(text).is_safe
