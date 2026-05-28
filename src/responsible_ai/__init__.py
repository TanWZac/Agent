"""Responsible AI module — content safety, PII protection, bias evaluation, and transparency."""

from src.responsible_ai.bias_evaluator import BiasEvalResult, BiasEvaluator
from src.responsible_ai.guardrails import Guardrails, GuardrailResult
from src.responsible_ai.testset_validator import TestSetValidator, ValidationReport

__all__ = [
    "BiasEvalResult",
    "BiasEvaluator",
    "Guardrails",
    "GuardrailResult",
    "TestSetValidator",
    "ValidationReport",
]

# Optional integrations — available when [rai] extras are installed
try:
    from src.responsible_ai.nemo_rails import NemoContentRail  # noqa: F401
    __all__.append("NemoContentRail")
except ImportError:
    pass

try:
    from src.responsible_ai.fairness import FairnessEvaluator, FairnessReport  # noqa: F401
    __all__.extend(["FairnessEvaluator", "FairnessReport"])
except ImportError:
    pass
