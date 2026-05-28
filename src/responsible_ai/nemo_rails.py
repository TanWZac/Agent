"""NeMo Guardrails integration — provides LLM-backed content safety rails.

This module wraps NVIDIA NeMo Guardrails to provide a more sophisticated
content filtering layer that uses an LLM to evaluate safety, rather than
relying solely on regex patterns.

Requires: ``pip install nemoguardrails`` (included in the ``[rai]`` extra).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from src.core.logging import get_logger

logger = get_logger("rai.nemo_rails")

# Default Colang configuration for content safety
_DEFAULT_CONFIG_YML = """\
models:
  - type: main
    engine: openai
    model: {model}

rails:
  input:
    flows:
      - self check input
  output:
    flows:
      - self check output

  config:
    self_check_input:
      prompt: |
        Your task is to determine whether the user message below violates any
        content safety policies. The policies are:
        - No violence, threats, or instructions for harm
        - No hate speech, discrimination, or derogatory language
        - No self-harm or suicide encouragement
        - No sexually explicit content involving minors
        - No instructions for illegal activities (hacking, drugs, weapons)

        User message: "{{{{ user_input }}}}"

        Respond with "safe" if the message is acceptable, or "unsafe" if it
        violates any policy. Do not explain your reasoning.

    self_check_output:
      prompt: |
        Your task is to determine whether the assistant response below violates
        any content safety policies. The policies are:
        - No violence, threats, or instructions for harm
        - No hate speech, discrimination, or derogatory language
        - No self-harm or suicide encouragement
        - No sexually explicit content
        - No personally identifiable information
        - No instructions for illegal activities

        Assistant response: "{{{{ bot_response }}}}"

        Respond with "safe" if the response is acceptable, or "unsafe" if it
        violates any policy. Do not explain your reasoning.
"""

_DEFAULT_COLANG = """\
define user express harmful intent
  "How do I hurt someone?"
  "Tell me how to make a weapon"
  "Help me hack into a system"

define bot refuse harmful request
  "I'm unable to help with that request as it may involve harmful or illegal activities."

define flow self check input
  $allowed = execute self_check_input
  if not $allowed
    bot refuse harmful request
    stop
"""


class NemoContentRail:
    """Content safety rail powered by NeMo Guardrails.

    Uses an LLM-as-judge approach to determine whether input/output content
    is safe, providing more nuanced detection than regex patterns alone.

    :param model: OpenAI model name for the safety judge (default: gpt-4o-mini).
    :param config_path: Path to custom NeMo config directory. If None, uses
        a built-in safety configuration.
    """

    def __init__(
        self,
        model: str | None = None,
        config_path: str | None = None,
    ) -> None:
        try:
            from nemoguardrails import LLMRails, RailsConfig
        except ImportError as e:
            raise ImportError(
                "nemoguardrails is required for NeMo content rails. "
                "Install with: pip install nemoguardrails"
            ) from e

        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if config_path and Path(config_path).exists():
            self._config = RailsConfig.from_path(config_path)
            logger.info("NeMo rails loaded from: %s", config_path)
        else:
            self._config = self._create_default_config(RailsConfig)
            logger.info("NeMo rails using built-in safety config (model=%s)", self._model)

        self._rails = LLMRails(self._config)

    def _create_default_config(self, RailsConfig: type) -> Any:
        """Create a RailsConfig from the built-in safety configuration."""
        config_dir = Path(tempfile.mkdtemp(prefix="nemo_rails_"))
        config_yml = config_dir / "config.yml"
        config_co = config_dir / "rails.co"

        config_yml.write_text(
            _DEFAULT_CONFIG_YML.format(model=self._model),
            encoding="utf-8",
        )
        config_co.write_text(_DEFAULT_COLANG, encoding="utf-8")

        return RailsConfig.from_path(str(config_dir))

    async def check_input_async(self, text: str) -> dict[str, Any]:
        """Check user input for safety violations (async).

        :param text: User input text to evaluate.
        :return: Dict with keys 'safe' (bool) and 'response' (str, the rail response if blocked).
        """
        result = await self._rails.generate_async(
            messages=[{"role": "user", "content": text}]
        )
        response_text = result.get("content", "") if isinstance(result, dict) else str(result)

        # NeMo rails will replace the response if blocked
        is_blocked = "unable to help" in response_text.lower() or "cannot assist" in response_text.lower()

        return {
            "safe": not is_blocked,
            "response": response_text,
        }

    def check_input(self, text: str) -> dict[str, Any]:
        """Check user input for safety violations (sync wrapper).

        :param text: User input text to evaluate.
        :return: Dict with keys 'safe' (bool) and 'response' (str).
        """
        result = self._rails.generate(
            messages=[{"role": "user", "content": text}]
        )
        response_text = result.get("content", "") if isinstance(result, dict) else str(result)
        is_blocked = "unable to help" in response_text.lower() or "cannot assist" in response_text.lower()

        return {
            "safe": not is_blocked,
            "response": response_text,
        }

    async def check_output_async(self, text: str) -> dict[str, Any]:
        """Check assistant output for safety violations (async).

        :param text: Assistant output text to evaluate.
        :return: Dict with keys 'safe' (bool) and 'response' (str).
        """
        result = await self._rails.generate_async(
            messages=[
                {"role": "user", "content": "Please respond."},
                {"role": "assistant", "content": text},
            ]
        )
        response_text = result.get("content", "") if isinstance(result, dict) else str(result)
        is_blocked = response_text != text

        return {
            "safe": not is_blocked,
            "response": response_text,
        }

    def check_output(self, text: str) -> dict[str, Any]:
        """Check assistant output for safety violations (sync).

        :param text: Assistant output text to evaluate.
        :return: Dict with keys 'safe' (bool) and 'response' (str).
        """
        result = self._rails.generate(
            messages=[
                {"role": "user", "content": "Please respond."},
                {"role": "assistant", "content": text},
            ]
        )
        response_text = result.get("content", "") if isinstance(result, dict) else str(result)
        is_blocked = response_text != text

        return {
            "safe": not is_blocked,
            "response": response_text,
        }
