"""
LLM factory — creates the appropriate LLM instance based on provider config.

Supports multi-model fallback: if the primary model fails, automatically
retries with configured fallback models.

:mod:`llm` provides a factory for instantiating LangChain-compatible LLMs based on configuration.
"""

from __future__ import annotations

import os
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.config import Settings
from src.core.logging import get_logger

logger = get_logger("agent.llm")

# Fallback models (env-configurable, comma-separated)
_FALLBACK_MODELS = os.getenv("LLM_FALLBACK_MODELS", "").split(",")
_FALLBACK_MODELS = [m.strip() for m in _FALLBACK_MODELS if m.strip()]


class FallbackLLM(BaseChatModel):
    """LLM wrapper with automatic fallback to alternate models on failure.

    If the primary model returns an error (rate limit, timeout, server error),
    retries with each fallback model in order.
    """

    primary: BaseChatModel
    fallbacks: list[BaseChatModel]
    model_names: list[str]

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "fallback_llm"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any):
        """Try primary, then each fallback in order."""
        all_models = [self.primary] + self.fallbacks
        last_error = None

        for i, model in enumerate(all_models):
            model_name = self.model_names[i] if i < len(self.model_names) else f"fallback_{i}"
            try:
                result = model._generate(messages, stop=stop, **kwargs)
                if i > 0:
                    logger.info("Fallback succeeded: using %s", model_name)
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    "Model %s failed (%s), trying next fallback...",
                    model_name, type(e).__name__,
                )
                continue

        raise last_error  # type: ignore[misc]

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_names": self.model_names}


def create_llm(settings: Settings) -> BaseChatModel:
    """
    Create an LLM instance based on the configured provider.

    If LLM_FALLBACK_MODELS is set, wraps the primary model in a FallbackLLM
    that automatically retries with alternate models on failure.

    :param settings: Application settings with LLM config.
    :return: A LangChain-compatible chat model.
    :raises ImportError: If the required package for the provider is not installed.
    :raises ValueError: If the provider is not recognized.
    """
    provider = settings.llm_provider

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        logger.info("Using OpenAI provider: model=%s", settings.openai_model)
        primary = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key,
        )

        # Build fallback chain if configured
        if _FALLBACK_MODELS:
            fallbacks = []
            model_names = [settings.openai_model]
            for fallback_model in _FALLBACK_MODELS:
                fallbacks.append(ChatOpenAI(
                    model=fallback_model,
                    temperature=settings.openai_temperature,
                    api_key=settings.openai_api_key,
                ))
                model_names.append(fallback_model)
            logger.info("Fallback chain: %s", " → ".join(model_names))
            return FallbackLLM(
                primary=primary,
                fallbacks=fallbacks,
                model_names=model_names,
            )

        return primary

    elif provider == "huggingface":
        try:
            from langchain_community.chat_models import ChatLlamaCpp
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for the huggingface provider. "
                "Install it with: pip install llama-cpp-python"
            ) from e

        from huggingface_hub import hf_hub_download

        logger.info(
            "Using HuggingFace local provider: model=%s, file=%s",
            settings.hf_model_id, settings.hf_model_file,
        )

        # Download GGUF model from HuggingFace Hub (cached after first download)
        model_path = hf_hub_download(
            repo_id=settings.hf_model_id,
            filename=settings.hf_model_file,
        )
        logger.info("Model path: %s", model_path)

        return ChatLlamaCpp(
            model_path=model_path,
            n_ctx=settings.hf_n_ctx,
            n_threads=settings.hf_n_threads,
            max_tokens=settings.hf_max_tokens,
            temperature=settings.openai_temperature,
            verbose=False,
        )

    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Supported: 'openai', 'huggingface'.")
