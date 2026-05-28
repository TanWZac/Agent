"""
LLM factory — creates the appropriate LLM instance based on provider config.

:mod:`llm` provides a factory for instantiating LangChain-compatible LLMs based on configuration.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.config import Settings
from src.core.logging import get_logger

logger = get_logger("agent.llm")


def create_llm(settings: Settings) -> BaseChatModel:
    """
    Create an LLM instance based on the configured provider.

    :param settings: Application settings with LLM config.
    :return: A LangChain-compatible chat model.
    :raises ImportError: If the required package for the provider is not installed.
    :raises ValueError: If the provider is not recognized.
    """
    provider = settings.llm_provider

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        logger.info("Using OpenAI provider: model=%s", settings.openai_model)
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key,
        )

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
