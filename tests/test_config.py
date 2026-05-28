"""Tests for configuration module."""

import os
import pytest

from src.config import Settings, get_settings
from src.core.exceptions import ConfigurationError


def test_settings_default_values():
    settings = get_settings(openai_api_key="test-key")
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.openai_temperature == 0
    assert settings.retrieval_top_k == 3


def test_settings_validate_missing_key():
    settings = get_settings(openai_api_key="", llm_provider="openai")
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        settings.validate()


def test_settings_validate_bad_temperature():
    settings = get_settings(openai_api_key="test", openai_temperature=3.0)
    with pytest.raises(ConfigurationError, match="OPENAI_TEMPERATURE"):
        settings.validate()


def test_settings_override():
    settings = get_settings(openai_api_key="key", note_file="/tmp/custom.txt")
    assert settings.note_file == "/tmp/custom.txt"


def test_settings_huggingface_provider_no_key_needed():
    settings = get_settings(llm_provider="huggingface", openai_api_key="")
    # Should NOT raise — HuggingFace doesn't need an OpenAI key
    settings.validate()
