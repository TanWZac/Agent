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
    settings = get_settings(openai_api_key="")
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        settings.validate()


def test_settings_validate_bad_temperature():
    settings = get_settings(openai_api_key="test", openai_temperature=3.0)
    with pytest.raises(ConfigurationError, match="OPENAI_TEMPERATURE"):
        settings.validate()


def test_settings_override():
    settings = get_settings(openai_api_key="key", note_file="/tmp/custom.txt")
    assert settings.note_file == "/tmp/custom.txt"
