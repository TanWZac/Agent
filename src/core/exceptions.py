"""
Custom exception hierarchy for the notepad-agent application.

:mod:`exceptions` defines all custom exceptions for agent errors, configuration, storage, and LLM failures.
"""

from __future__ import annotations


class AgentError(Exception):
    """
    Base exception for all agent errors.
    """


class ConfigurationError(AgentError):
    """
    Raised when configuration is missing or invalid.
    """


class NotepadError(AgentError):
    """
    Raised for notepad storage-related failures.
    """


class NotepadFullError(NotepadError):
    """
    Raised when the notepad file exceeds max allowed size.
    """


class RetrievalError(AgentError):
    """
    Raised when note retrieval fails.
    """


class ToolExecutionError(AgentError):
    """
    Raised when an agent tool fails during execution.
    """


class LLMError(AgentError):
    """
    Raised when the LLM API call fails.
    """
