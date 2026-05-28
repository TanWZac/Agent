"""Plugin tool registry — user-extensible tools via YAML configuration.

Allows users to register custom tools without modifying Python code.
Tools are defined in a YAML file and loaded at startup.

Configuration (env vars):
    PLUGIN_TOOLS_FILE: Path to YAML tool definitions (default: "config/tools.yml")

YAML Format::

    tools:
      - name: "calculate"
        description: "Perform a mathematical calculation"
        type: "python_eval"  # safe eval for math
        params:
          - name: "expression"
            type: "string"
            description: "Math expression to evaluate"

      - name: "fetch_url"
        description: "Fetch content from a URL"
        type: "http_get"
        params:
          - name: "url"
            type: "string"
            description: "URL to fetch"
        config:
          timeout: 10
          max_size_kb: 512
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from src.core.logging import get_logger

logger = get_logger("agent.plugins")

_PLUGIN_TOOLS_FILE = os.getenv("PLUGIN_TOOLS_FILE", "config/tools.yml")

# Safe math operations (no exec/eval of arbitrary code)
_SAFE_MATH_CHARS = re.compile(r"^[\d\s\+\-\*/\.\(\)\^%]+$")


def _safe_math_eval(expression: str) -> str:
    """Evaluate a safe mathematical expression (no code execution)."""
    expr = expression.strip()
    if not _SAFE_MATH_CHARS.match(expr):
        return f"Error: unsafe expression. Only numbers and operators (+, -, *, /, ^, %) are allowed."
    # Replace ^ with ** for Python exponentiation
    expr = expr.replace("^", "**")
    try:
        # Use compile + eval with no builtins for safety
        code = compile(expr, "<math>", "eval")
        # Ensure no names are used (only literals and operators)
        if code.co_names:
            return "Error: expressions with variable names are not allowed."
        result = eval(code, {"__builtins__": {}}, {})
        return str(result)
    except (SyntaxError, ZeroDivisionError, TypeError, ValueError) as e:
        return f"Error: {e}"


def _http_get_tool(url: str, timeout: int = 10, max_size_kb: int = 512) -> str:
    """Fetch content from a URL with size limits."""
    import urllib.request
    import urllib.error

    # Basic URL validation
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NotepadAgent/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read(max_size_kb * 1024)
            return content.decode("utf-8", errors="replace")[:max_size_kb * 1024]
    except urllib.error.URLError as e:
        return f"Error fetching URL: {e.reason}"
    except Exception as e:
        return f"Error: {e}"


class PluginToolRegistry:
    """Registry for user-defined plugin tools loaded from YAML config."""

    def __init__(self, config_path: str | None = None) -> None:
        self._tools: list = []
        self._config_path = Path(config_path or _PLUGIN_TOOLS_FILE)

    def load(self) -> list:
        """Load plugin tools from YAML config. Returns empty list if file doesn't exist."""
        if not self._config_path.exists():
            logger.debug("No plugin tools file at %s", self._config_path)
            return []

        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed — plugin tools disabled. Install with: pip install pyyaml")
            return []

        try:
            with self._config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error("Failed to load plugin tools: %s", e)
            return []

        if not config or "tools" not in config:
            return []

        for tool_def in config["tools"]:
            built_tool = self._build_tool(tool_def)
            if built_tool:
                self._tools.append(built_tool)
                logger.info("Loaded plugin tool: %s", tool_def["name"])

        return self._tools

    def _build_tool(self, tool_def: dict[str, Any]):
        """Build a LangChain tool from a YAML definition."""
        name = tool_def.get("name", "")
        description = tool_def.get("description", "")
        tool_type = tool_def.get("type", "")
        config = tool_def.get("config", {})

        if not name or not tool_type:
            logger.warning("Skipping invalid tool definition: %s", tool_def)
            return None

        if tool_type == "python_eval":
            @tool(name, description=description)
            def math_tool(expression: str) -> str:
                """Evaluate a mathematical expression safely."""
                return _safe_math_eval(expression)
            return math_tool

        elif tool_type == "http_get":
            timeout = config.get("timeout", 10)
            max_size_kb = config.get("max_size_kb", 512)

            @tool(name, description=description)
            def http_tool(url: str) -> str:
                """Fetch content from a URL."""
                return _http_get_tool(url, timeout=timeout, max_size_kb=max_size_kb)
            return http_tool

        elif tool_type == "static_response":
            response = config.get("response", "No response configured.")

            @tool(name, description=description)
            def static_tool(query: str) -> str:
                """Return a static configured response."""
                return response
            return static_tool

        else:
            logger.warning("Unknown plugin tool type: %s (tool: %s)", tool_type, name)
            return None

    @property
    def tools(self) -> list:
        """Return loaded tools."""
        return list(self._tools)


def load_plugin_tools(config_path: str | None = None) -> list:
    """Convenience function: load and return all plugin tools."""
    registry = PluginToolRegistry(config_path)
    return registry.load()
