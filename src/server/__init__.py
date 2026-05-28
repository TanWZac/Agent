"""Server package — HTTP API and MCP server."""

from src.server.api import app, start_server

__all__ = ["app", "start_server"]
