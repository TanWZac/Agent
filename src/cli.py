"""CLI entry point for the notepad-agent."""

from __future__ import annotations

import argparse
import sys

from src.agent import AgentSession
from src.config import get_settings
from src.core.exceptions import AgentError, ConfigurationError
from src.core.logging import get_logger, setup_logging

logger = get_logger("cli")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph notepad + websearch agent")
    parser.add_argument("--note-file", default=None, help="Path to the persistent note file")
    parser.add_argument("--serve", action="store_true", help="Start as HTTP API server")
    parser.add_argument("--log-level", default=None, help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    overrides: dict = {}
    if args.note_file:
        overrides["note_file"] = args.note_file
    if args.log_level:
        overrides["log_level"] = args.log_level

    settings = get_settings(**overrides)
    setup_logging(settings.log_level)

    try:
        settings.validate()
    except ConfigurationError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    if args.serve:
        from src.server.api import start_server

        start_server()
        return

    # Interactive CLI mode
    try:
        session = AgentSession(settings=settings)
    except ConfigurationError as e:
        logger.error("Failed to initialize agent: %s", e)
        sys.exit(1)

    print("Agent ready. Type 'exit' to quit.")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_text.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        if not user_text:
            continue

        try:
            response = session.chat(user_text)
            print(f"Assistant: {response}")
        except AgentError as e:
            logger.error("Agent error: %s", e)
            print(f"Error: {e}. Please try again.")


if __name__ == "__main__":
    main()
