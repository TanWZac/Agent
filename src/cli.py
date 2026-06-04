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
    parser.add_argument(
        "--persona", default=None,
        help="Agent persona (e.g., default, research_assistant, customer_support, code_reviewer, technical_writer, data_analyst)",
    )
    parser.add_argument("--list-personas", action="store_true", help="List available personas and exit")
    parser.add_argument("--file", default=None, help="Upload a file for the agent to analyze (uses MarkItDown)")
    args = parser.parse_args()

    if args.list_personas:
        from src.agent.prompts import get_persona, list_personas
        for name in list_personas():
            p = get_persona(name)
            print(f"  {name:20s} — {p.description}")
        return

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
        session = AgentSession(settings=settings, persona=args.persona)
    except ConfigurationError as e:
        logger.error("Failed to initialize agent: %s", e)
        sys.exit(1)

    persona_name = args.persona or "default"
    print(f"Agent ready (persona: {persona_name}). Type 'exit' to quit.")

    # If a file was provided, ingest it first
    if args.file:
        try:
            num_chunks = session.ingest_file(args.file)
            print(f"File '{args.file}' ingested ({num_chunks} chunks stored for retrieval).")
            prompt = (
                f"I've uploaded a file '{args.file}' ({num_chunks} chunks stored in memory). "
                f"Please summarize and analyze the content. Use the retrieve_notes tool to access it."
            )
            response = session.chat(prompt)
            print(f"Assistant: {response}")
        except Exception as e:
            logger.error("File ingestion failed: %s", e)
            print(f"Error ingesting file: {e}")

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
