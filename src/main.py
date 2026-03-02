from __future__ import annotations

import argparse

from langchain_core.messages import HumanMessage

from src.agent import build_agent, persist_turn


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph notepad + websearch agent")
    parser.add_argument("--note-file", default="data/notepad.txt", help="Path to the persistent note file")
    args = parser.parse_args()

    graph, notepad = build_agent(note_file=args.note_file)

    print("Agent ready. Type 'exit' to quit.")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        result = graph.invoke({"messages": [HumanMessage(content=user_text)]})
        assistant_msg = result["messages"][-1]
        assistant_text = getattr(assistant_msg, "content", str(assistant_msg))

        print(f"Assistant: {assistant_text}")
        persist_turn(notepad, user_text=user_text, assistant_text=assistant_text)


if __name__ == "__main__":
    main()
