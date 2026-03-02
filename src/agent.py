from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.notepad_rag import NotepadRAG


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_agent(note_file: str = "data/notepad.txt"):
    notepad = NotepadRAG(note_file=note_file)
    web_search = DuckDuckGoSearchResults(max_results=5)

    @tool
    def search_web(query: str) -> str:
        """Search the web for recent information."""
        return web_search.run(query)

    @tool
    def save_note(note: str) -> str:
        """Save an important fact into persistent notepad memory."""
        notepad.append(note)
        return "Saved to notepad."

    @tool
    def retrieve_notes(question: str) -> str:
        """Retrieve relevant notes from the persistent notepad memory."""
        hits = notepad.retrieve(question, k=3)
        if not hits:
            return "No relevant notes found."
        return "\n".join([f"- {h.text} (score={h.score:.3f})" for h in hits])

    tools = [search_web, save_note, retrieve_notes]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    system_prompt = (
        "You are a helpful assistant. Use tools when needed. "
        "Always check retrieve_notes for user-specific context before answering factual questions. "
        "When user shares durable preferences or facts, call save_note to store them."
    )

    def call_model(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("assistant", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "assistant")

    return graph.compile(), notepad


def persist_turn(notepad: NotepadRAG, user_text: str, assistant_text: str) -> None:
    notepad.append(f"USER: {user_text}")
    notepad.append(f"ASSISTANT: {assistant_text}")
