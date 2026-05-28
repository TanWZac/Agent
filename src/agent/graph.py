"""LangGraph state graph construction for the agent."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.llm import create_llm
from src.config import Settings
from src.core.exceptions import LLMError
from src.core.logging import get_logger

logger = get_logger("agent.graph")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools when needed. "
    "Always check retrieve_notes for user-specific context before answering factual questions. "
    "When user shares durable preferences or facts, call save_note to store them."
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(settings: Settings, tools: list):
    """Construct and compile the LangGraph state graph.

    Args:
        settings: App settings (model config, provider selection).
        tools: List of @tool-decorated callables.

    Returns:
        Compiled LangGraph runnable.
    """
    base_llm = create_llm(settings)

    # Only bind tools if the model supports it (OpenAI does natively;
    # local models may not support tool-calling via bind_tools)
    if hasattr(base_llm, "bind_tools"):
        try:
            llm = base_llm.bind_tools(tools)
        except NotImplementedError:
            logger.warning("LLM does not support bind_tools; tools will be described in prompt")
            llm = base_llm
    else:
        llm = base_llm

    def call_model(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        try:
            response = llm.invoke(messages)
        except Exception as e:
            logger.error("LLM invocation failed: %s", e)
            raise LLMError(f"LLM call failed: {e}") from e
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("assistant", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "assistant")

    return graph.compile()
