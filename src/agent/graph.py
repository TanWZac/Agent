"""LangGraph state graph construction for the agent."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.llm import create_llm
from src.config import Settings
from src.core.exceptions import LLMError
from src.core.logging import get_logger
from src.responsible_ai import Guardrails
from src.responsible_ai.config import RAIConfig

logger = get_logger("agent.graph")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools when needed. "
    "Always check retrieve_notes for user-specific context before answering factual questions. "
    "When user shares durable preferences or facts, call save_note to store them. "
    "You follow responsible AI principles: be helpful, harmless, and honest. "
    "Do not generate harmful, misleading, or biased content. "
    "Respect user privacy and do not request or store unnecessary personal information."
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str


def build_graph(settings: Settings, tools: list, rai_config: RAIConfig | None = None):
    """Construct and compile the LangGraph state graph.

    Args:
        settings: App settings (model config, provider selection).
        tools: List of @tool-decorated callables.
        rai_config: Optional Responsible AI configuration.

    Returns:
        Compiled LangGraph runnable.
    """
    base_llm = create_llm(settings)
    guardrails = Guardrails(config=rai_config)

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

    def input_guardrail(state: AgentState) -> AgentState:
        """Check user input against responsible AI guardrails."""
        messages = state["messages"]
        if not messages:
            return state

        last_message = messages[-1]
        user_text = getattr(last_message, "content", "")
        if not user_text:
            return state

        session_id = state.get("session_id", "")
        result = guardrails.check_input(user_text, session_id=session_id)

        if not result.allowed:
            # Replace the pipeline with a blocked response
            blocked_msg = AIMessage(content=result.blocked_reason)
            return {"messages": [blocked_msg], "session_id": session_id}

        # If PII was redacted, update the message content for LLM processing
        if result.processed_text != user_text:
            from langchain_core.messages import HumanMessage
            messages = list(messages[:-1]) + [HumanMessage(content=result.processed_text)]
            return {"messages": messages, "session_id": session_id}

        return state

    def input_was_blocked(state: AgentState) -> str:
        """Route based on whether input was blocked by guardrails."""
        messages = state["messages"]
        if messages and hasattr(messages[-1], "type") and messages[-1].type == "ai":
            return "blocked"
        return "allowed"

    def call_model(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        try:
            response = llm.invoke(messages)
        except Exception as e:
            logger.error("LLM invocation failed: %s", e)
            raise LLMError(f"LLM call failed: {e}") from e
        return {"messages": [response]}

    def output_guardrail(state: AgentState) -> AgentState:
        """Check assistant output against responsible AI guardrails."""
        messages = state["messages"]
        if not messages:
            return state

        last_message = messages[-1]
        if not hasattr(last_message, "type") or last_message.type != "ai":
            return state

        assistant_text = getattr(last_message, "content", "")
        if not assistant_text:
            return state

        session_id = state.get("session_id", "")
        result = guardrails.check_output(assistant_text, session_id=session_id)

        if not result.allowed or result.processed_text != assistant_text:
            modified_msg = AIMessage(content=result.processed_text)
            return {"messages": list(messages[:-1]) + [modified_msg], "session_id": session_id}

        return state

    def should_end_or_continue(state: AgentState):
        """Determine if we should end, use tools, or continue."""
        messages = state["messages"]
        if not messages:
            return END
        last = messages[-1]
        # If it's an AI message with tool calls, route to tools
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("input_guardrail", input_guardrail)
    graph.add_node("assistant", call_model)
    graph.add_node("output_guardrail", output_guardrail)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        input_was_blocked,
        {"blocked": "output_guardrail", "allowed": "assistant"},
    )
    graph.add_conditional_edges(
        "assistant",
        should_end_or_continue,
        {"tools": "tools", END: "output_guardrail"},
    )
    graph.add_edge("tools", "assistant")
    graph.add_edge("output_guardrail", END)

    return graph.compile()
