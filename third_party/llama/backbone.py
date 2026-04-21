from __future__ import annotations

import os
from typing import Any, Dict, List
from langchain.agents.middleware import before_model, after_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import AgentState
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI

guard = ChatOpenAI(
    model="meta-llama/llama-guard-4-12b",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0
)


def llama_guard_is_safe(text: str) -> bool:
    out = guard.invoke([HumanMessage(content=text)])
    label = (out.content or "").strip().lower()
    return label.startswith("safe")


@before_model()
def llama_guard_before_model(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    last_content = getattr(last_msg, "content", "")

    if not isinstance(last_content, str):
        return None

    if isinstance(last_msg, (HumanMessage, ToolMessage)):
        safe = llama_guard_is_safe(last_content)
        if not safe:
            state["messages"][-1].content = "[Blocked by safety filter: unsafe content removed.]"
    return None


@after_agent()
def llama_guard_after_agent(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    last_content = getattr(last_msg, "content", "")

    if not isinstance(last_content, str):
        return None

    if isinstance(last_msg, AIMessage):
        safe = llama_guard_is_safe(last_content)
        if not safe:
            state["messages"][-1].content = "[Blocked by safety filter: unsafe content removed.]"
    return None


def llama_guard_sanitize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    for msg in messages:
        content = getattr(msg, "content", "")

        should_check = False

        # Human / Tool 一律检查
        if isinstance(msg, (HumanMessage, ToolMessage)):
            should_check = True

        # AIMessage：只有“不带 tool_calls”才检查
        elif isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                should_check = True

        if should_check:
            safe = llama_guard_is_safe(content)
            if not safe:
                msg.content = "[Blocked by safety filter: unsafe content removed.]"

    return messages
