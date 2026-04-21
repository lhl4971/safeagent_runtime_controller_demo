from __future__ import annotations

from typing import Any, Dict, List
from langchain.agents.middleware import before_model, after_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import AgentState
from langgraph.runtime import Runtime

from llm_guard import scan_prompt, scan_output
from llm_guard.vault import Vault
from llm_guard.input_scanners import (
    Anonymize, BanTopics, InvisibleText, PromptInjection, Regex, Secrets, TokenLimit, Toxicity
)
from llm_guard.output_scanners import (
    BanTopics as OutputBanTopics, Bias, MaliciousURLs,
    Regex as OutputRegex, Relevance, Sensitive, Toxicity as OutputToxicity
)

input_scanners = [
    PromptInjection(),
    InvisibleText(),
    Regex(patterns=["Bearer [A-Za-z0-9-._~+/]+", "sk-[A-Za-z0-9]{20,}", "AKIA[0-9A-Z]{16}"]),
    TokenLimit(limit=4096),
    BanTopics(topics=["violence", "self-harm", "hate", "sexual"]),
]

output_scanners = [
    Sensitive(),
    OutputRegex(patterns=["Bearer [A-Za-z0-9-._~+/]+", "sk-[A-Za-z0-9]{20,}", "AKIA[0-9A-Z]{16}"]),
    MaliciousURLs(),
    OutputBanTopics(topics=["violence", "self-harm", "hate", "sexual"]),
]


def last_message_index(state: AgentState, MessageType: type) -> int | None:
    messages = state.get("messages", [])
    if not messages:
        return None

    idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], MessageType):
            idx = i
            break

    if idx is None:
        return None

    return idx


def guard_prompt(prompt: str) -> str:
    sanitized_prompt, results_valid, _ = scan_prompt(
        input_scanners,
        prompt,
        fail_fast=False
    )

    if any(valid is False for valid in results_valid.values()):
        return "[Blocked by safety filter: unsafe content removed.]"

    return sanitized_prompt


def guard_output(prompt: str, output: str) -> str:
    sanitized_output, results_valid, _ = scan_output(
        output_scanners,
        prompt,
        output,
        fail_fast=False
    )

    if any(valid is False for valid in results_valid.values()):
        return "[Blocked by safety filter: unsafe content removed.]"

    return sanitized_output


@before_model()
def llm_guard_before_model(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    last_content = getattr(last_msg, "content", "")

    if not isinstance(last_content, str):
        return None

    if isinstance(last_msg, (HumanMessage, ToolMessage)):
        state["messages"][-1].content = guard_prompt(last_content)
    return None


@after_agent()
def llm_guard_after_agent(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    last_content = getattr(last_msg, "content", "")

    if not isinstance(last_content, str):
        return None

    last_human_idx = last_message_index(state, HumanMessage)
    last_human_msg = messages[last_human_idx]
    last_human_content = getattr(last_human_msg, "content", "")

    if not isinstance(last_human_content, str):
        return None

    if isinstance(last_msg, AIMessage):
        state["messages"][-1].content = guard_output(last_human_content, last_content)
    return None


def llm_guard_sanitize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    last_human_content: str | None = None

    for msg in messages:
        content = getattr(msg, "content", "")

        # 1) HumanMessage: guard_prompt
        if isinstance(msg, HumanMessage):
            msg.content = guard_prompt(content)
            last_human_content = content
            continue

        # 2) ToolMessage: guard_prompt
        if isinstance(msg, ToolMessage):
            msg.content = guard_prompt(content)
            continue

        # 3) AIMessage:
        #    - 有 tool_calls: 跳过
        #    - 无 tool_calls: 用最近一条 human content 做 guard_output
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                continue
            if last_human_content is not None:
                msg.content = guard_output(last_human_content, content)
            continue

    return messages
