import json
import gradio as gr
from collections import deque
from typing import Deque, Any, Dict, Tuple, Literal
DecisionType = Literal["APPROVE", "SHADOW", "REJECT"]


def _format_args_block(args: dict) -> str:
    try:
        pretty = json.dumps(args, ensure_ascii=False, indent=2)
    except TypeError:
        pretty = str(args)
    return f"```json\n{pretty}\n```"


def render_hitl_modal(pending_calls: Deque[Dict[str, Any]] | None):
    """The HITL pop-up content is rendered based on the pending_calls_state."""
    if pending_calls is None:
        pending_calls = deque()

    if len(pending_calls) == 0:
        md = (
            "# 🛡️ SafeAgent Runtime Security Control System\n\n"
            "_No pending tool calls. The agent is running under normal protection._"
        )
        return md, gr.update(visible=False)

    item = pending_calls[0]
    pc = item.get("pending_call", {}) or {}
    name = pc.get("name", "unknown_tool")
    args = pc.get("args", {}) or {}

    lines: list[str] = [
        "# 🛡️ SafeAgent Runtime Security Control System\n",
        "**Agent plans to invoke a tool that requires review and confirmation:**\n",
        f"#### **Tool**: `{name}`\n",
        "#### **Arguments**:",
        _format_args_block(args),
    ]

    md = "\n".join(lines)
    return md, gr.update(visible=True)


def record_hitl_decision(
    pending_calls: Deque[Dict[str, Any]] | None,
    call_decisions: Deque[Dict[str, Any]] | None,
    decision: DecisionType,
) -> Tuple[Deque[Dict[str, Any]], Deque[Dict[str, Any]]]:
    """
    Consume the first call in the `pending_calls` queue and record
    the approval result up to the end of `call_decisions`.

    pending_calls: deque[{"idx": int, "pending_call": {...}, "status": "pending"}]
    call_decisions: deque[{"idx": int, "pending_call": {...}, "decision": "APPROVE"|"SHADOW"|"REJECT"}]
    decision: Approval results.
    """
    if pending_calls is None:
        pending_calls = deque()
    if call_decisions is None:
        call_decisions = deque()

    if not pending_calls:
        return pending_calls, call_decisions

    item = pending_calls.popleft()
    item["status"] = decision
    call_decisions.append(item)

    return pending_calls, call_decisions
