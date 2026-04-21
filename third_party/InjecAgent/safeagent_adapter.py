import json
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def parse_mcp_tool_response(resp: Any) -> Any:
    """
    MCP (LangChain adapter) 常见返回:
      list[{"type":"text","text":"...json...","id":"..."}]
    这里统一解析成 dict / str / 原样。
    """
    if isinstance(resp, list):
        texts: List[str] = []
        for b in resp:
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
                texts.append(b["text"])
        merged = "\n".join(texts).strip()
        if not merged:
            return resp
        try:
            return json.loads(merged)
        except json.JSONDecodeError:
            return merged

    if isinstance(resp, dict):
        return resp

    if isinstance(resp, str):
        s = resp.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s

    return resp


async def review_human_message(
    safe_agent,
    session_id: str,
    human_msg: HumanMessage,
) -> Dict[str, Any]:
    """
    Review a single HumanMessage with SafeAgent Core using the same
    request shape as the before_agent hook.
    """
    if not isinstance(human_msg, HumanMessage):
        raise TypeError("human_msg must be a HumanMessage")

    content = getattr(human_msg, "content", "")
    if not isinstance(content, str):
        raise TypeError("human_msg.content must be a string")

    core_request: Dict[str, Any] = {
        "hook": "before_agent",
        "observation": {
            "role": "user",
            "content": content,
        },
    }

    decision_raw = await safe_agent.ainvoke(
        {
            "session_id": session_id,
            "core_request": core_request,
        }
    )
    decision = parse_mcp_tool_response(decision_raw)
    return decision


async def review_ai_message_after_agent(
    safe_agent,
    session_id: str,
    ai_msg: AIMessage,
) -> Dict[str, Any]:
    """
    Review a single AIMessage without tool calls using the same
    request shape as the after_agent hook.
    """
    if not isinstance(ai_msg, AIMessage):
        raise TypeError("ai_msg must be an AIMessage")

    content = getattr(ai_msg, "content", "")
    if not isinstance(content, str):
        raise TypeError("ai_msg.content must be a string")

    tool_calls = getattr(ai_msg, "tool_calls", None)
    if tool_calls:
        raise ValueError("review_ai_message_after_agent only supports AIMessage without tool_calls")

    core_request: Dict[str, Any] = {
        "hook": "after_agent",
        "observation": {
            "role": "assistant",
            "content": content,
        },
    }

    decision_raw = await safe_agent.ainvoke(
        {
            "session_id": session_id,
            "core_request": core_request,
        }
    )
    decision = parse_mcp_tool_response(decision_raw)
    return decision


async def review_ai_message_after_model(
    safe_agent,
    session_id: str,
    ai_msg: AIMessage,
    last_user_content: str,
) -> Dict[str, Any]:
    """
    Review a single AIMessage with tool calls using the same
    request shape as the after_model hook.

    Parameters
    ----------
    last_user_content : str
        The latest user message content to send as `last_user`.
    """
    if not isinstance(ai_msg, AIMessage):
        raise TypeError("ai_msg must be an AIMessage")

    content = getattr(ai_msg, "content", "")
    if not isinstance(content, str):
        raise TypeError("ai_msg.content must be a string")

    if not isinstance(last_user_content, str):
        raise TypeError("last_user_content must be a string")

    tool_calls = getattr(ai_msg, "tool_calls", None)
    if not tool_calls:
        raise ValueError("review_ai_message_after_model only supports AIMessage with tool_calls")

    core_request: Dict[str, Any] = {
        "hook": "after_model",
        "observation": {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
            "last_user": last_user_content,
        },
    }

    decision_raw = await safe_agent.ainvoke(
        {
            "session_id": session_id,
            "core_request": core_request,
        }
    )
    decision = parse_mcp_tool_response(decision_raw)
    return decision


async def review_tool_message_before_model(
    safe_agent,
    session_id: str,
    tool_msg: ToolMessage,
) -> Dict[str, Any]:
    """
    Review a single ToolMessage using the same request shape
    as the before_model hook.
    """
    if not isinstance(tool_msg, ToolMessage):
        raise TypeError("tool_msg must be a ToolMessage")

    content = getattr(tool_msg, "content", "")
    if not isinstance(content, str):
        raise TypeError("tool_msg.content must be a string")

    name = getattr(tool_msg, "name", None)
    tool_call_id = getattr(tool_msg, "tool_call_id", None)

    core_request: Dict[str, Any] = {
        "hook": "before_model",
        "observation": {
            "role": "tool",
            "name": name,
            "tool_call_id": tool_call_id,
            "content": content,
        },
    }

    decision_raw = await safe_agent.ainvoke(
        {
            "session_id": session_id,
            "core_request": core_request,
        }
    )
    decision = parse_mcp_tool_response(decision_raw)
    return decision


async def process_triplet_with_safeagent(
    safe_agent,
    session_id: str,
    human_msg: HumanMessage,
    ai_msg: AIMessage,
    tool_msg: Optional[ToolMessage] = None,
    round: int = 0
) -> Dict[str, Any]:
    """
    Process one InjecAgent-style triplet through SafeAgent.

    Policy
    ------
    - APPROVE: continue
    - OVERRIDE: replace current message content and continue
    - any other action:
        - mark blocked=True
        - replace current message content with a safe fallback
        - return immediately with the current sanitized context

    Returns
    -------
    Dict[str, Any]
        {
            "blocked": bool,
            "blocked_at": str | None,   # "human" | "ai" | "tool" | None
            "messages": list,           # current sanitized context
            "human_msg": HumanMessage,
            "ai_msg": AIMessage,
            "tool_msg": ToolMessage | None,
            "human_decision": dict | None,
            "ai_decision": dict | None,
            "tool_decision": dict | None,
        }
    """
    if not isinstance(human_msg, HumanMessage):
        raise TypeError("human_msg must be a HumanMessage")
    if not isinstance(ai_msg, AIMessage):
        raise TypeError("ai_msg must be an AIMessage")
    if tool_msg is not None and not isinstance(tool_msg, ToolMessage):
        raise TypeError("tool_msg must be a ToolMessage or None")

    human_decision: Optional[Dict[str, Any]] = None
    ai_decision: Optional[Dict[str, Any]] = None
    tool_decision: Optional[Dict[str, Any]] = None

    def _action(decision: Dict[str, Any]) -> str:
        return str(decision.get("action", "")).upper().strip()

    def _replace_content(msg, new_text: str) -> None:
        msg.content = new_text

    def _apply_override_or_fallback(msg, decision: Dict[str, Any], fallback_text: str) -> None:
        override_text = decision.get("override")
        if isinstance(override_text, str) and override_text:
            _replace_content(msg, override_text)
        else:
            _replace_content(msg, fallback_text)

    def _current_messages():
        messages = [human_msg, ai_msg]
        if tool_msg is not None:
            messages.append(tool_msg)
        return messages

    # ------------------------------------------------------------------
    # Step 1: HumanMessage -> before_agent
    # ------------------------------------------------------------------
    if round == 0:
        human_decision = await review_human_message(
            safe_agent=safe_agent,
            session_id=session_id,
            human_msg=human_msg,
        )
        human_action = _action(human_decision)

        if human_action == "APPROVE":
            pass
        elif human_action == "OVERRIDE":
            _apply_override_or_fallback(
                human_msg,
                human_decision,
                "[SafeAgent Controller] User input has been sanitized.",
            )
        else:
            _apply_override_or_fallback(
                human_msg,
                human_decision,
                "[SafeAgent Controller] User input has been blocked and replaced by a safe version.",
            )
            return {
                "blocked": True,
                "blocked_at": "human",
                "messages": _current_messages(),
                "human_msg": human_msg,
                "ai_msg": ai_msg,
                "tool_msg": tool_msg,
                "human_decision": human_decision,
                "ai_decision": None,
                "tool_decision": None,
            }

    # ------------------------------------------------------------------
    # Step 2: AIMessage
    #   - with tool_calls    -> after_model
    #   - without tool_calls -> after_agent
    # ------------------------------------------------------------------
    tool_calls = getattr(ai_msg, "tool_calls", None)

    if tool_calls:
        ai_decision = await review_ai_message_after_model(
            safe_agent=safe_agent,
            session_id=session_id,
            ai_msg=ai_msg,
            last_user_content=human_msg.content,
        )
    else:
        ai_decision = await review_ai_message_after_agent(
            safe_agent=safe_agent,
            session_id=session_id,
            ai_msg=ai_msg,
        )

    ai_action = _action(ai_decision)

    if ai_action == "APPROVE":
        pass
    elif ai_action == "OVERRIDE":
        _apply_override_or_fallback(
            ai_msg,
            ai_decision,
            "[SafeAgent Controller] Assistant output has been sanitized.",
        )
    else:
        _apply_override_or_fallback(
            ai_msg,
            ai_decision,
            "[SafeAgent Controller] Assistant output or tool plan has been blocked and replaced by a safe version.",
        )
        return {
            "blocked": True,
            "blocked_at": "ai",
            "messages": _current_messages(),
            "human_msg": human_msg,
            "ai_msg": ai_msg,
            "tool_msg": tool_msg,
            "human_decision": human_decision,
            "ai_decision": ai_decision,
            "tool_decision": None,
        }

    # ------------------------------------------------------------------
    # Step 3: ToolMessage -> before_model
    # Only if AI planned tools and tool_msg exists
    # ------------------------------------------------------------------
    if tool_calls and tool_msg is not None:
        tool_decision = await review_tool_message_before_model(
            safe_agent=safe_agent,
            session_id=session_id,
            tool_msg=tool_msg,
        )
        tool_action = _action(tool_decision)

        if tool_action == "APPROVE":
            pass
        elif tool_action == "OVERRIDE":
            _apply_override_or_fallback(
                tool_msg,
                tool_decision,
                "[SafeAgent Controller] Tool output has been sanitized.",
            )
        else:
            _apply_override_or_fallback(
                tool_msg,
                tool_decision,
                "[SafeAgent Controller] Tool output has been blocked and replaced by a safe version.",
            )
            return {
                "blocked": True,
                "blocked_at": "tool",
                "messages": _current_messages(),
                "human_msg": human_msg,
                "ai_msg": ai_msg,
                "tool_msg": tool_msg,
                "human_decision": human_decision,
                "ai_decision": ai_decision,
                "tool_decision": tool_decision,
            }

    return {
        "blocked": False,
        "blocked_at": None,
        "messages": _current_messages(),
        "human_msg": human_msg,
        "ai_msg": ai_msg,
        "tool_msg": tool_msg,
        "human_decision": human_decision,
        "ai_decision": ai_decision,
        "tool_decision": tool_decision,
    }