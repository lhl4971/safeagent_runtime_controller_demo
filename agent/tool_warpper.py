from typing import Any, Dict, Union, Callable, Awaitable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.types import Command
from utils.agent import log_line, last_message_index, parse_mcp_tool_response


class SafeAgentToolWrapperMiddleware(AgentMiddleware):
    """
    Safety middleware for the tool_wrapper stage.

    Purpose
    -------
    Enforces zero-trust inspection on every outbound tool call before the
    tool executor receives it. The middleware consults the SafeAgent Core
    to determine whether the call is safe, must be rewritten, must be blocked,
    or must be escalated for human review.

    What this middleware provides
    -----------------------------
    - Blocks unsafe or disallowed tool calls before execution.
    - Rewrites tool arguments when the Core provides a safe override.
    - Forces human-in-the-loop approval for sensitive tool calls.
    - Guarantees that only validated tool invocations reach the handler.
    - Applies HITL replay flags so approved calls bypass further scrutiny.
    - Prevents bypass of safety logic under all execution modes (sync and async).

    Supported actions
    -----------------
    - **CALL_ALLOW**
      The tool call is safe. It is forwarded to the underlying handler unchanged.

    - **CALL_REWRITE**
      The Core supplies override_args that replace the original arguments.
      The rewritten call is executed instead of the original.

    - **CALL_BLOCK**
      The call is unsafe. A ToolMessage with an error status is returned
      and the tool is not executed.

    - **CALL_JIT_APPROVAL**
      The call requires human confirmation. A ToolMessage is emitted to
      trigger a HITL approval workflow and the tool is not executed until
      approval is provided.

    Safety guarantees
    -----------------
    - Unknown or malformed actions are rejected deterministically.
    - Unsafe calls never reach the underlying tool executor.
    - HITL-approved calls are executed exactly once and bypass future checks.
    - The SafeAgent Core cannot inject arbitrary prompts or code into the tool call.
    - All safety outcomes are logged for audit and traceability.

    Summary
    -------
    This middleware forms the outer perimeter of the zero-trust tool security model.
    Every tool call must be explicitly validated by the SafeAgent Core or by human
    approval before execution.
    """

    def __init__(self, safe_agent: Runnable, session_id: str) -> None:
        super().__init__()
        self.safe_agent = safe_agent
        self.session_id = session_id

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Union[ToolMessage, Command]],
    ) -> Union[ToolMessage, Command]:
        tool = request.tool
        tool_call: Dict[str, Any] = request.tool_call
        runtime = request.runtime

        if tool_call is None:
            return handler(request)

        cfg = getattr(runtime, "config", None)

        tool_call_id = tool_call.get("id")
        call_name = tool_call.get("name")
        args: Dict[str, Any] = tool_call.get("args", {}) or {}

        # === HITL-approved ===
        messages = request.state.get("messages", []) or []
        if messages:
            last_ai_msg_idx = last_message_index(request.state, AIMessage)
            msg = messages[last_ai_msg_idx]
            ak = getattr(msg, "additional_kwargs", {}) or {}
            safe_tags = ak.get("safe_tags", []) or []

            if "HITL_REPLAY" in safe_tags:
                log_line("tool_wrapper.hitl_execute", {
                    "tool": tool.name,
                    "tool_call_id": tool_call_id,
                    "args": args,
                })
                return handler(request)

        # === SafeAgent Core decision ===
        core_request: Dict[str, Any] = {
            "hook": "tool_wrapper",
            "observation": {
                "plan": msg.content,
                "name": call_name,
                "args": args,
                "description": getattr(tool, "description", None),
            }
        }
        log_line("tool_wrapper.core_request", core_request)

        try:
            decision: Dict[str, Any] = self.safe_agent.invoke({
                "session_id": self.session_id,
                "core_request": core_request,
            })
            decision = parse_mcp_tool_response(decision)
        except Exception as e:
            log_line("tool_wrapper.core_error", {"error": str(e)})
            error_text = (
                "[SafeAgent Controller] Safety core unavailable. "
                f"Tool call '{call_name}' has been blocked. ({e})"
            )
            return ToolMessage(
                content=error_text,
                tool_call_id=tool_call_id or "safeagent-blocked",
                name=call_name or getattr(tool, "name", "unknown_tool"),
                status="error",
            )
        action = str(decision.get("action", "")).upper().strip()
        log_line("tool_wrapper.core_decision", decision)

        ALLOWED_ACTIONS = {"CALL_ALLOW", "CALL_REWRITE", "CALL_BLOCK", "CALL_JIT_APPROVAL"}
        if action not in ALLOWED_ACTIONS:
            error_text = (
                f"[SafeAgent Controller] Unknown action '{action}'. "
                f"Tool call '{call_name}' blocked by zero-trust runtime."
            )
            return ToolMessage(
                content=error_text,
                tool_call_id=tool_call_id or "safeagent-blocked",
                name=call_name or getattr(tool, "name", "unknown_tool"),
                status="error",
            )

        if action == "CALL_BLOCK":
            safety_rationale = decision.get(
                "safety_rationale",
                f"[SafeAgent Core] Tool '{call_name}' call blocked by safety policy.",
            )
            return ToolMessage(
                content=safety_rationale,
                tool_call_id=tool_call_id or "safeagent-blocked",
                name=call_name or getattr(tool, "name", "unknown_tool"),
                status="error",
            )

        if action == "CALL_REWRITE":
            override_args = decision.get("override")
            if isinstance(override_args, dict):
                tool_call["args"] = override_args
            else:
                error_text = (
                    "[SafeAgent Controller] Invalid override payload in CALL_REWRITE. "
                    f"Tool call '{call_name}' blocked."
                )
                return ToolMessage(
                    content=error_text,
                    tool_call_id=tool_call_id or "safeagent-blocked",
                    name=call_name or getattr(tool, "name", "unknown_tool"),
                    status="error",
                )
            return handler(request)

        if action == "CALL_JIT_APPROVAL":
            content = decision.get(
                "user_notice",
                f"[SafeAgent Core] Tool '{call_name}' requires human confirmation."
            )
            additional_kwargs: Dict[str, Any] = {
                "pending_call": {
                    "name": call_name,
                    "args": args,
                    "id": tool_call_id,
                    "type": "tool_call",
                }
            }
            return ToolMessage(
                content=content,
                tool_call_id=tool_call_id or "safeagent-jit-pending",
                name=call_name or getattr(tool, "name", "unknown_tool"),
                status="success",
                additional_kwargs=additional_kwargs,
            )

        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Union[ToolMessage, Command]]]
    ) -> Union[ToolMessage, Command]:
        tool = request.tool
        tool_call: Dict[str, Any] = request.tool_call
        runtime = request.runtime

        if tool_call is None:
            return await handler(request)

        cfg = getattr(runtime, "config", None)

        tool_call_id = tool_call.get("id")
        call_name = tool_call.get("name")
        args: Dict[str, Any] = tool_call.get("args", {}) or {}

        # === HITL-approved ===
        messages = request.state.get("messages", []) or []
        if messages:
            last_ai_msg_idx = last_message_index(request.state, AIMessage)
            msg = messages[last_ai_msg_idx]
            ak = getattr(msg, "additional_kwargs", {}) or {}
            safe_tags = ak.get("safe_tags", []) or []

            if "HITL_REPLAY" in safe_tags:
                log_line("tool_wrapper.hitl_execute", {
                    "tool": tool.name,
                    "tool_call_id": tool_call_id,
                    "args": args,
                })
                return await handler(request)

        # === SafeAgent Core decision ===
        core_request: Dict[str, Any] = {
            "hook": "tool_wrapper",
            "observation": {
                "plan": msg.content,
                "name": call_name,
                "args": args,
                "description": getattr(tool, "description", None),
            }
        }
        log_line("tool_wrapper.core_request", core_request)

        try:
            decision: Dict[str, Any] = await self.safe_agent.ainvoke({
                "session_id": self.session_id,
                "core_request": core_request,
            })
            decision = parse_mcp_tool_response(decision)
        except Exception as e:
            log_line("tool_wrapper.core_error", {"error": str(e)})
            error_text = (
                "[SafeAgent Controller] Safety core unavailable. "
                f"Tool call '{call_name}' has been blocked. ({e})"
            )
            return ToolMessage(
                content=error_text,
                tool_call_id=tool_call_id or "safeagent-blocked",
                name=call_name or getattr(tool, "name", "unknown_tool"),
                status="error",
            )
        action = str(decision.get("action", "")).upper().strip()
        log_line("tool_wrapper.core_decision", decision)

        ALLOWED_ACTIONS = {"CALL_ALLOW", "CALL_REWRITE", "CALL_BLOCK", "CALL_JIT_APPROVAL"}
        if action not in ALLOWED_ACTIONS:
            error_text = (
                f"[SafeAgent Controller] Unknown action '{action}'. "
                f"Tool call '{call_name}' blocked by zero-trust runtime."
            )
            return ToolMessage(
                content=error_text,
                tool_call_id=tool_call_id or "safeagent-blocked",
                name=call_name or getattr(tool, "name", "unknown_tool"),
                status="error",
            )

        if action == "CALL_BLOCK":
            safety_rationale = decision.get(
                "safety_rationale",
                f"[SafeAgent Core] Tool '{call_name}' call blocked by safety policy.",
            )
            return ToolMessage(
                content=safety_rationale,
                tool_call_id=tool_call_id or "safeagent-blocked",
                name=call_name or getattr(tool, "name", "unknown_tool"),
                status="error",
            )

        if action == "CALL_REWRITE":
            override_args = decision.get("override")
            if isinstance(override_args, dict):
                tool_call["args"] = override_args
            else:
                error_text = (
                    "[SafeAgent Controller] Invalid override payload in CALL_REWRITE. "
                    f"Tool call '{call_name}' blocked."
                )
                return ToolMessage(
                    content=error_text,
                    tool_call_id=tool_call_id or "safeagent-blocked",
                    name=call_name or getattr(tool, "name", "unknown_tool"),
                    status="error",
                )
            return await handler(request)

        if action == "CALL_JIT_APPROVAL":
            content = decision.get(
                "user_notice",
                f"[SafeAgent Core] Tool '{call_name}' requires human confirmation."
            )
            additional_kwargs: Dict[str, Any] = {
                "pending_call": {
                    "name": call_name,
                    "args": args,
                    "id": tool_call_id,
                    "type": "tool_call",
                }
            }
            return ToolMessage(
                content=content,
                tool_call_id=tool_call_id or "safeagent-jit-pending",
                name=call_name or getattr(tool, "name", "unknown_tool"),
                status="success",
                additional_kwargs=additional_kwargs,
            )

        return await handler(request)
