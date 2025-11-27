from typing import Any, Dict, Union, Callable, Awaitable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable
from langgraph.types import Command
from utils.agent import log_line


class SafeAgentToolWrapperMiddleware(AgentMiddleware):
    """
    Tool-level safety middleware that delegates every tool call to a SafeAgent Core.

    This middleware targets LangChain 1.x `ToolCallRequest` objects, where
    `request.tool_call` is a dict produced from the model's `tool_calls`. It
    implements a zero-trust, two-stage policy for each tool invocation:

    1. Human-in-the-loop (HITL) fast path
       If the current `tool_call.id` appears in
       `runtime.config["configurable"]["approved_tool_calls"]`, the call is
       treated as already reviewed by a human. The corresponding record MUST
       have the form:

           {
               "id": <tool_call_id>,
               "name": <approved_tool_name>,
               "args": <approved_args_dict>
           }

       The wrapper then:
       - Verifies that the approved tool `name` matches the actual tool name.
         Any mismatch is treated as a protocol violation and the call is blocked
         with a `ToolMessage(status="error")`.
       - Overwrites `tool_call["args"]` with the approved `args`.
       - Delegates to the underlying tool-node via `handler(request)` without
         consulting the SafeAgent Core again.

    2. SafeAgent Core–mediated normal path
       For all other calls (i.e. those not present in `approved_tool_calls`),
       the wrapper delegates the decision to the SafeAgent Core (`safe_agent`):

       - It builds a structured request:

           {
               "hook": "tool_wrapper",
               "tool": {
                   "name": <call_name>,
                   "description": <tool.description or None>,
               },
               "args": <tool_call['args']>,
               "session_policy": request.state.get("session_policy", {})
           }

         and calls either:
           - `safe_agent.invoke(...)` in `wrap_tool_call`, or
           - `safe_agent.ainvoke(...)` in `awrap_tool_call`.

       - The Core is expected to return a dict with an `"action"` field, one of:
           * `"CALL_ALLOW"`      – execute the tool with current args
           * `"CALL_REWRITE"`    – execute the tool with `decision["override_args"]`
           * `"CALL_BLOCK"`      – do not execute the tool
           * `"CALL_JIT_APPROVAL"` – emit a pending ToolMessage for HITL review

       - Behavior by action:

           * CALL_ALLOW
               - No modification to `tool_call["args"]`.
               - Forward to the underlying tool-node via `handler(request)`.

           * CALL_REWRITE
               - If `decision["override_args"]` is a dict, replace
                 `tool_call["args"]` with it.
               - If the override payload is missing or invalid, return a
                 `ToolMessage(status="error")` with a deterministic error
                 message and do not execute the tool.
               - Otherwise, forward to `handler(request)`.

           * CALL_BLOCK
               - Return a `ToolMessage(status="error")` containing either
                 `decision["safety_rationale"]` or a fixed default message.
               - The underlying tool is never called.

           * CALL_JIT_APPROVAL
               - Return a `ToolMessage(status="success")` whose
                 `additional_kwargs["pending_call"]` embeds:
                     {
                         "name": <call_name>,
                         "args": <original args>,
                         "id": <tool_call_id>,
                         "type": "tool_call"
                     }
               - The tool is not executed at this stage. A later controller
                 (e.g. in a `before_model` hook) is responsible for collecting
                 the human decision, writing an entry into
                 `config.configurable["approved_tool_calls"]`, and letting the
                 same middleware treat it as HITL-approved on re-entry.

    Error handling and zero-trust properties
    ---------------------------------------
    - If the SafeAgent Core raises or is unavailable, the call is failed-closed:
      a `ToolMessage(status="error")` is returned and the tool is not
      executed.
    - If the Core response is not a dict, or if `decision["action"]` is not in
      the fixed allow-list
      `{"CALL_ALLOW", "CALL_REWRITE", "CALL_BLOCK", "CALL_JIT_APPROVAL"}`,
      the call is blocked by returning a deterministic `ToolMessage` with
      `status="error"`.
    - All argument mutations happen inside this middleware; the underlying
      tool implementation and LangChain/ LangGraph’s tracing/callbacks are
      exercised normally through `handler(request)`.
    - This middleware assumes LangChain >= 1.0 semantics where
      `ToolCallRequest.tool_call` is a dict containing at least `"id"`,
      `"name"` and `"args"` fields generated from the model's `tool_calls`.
    """

    def __init__(self, safe_agent: Runnable) -> None:
        super().__init__()
        self.safe_agent = safe_agent

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
        configurable: Dict[str, Any] = (cfg or {}).get("configurable", {}) or {}

        tool_call_id = tool_call.get("id")
        call_name = tool_call.get("name")
        args: Dict[str, Any] = tool_call.get("args", {}) or {}

        # === HITL-approved ===
        approved_tool_calls = configurable.get("approved_tool_calls", []) or []
        approved_call_ids = [tool_call.get("id") for tool_call in approved_tool_calls]
        idx = approved_call_ids.index(tool_call_id) if tool_call_id in approved_call_ids else -1

        if idx >= 0:
            approved_record = approved_tool_calls[idx]
            approved_args = approved_record.get("args", {}) or {}
            approved_name = approved_record.get("name")

            # Name mismatch -> fail
            if not approved_name or not call_name or approved_name != call_name:
                return ToolMessage(
                    content=(
                        "[SafeAgent Controller] HITL approval mismatch: "
                        f"approved tool '{approved_name}', but received '{call_name}'."
                    ),
                    tool_call_id=tool_call_id or "hitl-mismatch",
                    name=call_name,
                    status="error",
                )

            # Overwrite args with the approved ones
            tool_call["args"] = approved_args

            log_line("tool_wrapper.hitl_execute", {
                "tool": tool.name,
                "tool_call_id": tool_call_id,
                "approved_args": approved_args,
            })
            return handler(request)

        # === SafeAgent Core decision ===
        session_policy = request.state.get("session_policy", {}) or {}
        core_request: Dict[str, Any] = {
            "hook": "tool_wrapper",
            "tool": {
                "name": call_name,
                "description": getattr(tool, "description", None),
            },
            "args": args,
            "session_policy": session_policy,
        }
        log_line("tool_wrapper.core_request", core_request)

        try:
            decision: Dict[str, Any] = self.safe_agent.invoke(core_request, config=cfg)
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
            override_args = decision.get("override_args")
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
        configurable: Dict[str, Any] = (cfg or {}).get("configurable", {}) or {}

        tool_call_id = tool_call.get("id")
        call_name = tool_call.get("name")
        args: Dict[str, Any] = tool_call.get("args", {}) or {}

        # === HITL-approved ===
        approved_tool_calls = configurable.get("approved_tool_calls", []) or []
        approved_call_ids = [tool_call.get("id") for tool_call in approved_tool_calls]
        idx = approved_call_ids.index(tool_call_id) if tool_call_id in approved_call_ids else -1

        if idx >= 0:
            approved_record = approved_tool_calls[idx]
            approved_args = approved_record.get("args", {}) or {}
            approved_name = approved_record.get("name")

            # Name mismatch -> fail
            if not approved_name or not call_name or approved_name != call_name:
                return ToolMessage(
                    content=(
                        "[SafeAgent Controller] HITL approval mismatch: "
                        f"approved tool '{approved_name}', but received '{call_name}'."
                    ),
                    tool_call_id=tool_call_id or "hitl-mismatch",
                    name=call_name,
                    status="error",
                )

            # Overwrite args with the approved ones
            tool_call["args"] = approved_args

            log_line("tool_wrapper.hitl_execute", {
                "tool": tool.name,
                "tool_call_id": tool_call_id,
                "approved_args": approved_args,
            })
            return await handler(request)

        # === SafeAgent Core decision ===
        session_policy = request.state.get("session_policy", {}) or {}
        core_request: Dict[str, Any] = {
            "hook": "tool_wrapper",
            "tool": {
                "name": call_name,
                "description": getattr(tool, "description", None),
            },
            "args": args,
            "session_policy": session_policy,
        }
        log_line("tool_wrapper.core_request", core_request)

        try:
            decision: Dict[str, Any] = await self.safe_agent.ainvoke(core_request, config=cfg)
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
            override_args = decision.get("override_args")
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
