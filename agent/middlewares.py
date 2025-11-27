from typing import List, Dict, Any
from langgraph.runtime import Runtime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from langchain.agents.middleware import before_agent, after_agent, before_model, after_model
from langchain.agents.middleware import AgentState
from utils.agent import safe_agent, log_line


def _serialize_single_tool_call(tc: Any) -> Dict[str, Any]:
    """
    Serialize a single tool call (dict or ToolCall-like object) into a JSON-safe dict.

    This ensures SafeAgent Core only sees plain data structures and Pydantic/JSON
    can handle them safely.
    """
    if isinstance(tc, dict):
        return {
            "name": tc.get("name"),
            "args": tc.get("args"),
            "id": tc.get("id") or tc.get("tool_call_id"),
            "type": tc.get("type") or "tool_call",
        }

    if isinstance(tc, ToolCall) or (
        hasattr(tc, "name") and hasattr(tc, "args")
    ):
        data: Dict[str, Any] = {
            "name": getattr(tc, "name", None),
            "args": getattr(tc, "args", None),
        }
        tc_id = getattr(tc, "id", None)
        if tc_id is not None:
            data["id"] = tc_id
        tc_type = getattr(tc, "type", None)
        if tc_type is not None:
            data["type"] = tc_type
        else:
            data["type"] = "tool_call"
        return data

    return {
        "unknown_tool_call": repr(tc),
    }


def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """
    Normalize the `tool_calls` field into a JSON-safe list[dict] for SafeAgent Core.

    Accepts:
    - None
    - Single ToolCall / dict
    - list[ToolCall] / list[dict]
    """
    if tool_calls is None:
        return []

    if isinstance(tool_calls, list):
        return [_serialize_single_tool_call(tc) for tc in tool_calls]

    return [_serialize_single_tool_call(tool_calls)]


def _rollback_to_human_message(state: AgentState) -> int | None:
    """
    Roll back the conversation state to the most recent HumanMessage.

    This utility is shared by ROLLBACK, NO_MODEL_CALLING and other safety
    recovery paths that must discard untrusted LLM/tool outputs.

    Zero-trust guarantees:
    ----------------------
    - Only explicit HumanMessage nodes are considered valid rollback anchors.
    - If no HumanMessage exists in the trace, return `None` and let the caller
      decide whether to terminate the graph.
    - The function *only mutates* state["messages"], and never inserts any
      user-visible or model-visible text itself (the caller handles that).
    - Long-term memory poisoning is prevented because untrusted tail messages
      are always removed before being written to memory layers.

    Parameters
    ----------
    state : AgentState
        Mutable LangGraph agent state containing the `messages` list.

    Returns
    -------
    int | None
        The index of the retained HumanMessage (after rollback), or `None`
        if no HumanMessage existed in the state.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            idx = i
            break

    if idx is None:
        return None

    state["messages"] = messages[: idx + 1]
    return idx


@before_agent(can_jump_to=["end"])
def safe_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Zero-trust controller hook for the `before_agent` stage.

    This middleware sits at the ingress of the agent graph and delegates
    user-input safety decisions to the SafeAgent Core (`safe_agent`).
    It owns all **state mutation** and **control flow**, while the Core
    only returns declarative decisions.

    Behavior
    --------
    1. Read the last entry in ``state["messages"]``.
    2. If it is a ``HumanMessage`` with string ``content``, treat it as the
       current user input; otherwise do nothing and return ``None``.
    3. Build a Core request:

           {
               "hook": "before_agent",
               "last_message": {"role": "user", "content": <text>},
               "session_policy": state.get("session_policy", {})
           }

       and call ``safe_agent.invoke(...)``.
    4. Interpret ``decision["action"]`` under a **zero-trust** policy and
       optionally apply ``decision["policy_delta"]`` to the session policy.

    Supported actions
    -----------------
    - ``"APPROVE"``:
        Input is accepted as-is.
        * If ``policy_delta`` is a ``dict``, merge it into
          ``state["session_policy"]``.
        * Do not touch ``state["messages"]``.
        * Return ``None`` to continue normal execution.

    - ``"REJECT"``:
        The request is rejected before any model or tools are invoked.
        * Optionally merge a valid ``policy_delta`` into
          ``state["session_policy"]`` for future turns.
        * Build a user notice from
          ``decision["user_notice"]`` or a fixed default.
        * Return:

              {
                  "messages": [AIMessage(content=<reject_text>)],
                  "jump_to": "end"
              }

          which terminates the graph and returns the rejection to the caller.

    - ``"OVERRIDE"``:
        The input is unsafe but can be rewritten.
        * Optionally merge a valid ``policy_delta`` into
          ``state["session_policy"]``.
        * Read ``decision["override_context"]`` as a non-empty string.
        * On success:
            - Replace the last ``HumanMessage`` in ``state["messages"]``
              with a new one whose ``content`` is ``override_context`` and
              whose ``additional_kwargs`` are preserved.
            - Return ``None`` so that the updated input flows into the model.
        * If ``override_context`` is missing or invalid, treat this as a
          protocol violation and return a rejection with ``jump_to = "end"``.

    Session policy (policy_delta)
    -----------------------------
    - ``decision["policy_delta"]`` is optional and may accompany any action.
    - If present and a ``dict``, it is merged in place into
      ``state["session_policy"]`` so that downstream hooks and tool
      wrappers can observe updated capabilities / risk state.
    - If present but not a ``dict``, the Controller blocks the request and
      returns a deterministic error message with ``jump_to = "end"``.

    Zero-trust guarantees
    ---------------------
    - Only trailing ``HumanMessage`` with text content is inspected here;
      AI / Tool messages and non-text inputs are ignored.
    - Allowed actions are strictly limited to ``{"APPROVE", "REJECT",
      "OVERRIDE"}``. Any other value is treated as an error and leads to
      a hard block.
    - All mutations (messages and session_policy) are applied directly to
      the mutable ``state`` object; apart from explicit ``jump_to = "end"``,
      this hook does not rely on state-merging semantics.
    """
    messages = state.get("messages", [])
    session_policy = state.get("session_policy", {}) or {}
    if not messages:
        return None

    last_msg = messages[-1]
    last_content = getattr(last_msg, "content", "")

    # Ignore non-user messages (AI / Tool / others) and non-text content.
    if not isinstance(last_msg, HumanMessage) or not isinstance(last_content, str):
        return None

    core_request: Dict[str, Any] = {
        "hook": "before_agent",
        "last_message": {
            "role": "user",
            "content": last_content,
        },
        "session_policy": session_policy,
    }
    log_line("before_agent.core_request", core_request)

    try:
        decision = safe_agent.invoke(core_request)
    except Exception as e:
        log_line("before_agent.core_error", {"error": str(e)})
        return {
            "messages": [AIMessage(content="[SafeAgent Controller] Safety core unavailable. Request rejected.")],
            "jump_to": "end",
        }
    action = str(decision.get("action", "")).upper().strip()
    log_line("before_agent.core_decision", decision)

    # Block unknown actions (zero-trust policy).
    ALLOWED_ACTIONS = {"APPROVE", "REJECT", "OVERRIDE"}
    if action not in ALLOWED_ACTIONS:
        error_text = (
            f"[SafeAgent Controller] Unknown action '{action}'. "
            "Request blocked by zero-trust runtime."
        )
        return {
            "messages": [AIMessage(content=error_text)],
            "jump_to": "end"
        }

    # Update Runtime session policy
    policy_delta = decision.get("policy_delta", None)
    if policy_delta is not None:
        if not isinstance(policy_delta, dict):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "[SafeAgent Controller] Invalid policy_delta. "
                            "Request blocked by zero-trust runtime."
                        )
                    )
                ],
                "jump_to": "end",
            }
        session_policy.update(policy_delta)
        state["session_policy"] = session_policy

    if action == "APPROVE":
        return None

    if action == "REJECT":
        reject_text = decision.get(
            "user_notice",
            "[SafeAgent Core] The request has been rejected by security policy."
        )
        return {
            "messages": [AIMessage(content=reject_text)],
            "jump_to": "end",
        }

    if action == "OVERRIDE":
        new_input = decision.get("override_context")
        if not isinstance(new_input, str) or not new_input:
            return {
                "messages": [
                    AIMessage(content="[SafeAgent Controller] Invalid override_input. Request blocked.")
                ],
                "jump_to": "end"
            }

        state["messages"][-1] = HumanMessage(
            content=new_input,
            additional_kwargs=getattr(last_msg, "additional_kwargs", {}),
        )
        return None

    return None


@before_model(can_jump_to=["end"])
def safe_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Zero-trust controller hook for the `before_model` (context inspection) stage.

    This middleware runs after the environment produced a ToolMessage and right
    before the model is called again. It inspects the **last ToolMessage** and
    delegates the safety decision to the SafeAgent Core (`safe_agent`). The
    Controller performs all state mutations and flow control; the Core only
    returns a structured decision.

    Behavior
    --------
    1. Read the last entry in ``state["messages"]``.
    2. If it is **not** a ``ToolMessage`` or its ``content`` is non-text,
    return ``None`` (no-op).
    3. Otherwise, build a Core request:

        {
            "hook": "before_model",
            "last_message": {
                "role": "tool",
                "name": <tool_name>,
                "tool_call_id": <id>,
                "content": <tool_output_text>,
            },
            "session_policy": state.get("session_policy", {})
        }

    and call ``safe_agent.invoke(...)``.

    The Core responds with:

        {
            "action": "APPROVE" | "OVERRIDE" | "ROLLBACK" | "NO_MODEL_CALLING" | "REJECT",
            "policy_delta": { ... }?,              # optional session-policy update
            "allow_long_term_memory": bool?,       # optional LTM permission
            "override_context": "...",             # for OVERRIDE
            "user_notice": "..."               # for REJECT
        }

    Session policy (policy_delta)
    -----------------------------
    - If ``policy_delta`` is absent → unchanged.
    - If it is a ``dict`` → merged into ``state["session_policy"]`` in-place.
    - If present but not a ``dict`` → protocol violation; the last message is
    replaced by an error ``AIMessage`` and the graph ends via ``{"jump_to": "end"}``.

    Long-term memory tagging
    ------------------------
    - The Core may specify ``allow_long_term_memory``.
    - Controller enforces **default deny**:
        * ``True`` → tag message with ``allow_long_term_memory = True``.
        * anything else (missing / False / None / invalid) → ``False``.
    - Tag is written into ``last_msg.additional_kwargs`` for downstream memory writers.

    Supported actions
    -----------------
    - ``"APPROVE"``:
        Context is safe. After applying any policy updates and LTM tagging, the
        ToolMessage is passed to the model unchanged. Return ``None``.

    - ``"OVERRIDE"``:
        Tool output is unsafe but repairable. The Core must provide a non-empty
        ``override_context``.
        * Replace **only** ``last_msg.content`` with the safe version.
        * Preserve message type and structural fields.
        * LTM tag already applied above.
        Return ``None``. Missing/invalid ``override_context`` → protocol error → end.

    - ``"ROLLBACK"``:
        The entire short-term trajectory is untrustworthy.
        * Roll back to the most recent ``HumanMessage``. If none exists → block.
        * Append a fixed safety guidance ``AIMessage`` tagged with
        ``NO_LONG_TERM_MEMORY_WRITE``.
        Return ``None`` to re-invoke the model on the rolled-back context.

    - ``"NO_MODEL_CALLING"``:
        Model must **not** be called this turn.
        * Roll back to last ``HumanMessage`` if possible.
        * Append a final user-visible safety ``AIMessage`` using
        * Terminate via ``{"jump_to": "end"}``.

    - ``"REJECT"``:
        Tool output is unsafe; the producing tool must be disabled.
        * Add its name to ``session_policy["blocked_tools"]``.
        * Replace the ToolMessage with a fixed safety-guidance ``AIMessage`` tagged
        ``NO_LONG_TERM_MEMORY_WRITE``.
        * Allow the model to run again (no jump). Tools will be filtered by policy.

    Zero-trust guarantees
    ---------------------
    - Only the last textual ToolMessage is inspected.
    - Allowed actions are strictly:
        {"APPROVE", "OVERRIDE", "ROLLBACK", "NO_MODEL_CALLING", "REJECT"}.
    Anything else triggers a deterministic block.
    - All mutations are applied directly to ``state``. Only explicit ``jump_to``
    directives affect graph flow.
    - Core cannot inject arbitrary prompts; all model-visible text is synthesized
    by the Controller from its structured fields.
    """
    messages = state.get("messages", [])
    session_policy = state.get("session_policy", {}) or {}
    if not messages:
        return None

    last_msg = messages[-1]
    if not isinstance(last_msg, ToolMessage):
        return None

    last_content = getattr(last_msg, "content", "")
    if not isinstance(last_content, str):
        return None

    core_request = {
        "hook": "before_model",
        "last_message": {
            "role": "tool",
            "name": last_msg.name,
            "tool_call_id": last_msg.tool_call_id,
            "content": last_content,
        },
        "session_policy": session_policy,
    }
    log_line("before_model.core_request", core_request)

    try:
        decision = safe_agent.invoke(core_request)
    except Exception as e:
        log_line("before_model.core_error", {"error": str(e)})
        error_text = (
            "[SafeAgent Controller] Safety core unavailable. "
            "Tool plan cannot be executed for this turn."
        )
        state["messages"][-1] = AIMessage(
            content=error_text,
            additional_kwargs=getattr(last_msg, "additional_kwargs", {}),
        )
        setattr(state["messages"][-1], "tool_calls", [])
        return {"jump_to": "end"}
    action = str(decision.get("action", "")).upper().strip()
    log_line("before_model.core_decision", decision)

    # Allowed actions at this stage
    ALLOWED_ACTIONS = {"APPROVE", "OVERRIDE", "ROLLBACK", "NO_MODEL_CALLING", "REJECT"}
    if action not in ALLOWED_ACTIONS:
        error_text = (
            f"[SafeAgent Controller] Unknown before_model action '{action}'. "
            "Request blocked by zero-trust runtime."
        )
        state["messages"] = [
            AIMessage(content=error_text)
        ]
        return {"jump_to": "end"}

    # Update Runtime session policy
    policy_delta = decision.get("policy_delta", None)
    if policy_delta is not None:
        if not isinstance(policy_delta, dict):
            error_text = (
                "[SafeAgent Controller] Invalid policy_delta. "
                "Request blocked by zero-trust runtime."
            )
            state["messages"][-1] = AIMessage(content=error_text)
            return {"jump_to": "end"}
        session_policy.update(policy_delta)
        state["session_policy"] = session_policy

    # --- Apply per-message long-term memory policy if provided ---
    allow_ltm = decision.get("allow_long_term_memory", False)
    ak = getattr(last_msg, "additional_kwargs", {}) or {}
    ak["allow_long_term_memory"] = allow_ltm if allow_ltm is True else False
    setattr(last_msg, "additional_kwargs", ak)

    if action == "APPROVE":
        return None

    # === OVERRIDE: overwrite only the content of the last ToolMessage ===
    if action == "OVERRIDE":
        override_context = decision.get("override_context")
        if not isinstance(override_context, str) or not override_context:
            error_text = (
                "[SafeAgent Controller] Invalid override_context. "
                "Request blocked by zero-trust runtime."
            )
            state["messages"][-1] = AIMessage(content=error_text)
            return {"jump_to": "end"}

        state["messages"][-1].content = override_context
        return None

    # === ROLLBACK: roll back to last HumanMessage ===
    if action == "ROLLBACK":
        idx = _rollback_to_human_message(state)

        if idx is None:
            error_text = (
                "[SafeAgent Controller] ROLLBACK requested but no HumanMessage "
                "found in history. Request blocked by zero-trust runtime."
            )
            state["messages"] = [AIMessage(content=error_text)]
            return {"jump_to": "end"}

        # Insert safe guidance
        guidance = (
            "[SafeAgent Core] The previous reasoning path or tool output was "
            "determined to be unsafe or unreliable. You must re-evaluate the "
            "task carefully.\n"
            "- Think twice when planning a function calling, they could be unreliable.\n"
            "- Follow the session's safety policy strictly.\n"
            "- Provide a safer, more compliant alternative plan.\n"
        )

        state["messages"].append(
            AIMessage(
                content=guidance,
                additional_kwargs={"safe_tags": ["NO_LONG_TERM_MEMORY_WRITE"]},
            )
        )
        return None

    # === NO_MODEL_CALLING: do NOT call the model again ===
    if action == "NO_MODEL_CALLING":
        idx = _rollback_to_human_message(state)

        # if no HumanMessage, fallback to single canned response
        if idx is None:
            safe_resp = "[SafeAgent Controller] Model call skipped due to safety policy."
            state["messages"] = [AIMessage(content=safe_resp)]
            return {"jump_to": "end"}

        # Insert final user-visible safety response
        safe_resp = "[SafeAgent Core] This conversation turn has been terminated due to safety policy."
        state["messages"].append(AIMessage(content=safe_resp))

        return {"jump_to": "end"}

    # === REJECT: drop unsafe ToolMessage + disable the tool + continue model ===
    if action == "REJECT":
        last_tool_msg = messages[-1]

        # Identify the tool call and block the tool
        tool_name = None
        if hasattr(last_tool_msg, "name"):
            tool_name = last_tool_msg.name
        raw_blocked = session_policy.get("blocked_tools", [])
        blocked_tools = set(raw_blocked or [])
        if tool_name:
            blocked_tools.add(tool_name)
        session_policy["blocked_tools"] = list(blocked_tools)

        # Insert a guidance AIMessage for the next model step
        guidance = (
            "[SafeAgent Core] The previous tool result was unsafe and has "
            "been removed. The tool used in the last step is now disabled for "
            "the remainder of this session.\n"
            "- You MUST NOT call that tool again.\n"
            "- You MUST answer the user directly using only the available context.\n"
        )

        state["messages"][-1] = AIMessage(
            content=guidance,
            additional_kwargs={"safe_tags": ["NO_LONG_TERM_MEMORY_WRITE"]},
        )
        return None


@after_model(can_jump_to=["model", "end"])
def safe_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Zero-trust controller hook for the `after_model` stage (plan inspection).

    This hook runs *after* the model produces an AIMessage but *before* any
    tool is executed. It validates the tool-call plan using:

        1) local deterministic checks (blocked tools)
        2) SafeAgent Core decisions (APPROVE / REPLAN / REJECT)

    Controller owns all state mutation and control-flow decisions; Core only
    returns declarative judgments.

    Pipeline
    --------
    1. Read the last message:
        last_msg = state["messages"][-1]

    2. If last_msg is not AIMessage → no-op.
    If last_msg has no tool_calls (None/empty) → treat as normal text answer.

    3. Local forbidden-tool check:
    - session_policy["blocked_tools"] is a list of prohibited tool names.
    - If the plan calls any of them:
            * Rewrite last_msg.content with a fixed safety guidance template.
            * Clear tool_calls.
            * Set allow_long_term_memory = False.
            * return {"jump_to": "model"}  # force a new model call.

    4. Otherwise build a Core request:
        {
            "hook": "after_model",
            "last_message": {content, tool_calls},
            "session_policy": session_policy
        }
    and call:
        decision = safe_agent.invoke(request)

    5. On Core error:
        - Replace last message with a deterministic refusal.
        - Clear tool_calls.
        - return {"jump_to": "end"}

    Allowed Core actions
    --------------------
    - "APPROVE"
        Accept the plan as-is.
        Apply optional policy_delta (must be dict).
        return None → proceed to tool execution.

    - "REPLAN"
        Plan is unsafe, but replanning is allowed.
        Controller synthesizes a *fixed template* guidance message:
            - state reasons (if provided)
            - instruct model to generate a new plan compliant with policy
        Clear tool_calls, set allow_long_term_memory=False,
        return {"jump_to": "model"}.

    - "REJECT"
        Tools must not be used this turn.
        Controller synthesizes a *fixed template* natural-language guidance:
            - explain refusal
            - instruct model to answer directly with no tool calls
        Clear tool_calls, set allow_long_term_memory=False,
        return {"jump_to": "model"}.

    Policy delta
    ------------
    - If decision["policy_delta"] exists:
        * Must be a dict → merge into state["session_policy"].
        * Otherwise → protocol violation → replace with error AIMessage and
            return {"jump_to": "end"}.

    Zero-trust guarantees
    ---------------------
    - The Core cannot inject free-form prompt text; Controller synthesizes all
    guidance templates.
    - Only {"APPROVE", "REPLAN", "REJECT"} are accepted.
    - Any malformed payload (unknown action / invalid policy_delta /
    invalid override fields) results in a deterministic block.
    - All mutations happen directly on `state` (messages, tool_calls,
    session_policy), without relying on implicit merge semantics.
    """

    messages = state.get("messages", [])
    session_policy = state.get("session_policy", {}) or {}
    if not messages:
        return None

    last_msg = messages[-1]
    last_content = getattr(last_msg, "content", "")
    tool_calls = getattr(last_msg, "tool_calls", [])

    # Only inspect AI model outputs.
    if not isinstance(last_msg, AIMessage):
        return None

    # If no tool_calls, treat as a response to user and forward it.
    if not tool_calls:
        return None

    # === Check forbidden tools BEFORE asking the Core ===
    blocked_tools = set(session_policy.get("blocked_tools", []) or [])
    forbidden_called = False
    forbidden_list = []

    tool_calls = _normalize_tool_calls(tool_calls)
    for tc in tool_calls:
        name = tc.get("name")
        if name in blocked_tools:
            forbidden_called = True
            forbidden_list.append(name)

    if forbidden_called:
        tools_str = ", ".join(sorted(forbidden_list))
        guidance = (
            "[SafeAgent Controller] The previous tool plan attempted to call "
            f"forbidden tool(s): {tools_str}.\n"
            "You MUST propose a new plan WITHOUT using these tools.\n"
            "- Re-check the session policy.\n"
            "- Provide a safer alternative approach.\n"
        )

        guidance_content = (
            f"{last_content}\n\n{guidance}"
            if isinstance(last_content, str)
            else guidance
        )

        additional_kwargs = getattr(last_msg, "additional_kwargs", {}) or {}
        additional_kwargs["allow_long_term_memory"] = False
        state["messages"][-1] = AIMessage(
            content=guidance_content,
            additional_kwargs=additional_kwargs,
        )
        setattr(state["messages"][-1], "tool_calls", [])
        return {"jump_to": "model"}

    core_request: Dict[str, Any] = {
        "hook": "after_model",
        "last_message": {
            "role": "assistant",
            "content": last_content,
            "tool_calls": tool_calls,
        },
        "session_policy": session_policy,
    }
    log_line("after_model.core_request", core_request)

    try:
        decision = safe_agent.invoke(core_request)
    except Exception as e:
        log_line("after_model.core_error", {"error": str(e)})
        error_text = (
            "[SafeAgent Controller] Safety core unavailable. "
            "Tool plan cannot be executed for this turn."
        )
        state["messages"][-1] = AIMessage(
            content=error_text,
            additional_kwargs=getattr(last_msg, "additional_kwargs", {}),
        )
        setattr(state["messages"][-1], "tool_calls", [])
        return {"jump_to": "end"}
    action = str(decision.get("action", "")).upper().strip()
    log_line("after_model.core_decision", decision)

    # Set of allowed plan actions.
    ALLOWED_ACTIONS = {"APPROVE", "REPLAN", "REJECT"}
    if action not in ALLOWED_ACTIONS:
        error_text = (
            f"[SafeAgent Controller] Unknown plan action '{action}'. "
            "Plan rejected by zero-trust runtime."
        )
        # Replace last message with a plain error AIMessage and stop the graph.
        state["messages"][-1] = AIMessage(
            content=error_text,
            additional_kwargs=getattr(last_msg, "additional_kwargs", {}),
        )
        return {"jump_to": "end"}

    # Update Runtime session policy
    policy_delta = decision.get("policy_delta", None)
    if policy_delta is not None:
        if not isinstance(policy_delta, dict):
            error_text = (
                "[SafeAgent Controller] Invalid policy_delta. "
                "Plan rejected by zero-trust runtime."
            )
            state["messages"][-1] = AIMessage(
                content=error_text,
                additional_kwargs=getattr(last_msg, "additional_kwargs", {}),
            )
            return {"jump_to": "end"}

        session_policy.update(policy_delta)
        state["session_policy"] = session_policy

    if action == "APPROVE":
        return None

    if action == "REPLAN":
        safety_rationale = decision.get("safety_rationale")

        lines = [
            "[SafeAgent Notice] The previous tool plan has been rejected by the safety runtime.",
            "You must now propose a NEW tool plan that strictly follows the current safety policy "
            "and still helps the user achieve their goal.",
            "Do NOT call tools that would violate capability or safety constraints in the current session_policy.",
        ]
        if isinstance(safety_rationale, str) and safety_rationale.strip():
            lines.append(
                f"Safety rationale (for your internal reasoning only): {safety_rationale.strip()}"
            )

        guidance = "\n".join(lines)
        guidance_content = (
            f"{last_content}\n\n{guidance}"
            if isinstance(last_content, str)
            else guidance
        )

        # Overwrite last AIMessage content and clear tool_calls.
        additional_kwargs = getattr(last_msg, "additional_kwargs", {}) or {}
        additional_kwargs["allow_long_term_memory"] = False
        state["messages"][-1] = AIMessage(
            content=guidance_content,
            additional_kwargs=additional_kwargs,
        )
        setattr(state["messages"][-1], "tool_calls", [])
        return {"jump_to": "model"}

    if action == "REJECT":
        safety_rationale = decision.get("safety_rationale")
        user_notice = decision.get("user_notice")

        lines = [
            "[SafeAgent Notice] The proposed tool plan CANNOT be safely executed.",
            "For THIS TURN, you MUST NOT call any tools.",
            "Instead, you MUST answer the user's request directly in natural language, "
            "based only on the existing conversation context.",
        ]
        if isinstance(user_notice, str) and user_notice.strip():
            lines.append(
                f"When answering, you MUST explain to the user that tools were refused because: {user_notice.strip()}"
            )
        if isinstance(safety_rationale, str) and safety_rationale.strip():
            lines.append(
                f"Safety rationale (for your internal reasoning only): {safety_rationale.strip()}"
            )

        guidance = "\n".join(lines)
        guidance_content = (
            f"{last_content}\n\n{guidance}"
            if isinstance(last_content, str)
            else guidance
        )

        additional_kwargs = getattr(last_msg, "additional_kwargs", {}) or {}
        additional_kwargs["allow_long_term_memory"] = False
        state["messages"][-1] = AIMessage(
            content=guidance_content,
            additional_kwargs=additional_kwargs,
        )
        setattr(state["messages"][-1], "tool_calls", [])
        return {"jump_to": "model"}

    return None


@after_agent
def safe_after_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Zero-trust controller hook for the `after_agent` stage.

    This middleware runs at the egress of the agent graph, just before the
    final response is returned. It delegates safety decisions about the final
    assistant output to the SafeAgent Core (`safe_agent`) and enforces them by
    mutating the runtime state.

    Behavior
    --------
    1. Read the last entry in ``state["messages"]``.
    2. If it is an ``AIMessage`` with string ``content``, treat it as the
       candidate final response; otherwise do nothing and return ``None``.
    3. Build a Core request:

           {
               "hook": "after_agent",
               "last_message": {"role": "assistant", "content": <text>},
               "session_policy": state.get("session_policy", {})
           }

       and call ``safe_agent.invoke(...)``.
    4. Interpret ``decision["action"]`` under a **zero-trust** policy and
       apply an optional ``decision["policy_delta"]`` to the session policy
       before acting on the output.

    Supported actions
    -----------------
    - ``"APPROVE"``:
        Final output is accepted as-is.
        * If ``policy_delta`` is a ``dict``, merge it into
          ``state["session_policy"]``.
        * Do not change the last ``AIMessage``.
        * Return ``None`` so the original response is returned to the caller.

    - ``"REJECT"``:
        Final output is unsafe and must be replaced.
        * Optionally merge a valid ``policy_delta`` into
          ``state["session_policy"]`` for future turns.
        * Build a user notice from
          ``decision["user_notice"]`` or a fixed default.
        * Overwrite the last ``AIMessage`` in ``state["messages"]`` with the
          user notice text.
        * Return ``None`` (the graph is already at its end).

    - ``"OVERRIDE"``:
        Final output is unsafe but can be rewritten.
        * Optionally merge a valid ``policy_delta`` into
          ``state["session_policy"]``.
        * Read ``decision["override_context"]`` as a non-empty string.
        * On success:
            - Replace the last ``AIMessage`` with a new one whose ``content``
              is ``override_context``, preserving ``additional_kwargs``.
            - Return ``None``.
        * If ``override_context`` is missing or invalid, overwrite the last
          message with a deterministic error text.

    Session policy (policy_delta)
    -----------------------------
    - ``decision["policy_delta"]`` is optional and may accompany any action.
    - If present and a ``dict``, it is merged into
      ``state["session_policy"]`` in place so downstream components can
      observe updated risk/capability state.
    - If present but not a ``dict``, this is treated as a protocol violation
      and the final output is replaced by a fixed error message.

    Zero-trust guarantees
    ---------------------
    - Only the last textual ``AIMessage`` is inspected here; Tool / Human
      messages and non-text content are ignored.
    - Allowed actions are strictly limited to ``{"APPROVE", "REJECT",
      "OVERRIDE"}``. Any other value is treated as an error and causes the
      final output to be replaced by a deterministic error message.
    - All mutations (messages and session_policy) are applied directly to the
      mutable ``state`` object; this hook does not rely on state-merging
      semantics.
    - Malformed Core payloads (invalid ``override_context`` or ``policy_delta``)
      never result in an implicit allow; they deterministically degrade to a
      safe refusal.
    """
    messages = state.get("messages", [])
    session_policy = state.get("session_policy", {}) or {}
    if not messages:
        return None

    last_msg = messages[-1]
    last_content = getattr(last_msg, "content", "")

    # Only inspect textual AI output. Other message types are passed through.
    if not isinstance(last_msg, AIMessage) or not isinstance(last_content, str):
        return None

    core_request: Dict[str, Any] = {
        "hook": "after_agent",
        "last_message": {
            "role": "assistant",
            "content": last_content,
        },
        "session_policy": session_policy,
    }
    log_line("after_agent.core_request", core_request)

    try:
        decision = safe_agent.invoke(core_request)
    except Exception as e:
        log_line("after_agent.core_error", {"error": str(e)})
        error_text = (
            "[SafeAgent Controller] Safety core unavailable. "
            "Tool plan cannot be executed for this turn."
        )
        state["messages"][-1] = AIMessage(
            content=error_text,
            additional_kwargs=getattr(last_msg, "additional_kwargs", {}),
        )
        setattr(state["messages"][-1], "tool_calls", [])
        return None
    action = str(decision.get("action", "")).upper().strip()
    log_line("after_agent.core_decision", decision)

    # Zero-trust: only allow a closed set of actions.
    ALLOWED_ACTIONS = {"APPROVE", "REJECT", "OVERRIDE"}
    if action not in ALLOWED_ACTIONS:
        error_text = (
            f"[SafeAgent Controller] Unknown action '{action}'. "
            "Final response replaced by zero-trust runtime."
        )
        state["messages"][-1] = AIMessage(
            content=error_text,
            additional_kwargs=getattr(last_msg, "additional_kwargs", {}),
        )
        return None

    # Optional session_policy update (policy_delta) for any valid action.
    policy_delta = decision.get("policy_delta", None)
    if policy_delta is not None:
        if not isinstance(policy_delta, dict):
            error_text = (
                "[SafeAgent Controller] Invalid policy_delta. "
                "Final response replaced by zero-trust runtime."
            )
            state["messages"][-1] = AIMessage(
                content=error_text,
                additional_kwargs=getattr(last_msg, "additional_kwargs", {}),
            )
            return None

        session_policy.update(policy_delta)
        state["session_policy"] = session_policy

    if action == "APPROVE":
        return None

    if action == "REJECT":
        reject_text = decision.get(
            "user_notice",
            "[SafeAgent Core] The response has been rejected by security policy.",
        )
        state["messages"][-1] = AIMessage(
            content=reject_text,
            additional_kwargs=getattr(last_msg, "additional_kwargs", {}),
        )
        return None

    if action == "OVERRIDE":
        new_output = decision.get("override_context")
        if not isinstance(new_output, str) or not new_output:
            error_text = (
                "[SafeAgent Controller] Invalid override_context. "
                "Final response replaced by zero-trust runtime."
            )
            state["messages"][-1] = AIMessage(
                content=error_text,
                additional_kwargs=getattr(last_msg, "additional_kwargs", {}),
            )
            return None

        state["messages"][-1] = AIMessage(
            content=new_output,
            additional_kwargs=getattr(last_msg, "additional_kwargs", {}),
        )
        return None

    return None
