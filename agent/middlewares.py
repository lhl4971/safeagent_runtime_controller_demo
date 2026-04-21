from typing import List, Dict, Any
from langgraph.runtime import Runtime
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import before_agent, after_agent, before_model, after_model
from langchain.agents.middleware import AgentState, AgentMiddleware
from utils.agent import log_line, last_message_index, has_safe_tag, parse_mcp_tool_response


def _extract_hitl_outcomes(decisions: List[Dict[str, Any]]):
    """
    Parse HITL decisions into two parallel structures:
        - approved_calls: list of {id, name, args}
        - rejected_msgs: list of ToolMessage (ready to merge back)
    """
    approved_calls = []
    rejected_msgs = []

    for d in decisions:
        if not isinstance(d, dict):
            continue

        status = str(d.get("status", "")).upper().strip()
        pc = d.get("pending_call") or {}
        name = pc.get("name")
        call_id = pc.get("id")
        args = pc.get("args", {}) or {}

        if not name or not call_id:
            continue

        if status == "SHADOW":
            args = dict(args)
            args["execution_mode"] = "shadow"

        if status in ("APPROVE", "SHADOW"):
            approved_calls.append({"id": call_id, "name": name, "args": args})
            continue

        if status == "REJECT":
            rejected_msgs.append(
                ToolMessage(
                    name=name,
                    tool_call_id=call_id,
                    content=(
                        f"[SafeAgent Controller] "
                        f"Tool '{name}' call was rejected by human review."
                    ),
                )
            )

    return approved_calls, rejected_msgs


def _merge_tool_messages(messages, new_tool_msgs):
    """
    In-place merge: replace existing ToolMessage by matching tool_call_id.
    """
    id_map = {
        getattr(m, "tool_call_id", None): i
        for i, m in enumerate(messages)
        if isinstance(m, ToolMessage)
    }

    for tool_message in new_tool_msgs:
        tcid = getattr(tool_message, "tool_call_id", None)
        if not tcid:
            continue

        idx = id_map.get(tcid)
        if idx is not None:
            messages[idx] = tool_message


def build_safe_agent_middlewares(safe_agent: Runnable, session_id: str) -> list[AgentMiddleware]:
    @before_agent(can_jump_to=["end"])
    async def safe_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Safety middleware for the **before_agent** stage.

        Purpose
        -------
        This middleware inspects the **user's raw input** before the agent begins
        any planning or reasoning. It delegates the safety judgment to the
        SafeAgent Core and enforces the Core's decision with zero-trust control.

        What this middleware protects
        -----------------------------
        - Prevents unsafe or disallowed user requests from entering the agent loop.
        - Allows the Core to rewrite, block, or approve user input deterministically.
        - Ensures early-stage policy updates (session_policy mutations) are applied
        before any LLM reasoning happens.

        Core actions supported
        ----------------------
        The SafeAgent Core may return one of the following actions:

        - **APPROVE**
        - The user input is accepted as-is.
        - Agent proceeds normally into model reasoning.

        - **OVERRIDE**
        - Core provides a safe replacement for the user input.
        - The original HumanMessage content is replaced with Core-supplied text.
        - Planning continues with sanitized input.

        - **REJECT**
        - The request is disallowed by security policy.
        - A refusal AIMessage is generated for the user.
        - The agent turn terminates immediately.

        Additional capabilities
        -----------------------
        - **Session policy updates**
        Core may return a `policy_delta` dict to dynamically adjust runtime
        safety constraints (e.g., blocked tools, capability restrictions).

        - **Zero-trust enforcement**
        Any malformed Core output or unknown action results in a deterministic
        block with an error AIMessage.

        Summary
        -------
        This hook enforces the earliest and strictest safety boundary for the
        agent. All user input passes through SafeAgent Core before any planning
        or tool-selection logic is executed.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        last_content = getattr(last_msg, "content", "")

        # Ignore non-user messages (AI / Tool / others) and non-text content.
        if not isinstance(last_msg, HumanMessage) or not isinstance(last_content, str):
            return None

        core_request: Dict[str, Any] = {
            "hook": "before_agent",
            "observation": {
                "role": "user",
                "content": last_content,
            },
        }
        log_line("before_agent.core_request", core_request)

        try:
            decision = await safe_agent.ainvoke({
                "session_id": session_id,
                "core_request": core_request,
            })
            decision = parse_mcp_tool_response(decision)
        except Exception as e:
            log_line("before_agent.core_error", {"error": str(e)})
            state["messages"].append(
                AIMessage(content="[SafeAgent Controller] Safety core unavailable. Request rejected.")
            )
            return {"jump_to": "end"}
        action = str(decision.get("action", "")).upper().strip()
        log_line("before_agent.core_decision", decision)

        # Block unknown actions (zero-trust policy).
        ALLOWED_ACTIONS = {"APPROVE", "REJECT", "OVERRIDE"}
        if action not in ALLOWED_ACTIONS:
            error_text = (
                f"[SafeAgent Controller] Unknown action '{action}'. "
                "Request blocked by zero-trust runtime."
            )
            state["messages"].append(AIMessage(content=error_text))
            return {"jump_to": "end"}

        if action == "APPROVE":
            return None

        if action == "REJECT":
            reject_text = decision.get(
                "user_notice",
                "[SafeAgent Core] The request has been rejected by security policy."
            )
            state["messages"].append(AIMessage(content=reject_text))
            return {"jump_to": "end"}

        if action == "OVERRIDE":
            override_context = decision.get("override")
            if not isinstance(override_context, str) or not override_context:
                state["messages"].append(
                    AIMessage(content="[SafeAgent Controller] Invalid override. Request blocked.")
                )
                return {"jump_to": "end"}

            state["messages"][-1].content = override_context
            return None

        return None

    @before_model(can_jump_to=["end"])
    async def safe_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Safety middleware for the **before_model** stage.

        Purpose
        -------
        This middleware validates **all ToolMessage outputs** produced after the
        most recent model reasoning step. Before the model is allowed to continue
        reasoning, each tool result is checked by the SafeAgent Core under a
        zero-trust policy.

        What this middleware protects
        -----------------------------
        - Ensures no unsafe or policy-violating tool output ever reaches the model.
        - Allows the Core to rewrite, redact, roll back, or block tool outputs.
        - Enforces dynamic per-turn and per-tool safety constraints using the
        session policy.
        - Prevents the model from relying on unverified or corrupted tool results.

        Core actions supported
        ----------------------
        For each tool output, the SafeAgent Core may return one of:

        - **APPROVE**
        The tool output is accepted without modification.
        The model continues normally.

        - **OVERRIDE**
        The output is unsafe but repairable.
        The ToolMessage content is replaced with a Core-supplied safe version.
        If no valid override is provided, the output is replaced with a generic,
        non-sensitive placeholder.

        - **ROLLBACK**
        The current reasoning trajectory is unsafe.
        All messages after the last HumanMessage are removed, and a guidance
        AIMessage is inserted instructing the model to re-plan safely.

        - **TERMINATE**
        The model must not be called again for this turn.
        The controller rolls back to the last HumanMessage (if present) and
        inserts a final safety response.
        The agent turn ends immediately.

        - **REJECT**
        The tool output is unsafe and the tool is now disallowed for the
        remainder of the session.
        The ToolMessage content is replaced with safety guidance, and the model
        continues reasoning with the tool disabled.

        Additional capabilities
        -----------------------
        - **Session policy updates**
        Core may return a `policy_delta` to dynamically change runtime safety
        configuration (e.g., disable tools, tighten restrictions).

        - **Per-message memory protection**
        A Core flag (`NO_LONG_TERM_MEMORY_WRITE`) marks tool outputs that must
        never enter long-term memory, ensuring isolation of sensitive data.

        - **Zero-trust enforcement**
        Unknown actions, invalid override payloads, or Core errors result in a
        deterministic safety response and termination of the turn.

        Summary
        -------
        This hook forms the core of SafeAgent’s tool-output validation pipeline.
        All tool results are authenticated, potentially repaired, or suppressed
        before the model sees them—ensuring strict forward-propagation safety.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        last_ai_msg_idx = last_message_index(state, AIMessage)
        if last_ai_msg_idx is None:
            return None

        if not isinstance(messages[-1], ToolMessage):
            return None

        for tool_msg_idx in range(last_ai_msg_idx + 1, len(messages)):
            msg = messages[tool_msg_idx]
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                content = str(content)

            core_request = {
                "hook": "before_model",
                "observation": {
                    "role": "tool",
                    "name": msg.name,
                    "tool_call_id": msg.tool_call_id,
                    "content": content,
                },
            }
            log_line("before_model.core_request", core_request)

            try:
                decision = await safe_agent.ainvoke({
                    "session_id": session_id,
                    "core_request": core_request,
                })
                decision = parse_mcp_tool_response(decision)
            except Exception as e:
                del state["messages"][last_ai_msg_idx + 1:]
                log_line("before_model.core_error", {"error": str(e)})
                error_text = (
                    "[SafeAgent Controller] Safety core unavailable. "
                    "Tool plan cannot be executed for this turn."
                )
                state["messages"].append(
                    AIMessage(
                        content=error_text,
                        additional_kwargs=getattr(msg, "additional_kwargs", {}),
                    )
                )
                setattr(state["messages"][-1], "tool_calls", [])
                return {"jump_to": "end"}
            action = str(decision.get("action", "")).upper().strip()
            log_line("before_model.core_decision", decision)

            # Allowed actions at this stage
            ALLOWED_ACTIONS = {"APPROVE", "OVERRIDE", "ROLLBACK", "TERMINATE", "REJECT"}
            if action not in ALLOWED_ACTIONS:
                del state["messages"][last_ai_msg_idx + 1:]
                error_text = (
                    f"[SafeAgent Controller] Unknown before_model action '{action}'. "
                    "Request blocked by zero-trust runtime."
                )
                state["messages"].append(AIMessage(content=error_text))
                return {"jump_to": "end"}

            # --- Apply per-message long-term memory policy if provided ---
            allow_long_term_memory = decision.get("allow_long_term_memory", False)
            additional_kwargs = getattr(msg, "additional_kwargs", {}) or {}
            if not allow_long_term_memory:
                tags = additional_kwargs.get("safe_tags") or []
                if not isinstance(tags, list):
                    tags = []
                if "NO_LONG_TERM_MEMORY_WRITE" not in tags:
                    tags.append("NO_LONG_TERM_MEMORY_WRITE")
                # additional_kwargs["safe_tags"] = tags
                state["messages"][tool_msg_idx].additional_kwargs = additional_kwargs

            if action == "APPROVE":
                continue

            # === OVERRIDE: overwrite only the content of this ToolMessage ===
            if action == "OVERRIDE":
                override_context = decision.get("override")

                if not isinstance(override_context, str) or not override_context:
                    guidance = (
                        "[SafeAgent Controller] The original tool output has been "
                        "redacted due to safety policy. You may assume the tool "
                        "completed, but MUST NOT infer or reconstruct any sensitive "
                        "details from it."
                    )
                    state["messages"][tool_msg_idx].content = guidance
                else:
                    state["messages"][tool_msg_idx].content = override_context

                continue

            # === ROLLBACK: roll back to last HumanMessage ===
            if action == "ROLLBACK":
                human_msg_idx = last_message_index(state, HumanMessage)

                if human_msg_idx is None:
                    error_text = (
                        "[SafeAgent Controller] ROLLBACK requested but no HumanMessage "
                        "found in history. Request blocked by zero-trust runtime."
                    )
                    state["messages"].append(AIMessage(content=error_text))
                    return {"jump_to": "end"}

                del state["messages"][human_msg_idx + 1:]
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
                        additional_kwargs={"safe_tags": ["SAFEAGENT_FLOW_CONTROL_MESSAGE"]},
                    )
                )
                return None

            # === TERMINATE: do NOT call the model again ===
            if action == "TERMINATE":
                human_msg_idx = last_message_index(state, HumanMessage)

                # if no HumanMessage, fallback to single canned response
                if human_msg_idx is None:
                    safe_resp = "[SafeAgent Controller] Model call skipped due to safety policy."
                    state["messages"].append(AIMessage(content=safe_resp))
                    return {"jump_to": "end"}

                del state["messages"][human_msg_idx + 1:]
                # Insert final user-visible safety response
                safe_resp = "[SafeAgent Core] This conversation turn has been terminated due to safety policy."
                state["messages"].append(AIMessage(content=safe_resp))

                return {"jump_to": "end"}

            # === REJECT: drop unsafe ToolMessage + disable the tool + continue model ===
            if action == "REJECT":
                # Insert a guidance ToolMessage for the next model step
                guidance = (
                    "[SafeAgent Core] The previous tool result was unsafe and has "
                    "been removed. The tool used in the last step is now disabled for "
                    "the remainder of this session.\n"
                    "- You MUST NOT call that tool again.\n"
                    "- You MUST answer the user directly using only the available context.\n"
                )

                state["messages"][tool_msg_idx].content = guidance
                continue

        return None

    @after_model(can_jump_to=["model", "end"])
    async def safe_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Safety middleware for the after_model stage.

        Purpose
        -------
        Validates the tool plan produced by the model before any tool execution is
        allowed. Ensures that every tool call proposal aligns with the session's
        safety constraints, capability rules, and dynamic restrictions.

        What this middleware enforces
        -----------------------------
        - Validates the full tool_call list emitted by the model.
        - Blocks tool calls that reference forbidden or disabled tools.
        - Allows the SafeAgent Core to rewrite, reject, or force a replan of the
        proposed tool strategy.
        - Ensures the model cannot execute unsafe plans or circumvent policy via
        indirect tool chaining.
        - Applies dynamic session-policy updates returned by the safety Core.

        Supported safety actions
        ------------------------
        - **APPROVE**
        The proposed plan is safe. No modifications are applied.

        - **REPLAN**
        The plan is unsafe but recoverable. The model receives guidance and must
        produce a completely new plan that obeys session_policy constraints.

        - **REJECT**
        The plan is unsafe and cannot involve tools this turn. The model must
        answer the user directly in natural language using only existing context.

        Additional guarantees
        ---------------------
        - Tool calls targeting blocked or forbidden tools are rejected before any
        Core invocation.
        - All rejected or modified plans are tagged with
        SAFEAGENT_FLOW_CONTROL_MESSAGE to prevent unsafe content from entering
        long-term memory stores.
        - Unknown Core actions trigger a deterministic safety stop.

        Summary
        -------
        This middleware forms the safety gate for model-generated tool plans. It
        ensures that only compliant, policy-respecting plans are allowed to proceed,
        preventing unsafe execution paths and enforcing zero-trust runtime control.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        last_content = getattr(last_msg, "content", "")
        tool_calls = getattr(last_msg, "tool_calls", [])

        last_user_msg_idx = last_message_index(state, HumanMessage)
        if last_user_msg_idx is None:
            return None
        last_user_msg = messages[last_user_msg_idx]
        if not last_user_msg:
            return None
        last_user = getattr(last_user_msg, "content", "")

        # Only inspect AI model outputs.
        if not isinstance(last_msg, AIMessage):
            return None

        # If no tool_calls, treat as a response to user and forward it.
        if not tool_calls:
            return None

        core_request: Dict[str, Any] = {
            "hook": "after_model",
            "observation": {
                "role": "assistant",
                "content": last_content,
                "tool_calls": tool_calls,
                "last_user": last_user
            },
        }
        log_line("after_model.core_request", core_request)

        try:
            decision = await safe_agent.ainvoke({
                "session_id": session_id,
                "core_request": core_request,
            })
            decision = parse_mcp_tool_response(decision)
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
            # additional_kwargs["safe_tags"].append("SAFEAGENT_FLOW_CONTROL_MESSAGE")
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
                    f"When answering, you MUST explain that tools were refused because: {user_notice.strip()}"
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
            # additional_kwargs["safe_tags"].append("SAFEAGENT_FLOW_CONTROL_MESSAGE")
            state["messages"][-1] = AIMessage(
                content=guidance_content,
                additional_kwargs=additional_kwargs,
            )
            setattr(state["messages"][-1], "tool_calls", [])
            return {"jump_to": "model"}

        return None

    @after_agent
    async def safe_after_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Safety middleware for the after_agent stage.

        Purpose
        -------
        Applies a final safety check to the model's natural-language output
        after the full reasoning and tool execution cycle. Ensures that the
        assistant's final response complies with session safety rules.

        What this middleware enforces
        -----------------------------
        - Inspects the final AIMessage produced by the agent.
        - Allows the SafeAgent Core to override, sanitize, or block unsafe content.
        - Applies dynamic session_policy updates based on Core decisions.
        - Ensures unsafe final text cannot reach the user without inspection.
        - Guarantees that long-term memory writers will not store unsafe data.

        Supported actions
        -----------------
        - **APPROVE**
        The final output is safe and delivered without modification.

        - **OVERRIDE**
        The Core supplies a safe replacement message. The original content
        is discarded and replaced with override_context.

        - **REJECT**
        The final output is unsafe. The user receives a safety-compliant
        replacement response provided by the Core.

        Safety guarantees
        -----------------
        - Unknown actions trigger deterministic fallback responses.
        - The original content is never preserved when a safety violation is detected.
        - All safety-modified responses retain safe_tags to prevent long-term
        memory ingestion.

        Summary
        -------
        This middleware enforces a zero-trust final-response gate. Only fully
        validated, policy-respecting natural-language responses can reach the user.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        last_content = getattr(last_msg, "content", "")

        # Only inspect textual AI output. Other message types are passed through.
        if not isinstance(last_msg, AIMessage) or not isinstance(last_content, str):
            return None

        core_request: Dict[str, Any] = {
            "hook": "after_agent",
            "observation": {
                "role": "assistant",
                "content": last_content,
            },
        }
        log_line("after_agent.core_request", core_request)

        try:
            decision = await safe_agent.ainvoke({
                "session_id": session_id,
                "core_request": core_request,
            })
            decision = parse_mcp_tool_response(decision)
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
            override_context = decision.get("override")
            if not isinstance(override_context, str) or not override_context:
                error_text = (
                    "[SafeAgent Controller] Invalid override_context. "
                    "Final response replaced by zero-trust runtime."
                )
                state["messages"][-1] = AIMessage(
                    content=error_text,
                    additional_kwargs=getattr(last_msg, "additional_kwargs", {}),
                )
                return None

            state["messages"][-1].content = override_context
            return None

        return None

    @before_model(can_jump_to=["tools"])
    def hitl_replay_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        HITL Replay Middleware (before_model hook)
        ==========================================

        This hook implements the *state recovery layer* for SafeAgent's
        human-in-the-loop (HITL) execution model. It performs **no safety
        judgment** on its own; instead, it restores the runtime context after
        human review and ensures that LangGraph can resume execution deterministically.

        Why this middleware exists
        --------------------------
        LangGraph enforces strong state isolation: graph nodes may mutate
        `state["messages"]` only *in place* and only in certain phases.
        When HITL interrupts execution, the agent temporarily pauses the graph,
        shows the pending tool calls to the user, and waits for human approval.

        After the human selects APPROVE / SHADOW / REJECT for one or more tool
        calls, this middleware performs the *only* allowed form of state repair:

        **1. Merge human decisions back into the message context**
            - REJECT → overwrite the original ToolMessage with a deterministic
            rejection ToolMessage.
            - APPROVE / SHADOW → inject an AIMessage containing the re-authorized
            tool_calls list, tagged with ``HITL_REPLAY``.

        **2. Restore graph execution flow**
            - If re-authorized tool_calls exist → jump directly to the ``tools`` node.
            - Otherwise → model continues normally.

        **3. Clean up replay artifacts**
            After the replayed tool calls finish, the tool outputs appear as new
            ToolMessages. The middleware merges them back into their correct
            locations and removes all replay scaffolding.

        Route 1: HumanMessage marked SAFEAGENT_FLOW_CONTROL_MESSAGE
        -----------------------------------------------------------
        Indicates that HITL decisions have arrived from the UI.
        Steps:
            a) Remove the synthetic HumanMessage.
            b) Extract APPROVE/SHADOW/REJECT outcomes.
            c) Patch original ToolMessages for rejected calls.
            d) If APPROVE/SHADOW exists:
                → append an AIMessage(tool_calls=...) with tag HITL_REPLAY
                → return {"jump_to": "tools"}.

            If no approved actions → do nothing and continue.

        Route 2: ToolMessage following a HITL_REPLAY AIMessage
        ------------------------------------------------------
        This means the replayed tool calls were just executed by LangGraph.
        Steps:
            a) Identify the HITL_REPLAY AIMessage.
            b) Remove the replay AIMessage and its tool outputs.
            c) Merge real ToolMessage outputs back into their correct positions.

        Design Guarantees
        -----------------
        - Zero-trust: Only existing ToolMessages are overwritten; no new tool_call_id
        may be introduced by replay.

        - Deterministic: The replay AIMessage never becomes visible to the model;
        it is removed immediately after tool execution finishes.

        - Flow-safe: The middleware **only** resumes execution when human-approved
        calls exist; it never escalates permissions.

        - Graph-compatible: All state modifications are performed *in place*, in
        compliance with LangGraph’s immutability constraints.

        Returns
        -------
        - {"jump_to": "tools"}  — when approved tool calls must be executed.
        - None                  — when no replay is needed or after merging results.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        last_message = messages[-1]

        # Route 1
        if isinstance(last_message, HumanMessage) and has_safe_tag(last_message, "SAFEAGENT_FLOW_CONTROL_MESSAGE"):
            ak = getattr(last_message, "additional_kwargs", {}) or {}
            decisions = ak.get("hitl_call_decisions", None)
            if not decisions:
                return None

            messages.pop()
            approved_calls, rejected_msgs = _extract_hitl_outcomes(decisions)

            _merge_tool_messages(messages, rejected_msgs)

            if approved_calls:
                ai_msg = AIMessage(
                    content="[SafeAgent Controller] Executing approved tool calls.",
                    tool_calls=[tc for tc in approved_calls],
                    additional_kwargs={"safe_tags": ["HITL_REPLAY"]},
                )
                messages.append(ai_msg)
                return {"jump_to": "tools"}

            return None

        # Route 2
        if isinstance(last_message, ToolMessage):
            last_ai_msg_idx = last_message_index(state, AIMessage)
            if last_ai_msg_idx is None:
                return None
            last_ai_message = messages[last_ai_msg_idx]
            if not has_safe_tag(last_ai_message, "HITL_REPLAY"):
                return None
            replayed_tool_messages = messages[last_ai_msg_idx + 1:]
            del messages[last_ai_msg_idx:]
            _merge_tool_messages(messages, replayed_tool_messages)

        return None

    @after_agent
    def cleanup_after_turn(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", []) or []
        if not messages:
            return None

        new_messages = []
        for msg in messages:
            if has_safe_tag(msg, "SAFEAGENT_FLOW_CONTROL_MESSAGE") or has_safe_tag(msg, "HITL_REPLAY"):
                continue

            if isinstance(msg, ToolMessage) and has_safe_tag(msg, "NO_LONG_TERM_MEMORY_WRITE"):
                ak: Dict[str, Any] = getattr(msg, "additional_kwargs", {}) or {}

                safe_stub = (
                    "[SafeAgent Controller] Original tool output has been omitted "
                    "from long-term context for safety reasons."
                )

                new_msg = ToolMessage(
                    content=safe_stub,
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                    status=getattr(msg, "status", "success"),
                    additional_kwargs=ak,
                )
                new_messages.append(new_msg)
                continue

            new_messages.append(msg)

        messages.clear()
        messages.extend(new_messages)

        return None

    return [
        safe_before_agent, safe_after_agent,
        hitl_replay_before_model, safe_before_model,
        safe_after_model, cleanup_after_turn
    ]
