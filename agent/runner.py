import uuid
import asyncio
from typing import Dict, List, Deque, Any, Optional
from langchain_core.messages import HumanMessage
from agent.core import setup_agent
from utils.agent import collect_pending_calls, render_trace

agent = asyncio.run(setup_agent())


async def stream(
    user_msg: str,
    chat: List[Dict],
    trace: List[Dict],
    sid: uuid.UUID,
    call_decisions: Optional[Deque[Dict[str, Any]]],
):
    chat = chat or []
    trace = trace or []
    step = sum(1 for e in trace if "action" in e)

    if call_decisions:
        decisions_list = list(call_decisions)
        meta_msg = HumanMessage(
            content="[SafeAgent HITL] Applying human decisions for pending tool calls.",
            additional_kwargs={
                "hitl_call_decisions": decisions_list,
                "safe_tags": ["SAFEAGENT_FLOW_CONTROL_MESSAGE"],
            },
        )

        agent.update_state(
            {"configurable": {"thread_id": sid}},
            {"messages": [meta_msg]},
        )

        ai_idx = None
        for i in range(len(chat) - 1, -1, -1):
            if chat[i].get("role") == "assistant":
                ai_idx = i
                break
        ai_buf = chat[ai_idx].get("content", "") or ""
    else:
        chat = chat + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": ""},
        ]
        ai_idx = len(chat) - 1
        ai_buf = ""

    yield chat, render_trace(trace), chat, trace, None, None

    first_loop = True
    while True:
        if not call_decisions and first_loop:
            init = {"messages": [HumanMessage(content=user_msg)]}
        else:
            init = None

        first_loop = False
        call_decisions = None

        async for ev in agent.astream_events(
            init, version="v1",
            config={"configurable": {"thread_id": sid}, "recursion_limit": 100}
        ):
            kind = ev.get("event")
            data = ev.get("data", {})
            tool_name = data.get("name") or ev.get("name")
            inputs = data.get("inputs") or data.get("input") or {}
            output = data.get("output")

            if kind == "on_chat_model_stream":
                ch = data.get("chunk")
                if ch and getattr(ch, "content", None):
                    ai_buf += ch.content
                    chat[ai_idx]["content"] = ai_buf
                    yield chat, render_trace(trace), chat, trace, None, None
            elif kind == "on_tool_start":
                step += 1
                trace.append({"step": step, "action": {"tool": tool_name, "args": inputs}})
                yield chat, render_trace(trace), chat, trace, None, None
            elif kind == "on_tool_end":
                if not trace:
                    step = 1
                    trace.append({"step": step})
                trace[-1]["observation"] = {"output": output}
                yield chat, render_trace(trace), chat, trace, None, None

        state = agent.get_state({"configurable": {"thread_id": sid}})
        messages = state.values.get("messages", [])

        pending_calls = collect_pending_calls(messages)
        if len(pending_calls) > 0:
            yield chat, render_trace(trace), chat, trace, pending_calls, None
            return

        next_nodes = getattr(state, "next", None)
        if not next_nodes:
            yield chat, render_trace(trace), chat, trace, None, None
            return
