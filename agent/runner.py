import uuid
import json
import asyncio
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, BaseMessage
from agent.core import setup_agent

agent = asyncio.run(setup_agent())


def render_trace(events: List[Dict[str, Any]]) -> str:
    md = ["### 🧭 Function Calls"]

    def pretty(obj: Any) -> str:
        if isinstance(obj, BaseMessage):
            obj = json.loads(getattr(obj, "content", obj))
        return json.dumps(obj, ensure_ascii=False, indent=2)

    for ev in events:
        step = ev.get("step", "?")
        if "action" in ev:
            tool = ev["action"].get("tool", "?")
            args = ev["action"].get("args", {})
            md.append(
                f"<details><summary>🤖 Action #{step}: <code>{tool}</code></summary>\n\n"
                f"<pre>{pretty(args)}</pre>\n\n</details>"
            )
        if "observation" in ev:
            out = ev["observation"].get("output", {})
            md.append(
                f"<details><summary>🌍 Observation #{step}</summary>\n\n"
                f"<pre>{pretty(out)}</pre>\n\n</details>"
            )
    return "\n\n".join(md)


async def stream(user_msg: str, chat: List[Dict], trace: List[Dict], sid: uuid.UUID):
    chat = (chat or []) + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": ""}]
    ai_idx = len(chat) - 1
    ai_buf = ""
    trace = trace or []
    step = sum(1 for e in trace if "action" in e)

    yield chat, render_trace(trace), chat, trace

    init = {"messages": [HumanMessage(content=user_msg)]}

    async for ev in agent.astream_events(
        init, version="v1",
        config={"configurable": {"thread_id": sid}}
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
                yield chat, render_trace(trace), chat, trace
        elif kind == "on_tool_start":
            step += 1
            trace.append({"step": step, "action": {"tool": tool_name, "args": inputs}})
            yield chat, render_trace(trace), chat, trace
        elif kind == "on_tool_end":
            if not trace:
                step = 1
                trace.append({"step": step})
            trace[-1]["observation"] = {"output": output}
            yield chat, render_trace(trace), chat, trace

    yield chat, render_trace(trace), chat, trace
