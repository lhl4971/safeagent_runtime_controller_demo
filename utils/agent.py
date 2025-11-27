import json
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Deque
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

LOG_PATH = "webui.log"
safe_agent = RunnableLambda(lambda _: {"action": "APPROVE"})


def log_line(tag: str, data: Any) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        import json
        serialized = json.dumps(data, ensure_ascii=False)
    except Exception as e:
        serialized = f"<unserializable: {e}>"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] [{tag}] {serialized}\n")


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


def collect_pending_calls(messages: list) -> Deque[dict]:
    if not messages:
        return []

    ai_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            ai_idx = i
            break

    if ai_idx is None:
        return []

    pending_call_info = []
    for i in range(ai_idx + 1, len(messages)):
        msg = messages[i]
        if not isinstance(msg, ToolMessage):
            break
        aks = getattr(msg, "additional_kwargs", {}) or {}
        pc = aks.get("pending_call") or {}
        if pc:
            pending_call_info.append({"idx": i, "pending_call": pc, "status": "pending"})

    return deque(pending_call_info)
