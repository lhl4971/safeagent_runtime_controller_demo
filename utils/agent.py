import json
import yaml
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Deque, Tuple
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain.agents.middleware import AgentState
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

LOG_PATH = "webui.log"


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
        if isinstance(obj, ToolMessage):
            obj = parse_mcp_tool_response(getattr(obj, "content", obj))
        if isinstance(obj, BaseMessage):
            obj = json.loads(getattr(obj, "content", obj))
        return json.dumps(obj, ensure_ascii=False, indent=2)

    hidden_steps = {ev.get("step") for ev in events if "action" in ev and ev["action"].get("tool") == "safeagent_step"}
    filtered_events = [ev for ev in events if ev.get("step") not in hidden_steps]

    step_map = {}
    next_step = 1
    for ev in filtered_events:
        raw_step = ev.get("step")
        if raw_step not in step_map:
            step_map[raw_step] = next_step
            next_step += 1

    for ev in filtered_events:
        step = step_map[ev.get("step")]
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


def has_safe_tag(message: BaseMessage, tag: str) -> bool:
    if not isinstance(tag, str):
        return False

    ak = getattr(message, "additional_kwargs", {}) or {}
    if not isinstance(ak, Dict):
        return False

    safe_tags = ak.get("safe_tags", []) or []
    if not isinstance(safe_tags, List):
        return False

    return tag in safe_tags


def parse_mcp_tool_response(resp: Any) -> Any:
    # LangChain MCP adapter commonly returns: list[{"type":"text","text":"...json..."}]
    if isinstance(resp, list):
        texts = []
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


def load_yaml(path: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


async def get_safeagent_tools(
    mcp_url: str = "http://127.0.0.1:8000/mcp",
    server_name: str = "safeagent-core",
) -> Tuple[MultiServerMCPClient, BaseTool, BaseTool]:
    """
    返回:
      - client
      - register_tool (BaseTool)
      - step_tool     (BaseTool)
    """
    client = MultiServerMCPClient(
        {
            server_name: {
                "url": mcp_url,
                "transport": "streamable_http",
            }
        }
    )

    tools = await client.get_tools()

    register_tool = next((t for t in tools if t.name == "safeagent_register_session"), None)
    step_tool = next((t for t in tools if t.name == "safeagent_step"), None)

    if register_tool is None:
        raise RuntimeError("MCP tools missing: safeagent_register_session")
    if step_tool is None:
        raise RuntimeError("MCP tools missing: safeagent_step")

    return client, register_tool, step_tool
