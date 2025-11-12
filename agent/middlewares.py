from datetime import datetime
from langchain.agents.middleware import before_agent, after_agent, before_model, after_model, AgentState
from langgraph.runtime import Runtime
from typing import Any
import json

LOG_PATH = "webui.log"


def _log_line(tag: str, text: Any) -> None:
    """最简单日志写入函数：每次 append 一行"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        serialized = json.dumps(text, ensure_ascii=False)
    except Exception:
        serialized = str(text)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] [{tag}] {serialized}\n")


def _get_last_content(state: AgentState) -> str:
    """安全地提取最后一条消息内容"""
    try:
        msgs = state.get("messages", [])
        if not msgs:
            return ""
        last = msgs[-1]
        content = getattr(last, "content", last)
        # 如果内容本身是可序列化结构体，就序列化
        if isinstance(content, (dict, list)):
            return json.dumps(content, ensure_ascii=False)
        return str(content)
    except Exception as e:
        return f"<error extracting last message: {e}>"


# ---- 四个中间件 ---- #

@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    _log_line("before_agent", _get_last_content(state))
    return None


@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    _log_line("after_agent", _get_last_content(state))
    return None


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    _log_line("before_model", _get_last_content(state))
    return None


@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    _log_line("after_model", _get_last_content(state))
    return None
