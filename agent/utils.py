from datetime import datetime
from typing import Any
from langchain_core.runnables import RunnableLambda

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
