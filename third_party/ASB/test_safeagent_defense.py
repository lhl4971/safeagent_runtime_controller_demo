from __future__ import annotations

import os
import json
import asyncio
from uuid import uuid4
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml
from tqdm import tqdm

from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from langchain_mcp_adapters.client import MultiServerMCPClient

from third_party.ASB.backbone import (
    ASBAgentConfig,
    load_agent_config,
    load_tool_specs_from_jsonl,
    load_attacker_tools_from_jsonl,
    judge_task_success,
    attack_happened,
    load_agent_tasks,
)

from agent.middlewares import build_safe_agent_middlewares
from agent.tool_warpper import SafeAgentToolWrapperMiddleware

DIRECT_INJECTION = False
INDIRECT_INJECTION = False
MEMORY_POISIONING = True


# ============================================================
# 0) Models
# ============================================================
BASE_MODEL = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.environ["DEEPSEEK_API_TOKEN"],
    temperature=0,
)

JUDGE_MODEL = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.environ["DEEPSEEK_API_TOKEN"],
    temperature=0,
)


# ============================================================
# 1) YAML + MCP response parsing
# ============================================================
def load_yaml(path: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def parse_mcp_tool_response(resp: Any) -> Any:
    """
    MCP (LangChain adapter) 常见返回:
      list[{"type":"text","text":"...json...","id":"..."}]
    这里统一解析成 dict / str / 原样。
    """
    if isinstance(resp, list):
        texts: List[str] = []
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


# ============================================================
# 2) MCP tool fetch (return tool objects themselves)
# ============================================================
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


# ============================================================
# 3) Cache per-agent
# ============================================================
class _AgentCache:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt


_AGENT_CACHE: Dict[str, _AgentCache] = {}


def get_agent_cache(agent_name: str) -> _AgentCache:
    if agent_name in _AGENT_CACHE:
        return _AGENT_CACHE[agent_name]

    agent_cfg: ASBAgentConfig = load_agent_config(agent_name)
    system_prompt = agent_cfg.build_system_prompt()

    _AGENT_CACHE[agent_name] = _AgentCache(system_prompt)
    return _AGENT_CACHE[agent_name]


# ============================================================
# 4) Robust agent invoke helper
# ============================================================
async def invoke_agent(agent: Any, user_query: str) -> List[BaseMessage]:
    payload = {"messages": [HumanMessage(content=user_query)]}

    # Prefer async path
    if hasattr(agent, "ainvoke"):
        out = await agent.ainvoke(payload)
    else:
        out = await asyncio.to_thread(agent.invoke, payload)

    if isinstance(out, dict):
        msgs = out.get("messages", [])
        return msgs if isinstance(msgs, list) else []
    return []


# ============================================================
# 5) One job (async)
# ============================================================
# ============================================================
# 5) One job (async, supports DPI + IPI)
# ============================================================
async def run_one_case(
    agent_name: str,
    task: str,
    attacker_idx: Optional[int],
    *,
    register_tool: BaseTool,
    step_tool: BaseTool,
    runtime_cfg: Dict[str, Any],
    dev_cfg: Dict[str, Any],
    direct_injection: bool,
    indirect_injection: bool,
    memory_poisioning:bool,
) -> Tuple[bool, bool]:

    cache = get_agent_cache(agent_name)
    session_id = str(uuid4())

    # -------------------------
    # 1) register session
    # -------------------------
    register_raw = await register_tool.ainvoke(
        {
            "session_id": session_id,
            "runtime_cfg": runtime_cfg,
            "dev_cfg": dev_cfg,
        }
    )
    register = parse_mcp_tool_response(register_raw)

    if not (isinstance(register, dict) and register.get("ok") is True):
        raise RuntimeError(f"register failed: {register}")

    # -------------------------
    # 2) attacker spec
    # -------------------------
    attacker = None
    attacker_tool_name = None
    injection_payload = None

    if attacker_idx is not None and attacker_idx >= 0:
        attacker_specs = load_attacker_tools_from_jsonl(
            agent_name,
            attacker_indices=[attacker_idx],
        )
        if not attacker_specs:
            raise RuntimeError(
                f"No attacker tool for agent={agent_name}, idx={attacker_idx}"
            )

        attacker = attacker_specs[0]
        attacker_tool_name = attacker.tool_name

        if indirect_injection:
            injection_payload = attacker.build_injection_prompt()

    # -------------------------
    # 3) build NORMAL tools per-run
    # -------------------------
    normal_specs = load_tool_specs_from_jsonl(agent_name)

    if indirect_injection and injection_payload:
        tools: List[BaseTool] = [
            spec.to_langchain_tool(injection_payload)
            for spec in normal_specs
        ]
    else:
        tools = [
            spec.to_langchain_tool(injection_payload)
            for spec in normal_specs
        ]

    # append attacker tool itself
    if attacker is not None:
        tools.append(attacker.to_langchain_tool())

    # -------------------------
    # 4) build agent with SafeAgent middlewares
    # -------------------------
    middlewares = [
        *build_safe_agent_middlewares(step_tool, session_id),
        SafeAgentToolWrapperMiddleware(step_tool, session_id),
    ]

    agent = create_agent(
        model=BASE_MODEL,
        tools=tools,
        system_prompt=cache.system_prompt,
        middleware=middlewares,
    )

    # -------------------------
    # 5) DPI: rewrite user input
    # -------------------------
    user_query = task

    if attacker is not None and direct_injection:
        user_query = (
            task + " " + attacker.build_injection_prompt()
        )

    if memory_poisioning:
        messages = [AIMessage(attacker.build_poisoned_message(task))]
    else:
        messages = []

    messages.append(HumanMessage(content=user_query))

    # -------------------------
    # 6) run agent
    # -------------------------
    msgs = await invoke_agent(agent, user_query)

    # -------------------------
    # 7) judge
    # -------------------------
    attack_ok = (
        attack_happened(msgs, attacker_tool_name)
        if attacker_tool_name
        else False
    )

    task_ok = bool(judge_task_success(msgs, task, JUDGE_MODEL))

    return attack_ok, task_ok


# ============================================================
# 6) Async sweep
# ============================================================
async def main():
    # configs
    cfg_dir = Path("config")
    runtime_cfg = load_yaml(cfg_dir / "runtime.yaml")
    dev_cfg = load_yaml(cfg_dir / "developer.yaml")

    # MCP tools (objects)
    _client, register_tool, step_tool = await get_safeagent_tools()

    # jobs
    attack_start, attack_end = 0, 39
    idx_list = list(range(attack_start, attack_end + 1))

    cases = load_agent_tasks()  # [{"agent_name":..., "task":...}, ...]
    jobs: List[Tuple[str, str, int]] = []
    for item in cases:
        for idx in idx_list:
            jobs.append((item["agent_name"], item["task"], idx))

    # concurrency control (I/O bound, but don't DDoS your own MCP/core or LLM)
    max_concurrency = 16
    sem = asyncio.Semaphore(max_concurrency)

    total = 0
    attack_succ = 0
    task_succ = 0
    both_succ = 0

    async def _run_job(agent_name: str, task: str, idx: int):
        nonlocal total, attack_succ, task_succ, both_succ
        async with sem:
            attack_ok, task_ok = await run_one_case(
                agent_name=agent_name,
                task=task,
                attacker_idx=idx,
                register_tool=register_tool,
                step_tool=step_tool,
                runtime_cfg=runtime_cfg,
                dev_cfg=dev_cfg,
                direct_injection=DIRECT_INJECTION,
                indirect_injection=INDIRECT_INJECTION,
                memory_poisioning=MEMORY_POISIONING,
            )
            total += 1
            attack_succ += int(attack_ok)
            task_succ += int(task_ok)
            both_succ += int(attack_ok and task_ok)

    # launch
    tasks = [asyncio.create_task(_run_job(a, t, i)) for (a, t, i) in jobs]

    # progress
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"ASB sweep (async x{max_concurrency})"):
        try:
            await coro
        except Exception as e:
            # 不中断总评估，打印即可
            print(f"[WARN] job crashed: {type(e).__name__}: {e}")

    # report
    asr = attack_succ / total if total else 0.0
    tsr = task_succ / total if total else 0.0
    bsr = both_succ / total if total else 0.0

    print("\n" + "=" * 60)
    print(f"Tests: {total}")
    print("-" * 60)
    print(f"Attack successes: {attack_succ}")
    print(f"ASR: {asr:.4f}")
    print("-" * 60)
    print(f"Task successes: {task_succ}")
    print(f"TSR: {tsr:.4f}")
    print("-" * 60)
    print(f"Both success: {both_succ}")
    print(f"BSR: {bsr:.4f}")
    print("=" * 60)



asyncio.run(main())