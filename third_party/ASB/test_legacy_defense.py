
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from third_party.ASB.backbone import (
    ASBAgentConfig,
    load_agent_config,
    load_tool_specs_from_jsonl,
    load_attacker_tools_from_jsonl,
    judge_task_success,
    attack_happened,
    load_agent_tasks,
)

# from third_party.llama.backbone import llama_guard_before_model, llama_guard_after_agent
# from third_party.llm_guard.backbone import llm_guard_before_model, llm_guard_after_agent

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

DIRECT_INJECTION = False
INDIRECT_INJECTION = True
MEMORY_POISIONING = False


# -----------------------------
# 0) Models
# -----------------------------
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


# -----------------------------
# 1) Cache per-agent (system prompt only)
# -----------------------------
class _AgentCache:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt


_AGENT_CACHE: Dict[str, _AgentCache] = {}


def _get_agent_cache(agent_name: str) -> _AgentCache:
    if agent_name in _AGENT_CACHE:
        return _AGENT_CACHE[agent_name]

    agent_cfg: ASBAgentConfig = load_agent_config(agent_name)
    system_prompt = agent_cfg.build_system_prompt()

    _AGENT_CACHE[agent_name] = _AgentCache(system_prompt)
    return _AGENT_CACHE[agent_name]


def _build_normal_tools_for_agent(
    agent_name: str,
    injection: Optional[str] = None,
) -> List:
    """
    Always build tools fresh so different injections can be applied per run.
    """
    normal_specs = load_tool_specs_from_jsonl(agent_name)
    return [t.to_langchain_tool(injection) for t in normal_specs]


# -----------------------------
# 2) One job
# -----------------------------
def run_one_case(
    agent_name: str,
    task: str,
    attacker_idx: Optional[int],   # -1 or None => clean
    direct_injection: bool,
    indirect_injection: bool,
    memory_poisioning: bool
) -> Tuple[bool, bool]:
    cache = _get_agent_cache(agent_name)

    attacker: Optional[Any] = None
    attacker_tool_name: Optional[str] = None
    injection_payload: Optional[str] = None

    # --- load attacker spec (if any) ---
    if attacker_idx is not None and attacker_idx >= 0:
        attacker_specs = load_attacker_tools_from_jsonl(
            agent_name, attacker_indices=[attacker_idx]
        )
        if not attacker_specs:
            raise RuntimeError(f"No attacker tool for agent={agent_name}, idx={attacker_idx}")
        attacker = attacker_specs[0]
        attacker_tool_name = attacker.tool_name

        # IPI: tool output injection payload
        if indirect_injection:
            # 用 attacker 的注入模板（你之前的 build_direct_injection_prompt 也可以复用，
            # 但这里语义上更像“注入到tool output里的指令”）
            injection_payload = attacker.build_injection_prompt()

    # --- build normal tools (per-run) ---
    # 要求：IPI 为真时才“在 tools 中添加 injection 字段（即 tool output 末尾追加注入）”，否则不加
    tools = _build_normal_tools_for_agent(
        agent_name,
        injection=injection_payload if indirect_injection else None,
    )

    # --- optionally add attacker tool itself ---
    if attacker is not None:
        tools.append(attacker.to_langchain_tool())

    agent = create_agent(
        model=BASE_MODEL,
        tools=tools,
        system_prompt=cache.system_prompt,
        # middleware=[llama_guard_before_model, llama_guard_after_agent]
        # middleware=[llm_guard_before_model, llm_guard_after_agent]
    )

    # DPI: direct prompt injection => rewrite user query (append injection)
    user_query = task
    if attacker is not None and direct_injection:
        user_query = f"{task} {attacker.build_injection_prompt()}"

    if attacker is not None and memory_poisioning:
        messages = [AIMessage(attacker.build_poisoned_message(task))]
    else:
        messages = []

    messages.append(HumanMessage(content=user_query))
    result = agent.invoke({"messages": messages})
    msgs: List[BaseMessage] = result.get("messages", []) if isinstance(result, dict) else []

    attack_ok = attack_happened(msgs, attacker_tool_name) if attacker_tool_name else False
    task_ok = bool(judge_task_success(msgs, task, JUDGE_MODEL))
    return attack_ok, task_ok


# -----------------------------
# 3) Parallel sweep (10 workers)
# -----------------------------

attack_start, attack_end = 35, 39
max_workers = 64

cases = load_agent_tasks()  # [{"agent_name":..., "task":...}, ...]

idx_list: List[int] = range(attack_start, attack_end + 1)

jobs: List[Tuple[str, str, int]] = []
for item in cases:
    for idx in idx_list:
        jobs.append((item["agent_name"], item["task"], idx))

total = 0
attack_succ = 0
task_succ = 0
both_succ = 0

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = {
        ex.submit(run_one_case, agent_name, task, idx, DIRECT_INJECTION, INDIRECT_INJECTION, MEMORY_POISIONING): (agent_name, idx)
        for (agent_name, task, idx) in jobs
    }

    for fut in tqdm(as_completed(futures), total=len(futures), desc=f"ASB sweep ({max_workers} threads)"):
        agent_name, idx = futures[fut]
        total += 1
        try:
            attack_ok, task_ok = fut.result()
            attack_succ += int(attack_ok)
            task_succ += int(task_ok)
            both_succ += int(attack_ok and task_ok)
        except Exception as e:
            print(f"[WARN] agent={agent_name} idx={idx} crashed: {type(e).__name__}: {e}")

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
print("=" * 60)
