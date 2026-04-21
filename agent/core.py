import os
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from agent.prompt import SYSTEM_PROMPT
from agent.middlewares import build_safe_agent_middlewares
from agent.tool_warpper import SafeAgentToolWrapperMiddleware
# from utils.agent import load_yaml, parse_mcp_tool_response, get_safeagent_tools
import yaml
from utils.dummy_safeagent_core import build_dummy_safeagent_core


async def setup_agent(session_id: str):
    client = MultiServerMCPClient({
        "local": {
            "url": "http://127.0.0.1:22336/mcp",
            "transport": "streamable_http",
        }
    })
    tools = await client.get_tools()

    # configs
    # cfg_dir = Path("config")
    # runtime_cfg = load_yaml(cfg_dir / "runtime.yaml")
    # dev_cfg = load_yaml(cfg_dir / "developer.yaml")

    # MCP tools (objects)
    # _client, register_tool, step_tool = await get_safeagent_tools()

    # register_raw = await register_tool.ainvoke(
    #     {
    #         "session_id": session_id,
    #         "runtime_cfg": runtime_cfg,
    #         "dev_cfg": dev_cfg,
    #     }
    # )
    # register = parse_mcp_tool_response(register_raw)

    # if not (isinstance(register, dict) and register.get("ok") is True):
    #     raise RuntimeError(f"register failed: {register}")
    # safeagent_core = step_tool

    with open("config/dummy_safeagent_core.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    safeagent_core = build_dummy_safeagent_core(cfg)

    model = ChatOpenAI(
        model="openai/gpt-oss-120b",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
    )

    memory = MemorySaver()
    middlewares = [
        *build_safe_agent_middlewares(safeagent_core, session_id),
        SafeAgentToolWrapperMiddleware(safeagent_core, session_id)
    ]

    return create_agent(
        model=model,
        tools=tools,
        middleware=middlewares,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
        interrupt_after=["tools"],
        name="vibe_shell"
    )
