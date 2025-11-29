import os
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from agent.prompt import SYSTEM_PROMPT
from agent.middlewares import (
    safe_before_agent, safe_after_agent,
    safe_after_model, safe_before_model,
    hitl_replay_before_model)
from agent.tool_warpper import SafeAgentToolWrapperMiddleware


async def setup_agent():
    client = MultiServerMCPClient({
        "local": {
            "url": "http://127.0.0.1:22336/mcp",
            "transport": "streamable_http",
        }
    })
    tools = await client.get_tools()

    from langchain_core.runnables import RunnableLambda

    def _safe_agent(request, config=None):
        return {"action": "CALL_JIT_APPROVAL"}

    safe_agent = RunnableLambda(_safe_agent)

    model = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ["DEEPSEEK_API_TOKEN"],
        temperature=0,
    )

    memory = MemorySaver()
    middlewares = [
        safe_before_agent, safe_after_agent,
        hitl_replay_before_model, safe_before_model,
        safe_after_model,
        SafeAgentToolWrapperMiddleware(safe_agent)
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
