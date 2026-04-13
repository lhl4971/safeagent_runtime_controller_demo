import os
import json
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from agent.prompt import SYSTEM_PROMPT
from agent.middlewares import build_safe_agent_middlewares
from agent.tool_warpper import SafeAgentToolWrapperMiddleware


async def setup_agent(session_id: str):
    client = MultiServerMCPClient({
        "local": {
            "url": "http://127.0.0.1:22336/mcp",
            "transport": "streamable_http",
        }
    })
    tools = await client.get_tools()

    from langchain_core.runnables import RunnableLambda

    def _safe_agent(request, config=None):
        hook = request.get("core_request").get("hook", "") or ""
        if hook == "tool_wrapper":
            return [
                {
                    "type": "text",
                    "text": json.dumps({
                        "action": "CALL_JIT_APPROVAL"
                    })
                }
            ]
        return [
            {
                "type": "text",
                "text": json.dumps({
                    "action": "APPROVE",
                    "allow_long_term_memory": True
                })
            }
        ]

    safe_agent = RunnableLambda(_safe_agent)

    # model = ChatOpenAI(
    #     model="deepseek-chat",
    #     base_url="https://api.deepseek.com/v1",
    #     api_key=os.environ["DEEPSEEK_API_TOKEN"],
    #     temperature=0,
    # )

    model = ChatOpenAI(
        model="openai/gpt-oss-120b",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
    )

    memory = MemorySaver()
    middlewares = [
        *build_safe_agent_middlewares(safe_agent, session_id),
        SafeAgentToolWrapperMiddleware(safe_agent, session_id)
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
