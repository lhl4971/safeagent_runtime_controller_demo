import os
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from agent.prompt import SYSTEM_PROMPT
from agent.middlewares import safe_before_agent, safe_before_model, safe_after_model, safe_after_agent
from agent.attach_runtime_safety import attach_runtime_safety


async def setup_agent():
    client = MultiServerMCPClient({
        "local": {
            "url": "http://127.0.0.1:22336/mcp",
            "transport": "streamable_http",
        }
    })
    tools = await client.get_tools()

    from langchain_core.runnables import RunnableLambda

    def safe_agent(request, config=None):
        print(request)
        return {"action": "ALLOW"}

    dummy_safe_agent = RunnableLambda(safe_agent)

    w_tools = [attach_runtime_safety(tool, dummy_safe_agent) for tool in tools]

    model = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ["DEEPSEEK_API_TOKEN"],
        temperature=0,
    )

    memory = MemorySaver()
    middlewares = [safe_before_agent, safe_before_model, safe_after_model, safe_after_agent]

    return create_agent(
        model=model,
        tools=w_tools,
        middleware=middlewares,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
        name="vibe_shell"
    )
