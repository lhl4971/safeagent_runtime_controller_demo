import os
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from agent.prompt import SYSTEM_PROMPT
from agent.middlewares import log_before_agent, log_before_model, log_after_model, log_after_agent


async def setup_agent():
    client = MultiServerMCPClient({
        "local": {
            "url": "http://127.0.0.1:22336/mcp",
            "transport": "streamable_http",
        }
    })
    tools = await client.get_tools()

    model = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ["DEEPSEEK_API_TOKEN"],
        temperature=0,
    )

    memory = MemorySaver()
    middlewares = [log_before_agent, log_before_model, log_after_model, log_after_agent]

    return create_agent(
        model=model,
        tools=tools,
        middleware=middlewares,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
        # interrupt_before=["tools"],
        # interrupt_after=["tools"],
        name="vibe_shell"
    )
