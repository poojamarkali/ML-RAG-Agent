"""Agent orchestration: the LLM decides which tools to call (documents, SQL,
MCP tools), loops until it has an answer. `create_agent` runs that loop for us."""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from src.agent import cache
from src.agent.llm import get_llm
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import get_all_tools

_agent = None


async def get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent(get_llm(), await get_all_tools(), system_prompt=SYSTEM_PROMPT)
    return _agent


async def ask_agent(question: str, history: list[dict] | None = None) -> dict:
    """history: earlier turns as [{"role": "user"|"assistant", "content": "..."}]."""
    # Cache only standalone questions -- a follow-up like "and its latency?" depends on history.
    if not history and (cached := cache.lookup(question, cache.AGENT_CACHE)):
        return {**cached, "cached": True}

    agent = await get_agent()
    result = await agent.ainvoke({"messages": (history or []) + [{"role": "user", "content": question}]})

    messages = result["messages"]
    answer = messages[-1].text
    tools_used = [
        {"tool": call["name"], "args": call["args"]}
        for m in messages if isinstance(m, AIMessage) for call in m.tool_calls
    ]
    result = {"answer": answer, "tools_used": tools_used}
    if not history:
        cache.save(question, result, cache.AGENT_CACHE)
    return {**result, "cached": False}
