"""Agent orchestration: the LLM decides which tools to call (documents, SQL,
MCP tools), loops until it has an answer. `create_agent` runs that loop for us.
This is what the chat page uses."""

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from src.agent import cache
from src.agent.llm import get_llm
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import get_all_tools

_agent = None


async def get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent(
            get_llm(), await get_all_tools(), system_prompt=SYSTEM_PROMPT,
            # Hard limits per question (some models over-search, which only adds seconds):
            # extra calls are blocked and the agent answers with what it already has.
            middleware=[
                ToolCallLimitMiddleware(tool_name="search_documents", run_limit=2),
                ToolCallLimitMiddleware(run_limit=8),
            ],
        )
    return _agent


def collect_sources(messages) -> tuple[list[dict], list[dict]]:
    """Go through the tool results and pull out what each tool used:
    PDF passages (from search_documents) and SQL queries + rows (from query_database)."""
    passages, sql = [], []
    for m in messages:
        if isinstance(m, ToolMessage) and m.artifact:
            if m.name == "search_documents":
                passages += m.artifact
            elif m.name == "query_database":
                sql.append(m.artifact)
    # the same passage can come back from several searches -- keep it once
    unique = {(p["source"], p["page"], p["text"]): p for p in passages}
    return list(unique.values()), sql


async def ask_agent(question: str, history: list[dict] | None = None) -> dict:
    """history: earlier turns as [{"role": "user"|"assistant", "content": "..."}].
    Returns {"answer", "sources" (PDF passages), "sql" (queries + rows), "tools_used", "cached"}."""
    # Cache only standalone questions -- a follow-up like "and its latency?" depends on history.
    if not history and (cached := cache.lookup(question, cache.AGENT_CACHE)):
        return {**cached, "cached": True}

    agent = await get_agent()
    output = await agent.ainvoke({"messages": (history or []) + [{"role": "user", "content": question}]})

    messages = output["messages"]
    sources, sql = collect_sources(messages)
    result = {
        "answer": messages[-1].text,
        "sources": sources,
        "sql": sql,
        "tools_used": [{"tool": call["name"], "args": call["args"]}
                       for m in messages if isinstance(m, AIMessage) for call in m.tool_calls],
    }
    if not history:
        cache.save(question, result, cache.AGENT_CACHE)
    return {**result, "cached": False}
