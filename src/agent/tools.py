"""Tools the agent can call.

Each tool returns two things:
  - text for the LLM to read
  - an "artifact" for us: the PDF passages or SQL rows it used, so the UI can show sources.
The MCP server gets plain versions (text only) of the same tools.

External MCP tools: add servers to config/mcp_servers.json and they are loaded
automatically (used in Research mode). Secrets stay in .env: write "${HF_TOKEN}" in the
JSON and it is replaced by the value from .env. Example:
{
  "huggingface": {"transport": "streamable_http", "url": "https://huggingface.co/mcp",
                  "headers": {"Authorization": "Bearer ${HF_TOKEN}"}}
}
"""

import json
import os
import re

from langchain_core.tools import StructuredTool

from config import settings
from src.database.executor import get_schema, run_query
from src.ingestion.pipeline import list_documents
from src.retrieval.search import format_docs, search


# ---- tools with sources (used by the agent) ------------------------------------
def search_documents_with_sources(query: str, source: str | None = None) -> tuple[str, list[dict]]:
    docs = search(query, k=4, source=source)
    text = format_docs(docs) if docs else "No relevant passages found."
    sources = [{"text": d.page_content, "source": d.metadata.get("source"), "page": d.metadata.get("page"),
                "score": d.metadata.get("score")} for d in docs]
    return text, sources


def query_database_with_rows(sql: str) -> tuple[str, dict]:
    try:
        rows = run_query(sql)
    except Exception as e:  # return the error so the agent can fix its query
        return f"SQL error: {e}", {"sql": sql, "rows": [], "error": str(e)}
    text = json.dumps(rows, indent=2, default=str) if rows else "Query returned 0 rows."
    return text, {"sql": sql, "rows": rows[:50]}


# ---- plain tools (text only) -- also what the MCP server exposes ---------------
def search_documents(query: str, source: str | None = None) -> str:
    """Search the ML PDF documents (hybrid search + re-ranking). Returns passages with
    [file, page] citations. Optionally set `source` to one PDF file name to search only that file."""
    return search_documents_with_sources(query, source)[0]


def query_database(sql: str) -> str:
    """Run one read-only SQL SELECT on the ML benchmark database and return the rows as JSON.
    Model names include their settings, e.g. 'Random Forest (n_estimators=100)' -- use
    LIKE '%Random Forest%' when you don't know the exact name."""
    return query_database_with_rows(sql)[0]


def list_pdf_documents() -> str:
    """List the PDF file names in the knowledge base (use one as `source` in search_documents)."""
    return "\n".join(list_documents()) or "No documents ingested."


def get_database_schema() -> str:
    """Show the SQL tables and columns. Call this before writing a SQL query."""
    return get_schema()


LOCAL_TOOLS = [search_documents, list_pdf_documents, get_database_schema, query_database]


def agent_tools() -> list:
    """The same tools for the agent, but search/query also hand back their sources (artifact)."""
    with_sources = {"search_documents": search_documents_with_sources, "query_database": query_database_with_rows}
    tools = []
    for fn in LOCAL_TOOLS:
        if fn.__name__ in with_sources:
            tools.append(StructuredTool.from_function(
                func=with_sources[fn.__name__], name=fn.__name__, description=fn.__doc__,
                args_schema=StructuredTool.from_function(fn).args_schema,
                response_format="content_and_artifact"))
        else:
            tools.append(StructuredTool.from_function(fn))
    return tools


def fill_env_vars(servers: dict) -> dict:
    """Replace "${NAME}" with the value from .env. A header whose variable is empty is
    dropped, so a server that works without a token (like Hugging Face) still connects."""
    for server in servers.values():
        headers = server.get("headers", {})
        for name, value in list(headers.items()):
            filled = re.sub(r"\$\{(\w+)\}", lambda m: os.getenv(m.group(1), ""), value)
            if re.search(r"\$\{\w+\}", value) and not filled.replace("Bearer", "").strip():
                del headers[name]
            else:
                headers[name] = filled
    return servers


async def load_mcp_tools() -> list:
    """Tools from the external MCP servers in config/mcp_servers.json. If a server can't be
    reached, the app keeps working without its tools instead of failing."""
    if not settings.mcp_servers_file.exists():
        return []
    servers = json.loads(settings.mcp_servers_file.read_text())
    if not servers:
        return []
    from langchain_mcp_adapters.client import MultiServerMCPClient

    try:
        return await MultiServerMCPClient(fill_env_vars(servers)).get_tools()
    except Exception as e:
        print(f"[warn] could not load MCP tools, continuing without them: {e}")
        return []
