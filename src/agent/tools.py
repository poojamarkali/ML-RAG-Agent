"""Tools the agent can call. Plain functions, so the same ones are also
exposed by our MCP server (src/mcp_server/server.py).

External MCP tools: add servers to config/mcp_servers.json and they are loaded
automatically -- no code change needed. Example:
{
  "filesystem": {"transport": "stdio", "command": "npx",
                 "args": ["-y", "@modelcontextprotocol/server-filesystem", "./data"]},
  "my_http_server": {"transport": "streamable_http", "url": "http://localhost:9000/mcp"}
}
"""

import json

from langchain_core.tools import tool

from config import settings
from src.database.executor import get_schema, run_query
from src.ingestion.pipeline import list_documents
from src.retrieval.search import format_docs, search


def search_documents(query: str, source: str | None = None) -> str:
    """Search the ML PDF documents (hybrid search + re-ranking). Returns passages with
    [file, page] citations. Optionally set `source` to one PDF file name to search only that file."""
    docs = search(query, k=4, source=source)
    return format_docs(docs) if docs else "No relevant passages found."


def list_pdf_documents() -> str:
    """List the PDF file names in the knowledge base (use one as `source` in search_documents)."""
    return "\n".join(list_documents()) or "No documents ingested."


def get_database_schema() -> str:
    """Show the SQL tables and columns. Call this before writing a SQL query."""
    return get_schema()


def query_database(sql: str) -> str:
    """Run a read-only SQL SELECT on the ML benchmark database and return the rows as JSON."""
    try:
        rows = run_query(sql)
    except Exception as e:  # return the error so the agent can fix its query
        return f"SQL error: {e}"
    return json.dumps(rows, indent=2, default=str) if rows else "Query returned 0 rows."


LOCAL_TOOLS = [search_documents, list_pdf_documents, get_database_schema, query_database]


async def load_mcp_tools() -> list:
    """Tools from the external MCP servers listed in config/mcp_servers.json."""
    if not settings.mcp_servers_file.exists():
        return []
    servers = json.loads(settings.mcp_servers_file.read_text())
    if not servers:
        return []
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return await MultiServerMCPClient(servers).get_tools()


async def get_all_tools() -> list:
    return [tool(f) for f in LOCAL_TOOLS] + await load_mcp_tools()
