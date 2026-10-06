"""Our own MCP server: exposes the same tools the agent uses, so any MCP client
(Claude Desktop, an IDE, another agent) can search the PDFs and query the database.
Run: uv run python run_server.py mcp"""

from mcp.server.fastmcp import FastMCP

from src.agent.tools import LOCAL_TOOLS
from src.database.schema import init_db

mcp = FastMCP("ml-rag-agent")
for fn in LOCAL_TOOLS:
    mcp.tool()(fn)


if __name__ == "__main__":
    init_db()
    mcp.run(transport="stdio")
