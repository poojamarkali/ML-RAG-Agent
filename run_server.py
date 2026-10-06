"""One entry point for everything.

  uv run python run_server.py ingest                      # every PDF in data/documents (strategy from config/chunking.json)
  uv run python run_server.py ingest --file "data/documents/x.pdf" --strategy semantic
  uv run python run_server.py ingest --sql "SELECT * FROM model_benchmarks" --name benchmarks
  uv run python run_server.py ingest --reset              # wipe the collection first
  uv run python run_server.py api                         # FastAPI on :8000
  uv run python run_server.py ui                          # Streamlit on :8501 (needs the api running)
  uv run python run_server.py eval                        # retrieval metrics (hit rate, MRR)
  uv run python run_server.py eval --llm                  # + LLM-as-judge (calls the LLM)
  uv run python run_server.py eval --chunking             # compare chunking strategies (slow)
  uv run python run_server.py eval --chunking --file "x.pdf" --pages 140-210 --only "fixed 600,agentic"
  uv run python run_server.py mcp                         # our MCP server over stdio
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ingest", "api", "ui", "eval", "mcp"])
    parser.add_argument("--file", help="ingest: only this PDF")
    parser.add_argument("--strategy", help="ingest: chunking strategy (default: config/chunking.json)")
    parser.add_argument("--sql", help="ingest: a SELECT whose rows become searchable chunks")
    parser.add_argument("--name", default="table", help="ingest: name for the --sql source")
    parser.add_argument("--reset", action="store_true", help="ingest: delete all indexed chunks first")
    parser.add_argument("--llm", action="store_true", help="eval: also run LLM-as-judge")
    parser.add_argument("--chunking", action="store_true", help="eval: compare chunking strategies")
    parser.add_argument("--pages", help="eval --chunking: page range like 140-210")
    parser.add_argument("--only", help='eval --chunking: comma-separated strategies, e.g. "fixed 600,semantic"')
    args = parser.parse_args()

    if args.command == "ingest":
        from src.database.schema import init_db
        from src.ingestion import pipeline
        from src.retrieval.vector_store import reset_collection

        init_db()
        if args.reset:
            reset_collection()
        if args.sql:
            pipeline.ingest_sql(args.sql, args.name)
        elif args.file:
            pipeline.ingest_pdf(args.file, args.strategy)
        else:
            pipeline.ingest_folder(strategy=args.strategy)

    elif args.command == "api":
        import uvicorn

        uvicorn.run("src.api.main:app", host=settings.api_host, port=settings.api_port)

    elif args.command == "ui":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"])

    elif args.command == "eval":
        from src.evaluation import chunking_eval, llm_eval, retrieval_eval

        if args.chunking:
            kwargs = {}
            if args.file:
                kwargs["file_name"] = Path(args.file).name
            if args.pages:
                kwargs["pages"] = tuple(int(x) for x in args.pages.split("-"))
            if args.only:
                kwargs["candidates"] = [s.strip() for s in args.only.split(",")]
            chunking_eval.print_report(chunking_eval.compare_chunking(**kwargs))
        else:
            retrieval_eval.print_report(retrieval_eval.compare_strategies())
        if args.llm:
            llm_eval.print_report(llm_eval.run_llm_eval())

    elif args.command == "mcp":
        from src.database.schema import init_db
        from src.mcp_server.server import mcp

        init_db()
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
