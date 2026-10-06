"""Read-only SQL for the agent, the API and the MCP server."""

from sqlalchemy import create_engine, event, inspect, text

from config import settings
from src.database.schema import engine

# A second engine whose connections can only read -- the database itself blocks writes,
# so the agent can't change data even if it writes a bad query.
if engine.url.get_backend_name() == "sqlite":
    read_only_engine = create_engine(settings.database_url)
    event.listen(read_only_engine, "connect", lambda conn, _: conn.execute("PRAGMA query_only = ON"))
else:
    read_only_engine = create_engine(
        settings.database_url, connect_args={"options": "-c default_transaction_read_only=on"}
    )


def get_schema() -> str:
    """Every table with its columns -- read live from the DB, so new tables show up automatically."""
    insp = inspect(engine)
    return "\n".join(
        f"{table}({', '.join(f'{c['name']} {c['type']}' for c in insp.get_columns(table))})"
        for table in insp.get_table_names()
    )


def run_query(sql: str, max_rows: int = 200) -> list[dict]:
    """Run one read-only SELECT and return rows as dicts. Raises ValueError for anything else."""
    sql = sql.strip().rstrip(";")
    if ";" in sql or sql.split(maxsplit=1)[0].lower() not in ("select", "with"):
        raise ValueError("Only a single SELECT (or WITH ... SELECT) query is allowed.")
    with read_only_engine.connect() as conn:
        result = conn.execute(text(sql))
        return [dict(row._mapping) for row in result.fetchmany(max_rows)]
