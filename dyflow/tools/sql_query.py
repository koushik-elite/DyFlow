"""
dyflow/tools/sql_query.py
─────────────────────────
SQLQueryTool — executes SQL against a database via SQLAlchemy.

Supports any SQLAlchemy-compatible URL:
  - sqlite:///path/to/file.db
  - postgresql://user:pass@host/db
  - mysql+pymysql://user:pass@host/db

Also provides:
  - SchemaInspector  — dumps table schemas for prompt injection
  - MockSQLQueryTool — deterministic stub for offline testing

Params accepted by execute()
-----------------------------
  query  : str  — the SQL query to run
  limit  : int  — max rows to return (default 50, hard cap 500)
"""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolResult, ToolStatus


# ── Formatting helpers ────────────────────────────────────────────────────────

def _rows_to_table(columns: List[str], rows: List[tuple]) -> str:
    """Render result rows as a simple ASCII table."""
    if not rows:
        return "(0 rows returned)"
    col_widths = [len(c) for c in columns]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    def fmt_row(vals):
        return " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(vals))

    sep   = "-+-".join("-" * w for w in col_widths)
    lines = [fmt_row(columns), sep] + [fmt_row(row) for row in rows]
    return "\n".join(lines)


# ── Live tool (SQLAlchemy) ────────────────────────────────────────────────────

class SQLQueryTool(BaseTool):
    """
    Executes a SQL query against a database using SQLAlchemy.

    Parameters
    ----------
    db_url     : str
        SQLAlchemy connection URL. Falls back to SQL_DB_URL env var.
    max_rows   : int
        Hard cap on returned rows to avoid token overflow.
    read_only  : bool
        If True, rejects any query that is not a SELECT statement.
    """

    tool_name = "SQL_QUERY"
    _HARD_ROW_CAP = 500

    def __init__(
        self,
        db_url:    Optional[str] = None,
        max_rows:  int = 50,
        read_only: bool = True,
    ) -> None:
        self.db_url    = db_url or os.getenv("SQL_DB_URL", "")
        self.max_rows  = min(max_rows, self._HARD_ROW_CAP)
        self.read_only = read_only
        self._engine   = None  # lazy-initialised

    # ── Engine initialisation ─────────────────────────────────────────────────

    def _get_engine(self):
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
            except ImportError:
                raise RuntimeError(
                    "'sqlalchemy' is not installed. Run: pip install sqlalchemy"
                )
            if not self.db_url:
                raise RuntimeError(
                    "No database URL configured. "
                    "Pass db_url= or set the SQL_DB_URL environment variable."
                )
            self._engine = create_engine(self.db_url)
        return self._engine

    # ── Schema inspection ─────────────────────────────────────────────────────

    def get_schema(self) -> str:
        """
        Return a concise schema dump for all tables.
        Inject this into the SQL_QUERY prompt via {schema}.
        """
        try:
            from sqlalchemy import inspect as sa_inspect
            inspector = sa_inspect(self._get_engine())
            lines = []
            for table_name in inspector.get_table_names():
                cols = inspector.get_columns(table_name)
                col_defs = ", ".join(
                    f"{c['name']} {c['type']}" for c in cols
                )
                lines.append(f"  {table_name}({col_defs})")
            return "Tables:\n" + "\n".join(lines) if lines else "(no tables found)"
        except Exception as exc:
            return f"(schema unavailable: {exc})"

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "").strip()
        limit = int(params.get("limit", self.max_rows))
        limit = min(limit, self._HARD_ROW_CAP)

        if not query:
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message="Empty SQL query.",
            )

        # Read-only guard
        if self.read_only:
            first_token = query.lstrip().split()[0].upper()
            if first_token not in ("SELECT", "WITH", "EXPLAIN"):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    raw_output="",
                    tool_name=self.tool_name,
                    query=query,
                    error_message=(
                        f"Read-only mode is enabled. "
                        f"Only SELECT / WITH / EXPLAIN queries are permitted. "
                        f"Received: {first_token}"
                    ),
                )

        try:
            from sqlalchemy import text
            engine = self._get_engine()
            with engine.connect() as conn:
                result = conn.execute(text(query))
                columns = list(result.keys())
                rows    = result.fetchmany(limit)
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message=str(exc),
            )

        if not rows:
            return ToolResult(
                status=ToolStatus.EMPTY,
                raw_output="Query executed successfully. 0 rows returned.",
                tool_name=self.tool_name,
                query=query,
                structured={"columns": columns, "rows": [], "row_count": 0},
            )

        table_str = _rows_to_table(columns, rows)
        structured = {
            "columns":   columns,
            "rows":      [dict(zip(columns, row)) for row in rows],
            "row_count": len(rows),
            "truncated": len(rows) == limit,
        }

        raw = f"Columns: {', '.join(columns)}\nRows returned: {len(rows)}\n\n{table_str}"
        if len(rows) == limit:
            raw += f"\n\n[Results truncated at {limit} rows]"

        return ToolResult(
            status=ToolStatus.SUCCESS,
            raw_output=raw,
            tool_name=self.tool_name,
            query=query,
            structured=structured,
        )


# ── Schema inspector (standalone helper) ─────────────────────────────────────

class SchemaInspector:
    """
    Standalone utility to dump a database schema string suitable for
    injection into the SQL_QUERY prompt {schema} placeholder.

    Usage
    -----
        inspector = SchemaInspector("sqlite:///spider/database/world.db")
        schema_str = inspector.dump()
        # inject into prompt: prompt.format(schema=schema_str, ...)
    """

    def __init__(self, db_url: str) -> None:
        self._tool = SQLQueryTool(db_url=db_url)

    def dump(self) -> str:
        return self._tool.get_schema()


# ── Offline stub ──────────────────────────────────────────────────────────────

class MockSQLQueryTool(BaseTool):
    """
    Returns a deterministic fake result.
    Useful for unit tests when no database is available.
    """

    tool_name = "SQL_QUERY"

    # Tiny in-memory dataset
    MOCK_DATA = {
        "SELECT": {
            "columns": ["id", "name", "value"],
            "rows": [
                (1, "Alpha", 100),
                (2, "Beta",  200),
                (3, "Gamma", 300),
            ],
        }
    }

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        query  = params.get("query", "SELECT *").strip()
        mock   = self.MOCK_DATA["SELECT"]
        cols   = mock["columns"]
        rows   = mock["rows"]
        table  = _rows_to_table(cols, rows)
        raw    = f"[MOCK] Query: {query}\n\nColumns: {', '.join(cols)}\nRows: {len(rows)}\n\n{table}"
        return ToolResult(
            status=ToolStatus.SUCCESS,
            raw_output=raw,
            tool_name=self.tool_name,
            query=query,
            structured={"columns": cols, "rows": [dict(zip(cols, r)) for r in rows], "row_count": len(rows)},
            metadata={"mock": True},
        )
