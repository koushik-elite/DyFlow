import sys
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dyflow import ModelService
from dyflow.core.tool_workflow import ToolAwareWorkflowExecutor
from dyflow.tools.registry import ToolRegistry
from dyflow.tools.web_search import WebSearchTool, MockWebSearchTool
from dyflow.tools.sql_query import SQLQueryTool, MockSQLQueryTool


def build_tool_registry() -> ToolRegistry:
    """
    Build and return a ToolRegistry with WebSearch and SQL tools.

    Uses live tools when API keys / DB URLs are set in the environment,
    falls back to mock tools automatically for offline / local runs.
    """
    registry = ToolRegistry()

    # ── Web Search ────────────────────────────────────────────────────────────
    # Support both SERPAPI_API_KEY (new) and SERPER_API_KEY (legacy) env var names
    serpapi_key = os.getenv("SERPAPI_API_KEY", "") or os.getenv("SERPER_API_KEY", "")
    if serpapi_key:
        print(f"[Tools] WebSearchTool → live (SerpAPI) key=...{serpapi_key[-6:]}")
        registry.register("WEB_SEARCH", WebSearchTool(api_key=serpapi_key))
    else:
        print("[Tools] WebSearchTool → mock")
        print("[Tools]   ⚠  Set SERPAPI_API_KEY in your .env for live search")
        registry.register("WEB_SEARCH", MockWebSearchTool())

    # ── SQL Query ─────────────────────────────────────────────────────────────
    db_url = os.getenv("SQL_DB_URL", "")
    if db_url:
        print(f"[Tools] SQLQueryTool → live ({db_url})")
        registry.register("SQL_QUERY", SQLQueryTool(db_url=db_url, read_only=True))
    else:
        print("[Tools] SQLQueryTool → mock (set SQL_DB_URL for live database)")
        registry.register("SQL_QUERY", MockSQLQueryTool())

    return registry


def main():
    """
    Run DyFlow-T (tool-augmented) on a sample problem that exercises
    both web search and SQL retrieval.
    """
    # Problem that benefits from both web search and structured data lookup
    problem_description = """
    Answer the following research question using available tools:

    What is the current population of Tokyo, and how does it compare
    to the populations of New York City and London?
    Provide a summary table with the three cities and their populations.
    """

    print("=" * 60)
    print("DyFlow-T  |  Tool-Augmented Workflow")
    print("=" * 60)

    # ── Model services (both Gemini) ──────────────────────────────────────────
    designer_service = ModelService(model="gemini-2.5-flash")
    executor_service = ModelService(model="gemini-2.5-flash")

    # ── Tool registry (WebSearch + SQL) ───────────────────────────────────────
    tool_registry = build_tool_registry()
    print(f"[Tools] Registered: {tool_registry.registered_tools()}\n")

    # ── ToolAwareWorkflowExecutor ─────────────────────────────────────────────
    executor = ToolAwareWorkflowExecutor(
        problem_description=problem_description,
        designer_service=designer_service,
        executor_service=executor_service,
        tool_registry=tool_registry,
        save_design_history=True,
        max_tool_retries=2,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    print("Starting DyFlow-T execution...")
    print("-" * 60)
    final_answer, trajectory = executor.run(max_steps=4)

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("WORKFLOW EXECUTION COMPLETE")
    print("=" * 60)

    print("\n=== Final Answer ===")
    print(final_answer)

    print("\n=== Tool Usage Summary ===")
    tool_results = getattr(executor.state, "tool_results", {})
    if tool_results:
        for key, result in tool_results.items():
            print(f"  [{key}] {result.tool_name} | status={result.status.value} | elapsed={result.elapsed_sec:.2f}s")
    else:
        print("  No tool calls were made.")

    print("\n=== Operator Execution Log ===")
    for entry in trajectory[-5:]:   # show last 5 log entries
        print(f"  {entry.get('operator_type', '?')} [{entry.get('operator_id', '?')}] → {entry.get('status', '?')}")


if __name__ == "__main__":
    main()
