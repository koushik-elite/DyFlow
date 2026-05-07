import sys
import os
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dyflow import ModelService
from dyflow.core.workflow import WorkflowExecutor
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
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        print("[Tools] WebSearchTool → live (Tavily)")
        registry.register("WEB_SEARCH", WebSearchTool(api_key=tavily_key))
    else:
        print("[Tools] WebSearchTool → mock")
        print("[Tools]   ⚠  Set TAVILY_API_KEY in your .env for live search")
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


def run_dyflow(problem: str, designer: ModelService, executor: ModelService):
    """
    DyFlow baseline — no tools, parametric memory only.
    Returns (answer, design_history).
    """
    wf = WorkflowExecutor(
        problem_description=problem,
        designer_service=designer,
        executor_service=executor,
        save_design_history=True,
    )
    try:
        answer = wf.execute()
        history = getattr(wf.state, "design_history", [])
        return str(answer) if answer else "", history
    except Exception as e:
        return f"ERROR: {e}", []


def run_dyflow_t(problem: str, designer: ModelService, executor: ModelService,
                 tool_registry: ToolRegistry):
    """
    DyFlow-T — tool-augmented (web search + SQL).
    Returns (answer, design_history, trajectory, tool_results).
    """
    wf = ToolAwareWorkflowExecutor(
        problem_description=problem,
        designer_service=designer,
        executor_service=executor,
        tool_registry=tool_registry,
        save_design_history=True,
        max_tool_retries=2,
    )
    try:
        answer, trajectory = wf.run(max_steps=15)
        history      = getattr(wf.state, "design_history", [])
        tool_results = getattr(wf.state, "tool_results", {})
        return answer or "", history, trajectory, tool_results
    except Exception as e:
        return f"ERROR: {e}", [], [], {}


def main():
    """
    Run DyFlow (no tools) and DyFlow-T (tool-augmented) side-by-side
    on a sample problem that exercises both web search and structured lookup.
    """
    
    problem_description = """
    I am a US-based investor with $10,000. 
    Convert it to INR at today's exchange rate, then calculate how many grams of gold I can buy at today's Indian gold rate. 
    What is my investment worth in USD if gold grows '15%' annually over 9 months?
    """

    print("=" * 70)
    print("DyFlow vs DyFlow-T  |  Side-by-side workflow comparison")
    print("=" * 70)
    print("\nProblem:")
    print(problem_description.strip())
    print()

    # ── Model services (both Gemini) ──────────────────────────────────────────
    designer_service = ModelService(model="gemini-2.5-flash")
    executor_service = ModelService(model="gemini-2.5-flash")

    # ── Tool registry (only DyFlow-T uses it) ─────────────────────────────────
    print("-" * 70)
    print("Building tool registry (used by DyFlow-T only)...")
    tool_registry = build_tool_registry()
    print(f"[Tools] Registered: {tool_registry.registered_tools()}")
    print("-" * 70)

    # ── DyFlow baseline (no tools) ────────────────────────────────────────────
    print("\n[1/2] Running DyFlow (no tools — parametric memory only)...")
    t0 = time.time()
    df_answer, df_history = run_dyflow(problem_description, designer_service, executor_service)
    t_df = time.time() - t0
    print(f"      done in {t_df:.1f}s  |  design stages: {len(df_history)}")

    # ── DyFlow-T (web + SQL) ──────────────────────────────────────────────────
    print("\n[2/2] Running DyFlow-T (tool-augmented)...")
    t0 = time.time()
    dft_answer, dft_history, dft_trajectory, dft_tool_results = run_dyflow_t(
        problem_description, designer_service, executor_service, tool_registry
    )
    t_dft = time.time() - t0
    print(f"      done in {t_dft:.1f}s  |  design stages: {len(dft_history)}")

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- DyFlow (no tools) ---")
    print(f"Elapsed : {t_df:.1f}s")
    print(f"Stages  : {len(df_history)}")
    print(f"Answer  :\n{df_answer}\n")

    print("--- DyFlow-T (tool-augmented) ---")
    print(f"Elapsed : {t_dft:.1f}s")
    print(f"Stages  : {len(dft_history)}")
    print(f"Answer  :\n{dft_answer}\n")

    # ── Tool usage summary (DyFlow-T only) ────────────────────────────────────
    print("--- DyFlow-T Tool Usage ---")
    if dft_tool_results:
        for key, result in dft_tool_results.items():
            print(f"  [{key}] {result.tool_name} | status={result.status.value} | elapsed={result.elapsed_sec:.2f}s")
    else:
        print("  No tool calls were made.")

    # ── Operator log (last 5) ─────────────────────────────────────────────────
    print("\n--- DyFlow-T Operator Log (last 5) ---")
    for entry in dft_trajectory[-5:]:
        print(f"  {entry.get('operator_type', '?')} [{entry.get('operator_id', '?')}] → {entry.get('status', '?')}")

    print("=" * 70)


if __name__ == "__main__":
    main()
