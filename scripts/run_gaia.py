#!/usr/bin/env python3
"""
Run DyFlow-T on the GAIA benchmark.

Usage
─────
# Validation split, first 20 questions, 5 workers
python scripts/run_gaia.py --mode validation --size 20 --workers 5

# Full validation run
python scripts/run_gaia.py --mode validation

# Show metrics from a previous run (no new inference)
python scripts/run_gaia.py --metrics-only

Dataset
───────
Download from https://huggingface.co/datasets/gaia-benchmark/GAIA
Place at:  benchmarks/data/GAIA/GAIA_validation.json
           benchmarks/data/GAIA/GAIA_test.json

Each record must contain: task_id, question, final_answer, level
"""

import sys
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dyflow.model_service import ModelService
from dyflow.core.tool_workflow import ToolAwareWorkflowExecutor
from dyflow.tools.registry import ToolRegistry
from dyflow.tools.web_search import WebSearchTool, MockWebSearchTool
from dyflow.tools.sql_query import MockSQLQueryTool
from benchmarks.gaia import GAIABenchmark


# ── Tool registry ──────────────────────────────────────────────────────────────

def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    # Support both SERPAPI_API_KEY (new) and SERPER_API_KEY (legacy) env var names
    serpapi_key = os.getenv("SERPAPI_API_KEY", "") or os.getenv("SERPER_API_KEY", "")
    if serpapi_key:
        print(f"[Tools] WebSearchTool → live (SerpAPI) key=...{serpapi_key[-6:]}")
        registry.register("WEB_SEARCH", WebSearchTool(api_key=serpapi_key))
    else:
        print("[Tools] WebSearchTool → mock")
        print("[Tools]   ⚠  Set SERPAPI_API_KEY in your .env for live search")
        registry.register("WEB_SEARCH", MockWebSearchTool())

    # GAIA does not primarily test SQL — register mock to keep registry complete
    registry.register("SQL_QUERY", MockSQLQueryTool())
    print("[Tools] SQLQueryTool  → mock (not needed for GAIA)")

    return registry


# ── Workflow function (called once per question) ───────────────────────────────

def make_workflow_fn(
    designer_service: ModelService,
    executor_service: ModelService,
    tool_registry: ToolRegistry,
):
    """Return a callable that runs DyFlow-T on a single GAIA question."""

    def run_single(question: str):
        executor = ToolAwareWorkflowExecutor(
            problem_description=question,
            designer_service=designer_service,
            executor_service=executor_service,
            tool_registry=tool_registry,
            save_design_history=True,
            max_tool_retries=2,
        )
        final_answer, trajectory = executor.run(max_steps=4)
        design_history = getattr(executor.state, "design_history", [])
        return final_answer, design_history

    return run_single


# ── Metrics-only mode ──────────────────────────────────────────────────────────

def show_metrics(args):
    benchmark = GAIABenchmark(
        execution_model="gemini-2.5-flash",
        baseline=args.baseline,
        mode=args.mode,
    )
    benchmark.calculate_metrics()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("DyFlow-T  |  GAIA Benchmark Evaluation")
    print("=" * 60)
    print(f"Mode      : {args.mode}")
    print(f"Size      : {args.size or 'all'}")
    print(f"Workers   : {args.workers}")
    print(f"Baseline  : {args.baseline}")
    print("=" * 60 + "\n")

    # ── Services ───────────────────────────────────────────────────────────────
    designer_service = ModelService(model="gemini-2.5-flash")
    executor_service = ModelService(model="gemini-2.5-flash")
    judge_service    = ModelService(model="gemini-2.5-flash")

    # ── Tools ──────────────────────────────────────────────────────────────────
    tool_registry = build_tool_registry()

    # ── Benchmark ──────────────────────────────────────────────────────────────
    benchmark = GAIABenchmark(
        execution_model="gemini-2.5-flash",
        baseline=args.baseline,
        mode=args.mode,
    )
    benchmark.executor_service = executor_service

    # ── Workflow function ──────────────────────────────────────────────────────
    workflow_fn = make_workflow_fn(designer_service, executor_service, tool_registry)

    # ── Run ────────────────────────────────────────────────────────────────────
    benchmark.run(
        generate_service=designer_service,
        judge_service=judge_service,
        function=workflow_fn,
        size=args.size,
        max_workers=args.workers,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DyFlow-T on GAIA benchmark")
    parser.add_argument(
        "--mode", type=str, default="validation", choices=["validation", "test"],
        help="Dataset split to evaluate on (default: validation)",
    )
    parser.add_argument(
        "--size", type=int, default=None,
        help="Number of questions to evaluate (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Max parallel workers (default: 5)",
    )
    parser.add_argument(
        "--baseline", type=str, default="DyFlow-T",
        help="Baseline label for result filenames (default: DyFlow-T)",
    )
    parser.add_argument(
        "--metrics-only", action="store_true",
        help="Skip inference, just print metrics from the last saved results",
    )
    args = parser.parse_args()

    if args.metrics_only:
        show_metrics(args)
    else:
        main(args)
