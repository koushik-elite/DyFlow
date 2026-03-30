#!/usr/bin/env python3
"""
run_spider.py — Evaluate DyFlow-T on Spider Text-to-SQL benchmark.

Usage
─────
    # Quick test using the sample e_commerce database (no download needed)
    python scripts/run_spider.py --sample

    # Dev split, first 20 questions, 5 workers
    python scripts/run_spider.py --mode dev --size 20 --workers 5

    # Full Spider dev evaluation
    python scripts/run_spider.py --mode dev

    # Show metrics from a previous run
    python scripts/run_spider.py --metrics-only

Setup
─────
    # Sample DB (instant, no download):
    python scripts/install_spider.py --sample-only

    # Full Spider dataset:
    python scripts/install_spider.py
"""

import sys
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dyflow.model_service import ModelService
from dyflow.core.tool_workflow import ToolAwareWorkflowExecutor
from dyflow.tools.registry import ToolRegistry
from dyflow.tools.web_search import MockWebSearchTool
from dyflow.tools.sql_query import SQLQueryTool, MockSQLQueryTool, SchemaInspector
from benchmarks.spider import SpiderBenchmark

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(REPO_ROOT, "benchmarks", "data", "Spider")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")


# ── Tool registry ──────────────────────────────────────────────────────────────

def build_tool_registry(db_url: str) -> ToolRegistry:
    registry = ToolRegistry()

    # SQL tool — live, pointed at the current benchmark database
    print(f"[Tools] SQLQueryTool → {db_url}")
    registry.register("SQL_QUERY", SQLQueryTool(db_url=db_url, read_only=True))

    # WEB_SEARCH not needed for Spider — register mock to keep registry complete
    registry.register("WEB_SEARCH", MockWebSearchTool())
    print("[Tools] WebSearchTool → mock (not needed for Spider)")

    return registry


# ── Per-question workflow function ─────────────────────────────────────────────

def make_workflow_fn(
    designer_service: ModelService,
    executor_service: ModelService,
    db_root: str,
):
    """
    Returns a callable that runs DyFlow-T for a single Spider question.
    The SQL tool is pointed at the correct per-question database.
    """

    def run_single(enriched_question: str):
        # Extract db_id from the enriched prompt to build the correct DB URL
        db_id = _parse_db_id(enriched_question)
        db_path = os.path.join(db_root, db_id, f"{db_id}.db")
        db_url  = f"sqlite:///{db_path}"

        tool_registry = build_tool_registry(db_url)

        executor = ToolAwareWorkflowExecutor(
            problem_description=enriched_question,
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


def _parse_db_id(prompt: str) -> str:
    """Extract database name from the enriched question prompt."""
    import re
    match = re.search(r"Database:\s*(\S+)", prompt)
    return match.group(1).strip() if match else "unknown"


# ── Sample mode ────────────────────────────────────────────────────────────────

def run_sample(args):
    """Run evaluation on the sample e_commerce database (no Spider download)."""
    sample_db   = os.path.join(SAMPLE_DIR, "e_commerce", "e_commerce.db")
    sample_json = os.path.join(SAMPLE_DIR, "spider_sample.json")

    if not os.path.exists(sample_db) or not os.path.exists(sample_json):
        print("[ERROR] Sample database not found. Run first:")
        print("        python scripts/install_spider.py --sample-only")
        sys.exit(1)

    print("=" * 60)
    print("DyFlow-T  |  Spider (sample e_commerce DB)")
    print("=" * 60)

    # Print schema for reference
    print("\n[Schema] e_commerce database:")
    inspector = SchemaInspector(f"sqlite:///{sample_db}")
    print(inspector.dump())
    print()

    designer_service = ModelService(model="gemini-2.5-flash")
    executor_service = ModelService(model="gemini-2.5-flash")

    # Point SQL tool at the sample database
    db_url = f"sqlite:///{sample_db}"
    tool_registry = build_tool_registry(db_url)

    def run_single_sample(enriched_question: str):
        executor = ToolAwareWorkflowExecutor(
            problem_description=enriched_question,
            designer_service=designer_service,
            executor_service=executor_service,
            tool_registry=tool_registry,
            save_design_history=True,
            max_tool_retries=2,
        )
        final_answer, trajectory = executor.run(max_steps=4)
        design_history = getattr(executor.state, "design_history", [])
        return final_answer, design_history

    # Patch db_root to sample dir
    benchmark = SpiderBenchmark(
        execution_model="gemini-2.5-flash",
        baseline="DyFlow-T-sample",
        mode="sample",
        dataset="spider",
        db_root=SAMPLE_DIR,
    )
    benchmark.dataset_path = sample_json

    # Override output paths to sample results
    results_dir = os.path.join(REPO_ROOT, "benchmarks", "results", "Spider", "sample")
    os.makedirs(results_dir, exist_ok=True)
    benchmark.output_path  = os.path.join(results_dir, "DyFlow-T-sample_gemini-2.5-flash.json")
    benchmark.cost_path    = os.path.join(results_dir, "DyFlow-T-sample_gemini-2.5-flash_cost.json")
    benchmark.metrics_path = os.path.join(results_dir, "DyFlow-T-sample_gemini-2.5-flash_metrics.json")

    benchmark.run(
        generate_service=designer_service,
        function=run_single_sample,
        size=args.size,
        max_workers=args.workers,
    )


# ── Main evaluation ────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("DyFlow-T  |  Spider Benchmark Evaluation")
    print("=" * 60)
    print(f"Mode      : {args.mode}")
    print(f"Dataset   : {args.dataset}")
    print(f"Size      : {args.size or 'all'}")
    print(f"Workers   : {args.workers}")
    print(f"Baseline  : {args.baseline}")
    print("=" * 60 + "\n")

    db_root = os.path.join(DATA_DIR, "databases")
    if not os.path.isdir(db_root):
        print(f"[ERROR] DB root not found: {db_root}")
        print("        Run:  python scripts/install_spider.py")
        sys.exit(1)

    designer_service = ModelService(model="gemini-2.5-flash")
    executor_service = ModelService(model="gemini-2.5-flash")

    benchmark = SpiderBenchmark(
        execution_model="gemini-2.5-flash",
        baseline=args.baseline,
        mode=args.mode,
        dataset=args.dataset,
        db_root=db_root,
    )
    benchmark.executor_service = executor_service

    workflow_fn = make_workflow_fn(designer_service, executor_service, db_root)

    benchmark.run(
        generate_service=designer_service,
        function=workflow_fn,
        size=args.size,
        max_workers=args.workers,
    )


# ── Metrics only ───────────────────────────────────────────────────────────────

def show_metrics(args):
    benchmark = SpiderBenchmark(
        execution_model="gemini-2.5-flash",
        baseline=args.baseline,
        mode=args.mode,
        dataset=args.dataset,
    )
    benchmark.calculate_metrics()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DyFlow-T on Spider Text-to-SQL benchmark"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Run on sample e_commerce DB (no Spider download needed)",
    )
    parser.add_argument(
        "--mode", type=str, default="dev", choices=["dev", "train"],
        help="Dataset split (default: dev)",
    )
    parser.add_argument(
        "--dataset", type=str, default="spider", choices=["spider", "bird"],
        help="Which dataset to evaluate (default: spider)",
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
        help="Skip inference, just print metrics from last saved results",
    )
    args = parser.parse_args()

    if args.metrics_only:
        show_metrics(args)
    elif args.sample:
        run_sample(args)
    else:
        main(args)
