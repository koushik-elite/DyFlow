#!/usr/bin/env python3
"""
compare_spider.py — DyFlow vs DyFlow-T comparison on Spider Text-to-SQL.

Runs both systems on the same Spider questions and saves a JSON report
with side-by-side execution accuracy, per-hardness breakdown, and
per-question correctness for error analysis.

Usage
─────
    # Sample DB (instant, no download)
    python scripts/compare_spider.py --sample

    # Spider dev split, first 50 questions
    python scripts/compare_spider.py --mode dev --size 50 --workers 3

    # View last report
    python scripts/compare_spider.py --report-only
"""

import sys
import os
import json
import datetime
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

from dyflow.model_service import ModelService
from dyflow.core.workflow import WorkflowExecutor
from dyflow.core.tool_workflow import ToolAwareWorkflowExecutor
from dyflow.tools.registry import ToolRegistry
from dyflow.tools.web_search import MockWebSearchTool
from dyflow.tools.sql_query import SQLQueryTool, MockSQLQueryTool
from benchmarks.spider import SpiderBenchmark

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(REPO_ROOT, "benchmarks", "data", "Spider")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
REPORT_DIR = os.path.join(REPO_ROOT, "benchmarks", "results", "Spider", "comparison")


# ── Workflow functions ─────────────────────────────────────────────────────────

def make_dyflow_fn(designer_service: ModelService, executor_service: ModelService):
    """DyFlow baseline — original WorkflowExecutor, no tools."""
    def run(enriched_question: str):
        executor = WorkflowExecutor(
            problem_description=enriched_question,
            designer_service=designer_service,
            executor_service=executor_service,
            save_design_history=True,
        )
        try:
            answer = executor.execute()
            history = getattr(executor.state, "design_history", [])
            return str(answer) if answer else "", history
        except Exception as e:
            return f"ERROR: {e}", []
    return run


def make_dyflow_t_fn(
    designer_service: ModelService,
    executor_service: ModelService,
    db_root: str,
):
    """DyFlow-T — ToolAwareWorkflowExecutor with live SQLQueryTool."""
    def run(enriched_question: str):
        # Parse db_id from the enriched prompt to point SQL tool at correct DB
        import re
        match = re.search(r"Database:\s*(\S+)", enriched_question)
        db_id = match.group(1).strip() if match else "unknown"
        db_path = os.path.join(db_root, db_id, f"{db_id}.db")
        db_url  = f"sqlite:///{db_path}"

        registry = ToolRegistry()
        if os.path.exists(db_path):
            registry.register("SQL_QUERY", SQLQueryTool(db_url=db_url, read_only=True))
        else:
            registry.register("SQL_QUERY", MockSQLQueryTool())
        registry.register("WEB_SEARCH", MockWebSearchTool())

        executor = ToolAwareWorkflowExecutor(
            problem_description=enriched_question,
            designer_service=designer_service,
            executor_service=executor_service,
            tool_registry=registry,
            save_design_history=True,
            max_tool_retries=2,
        )
        try:
            answer, trajectory = executor.run(max_steps=4)
            history = getattr(executor.state, "design_history", [])
            return answer or "", history
        except Exception as e:
            return f"ERROR: {e}", []
    return run


# ── Report builder ─────────────────────────────────────────────────────────────

def build_report(
    dyflow_results:   list,
    dyflow_t_results: list,
    args,
    elapsed_dyflow:   float,
    elapsed_dyflow_t: float,
    dataset:          str = "spider",
) -> dict:
    """Merge two result lists into a comparison report."""

    def summarise(results: list) -> dict:
        total = correct = errors = 0
        hardness: dict = {}
        for r in results:
            if r.get("error") and not r.get("predicted_sql"):
                errors += 1
                continue
            total += 1
            h = r.get("hardness", "unknown")
            hardness.setdefault(h, {"total": 0, "correct": 0})
            hardness[h]["total"] += 1
            if r.get("judge_result", False):
                correct += 1
                hardness[h]["correct"] += 1
        acc = correct / total if total else 0.0
        return {
            "total":              total,
            "correct":            correct,
            "errors":             errors,
            "execution_accuracy": round(acc, 4),
            "hardness_accuracy": {
                h: round(s["correct"] / s["total"], 4) if s["total"] else 0.0
                for h, s in hardness.items()
            },
            "hardness_counts": hardness,
        }

    dyflow_summary   = summarise(dyflow_results)
    dyflow_t_summary = summarise(dyflow_t_results)

    # Build per-question comparison
    question_map = {}
    for r in dyflow_results:
        key = r.get("db_id", "") + "|" + r.get("question", "")[:60]
        question_map[key] = {"dyflow": r}
    for r in dyflow_t_results:
        key = r.get("db_id", "") + "|" + r.get("question", "")[:60]
        if key not in question_map:
            question_map[key] = {}
        question_map[key]["dyflow_t"] = r

    per_question = []
    dyflow_wins = dyflow_t_wins = both_correct = both_wrong = 0
    for key, pair in question_map.items():
        df  = pair.get("dyflow", {})
        dft = pair.get("dyflow_t", {})
        df_ok  = df.get("judge_result", False)
        dft_ok = dft.get("judge_result", False)

        if df_ok and dft_ok:   both_correct  += 1
        elif not df_ok and not dft_ok: both_wrong += 1
        elif dft_ok and not df_ok: dyflow_t_wins += 1
        elif df_ok and not dft_ok: dyflow_wins   += 1

        per_question.append({
            "db_id":             df.get("db_id", dft.get("db_id", "")),
            "question":          df.get("question", dft.get("question", "")),
            "hardness":          df.get("hardness", dft.get("hardness", "")),
            "gold_sql":          df.get("gold_sql", dft.get("gold_sql", "")),
            "dyflow_correct":    df_ok,
            "dyflow_t_correct":  dft_ok,
            "dyflow_sql":        df.get("predicted_sql", ""),
            "dyflow_t_sql":      dft.get("predicted_sql", ""),
            "dyflow_error":      df.get("execution_error", ""),
            "dyflow_t_error":    dft.get("execution_error", ""),
            "outcome":           (
                "both_correct"    if df_ok  and dft_ok  else
                "dyflow_t_only"   if dft_ok and not df_ok else
                "dyflow_only"     if df_ok  and not dft_ok else
                "both_wrong"
            ),
        })

    total_q = max(len(dyflow_results), len(dyflow_t_results))
    delta = dyflow_t_summary["execution_accuracy"] - dyflow_summary["execution_accuracy"]

    report = {
        "meta": {
            "timestamp":  str(datetime.datetime.now()),
            "dataset":    dataset.upper(),
            "mode":       args.mode if not args.sample else "sample",
            "size":       args.size or "all",
            "model":      "gemini-2.5-flash",
        },
        "summary": {
            "dyflow": {
                **dyflow_summary,
                "elapsed_sec": round(elapsed_dyflow, 1),
                "label": "DyFlow (no tools)",
            },
            "dyflow_t": {
                **dyflow_t_summary,
                "elapsed_sec": round(elapsed_dyflow_t, 1),
                "label": "DyFlow-T (SQL tool)",
            },
            "delta_execution_accuracy": round(delta, 4),
            "improvement_pct":          round(delta * 100, 2),
        },
        "outcome_breakdown": {
            "total_questions":  total_q,
            "both_correct":     both_correct,
            "dyflow_t_only":    dyflow_t_wins,
            "dyflow_only":      dyflow_wins,
            "both_wrong":       both_wrong,
        },
        "per_question": per_question,
    }
    return report


def print_report(report: dict) -> None:
    s = report["summary"]
    df  = s["dyflow"]
    dft = s["dyflow_t"]
    ob  = report["outcome_breakdown"]

    print("\n" + "=" * 65)
    print("SPIDER COMPARISON: DyFlow vs DyFlow-T")
    print("=" * 65)
    print(f"{'Metric':<30} {'DyFlow':>12} {'DyFlow-T':>12}")
    print("-" * 65)
    print(f"{'Execution Accuracy':<30} {df['execution_accuracy']:>11.2%} {dft['execution_accuracy']:>11.2%}")
    print(f"{'Correct':<30} {df['correct']:>12} {dft['correct']:>12}")
    print(f"{'Total evaluated':<30} {df['total']:>12} {dft['total']:>12}")
    print(f"{'Elapsed (sec)':<30} {df['elapsed_sec']:>12} {dft['elapsed_sec']:>12}")
    print("-" * 65)

    # Per-hardness: Spider labels + BIRD labels
    all_levels = sorted(set(
        list(df["hardness_accuracy"].keys()) +
        list(dft["hardness_accuracy"].keys())
    ))
    for h in ["easy", "simple", "medium", "moderate", "hard", "challenging", "extra", "unknown"]:
        if h in all_levels:
            dfa  = df["hardness_accuracy"].get(h, 0.0)
            dfta = dft["hardness_accuracy"].get(h, 0.0)
            print(f"  {h:<28} {dfa:>11.2%} {dfta:>11.2%}")

    print("-" * 65)
    delta = s["delta_execution_accuracy"]
    sign  = "+" if delta >= 0 else ""
    print(f"  DyFlow-T improvement: {sign}{delta:.2%}  ({sign}{s['improvement_pct']}pp)")
    print("-" * 65)
    print(f"  Both correct:    {ob['both_correct']}")
    print(f"  DyFlow-T only:   {ob['dyflow_t_only']}  ← tool helped")
    print(f"  DyFlow only:     {ob['dyflow_only']}   ← tool hurt")
    print(f"  Both wrong:      {ob['both_wrong']}")
    print("=" * 65)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_comparison(args):
    import time

    dataset   = args.dataset.lower()
    data_dir  = os.path.join(REPO_ROOT, "benchmarks", "data",
                             "BIRD" if dataset == "bird" else "Spider")

    if args.sample:
        db_root      = SAMPLE_DIR
        dataset_path = os.path.join(SAMPLE_DIR, "spider_sample.json")
        mode_label   = "sample"
        dataset      = "spider"
        if not os.path.exists(dataset_path):
            print("[ERROR] Sample DB not found. Run:")
            print("        python scripts/install_spider.py --sample-only")
            sys.exit(1)
    else:
        db_root      = os.path.join(data_dir, "databases")
        dataset_path = os.path.join(data_dir, f"{dataset}_{args.mode}.json")
        mode_label   = args.mode
        if not os.path.exists(dataset_path):
            installer = "install_bird.py" if dataset == "bird" else "install_spider.py"
            print(f"[ERROR] Dataset not found: {dataset_path}")
            print(f"        Run: python scripts/{installer}")
            sys.exit(1)

    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 65)
    print(f"{dataset.upper()} Comparison: DyFlow (no tool) vs DyFlow-T (SQL tool)")
    print("=" * 65)
    if dataset == "bird":
        print("BIRD advantage for tool comparison:")
        print("  • Real databases with up to 33,000 rows")
        print("  • LLM cannot hallucinate correct COUNT/SUM/AVG over real data")
        print("  • Tool execution gap expected to be larger than Spider")
    print(f"Mode     : {mode_label}")
    print(f"Size     : {args.size or 'all'}")
    print(f"Workers  : {args.workers}")
    print("=" * 65)

    designer_service = ModelService(model="gemini-2.5-flash")
    executor_service = ModelService(model="gemini-2.5-flash")

    # ── Run DyFlow (no tools) ──────────────────────────────────────────────────
    print("\n[1/2] Running DyFlow (original — no tools)...")
    bench_df = SpiderBenchmark(
        execution_model="gemini-2.5-flash",
        baseline="DyFlow",
        mode=mode_label,
        db_root=db_root,
        dataset=dataset,
    )
    bench_df.dataset_path = dataset_path
    bench_df.output_path  = os.path.join(REPORT_DIR, f"dyflow_{dataset}_{mode_label}.json")
    bench_df.cost_path    = os.path.join(REPORT_DIR, f"dyflow_{dataset}_{mode_label}_cost.json")
    bench_df.metrics_path = os.path.join(REPORT_DIR, f"dyflow_{dataset}_{mode_label}_metrics.json")

    t0 = time.time()
    dyflow_results = bench_df.evaluate_all_problems(
        function=make_dyflow_fn(designer_service, executor_service),
        max_workers=args.workers,
        size=args.size,
    )
    elapsed_dyflow = time.time() - t0
    with open(bench_df.output_path, "w") as f:
        json.dump(dyflow_results, f, indent=2)

    # ── Run DyFlow-T (with SQL tool) ───────────────────────────────────────────
    print("\n[2/2] Running DyFlow-T (tool-augmented — SQL tool)...")
    bench_dft = SpiderBenchmark(
        execution_model="gemini-2.5-flash",
        baseline="DyFlow-T",
        mode=mode_label,
        db_root=db_root,
        dataset=dataset,
    )
    bench_dft.dataset_path = dataset_path
    bench_dft.output_path  = os.path.join(REPORT_DIR, f"dyflow_t_{dataset}_{mode_label}.json")
    bench_dft.cost_path    = os.path.join(REPORT_DIR, f"dyflow_t_{dataset}_{mode_label}_cost.json")
    bench_dft.metrics_path = os.path.join(REPORT_DIR, f"dyflow_t_{dataset}_{mode_label}_metrics.json")

    t0 = time.time()
    dyflow_t_results = bench_dft.evaluate_all_problems(
        function=make_dyflow_t_fn(designer_service, executor_service, db_root),
        max_workers=args.workers,
        size=args.size,
    )
    elapsed_dyflow_t = time.time() - t0
    with open(bench_dft.output_path, "w") as f:
        json.dump(dyflow_t_results, f, indent=2)

    # ── Build and save report ──────────────────────────────────────────────────
    report = build_report(
        dyflow_results, dyflow_t_results, args,
        elapsed_dyflow, elapsed_dyflow_t,
        dataset=dataset,
    )

    report_path = os.path.join(
        REPORT_DIR,
        f"comparison_{dataset}_{mode_label}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_report(report)
    print(f"\nReport saved → {report_path}")
    return report_path


def show_latest_report(args):
    """Print metrics from the most recent comparison report."""
    if not os.path.isdir(REPORT_DIR):
        print("[ERROR] No reports found. Run the comparison first.")
        sys.exit(1)
    reports = sorted(
        [f for f in os.listdir(REPORT_DIR) if f.startswith("comparison_")],
        reverse=True,
    )
    if not reports:
        print("[ERROR] No comparison reports found in", REPORT_DIR)
        sys.exit(1)
    path = os.path.join(REPORT_DIR, reports[0])
    print(f"Loading: {path}")
    with open(path) as f:
        report = json.load(f)
    print_report(report)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare DyFlow vs DyFlow-T on Spider Text-to-SQL"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Use sample e_commerce DB (no Spider download needed)",
    )
    parser.add_argument(
        "--mode", type=str, default="dev", choices=["dev", "train"],
        help="Spider split to evaluate (default: dev)",
    )
    parser.add_argument(
        "--dataset", type=str, default="spider", choices=["spider", "bird"],
        help="Which dataset to compare on (default: spider)",
    )
    parser.add_argument(
        "--size", type=int, default=None,
        help="Number of questions per system (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Parallel workers per system (default: 3)",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Print metrics from the last saved report without running evaluation",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete any cached temp files and re-run both systems from scratch",
    )
    args = parser.parse_args()

    if args.force:
        import glob
        for f in glob.glob(os.path.join(REPORT_DIR, "temp_*.json")):
            os.remove(f)
            print(f"[force] Removed cache: {f}")

    if args.report_only:
        show_latest_report(args)
    else:
        run_comparison(args)
