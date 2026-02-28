#!/usr/bin/env python3
"""
compare_gaia.py — DyFlow vs DyFlow-T comparison on the GAIA benchmark.

GAIA tests real-world question answering that requires:
  - Current factual knowledge (today's prices, recent events, live data)
  - Multi-step web research (finding a paper, then reading its date)
  - Tool-grounded answers that cannot be hallucinated

Why this proves the tool's value:
  DyFlow (no tool) : must rely on parametric LLM memory — outdated, unreliable
  DyFlow-T (web tool): fetches real current information — grounded answers

e.g. "What is the USD/EUR rate today?"
  DyFlow:   "The rate is approximately 0.92" ← hallucinated assumption
  DyFlow-T: searches web → "0.9187" ← real current value

Usage
─────
    python scripts/compare_gaia.py --size 10 --workers 2
    python scripts/compare_gaia.py --mode validation --size 50 --workers 5
    python scripts/compare_gaia.py --report-only

Requirements
────────────
    GAIA dataset at benchmarks/data/GAIA/GAIA_validation.json
    Download: https://huggingface.co/datasets/gaia-benchmark/GAIA
    SERPAPI_API_KEY set in .env for live web search
"""

import sys
import os
import json
import datetime
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

from dyflow.model_service import ModelService
from dyflow.core.workflow import WorkflowExecutor
from dyflow.core.tool_workflow import ToolAwareWorkflowExecutor
from dyflow.tools.registry import ToolRegistry
from dyflow.tools.web_search import WebSearchTool, MockWebSearchTool
from dyflow.tools.sql_query import MockSQLQueryTool
from benchmarks.gaia import GAIABenchmark

REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR    = os.path.join(REPO_ROOT, "benchmarks", "data", "GAIA")
REPORT_DIR  = os.path.join(REPO_ROOT, "benchmarks", "results", "GAIA", "comparison")


# ── Workflow factories ─────────────────────────────────────────────────────────

def make_dyflow_fn(designer: ModelService, executor: ModelService):
    """
    DyFlow baseline — original WorkflowExecutor with NO web search tool.
    Must rely entirely on LLM parametric memory to answer questions.
    For current-knowledge questions this is unreliable and often wrong.
    """
    def run(question: str):
        wf = WorkflowExecutor(
            problem_description=question,
            designer_service=designer,
            executor_service=executor,
            save_design_history=True,
        )
        try:
            answer  = wf.execute()
            history = getattr(wf.state, "design_history", [])
            return str(answer) if answer else "", history
        except Exception as e:
            return f"ERROR: {e}", []
    return run


def make_dyflow_t_fn(designer: ModelService, executor: ModelService):
    """
    DyFlow-T — ToolAwareWorkflowExecutor with live WebSearchTool.
    Fetches real current information from the web before answering.
    For current-knowledge questions this is grounded and reliable.
    """
    serpapi_key = os.getenv("SERPAPI_API_KEY", "") or os.getenv("SERPER_API_KEY", "")

    def build_registry():
        registry = ToolRegistry()
        if serpapi_key:
            print(f"  [Tools] WebSearchTool → live (SerpAPI) key=...{serpapi_key[-6:]}")
            registry.register("WEB_SEARCH", WebSearchTool(api_key=serpapi_key))
        else:
            print("  [Tools] WebSearchTool → mock ⚠  Set SERPAPI_API_KEY for live search")
            registry.register("WEB_SEARCH", MockWebSearchTool())
        registry.register("SQL_QUERY", MockSQLQueryTool())
        return registry

    def run(question: str):
        wf = ToolAwareWorkflowExecutor(
            problem_description=question,
            designer_service=designer,
            executor_service=executor,
            tool_registry=build_registry(),
            save_design_history=True,
            max_tool_retries=2,
        )
        try:
            answer, trajectory = wf.run(max_steps=8)
            history = getattr(wf.state, "design_history", [])
            return answer or "", history
        except Exception as e:
            return f"ERROR: {e}", []
    return run


# ── Report builder ─────────────────────────────────────────────────────────────

def summarise(results: list) -> dict:
    total = correct = errors = 0
    levels: dict = {}
    for r in results:
        if r.get("error") and not r.get("generated_solution"):
            errors += 1
            continue
        total += 1
        lv = str(r.get("level", "unknown"))
        levels.setdefault(lv, {"total": 0, "correct": 0})
        levels[lv]["total"] += 1
        if r.get("judge_result", False):
            correct += 1
            levels[lv]["correct"] += 1
    return {
        "total":    total,
        "correct":  correct,
        "errors":   errors,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "level_accuracy": {
            lv: round(s["correct"] / s["total"], 4) if s["total"] else 0.0
            for lv, s in levels.items()
        },
        "level_counts": levels,
    }


def build_report(df_res, dft_res, args, t_df, t_dft) -> dict:
    df_sum  = summarise(df_res)
    dft_sum = summarise(dft_res)

    # Per-question comparison
    qmap = {}
    for r in df_res:
        k = r.get("task_id", r.get("question","")[:60])
        qmap[k] = {"dyflow": r}
    for r in dft_res:
        k = r.get("task_id", r.get("question","")[:60])
        qmap.setdefault(k, {})["dyflow_t"] = r

    both_ok = dft_only = df_only = both_wrong = 0
    per_q   = []
    for k, pair in qmap.items():
        df  = pair.get("dyflow",   {})
        dft = pair.get("dyflow_t", {})
        df_ok  = df.get("judge_result",  False)
        dft_ok = dft.get("judge_result", False)

        if   df_ok and dft_ok:        both_ok   += 1
        elif dft_ok and not df_ok:    dft_only  += 1
        elif df_ok  and not dft_ok:   df_only   += 1
        else:                         both_wrong += 1

        per_q.append({
            "task_id":          k,
            "question":         df.get("question",  dft.get("question",  "")),
            "level":            df.get("level",     dft.get("level",     "")),
            "ground_truth":     df.get("final_answer", dft.get("final_answer", "")),
            "dyflow_correct":   df_ok,
            "dyflow_t_correct": dft_ok,
            "dyflow_answer":    df.get("extracted_answer",  ""),
            "dyflow_t_answer":  dft.get("extracted_answer", ""),
            "outcome": (
                "both_correct"  if df_ok  and dft_ok  else
                "dyflow_t_only" if dft_ok and not df_ok else
                "dyflow_only"   if df_ok  and not dft_ok else
                "both_wrong"
            ),
        })

    delta = dft_sum["accuracy"] - df_sum["accuracy"]

    return {
        "meta": {
            "timestamp": str(datetime.datetime.now()),
            "dataset":   "GAIA",
            "mode":      args.mode,
            "size":      args.size or "all",
            "model":     "gemini-2.5-flash",
            "web_search_live": bool(
                os.getenv("SERPAPI_API_KEY", "") or os.getenv("SERPER_API_KEY", "")
            ),
        },
        "summary": {
            "dyflow": {
                **df_sum,
                "elapsed_sec": round(t_df, 1),
                "label": "DyFlow (no tools — relies on LLM memory)",
            },
            "dyflow_t": {
                **dft_sum,
                "elapsed_sec": round(t_dft, 1),
                "label": "DyFlow-T (WebSearchTool — fetches real current info)",
            },
            "delta_accuracy":    round(delta, 4),
            "improvement_pct":   round(delta * 100, 2),
        },
        "outcome_breakdown": {
            "total_questions": max(len(df_res), len(dft_res)),
            "both_correct":    both_ok,
            "dyflow_t_only":   dft_only,
            "dyflow_only":     df_only,
            "both_wrong":      both_wrong,
        },
        "key_insight": (
            "GAIA questions require current or niche factual knowledge "
            "the LLM cannot reliably recall. DyFlow hallucinates plausible "
            "but wrong answers. DyFlow-T fetches real information via "
            "WebSearchTool and grounds its answer in actual retrieved content."
        ),
        "per_question": per_q,
    }


def print_report(report: dict) -> None:
    s    = report["summary"]
    df   = s["dyflow"]
    dft  = s["dyflow_t"]
    ob   = report["outcome_breakdown"]
    delta = s["delta_accuracy"]
    sign  = "+" if delta >= 0 else ""
    live  = report["meta"].get("web_search_live", False)

    print("\n" + "=" * 68)
    print("GAIA COMPARISON: DyFlow (no tool) vs DyFlow-T (web search)")
    print("=" * 68)
    print(f"  Web search: {'LIVE (SerpAPI)' if live else '⚠  MOCK (set SERPAPI_API_KEY)'}")
    print(f"{'Metric':<32} {'DyFlow':>14} {'DyFlow-T':>14}")
    print("-" * 68)
    print(f"{'Accuracy':<32} {df['accuracy']:>13.2%} {dft['accuracy']:>13.2%}")
    print(f"{'Correct':<32} {df['correct']:>14} {dft['correct']:>14}")
    print(f"{'Total evaluated':<32} {df['total']:>14} {dft['total']:>14}")
    print(f"{'Elapsed (sec)':<32} {df['elapsed_sec']:>14} {dft['elapsed_sec']:>14}")
    print("-" * 68)

    # Per-level breakdown
    all_levels = sorted(set(
        list(df["level_accuracy"].keys()) +
        list(dft["level_accuracy"].keys())
    ))
    for lv in all_levels:
        da  = df["level_accuracy"].get(lv, 0.0)
        dta = dft["level_accuracy"].get(lv, 0.0)
        n   = df["level_counts"].get(lv, {}).get("total", 0)
        print(f"  Level {lv:<26} {da:>13.2%} {dta:>13.2%}  (n={n})")

    print("-" * 68)
    print(f"  Tool improvement:    {sign}{delta:.2%}  ({sign}{s['improvement_pct']}pp)")
    print("-" * 68)
    print(f"  Both correct:        {ob['both_correct']}")
    print(f"  DyFlow-T only:       {ob['dyflow_t_only']}  ← web search helped")
    print(f"  DyFlow only:         {ob['dyflow_only']}   ← web search hurt")
    print(f"  Both wrong:          {ob['both_wrong']}")
    print("-" * 68)
    print(f"  Key insight:")
    insight = report.get("key_insight", "")
    # Word-wrap at 64 chars
    words, line = insight.split(), ""
    for w in words:
        if len(line) + len(w) + 1 > 64:
            print(f"    {line}")
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        print(f"    {line}")
    print("=" * 68)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_comparison(args):
    dataset_path = os.path.join(DATA_DIR, f"GAIA_{args.mode}.json")
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print("  Download from: https://huggingface.co/datasets/gaia-benchmark/GAIA")
        print(f"  Place at: {dataset_path}")
        sys.exit(1)

    os.makedirs(REPORT_DIR, exist_ok=True)

    if args.force:
        import glob
        for f in glob.glob(os.path.join(REPORT_DIR, "temp_*.json")):
            os.remove(f)
            print(f"[force] Removed cache: {f}")

    print("=" * 68)
    print("GAIA Comparison: DyFlow vs DyFlow-T")
    print("=" * 68)
    print(f"Mode     : {args.mode}")
    print(f"Size     : {args.size or 'all'}")
    print(f"Workers  : {args.workers}")
    serpapi_key = os.getenv("SERPAPI_API_KEY", "") or os.getenv("SERPER_API_KEY", "")
    print(f"Web search: {'live (SerpAPI)' if serpapi_key else '⚠  mock — set SERPAPI_API_KEY'}")
    print("Why GAIA : Current facts the LLM cannot reliably recall")
    print("=" * 68)

    designer = ModelService(model="gemini-2.5-flash")
    executor = ModelService(model="gemini-2.5-flash")

    def _make_bench(baseline):
        b = GAIABenchmark(
            execution_model="gemini-2.5-flash",
            baseline=baseline,
            mode=args.mode,
        )
        b.dataset_path = dataset_path
        b.output_path  = os.path.join(REPORT_DIR, f"{baseline.lower()}_{args.mode}.json")
        b.cost_path    = os.path.join(REPORT_DIR, f"{baseline.lower()}_{args.mode}_cost.json")
        b.metrics_path = os.path.join(REPORT_DIR, f"{baseline.lower()}_{args.mode}_metrics.json")
        return b

    judge_service = ModelService(model="gemini-2.5-flash")

    # ── Run DyFlow (no tools) ──────────────────────────────────────────────────
    print("\n[1/2] DyFlow (no tools — relies on LLM parametric memory)...")
    bench_df = _make_bench("DyFlow")
    t0 = time.time()
    df_results = bench_df.evaluate_all_problems(
        function=make_dyflow_fn(designer, executor),
        judge_service=judge_service,
        max_workers=args.workers,
        size=args.size,
    )
    t_df = time.time() - t0
    with open(bench_df.output_path, "w", encoding="utf-8") as f:
        json.dump(df_results, f, indent=2, ensure_ascii=False)

    # ── Run DyFlow-T (web search) ──────────────────────────────────────────────
    print("\n[2/2] DyFlow-T (WebSearchTool — fetches real current info)...")
    bench_dft = _make_bench("DyFlow-T")
    t0 = time.time()
    dft_results = bench_dft.evaluate_all_problems(
        function=make_dyflow_t_fn(designer, executor),
        judge_service=judge_service,
        max_workers=args.workers,
        size=args.size,
    )
    t_dft = time.time() - t0
    with open(bench_dft.output_path, "w", encoding="utf-8") as f:
        json.dump(dft_results, f, indent=2, ensure_ascii=False)

    # ── Build and save report ──────────────────────────────────────────────────
    report = build_report(df_results, dft_results, args, t_df, t_dft)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"comparison_{args.mode}_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_report(report)
    print(f"\nReport → {report_path}")


def show_latest(args):
    if not os.path.isdir(REPORT_DIR):
        print("[ERROR] No reports found. Run the comparison first.")
        sys.exit(1)
    reports = sorted(
        [f for f in os.listdir(REPORT_DIR) if f.startswith("comparison_")],
        reverse=True,
    )
    if not reports:
        print(f"[ERROR] No comparison reports in {REPORT_DIR}")
        sys.exit(1)
    path = os.path.join(REPORT_DIR, reports[0])
    print(f"Loading: {path}")
    print_report(json.load(open(path)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare DyFlow vs DyFlow-T on GAIA benchmark"
    )
    p.add_argument(
        "--mode", default="validation", choices=["validation", "test"],
        help="GAIA split to evaluate (default: validation)",
    )
    p.add_argument(
        "--size", type=int, default=None,
        help="Number of questions per system (default: all)",
    )
    p.add_argument(
        "--workers", type=int, default=2,
        help="Parallel workers per system (default: 2)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Clear cached temp files and re-run from scratch",
    )
    p.add_argument(
        "--report-only", action="store_true",
        help="Print metrics from last saved report without running",
    )
    args = p.parse_args()

    if args.report_only:
        show_latest(args)
    else:
        run_comparison(args)
