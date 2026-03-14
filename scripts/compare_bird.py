#!/usr/bin/env python3
"""
compare_bird.py — DyFlow vs DyFlow-T comparison on BIRD benchmark.

BIRD is a better benchmark than Spider for proving the tool's value because:
  - Databases have 33k+ rows — the LLM cannot guess aggregate results
  - Questions require joins across dirty, real-world data
  - Exchange rates, counts, sums — impossible without executing the query

Usage
─────
    # Sample finance DB (instant, no download)
    python scripts/compare_bird.py --sample

    # BIRD dev split, 50 questions per system
    python scripts/compare_bird.py --mode dev --size 50 --workers 3

    # View last saved report
    python scripts/compare_bird.py --report-only
"""

import sys, os, json, datetime, argparse, time, re
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
from dyflow.tools.web_search import MockWebSearchTool
from dyflow.tools.sql_query import SQLQueryTool, MockSQLQueryTool
from benchmarks.spider import SpiderBenchmark

REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR    = os.path.join(REPO_ROOT, "benchmarks", "data", "BIRD")
SAMPLE_DIR  = os.path.join(DATA_DIR, "sample")
REPORT_DIR  = os.path.join(REPO_ROOT, "benchmarks", "results", "BIRD", "comparison")


# ── Workflow factories ─────────────────────────────────────────────────────────

def make_dyflow_fn(designer, executor):
    """DyFlow baseline — original WorkflowExecutor, no SQL execution tool."""
    def run(enriched_question: str):
        wf = WorkflowExecutor(
            problem_description=enriched_question,
            designer_service=designer,
            executor_service=executor,
            save_design_history=True,
        )
        try:
            answer   = wf.execute()
            history  = getattr(wf.state, "design_history", [])
            return str(answer) if answer else "", history
        except Exception as e:
            return f"ERROR: {e}", []
    return run


def make_dyflow_t_fn(designer, executor, db_root: str):
    """DyFlow-T — same LLM, but SQL_QUERY executes against real database."""
    def run(enriched_question: str):
        match   = re.search(r"Database:\s*(\S+)", enriched_question)
        db_id   = match.group(1).strip() if match else "unknown"
        db_path = os.path.join(db_root, db_id, f"{db_id}.db")
        db_url  = f"sqlite:///{db_path}"

        registry = ToolRegistry()
        registry.register(
            "SQL_QUERY",
            SQLQueryTool(db_url=db_url, read_only=True) if os.path.exists(db_path)
            else MockSQLQueryTool()
        )
        registry.register("WEB_SEARCH", MockWebSearchTool())

        wf = ToolAwareWorkflowExecutor(
            problem_description=enriched_question,
            designer_service=designer,
            executor_service=executor,
            tool_registry=registry,
            save_design_history=True,
            max_tool_retries=2,
        )
        try:
            answer, traj = wf.run(max_steps=4)
            history = getattr(wf.state, "design_history", [])
            return answer or "", history
        except Exception as e:
            return f"ERROR: {e}", []
    return run


# ── Report builder ─────────────────────────────────────────────────────────────

def summarise(results: list) -> dict:
    total = correct = errors = 0
    hardness: dict = {}
    for r in results:
        if r.get("error") and not r.get("predicted_sql"):
            errors += 1; continue
        total += 1
        h = r.get("hardness", "unknown")
        hardness.setdefault(h, {"total":0,"correct":0})
        hardness[h]["total"] += 1
        if r.get("judge_result", False):
            correct += 1
            hardness[h]["correct"] += 1
    return {
        "total": total, "correct": correct, "errors": errors,
        "execution_accuracy": round(correct/total,4) if total else 0.0,
        "hardness_accuracy": {
            h: round(s["correct"]/s["total"],4) if s["total"] else 0.0
            for h,s in hardness.items()
        },
        "hardness_counts": hardness,
    }


def build_report(df_res, dft_res, args, t_df, t_dft) -> dict:
    df_sum  = summarise(df_res)
    dft_sum = summarise(dft_res)

    # Per-question outcome breakdown
    qmap = {}
    for r in df_res:
        k = r.get("db_id","") + "|" + r.get("question","")[:60]
        qmap[k] = {"dyflow": r}
    for r in dft_res:
        k = r.get("db_id","") + "|" + r.get("question","")[:60]
        qmap.setdefault(k, {})["dyflow_t"] = r

    both_ok = dft_only = df_only = both_wrong = 0
    per_q = []
    for k, pair in qmap.items():
        df  = pair.get("dyflow",  {})
        dft = pair.get("dyflow_t",{})
        df_ok  = df.get("judge_result",  False)
        dft_ok = dft.get("judge_result", False)
        if df_ok and dft_ok:        both_ok   += 1
        elif dft_ok and not df_ok:  dft_only  += 1
        elif df_ok  and not dft_ok: df_only   += 1
        else:                       both_wrong+= 1
        per_q.append({
            "db_id":           df.get("db_id",  dft.get("db_id","")),
            "question":        df.get("question",dft.get("question","")),
            "hardness":        df.get("hardness",dft.get("hardness","")),
            "evidence":        df.get("evidence",dft.get("evidence","")),
            "gold_sql":        df.get("gold_sql", dft.get("gold_sql","")),
            "dyflow_correct":  df_ok,
            "dyflow_t_correct":dft_ok,
            "dyflow_sql":      df.get("predicted_sql",""),
            "dyflow_t_sql":    dft.get("predicted_sql",""),
            "dyflow_error":    df.get("execution_error",""),
            "dyflow_t_error":  dft.get("execution_error",""),
            "outcome": (
                "both_correct"  if df_ok  and dft_ok  else
                "dyflow_t_only" if dft_ok and not df_ok else
                "dyflow_only"   if df_ok  and not dft_ok else
                "both_wrong"
            ),
        })

    delta = dft_sum["execution_accuracy"] - df_sum["execution_accuracy"]
    return {
        "meta": {
            "timestamp": str(datetime.datetime.now()),
            "dataset": "BIRD", "mode": args.mode,
            "size": args.size or "all", "model": "gemini-2.5-flash",
        },
        "summary": {
            "dyflow":   {**df_sum,  "elapsed_sec": round(t_df,1),  "label": "DyFlow (no tools — LLM guesses SQL results)"},
            "dyflow_t": {**dft_sum, "elapsed_sec": round(t_dft,1), "label": "DyFlow-T (SQL tool — real database execution)"},
            "delta_execution_accuracy": round(delta,4),
            "improvement_pct":          round(delta*100,2),
        },
        "outcome_breakdown": {
            "total_questions": max(len(df_res),len(dft_res)),
            "both_correct":    both_ok,
            "dyflow_t_only":   dft_only,
            "dyflow_only":     df_only,
            "both_wrong":      both_wrong,
        },
        "key_insight": (
            "DyFlow must guess SQL result rows from schema alone (unreliable for "
            "aggregates, joins, and domain-specific values). DyFlow-T executes "
            "the SQL against the real BIRD database and reads actual returned rows."
        ),
        "per_question": per_q,
    }


def print_report(report: dict) -> None:
    s   = report["summary"]
    df  = s["dyflow"]
    dft = s["dyflow_t"]
    ob  = report["outcome_breakdown"]
    delta = s["delta_execution_accuracy"]
    sign  = "+" if delta >= 0 else ""

    print("\n" + "=" * 68)
    print("BIRD COMPARISON: DyFlow (no tool) vs DyFlow-T (SQL execution)")
    print("=" * 68)
    print(f"{'Metric':<32} {'DyFlow':>14} {'DyFlow-T':>14}")
    print("-" * 68)
    print(f"{'Execution Accuracy':<32} {df['execution_accuracy']:>13.2%} {dft['execution_accuracy']:>13.2%}")
    print(f"{'Correct':<32} {df['correct']:>14} {dft['correct']:>14}")
    print(f"{'Total evaluated':<32} {df['total']:>14} {dft['total']:>14}")
    print(f"{'Elapsed (sec)':<32} {df['elapsed_sec']:>14} {dft['elapsed_sec']:>14}")
    print("-" * 68)

    for h in ("simple","moderate","challenging","unknown"):
        da  = df["hardness_accuracy"].get(h)
        dta = dft["hardness_accuracy"].get(h)
        if da is not None or dta is not None:
            da  = da  or 0.0
            dta = dta or 0.0
            dc  = df["hardness_counts"].get(h,{}).get("total",0)
            print(f"  {h:<30} {da:>13.2%} {dta:>13.2%}  (n={dc})")

    print("-" * 68)
    print(f"  Tool improvement:    {sign}{delta:.2%}  ({sign}{s['improvement_pct']}pp)")
    print("-" * 68)
    print(f"  Both correct:        {ob['both_correct']}")
    print(f"  DyFlow-T only:       {ob['dyflow_t_only']}  ← tool execution helped")
    print(f"  DyFlow only:         {ob['dyflow_only']}   ← tool hurt (SQL error)")
    print(f"  Both wrong:          {ob['both_wrong']}")
    print("-" * 68)
    print(f"  Key insight: {report.get('key_insight','')[:70]}...")
    print("=" * 68)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_comparison(args):
    if args.sample:
        db_root      = SAMPLE_DIR
        dataset_path = os.path.join(SAMPLE_DIR, "bird_sample.json")
        mode_label   = "sample"
        if not os.path.exists(dataset_path):
            print("[ERROR] Run first: python scripts/install_bird.py --sample-only")
            sys.exit(1)
    else:
        db_root      = os.path.join(DATA_DIR, "databases")
        dataset_path = os.path.join(DATA_DIR, f"bird_{args.mode}.json")
        mode_label   = args.mode
        if not os.path.exists(dataset_path):
            print(f"[ERROR] Not found: {dataset_path}")
            print("  Run: python scripts/install_bird.py")
            sys.exit(1)

    os.makedirs(REPORT_DIR, exist_ok=True)

    if args.force:
        import glob
        for f in glob.glob(os.path.join(REPORT_DIR, "temp_*.json")):
            os.remove(f); print(f"[force] Removed: {f}")

    print("=" * 68)
    print("BIRD Comparison: DyFlow vs DyFlow-T")
    print("=" * 68)
    print(f"Dataset  : BIRD ({mode_label})")
    print(f"Size     : {args.size or 'all'}")
    print(f"Workers  : {args.workers}")
    print("Why BIRD : Large real DBs — LLM cannot guess aggregate answers")
    print("=" * 68)

    designer = ModelService(model="gemini-2.5-flash")
    executor = ModelService(model="gemini-2.5-flash")

    def _make_bench(baseline, out_suffix):
        b = SpiderBenchmark(
            execution_model="gemini-2.5-flash",
            baseline=baseline,
            mode=mode_label,
            dataset="bird",
            db_root=db_root,
        )
        b.dataset_path = dataset_path
        b.output_path  = os.path.join(REPORT_DIR, f"{out_suffix}_{mode_label}.json")
        b.cost_path    = os.path.join(REPORT_DIR, f"{out_suffix}_{mode_label}_cost.json")
        b.metrics_path = os.path.join(REPORT_DIR, f"{out_suffix}_{mode_label}_metrics.json")
        return b

    # ── Run DyFlow ─────────────────────────────────────────────────────────────
    print("\n[1/2] DyFlow (no tools — LLM must guess SQL result rows)...")
    bench_df = _make_bench("DyFlow", "dyflow")
    t0 = time.time()
    df_results = bench_df.evaluate_all_problems(
        function=make_dyflow_fn(designer, executor),
        max_workers=args.workers, size=args.size,
    )
    t_df = time.time() - t0
    with open(bench_df.output_path,"w") as f:
        json.dump(df_results, f, indent=2)

    # ── Run DyFlow-T ───────────────────────────────────────────────────────────
    print("\n[2/2] DyFlow-T (SQL tool — executes query against real DB)...")
    bench_dft = _make_bench("DyFlow-T", "dyflow_t")
    t0 = time.time()
    dft_results = bench_dft.evaluate_all_problems(
        function=make_dyflow_t_fn(designer, executor, db_root),
        max_workers=args.workers, size=args.size,
    )
    t_dft = time.time() - t0
    with open(bench_dft.output_path,"w") as f:
        json.dump(dft_results, f, indent=2)

    # ── Report ─────────────────────────────────────────────────────────────────
    report = build_report(df_results, dft_results, args, t_df, t_dft)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"comparison_{mode_label}_{ts}.json")
    with open(report_path,"w",encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_report(report)
    print(f"\nReport → {report_path}")


def show_latest(args):
    if not os.path.isdir(REPORT_DIR):
        print("No reports found."); sys.exit(1)
    reports = sorted(
        [f for f in os.listdir(REPORT_DIR) if f.startswith("comparison_")],
        reverse=True,
    )
    if not reports:
        print("No comparison reports found."); sys.exit(1)
    path = os.path.join(REPORT_DIR, reports[0])
    print(f"Loading: {path}")
    print_report(json.load(open(path)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare DyFlow vs DyFlow-T on BIRD benchmark"
    )
    p.add_argument("--sample",      action="store_true", help="Use sample finance DB")
    p.add_argument("--mode",        default="dev", choices=["dev","train"])
    p.add_argument("--size",        type=int,  default=None)
    p.add_argument("--workers",     type=int,  default=3)
    p.add_argument("--force",       action="store_true", help="Clear cached temp files")
    p.add_argument("--report-only", action="store_true", help="Print last saved report")
    args = p.parse_args()

    if args.report_only:
        show_latest(args)
    else:
        run_comparison(args)
